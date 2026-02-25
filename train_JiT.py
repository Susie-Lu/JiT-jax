"""
Train JiT
"""

import os
import logging
from flax import nnx
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid
import optax
import orbax.checkpoint as ocp
import functools
import numpy as np
from omegaconf import DictConfig
from collections import deque
import wandb
import json

from utils.input_pipeline_imagenet import create_imagenet_loaders, prepare_batch_data
from utils.sample_images import sample, process_image_for_save
from utils.fid_evaluate import get_fid
from utils.fid_utils import get_fid_network
from utils.util_ckpt import get_global_array

log = logging.getLogger(__name__)


def loss_fn(model, x, t, y, noise):
    z = t[:, None, None, None] * x + (1 - t)[:, None, None, None] * noise
    v = (x - z) / jnp.clip((1 - t)[:, None, None, None], min=0.05)
    v_pred = model.forward_x_to_v(z, t, y)
    return jnp.mean((v - v_pred) ** 2)


@functools.partial(jax.pmap, axis_name="G")
def compute_loss(
    model_state: nnx.State,
    model_def: nnx.GraphDef,
    replicated_non_trainable_params: nnx.State,
    x: jax.Array,
    t: jax.Array,
    y: jax.Array,
    noise: jax.Array,
):
    # Reconstruct the model from its state and static definition
    model = nnx.merge(model_def, model_state, replicated_non_trainable_params)
    loss = loss_fn(model, x, t, y, noise)
    return jax.lax.pmean(loss, axis_name="G")


def get_update_func(model_state, decay):
    warmup_steps = 5 * 1251
    schedule_fn = optax.linear_schedule(
        init_value=0.0, end_value=2e-4, transition_steps=warmup_steps
    )
    optimizer = optax.adamw(learning_rate=schedule_fn, b1=0.9, b2=0.95, weight_decay=0)
    opt_state = optimizer.init(model_state)

    @functools.partial(jax.pmap, axis_name="G")
    def update_parallel(
        model_state: nnx.State,
        model_def: nnx.GraphDef,
        replicated_non_trainable_params: nnx.State,
        opt_state: optax.OptState,
        replicated_ema_state: nnx.State,
        x: jax.Array,
        t: jax.Array,
        y: jax.Array,
        noise: jax.Array,
    ):
        model = nnx.merge(model_def, model_state, replicated_non_trainable_params)

        loss, grads = nnx.value_and_grad(loss_fn)(model, x, t, y, noise)

        # Combine gradients and loss across all devices by taking the mean
        grads = jax.lax.pmean(grads, axis_name="G")
        loss = jax.lax.pmean(loss, axis_name="G")

        # Each device performs its own update. Since gradients are synchronized,
        # the model state remains consistent across all devices.
        updates, new_opt_state = optimizer.update(grads, opt_state, model_state)
        new_model_state = optax.apply_updates(model_state, updates)

        new_ema_state = jax.tree.map(
            lambda ema_p, p: decay * ema_p + (1 - decay) * p,
            replicated_ema_state,
            new_model_state,
        )

        return new_model_state, new_opt_state, new_ema_state, loss

    return update_parallel, opt_state, schedule_fn


def get_num_param(model):
    """
    Returns total number of trainable parameters in nnx model
    """
    params = nnx.state(model, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    return total_params


def train(model, base_key, cfg: DictConfig):
    """
    Execute model training and evaluation loop, supporting multihost.

    Args:
        model: the JiT model to train
        base_key: the base jax.random key
        cfg: Hydra config
    """
    log.info(f"Process index: {jax.process_index()}")
    log.info(f"Process count: {jax.process_count()}")
    log.info(f"Local device count: {jax.local_device_count()}")
    log.info(f"Global device count: {jax.device_count()}")

    key = random.fold_in(base_key, jax.process_index())

    class_labels_unique = list(cfg.training.class_labels)
    class_labels = jnp.array(
        [num for num in class_labels_unique for _ in range(cfg.training.num_per_class)]
    )
    num_images = class_labels.shape[0]

    G = jax.local_device_count()

    batch_size_for_1_process = cfg.training.batch_size // jax.process_count()

    train_loader = create_imagenet_loaders(
        dataset_root=cfg.dataset.data_dir,
        split="train",
        batch_size=batch_size_for_1_process,
        image_size=cfg.dataset.image_size,
        num_workers=cfg.training.num_dataloader_workers,
        prefetch=cfg.training.prefetch,
    )

    eval_loader = create_imagenet_loaders(
        dataset_root=cfg.dataset.data_dir,
        split="val",
        batch_size=batch_size_for_1_process,
        image_size=cfg.dataset.image_size,
        num_workers=cfg.training.num_dataloader_workers,
        prefetch=cfg.training.prefetch,
        val_drop_last=True,
    )

    log.info(f"Train loader: {len(train_loader)}")
    log.info(f"Eval loader: {len(eval_loader)}")

    # Create EMA version of the model
    ema_model = nnx.clone(model)

    # Prepare models for training
    model.training = True
    ema_model.training = False

    if jax.process_index() == 0 and not os.path.exists(cfg.training.ckpt_dir):
        os.makedirs(cfg.training.ckpt_dir)
        log.info(f"Directory '{cfg.training.ckpt_dir}' created successfully.")

    # Create checkpoint manager
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.training.num_ckpt_kept,
        save_interval_steps=cfg.training.ckpt_every,
        step_prefix="epoch",
    )
    mngr = ocp.CheckpointManager(
        cfg.training.ckpt_dir,
        options=options,
        item_names=("model_state", "ema_state", "optimizer_state"),
    )

    log.info(f"Training for {cfg.training.num_epochs} epochs...")

    model_def, model_state, non_trainable_params = nnx.split(model, nnx.Param, ...)
    ema_def, ema_state, ema_non_trainable_params = nnx.split(ema_model, nnx.Param, ...)
    update_parallel, opt_state, schedule_fn = get_update_func(
        model_state, cfg.training.decay
    )

    # Restore checkpoint
    resuming_from_epoch = mngr.latest_step()

    if resuming_from_epoch is None:
        log.info("No checkpoints found. Starting from scratch.")
        resuming_from_epoch = 0
    else:
        log.info(f"Restoring from latest checkpoint: epoch {resuming_from_epoch}")

        abstract_model_state = jax.tree.map(
            ocp.utils.to_shape_dtype_struct, model_state
        )
        abstract_ema_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, ema_state)
        abstract_opt_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, opt_state)
        restored = mngr.restore(
            resuming_from_epoch,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(abstract_model_state),
                ema_state=ocp.args.StandardRestore(abstract_ema_state),
                optimizer_state=ocp.args.StandardRestore(abstract_opt_state),
            ),
        )

        model_state = restored.model_state
        ema_state = restored.ema_state
        opt_state = restored.optimizer_state
        model = nnx.merge(model_def, restored.model_state, non_trainable_params)
        ema_model = nnx.merge(ema_def, restored.ema_state, ema_non_trainable_params)

    # Replicate the model's state across all available devices
    # This adds a leading axis of size G to each parameter array
    replicated_state = jax.tree.map(lambda x: jnp.array([x] * G), model_state)
    replicated_non_trainable_params = jax.tree.map(
        lambda x: jnp.array([x] * G), non_trainable_params
    )
    replicated_opt_state = jax.tree.map(lambda x: jnp.array([x] * G), opt_state)
    replicated_ema_state = jax.tree.map(lambda x: jnp.array([x] * G), ema_state)

    # Set up FID evaluation utilities
    log.info("Loading Inception")
    get_activations = get_fid_network()
    log.info("Loading Inception: done")
    log.info("Loading FID ref stats")
    truth_stats = np.load(cfg.dataset.fid_stats_dir)
    key_fid = random.key(cfg.training.key_fid)

    # Set up wandb run
    if jax.process_index() == 0:
        wandb.login(key=cfg.wandb_info.key)

        if not os.path.exists(cfg.wandb_info.saved_run_id):
            run = wandb.init(
                entity=cfg.wandb_info.entity,
                project=cfg.wandb_info.project,
                name=cfg.wandb_info.name,
                dir=cfg.output_dir,
            )
            run_id = run.id

            with open(cfg.wandb_info.saved_run_id, "w") as f:
                json.dump({"run_id": run_id}, f)

            log.info(f"Started new wandb run {run_id}")
        else:
            log.info(f"Resuming wandb run")
            with open(cfg.wandb_info.saved_run_id, "r") as f:
                run_id = json.load(f)["run_id"]

            run = wandb.init(
                entity=cfg.wandb_info.entity,
                project=cfg.wandb_info.project,
                id=run_id,
                resume="allow",
                dir=cfg.output_dir,
            )

    num_param = get_num_param(model) / 1e6
    log.info(f"Number of trainable parameters: {num_param} x 10^6")
    if jax.process_index() == 0:
        wandb.log({"num_params": num_param}, step=0)

    for epoch in range(resuming_from_epoch + 1, cfg.training.num_epochs + 1):
        log.info(f"Beginning epoch {epoch}...")

        # For moving average of the loss
        loss_avg = deque(maxlen=cfg.training.log_loss_every)

        for data_iter_step, (x, y) in enumerate(train_loader):
            # Split the JAX rng key
            key, subkey_t = random.split(key)
            key, subkey_noise = random.split(key)

            # Shard the data batch across devices
            x, y = prepare_batch_data(x, y)

            # Logit-normal time step sampler
            z = random.normal(subkey_t, shape=(G, x.shape[1])) * 0.8 - 0.8
            t = sigmoid(z)
            noise = random.normal(subkey_noise, shape=x.shape)

            # Shapes:
            #    x, noise: (G, device_batch_size, H, W, C)
            #    y, t: (G, device_batch_size)

            # Perform the update in parallel
            replicated_state, replicated_opt_state, replicated_ema_state, loss = (
                update_parallel(
                    replicated_state,
                    model_def,
                    replicated_non_trainable_params,
                    replicated_opt_state,
                    replicated_ema_state,
                    x,
                    t,
                    y,
                    noise,
                )
            )
            loss_val = loss[0]

            if data_iter_step % cfg.training.log_every == 0:
                log.info(f"(epoch {epoch}) (step={data_iter_step}) done")

            # Log loss
            loss_avg.append(loss_val)
            if jax.process_index() == 0:
                if data_iter_step % cfg.training.log_loss_every == 0:
                    epoch_1000x = int(
                        (data_iter_step / len(train_loader) + epoch - 1) * 1000
                    )
                    actual_step = (epoch - 1) * len(train_loader) + data_iter_step
                    wandb.log(
                        {"train_loss": loss_val, "lr": schedule_fn(actual_step)},
                        step=epoch_1000x,
                    )
                    log.info(f"(epoch {epoch}) (step={data_iter_step}) loss {loss_val}")

                    if data_iter_step > 0:
                        avg100_loss = sum(loss_avg) / len(loss_avg)
                        wandb.log({"train_loss_avg": avg100_loss}, step=epoch_1000x)
                        log.info(
                            f"(epoch {epoch}) (step={data_iter_step}) loss {avg100_loss}"
                        )

        # Save checkpoint
        if epoch % cfg.training.ckpt_every == 0:
            mngr.save(
                epoch,
                args=ocp.args.Composite(
                    model_state=ocp.args.StandardSave(
                        get_global_array(replicated_state)
                    ),
                    ema_state=ocp.args.StandardSave(
                        get_global_array(replicated_ema_state)
                    ),
                    optimizer_state=ocp.args.StandardSave(
                        get_global_array(replicated_opt_state)
                    ),
                ),
            )
            mngr.wait_until_finished()

            log.info(f"(epoch={epoch:07d}) saved checkpoint")

        # Sample images
        if epoch % cfg.training.sample_every == 0:
            log.info(f"(epoch={epoch:07d}) sampling images")
            cur_ema_state = jax.tree.map(lambda x: x[0], replicated_ema_state)
            ema_model = nnx.merge(ema_def, cur_ema_state, ema_non_trainable_params)
            key, subkey_sample = random.split(key)

            log.info("Handing over to the sampling function")
            output_images = sample(
                ema_model,
                epoch,
                subkey_sample,
                class_labels,
                image_size=cfg.dataset.image_size,
                num_channels=cfg.dataset.num_channels,
            )
            log.info(f"Output images shape: {output_images.shape}")

            if jax.process_index() == 0:
                grid = process_image_for_save(output_images, num_images)
                log.info("Saving image")
                wandb.log(
                    {
                        "sampled_images": wandb.Image(
                            grid, caption=f"Samples at epoch {epoch}"
                        )
                    },
                    step=epoch * 1000,
                )

        # Compute FID
        if epoch % cfg.training.compute_fid_every == 0:
            ema_replicated_non_trainable_params = jax.tree.map(
                lambda x: jnp.array([x] * G), ema_non_trainable_params
            )
            fid_score = get_fid(
                epoch,
                ema_def,
                replicated_ema_state,
                ema_replicated_non_trainable_params,
                get_activations,
                truth_stats,
                key_fid,
                num_images_per_class=8,
                batch_size=800,
                image_size=cfg.dataset.image_size,
                num_channels=cfg.dataset.num_channels,
            )

            if jax.process_index() == 0:
                wandb.log({"fid": fid_score}, step=epoch * 1000)

    mngr.wait_until_finished()
    wandb.finish()
    log.info("Done!")
