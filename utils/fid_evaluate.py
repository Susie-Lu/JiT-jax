"""
Evaluate FID of model
"""

import logging
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
import optax

from utils.ode_solvers import sample_heun
from utils.fid_utils import get_fid_network, fid_from_stats

log = logging.getLogger(__name__)


def get_fid(
    epoch,
    model_def,
    replicated_state,
    replicated_non_trainable_params,
    get_activations,
    truth_stats,
    key,
    num_images_per_class=8,
    batch_size=800,
    image_size=32,
    num_channels=3,
):
    activations = (
        []
    )  # Store the Inception activations for the images sampled from the model

    num_devices = jax.local_device_count()

    # y: sample class labels so that we get num_images_per_class samples from each of the 1,000 classes
    num_batches = (num_images_per_class * 1000) // batch_size
    y = jnp.repeat(jnp.arange(1000), num_images_per_class)
    key, subkey_y = random.split(key)
    y = random.permutation(subkey_y, y)
    y = y.reshape((num_batches, num_devices, batch_size // num_devices))

    for i in range(num_batches):
        # z: create sampling noise
        key, subkey_z = random.split(key)
        z = random.normal(
            key=subkey_z,
            shape=(
                num_devices,
                batch_size // num_devices,
                image_size,
                image_size,
                num_channels,
            ),
        )

        log.info(f"(epoch {epoch}) (batch {i}) sampling images")
        output_images = sample_heun(
            z, y[i], replicated_state, model_def, replicated_non_trainable_params
        )[0][0:batch_size]
        log.info(f"(epoch {epoch}) (batch {i}) sampling images: done")
        log.info(
            f"(epoch {epoch}) (batch {i}) output_images.shape {output_images.shape}"
        )

        output_images = jax.image.resize(
            output_images, (batch_size, 299, 299, 3), method="bilinear", antialias=False
        )
        output_images = output_images.reshape(
            (num_devices, batch_size // num_devices, 299, 299, 3)
        )

        log.info(f"(epoch {epoch}) (batch {i}) compute activations")
        preds = get_activations(output_images)
        log.info(f"(epoch {epoch}) (batch {i}) compute activations: done")

        preds = np.array(preds.reshape((batch_size, -1)))
        log.info(f"(epoch {epoch}) (batch {i}) preds.shape {preds.shape} after reshape")
        activations.append(preds)

    activations = np.concatenate(activations, axis=0)
    mu1 = np.mean(activations, axis=0)
    sigma1 = np.cov(activations, rowvar=False)

    mu_truth = truth_stats["mu"]
    sigma_truth = truth_stats["sigma"]

    log.info(f"(epoch {epoch}) Computing FID score")
    fid_score = fid_from_stats(mu1, sigma1, mu_truth, sigma_truth)
    log.info(f"(epoch {epoch}) FID {fid_score}")
    return fid_score


## Utilities for FID evaluation after training


def restore_ckpt(model, epoch, cfg):
    # Create abstract states for restoring checkpoint
    _, init_model_state, _ = nnx.split(model, nnx.Param, ...)

    warmup_steps = 5 * 1251
    schedule_fn = optax.linear_schedule(
        init_value=0.0, end_value=2e-4, transition_steps=warmup_steps
    )
    optimizer = optax.adamw(learning_rate=schedule_fn, b1=0.9, b2=0.95, weight_decay=0)

    init_opt_state = optimizer.init(init_model_state)
    ema_model = nnx.clone(model)
    ema_def, init_ema_state, ema_non_trainable_params = nnx.split(
        ema_model, nnx.Param, ...
    )

    abstract_model_state = jax.tree.map(
        ocp.utils.to_shape_dtype_struct, init_model_state
    )
    abstract_ema_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, init_ema_state)
    abstract_opt_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, init_opt_state)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.training.num_ckpt_kept,
        save_interval_steps=cfg.training.ckpt_every,
        step_prefix="epoch",
        create=True,
    )
    mngr = ocp.CheckpointManager(
        cfg.training.ckpt_dir,
        options=options,
        item_names=("model_state", "ema_state", "optimizer_state"),
    )

    restored = mngr.restore(
        epoch,
        args=ocp.args.Composite(
            model_state=ocp.args.StandardRestore(abstract_model_state),
            ema_state=ocp.args.StandardRestore(abstract_ema_state),
            optimizer_state=ocp.args.StandardRestore(abstract_opt_state),
        ),
    )

    ema_model = nnx.merge(ema_def, restored.ema_state, ema_non_trainable_params)
    ema_model.training = False
    return ema_model


def get_fid_cfg_sweep(
    model,
    epoch,
    cfg,
    cfg_scales,
    num_images_per_class=50,
    batch_size=1000,
    key_seed=365,
):
    log.info("Loading Inception")
    get_activations = get_fid_network()
    log.info("Loading Inception: done")

    log.info("Loading FID ref stats")
    truth_stats = np.load(cfg.dataset.fid_stats_dir)
    key = random.key(key_seed)
    num_devices = jax.local_device_count()

    # Restore EMA ckpt
    ema_model = restore_ckpt(model, epoch, cfg)

    fid_list = []
    for cfg_scale in cfg_scales:
        log.info(f"CFG scale: {cfg_scale}")
        ema_model.cfg_scale = cfg_scale

        # Replicate the model's state across all available devices
        model_def, model_state, non_trainable_params = nnx.split(
            ema_model, nnx.Param, ...
        )
        replicated_state = jax.tree.map(
            lambda x: jnp.array([x] * num_devices), model_state
        )
        replicated_non_trainable_params = jax.tree.map(
            lambda x: jnp.array([x] * num_devices), non_trainable_params
        )

        fid_list.append(
            get_fid(
                epoch,
                model_def,
                replicated_state,
                replicated_non_trainable_params,
                get_activations,
                truth_stats,
                key,
                num_images_per_class,
                batch_size,
                cfg.dataset.image_size,
                cfg.dataset.num_channels,
            )
        )

    log.info(f"Summary: {fid_list}")
