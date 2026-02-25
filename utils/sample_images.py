"""
Sample images from model
"""

import logging
import jax
import jax.numpy as jnp
from jax import random
from utils.ode_solvers import sample_heun
from flax import nnx
import torch
import numpy as np
from torchvision.utils import make_grid

log = logging.getLogger(__name__)


def sample(model, epoch, key, class_list, image_size=32, num_channels=3):
    """
    Samples image from each class given in class_list.
    Requires that len(class_list) is divisible by the number of local devices.
    """
    num_devices = jax.local_device_count()

    num_images = len(class_list)
    y = class_list.reshape((num_devices, num_images // num_devices))

    # z: create sampling noise
    key, subkey_z = random.split(key)
    z = random.normal(
        key=subkey_z,
        shape=(
            num_devices,
            num_images // num_devices,
            image_size,
            image_size,
            num_channels,
        ),
    )

    # Split model
    model.training = False
    model_def, model_state, non_trainable_params = nnx.split(model, nnx.Param, ...)
    replicated_state = jax.tree.map(lambda x: jnp.array([x] * num_devices), model_state)
    replicated_non_trainable_params = jax.tree.map(
        lambda x: jnp.array([x] * num_devices), non_trainable_params
    )

    log.info(f"(epoch {epoch}) sampling images...")
    output_images = sample_heun(
        z, y, replicated_state, model_def, replicated_non_trainable_params
    )[0]
    log.info(f"(epoch {epoch}) sampling images: done")

    output_images = jax.device_get(output_images)
    return output_images


def process_image_for_save(output_images, num_images):
    """
    Process images on host 0 for saving
    """
    samples_tensor = torch.from_numpy(
        np.transpose(np.array(output_images[0:num_images]), (0, 3, 1, 2))
    )  # (N, C, H, W)
    samples_tensor = torch.clamp(samples_tensor, min=-1, max=1)
    grid = make_grid(samples_tensor, nrow=5, normalize=True, value_range=(-1, 1))
    return grid
