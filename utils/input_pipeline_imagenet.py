"""
Create dataloaders for ImageNet
"""

import os
import jax
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from PIL import Image
from functools import partial
import random

MEAN_RGB = [0.5, 0.5, 0.5]
STDDEV_RGB = [0.5, 0.5, 0.5]


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_imagenet_loaders(
    dataset_root,
    split,
    batch_size,
    image_size,
    num_workers=32,
    prefetch=3,
    val_drop_last=False,
):
    """
    Create dataloaders that load ImageNet and crop to image_size x image_size.
    Has sampler for multi-process.
    """
    rank = jax.process_index()

    dataset = datasets.ImageFolder(
        os.path.join(dataset_root, split),
        transform=transforms.Compose(
            [
                transforms.Lambda(
                    lambda pil_image: center_crop_arr(pil_image, image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB, inplace=True),
            ]
        ),
    )

    if split == "train":
        sampler = DistributedSampler(
            dataset,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            persistent_workers=True,
        )

    elif split == "val":
        sampler = DistributedSampler(
            dataset,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=val_drop_last,
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch,
            persistent_workers=True,
        )

    return dataloader


def prepare_batch_data(image, label):
    """
    Reformat a input batch from PyTorch Dataloader and shard it across local devices.
    """
    local_device_count = jax.local_device_count()

    assert image.shape[0] % local_device_count == 0

    # Reshape to (batch_size, height, width, num_channels)
    image = image.permute(0, 2, 3, 1)

    # Reshape to (local_device_count, device_batch_size, ...) for pmap
    image = image.reshape((local_device_count, -1) + image.shape[1:])
    label = label.reshape(local_device_count, -1)

    # Convert to numpy arrays for JAX
    image = image.numpy()
    label = label.numpy()

    return image, label
