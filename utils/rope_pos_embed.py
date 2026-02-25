"""
JAX implementation of RoPE, based on
https://github.com/baaivision/EVA/EVA02
https://github.com/lucidrains/rotary-embedding-torch
"""

import jax.numpy as jnp
from flax import nnx
from math import pi


def broadcat_jax(tensor1, tensor2):
    target_shape = (tensor1.shape[0], tensor1.shape[0], tensor1.shape[2])
    tensor1 = jnp.broadcast_to(tensor1, target_shape)
    tensor2 = jnp.broadcast_to(tensor2, target_shape)
    return jnp.concat([tensor1, tensor2], axis=-1)


def rotate_half(x):
    """
    Args:
        x: Input array, e.g. query or key array, expected shape (batch_size, seq_len, dim),
        where dim is even.
    """
    orig_shape = x.shape

    if orig_shape[-1] % 2 != 0:
        raise ValueError(
            f"Last dimension of input tensor must be even, but got {orig_shape[-1]}"
        )

    d = orig_shape[-1] // 2

    # Reshape to (..., d, 2)
    x = jnp.reshape(x, (*orig_shape[:-1], d, 2))

    # Split into two halves
    x1 = x[..., 0]  # (..., d)
    x2 = x[..., 1]  # (..., d)

    # Stack them back as (-x2, x1)
    x_rotated = jnp.stack([-x2, x1], axis=-1)  # (..., d, 2)

    # Reshape back to the original last dimension
    return jnp.reshape(x_rotated, orig_shape)


class VisionRotaryEmbeddingFast(nnx.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0,
    ):
        """
        Initializes the module and pre-computes the rotary embedding frequencies.

        Args:
            dim: The dimension of the features for *one* direction (e.g., height OR width).
                 The input tensor t is expected to have a feature dimension of `2 * dim`.
            pt_seq_len: Pre-training sequence length (e.g., grid size).
            ft_seq_len: Fine-tuning sequence length. If None, defaults to `pt_seq_len`.
            custom_freqs: Optional custom frequencies to use instead of calculating them.
            freqs_for: Modality for frequency calculation ('lang', 'pixel', 'constant').
            theta: Theta value for 'lang' frequency calculation.
            max_freq: Max frequency for 'pixel' frequency calculation.
            num_freqs: Number of frequencies for 'constant' frequency calculation.
        """
        super().__init__()
        # Calculate base frequencies
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
            )
        elif freqs_for == "pixel":
            freqs = jnp.linspace(1.0, max_freq / 2, dim // 2, dtype=jnp.float32) * pi
        elif freqs_for == "constant":
            freqs = jnp.ones(num_freqs, dtype=jnp.float32)
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        t = jnp.arange(ft_seq_len, dtype=jnp.float32) / ft_seq_len * pt_seq_len
        freqs = jnp.einsum("..., f -> ... f", t, freqs)  # Shape: (ft_seq_len, dim // 2)
        freqs = jnp.repeat(freqs, 2, axis=-1)  # Shape: (ft_seq_len, dim)
        freqs = broadcat_jax(
            freqs[:, None, :], freqs[None, :, :]
        )  # Shape: (ft_seq_len, ft_seq_len, 2 * dim)

        if num_cls_token > 0:
            cos_img = jnp.cos(freqs).reshape(-1, freqs.shape[-1])  # [N_img, D]
            sin_img = jnp.sin(freqs).reshape(-1, freqs.shape[-1])

            # pad CLS rows: cos=1 (no rotation), sin=0
            D = cos_img.shape[1]
            cos_pad = jnp.ones(shape=(num_cls_token, D), dtype=cos_img.dtype)
            sin_pad = jnp.zeros(shape=(num_cls_token, D), dtype=sin_img.dtype)

            self.freqs_cos = jnp.concat([cos_pad, cos_img], axis=0)  # [N_cls+N_img, D]
            self.freqs_sin = jnp.concat([sin_pad, sin_img], axis=0)

        else:
            self.freqs_cos = jnp.cos(freqs).reshape(
                -1, freqs.shape[-1]
            )  # Shape: (ft_seq_len * ft_seq_len, 2 * dim)
            self.freqs_sin = jnp.sin(freqs).reshape(-1, freqs.shape[-1])

    def __call__(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
