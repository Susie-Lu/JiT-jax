"""
Euler & Heun ODE solvers for sampling images
"""

import jax
import jax.numpy as jnp
from flax import nnx
import functools


@functools.partial(jax.pmap, axis_name="batch")
def sample_euler(x, y, model_state, model_def, non_trainable_params):
    model = nnx.merge(model_def, model_state, non_trainable_params)
    batch_size = y.shape[0]

    def euler_step(t, x):
        t_start = t / 200
        t_end = (t + 1) / 200

        t_batch = jnp.repeat(jnp.array([t_start]), repeats=batch_size)
        x = x + (t_end - t_start) * model.forward_x_to_v_with_cfg(x, t_batch, y)
        return x

    x = jax.lax.fori_loop(0, 200, euler_step, x)
    x_all = jax.lax.all_gather(x, axis_name="batch", tiled=True)
    return x_all


@functools.partial(jax.pmap, axis_name="batch")
def sample_heun(x, y, model_state, model_def, non_trainable_params):
    model = nnx.merge(model_def, model_state, non_trainable_params)
    batch_size = y.shape[0]

    def heun_step(t, x):
        t_start = t / 50
        t_end = (t + 1) / 50

        t_start_batch = jnp.repeat(jnp.array([t_start]), repeats=batch_size)
        t_end_batch = jnp.repeat(jnp.array([t_end]), repeats=batch_size)

        left_derivative = model.forward_x_to_v_with_cfg(x, t_start_batch, y)
        euler = x + (t_end - t_start) * left_derivative  # Euler guess
        right_derivative = model.forward_x_to_v_with_cfg(euler, t_end_batch, y)
        x = x + (t_end - t_start) * 0.5 * (left_derivative + right_derivative)

        return x

    x = jax.lax.fori_loop(0, 49, heun_step, x)

    # Use Euler for the last step
    t = 49
    t_start = t / 50
    t_end = (t + 1) / 50
    t_start_batch = jnp.repeat(jnp.array([t_start]), repeats=batch_size)
    x = x + (t_end - t_start) * model.forward_x_to_v_with_cfg(x, t_start_batch, y)

    x_all = jax.lax.all_gather(x, axis_name="batch", tiled=True)
    return x_all
