import jax.random as jrandom
import jax.nn as jnn
import jax.numpy as jnp
import jax.numpy.linalg as la
import equinox as eqx
import chex
from jax import jit
from pax.core.environment import Environment
from pax import make
from typing import List
from tensorflow_probability.substrates import jax as tfp

    
@jit
def orthogonal_init(array, gain, key):
    r"""Fills the input `array` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        array: an n-dimensional `jax.array`, where :math:`n \geq 2`
        gain: optional scaling factor
    """
    if len(array.shape) < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    if array.size == 0:
        # no-op
        return array
    rows = array.shape[0]
    cols = array.size // rows
    flattened = jrandom.normal(key, (rows, cols))

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, _ = la.qr(flattened)

    if rows < cols:
        q = q.T

    return gain * q


class QRLinear(eqx.nn.Linear):
    def __init__(self, input_dim, output_dim, gain, key):
        super().__init__(input_dim, output_dim, key=key)
        self.weight = orthogonal_init(self.weight, gain, key)
        self.bias = jnp.zeros_like(self.bias)

class Agent(eqx.Module):
    critic: List
    actor: List

    def __init__(self, env: Environment, key):
        obs_shape = env.observation_space.shape
        keys = jrandom.split(key, 6)
        self.critic = [
            eqx.filter_vmap(
                QRLinear(
                    jnp.array(obs_shape).prod(),
                    64,
                    jnp.sqrt(2.0),
                    key=keys[0],
                ),
                in_axes=0,
            ),
            jnn.tanh,
            eqx.filter_vmap(QRLinear(64, 64, jnp.sqrt(2.0), key=keys[1]), in_axes=0),
            jnn.tanh,
            eqx.filter_vmap(QRLinear(64, 1, jnp.array([1.0]), key=keys[2]), in_axes=0),
        ]

        self.actor = [
            eqx.filter_vmap(
                QRLinear(
                    jnp.array(obs_shape).prod(),
                    64,
                    jnp.sqrt(2),
                    key=keys[3],
                ),
                in_axes=0,
            ),
            jnn.tanh,
            eqx.filter_vmap(QRLinear(64, 64, jnp.sqrt(2.0), key=keys[4]), in_axes=0),
            jnn.tanh,
            eqx.filter_vmap(
                QRLinear(
                    64, env.action_space.size, jnp.array([0.1]), key=keys[5]
                ),
                in_axes=0,
            ),
        ]

    @eqx.filter_jit
    def get_value(self, x):
        x_v = x
        for layer in self.critic:
            x_v = layer(x_v)
        return x_v

    
    #@eqx.filter_jit
    def get_action_and_value(self, x):
        x_a = x
        for layer in self.actor:
            x_a = layer(x_a)
            
        pi = tfp.distributions.Categorical(logits=x_a)

        return self.get_value(x), pi