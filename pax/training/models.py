import jax.random as jrandom
import jax.nn as jnn
import jax.numpy as jnp
import jax.numpy.linalg as la
import equinox as eqx
import chex
from jax import jit
from pax.core.environment import Environment, EnvParams
from typing import List, Tuple
from jax.debug import print as dprint


def orthogonal_init(
    array: chex.ArrayDevice, gain: float, key: chex.PRNGKey
) -> chex.ArrayDevice:
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
    """
    Linear layer with orthogonal initialization of weights.
    """

    def __init__(self, input_dim: int, output_dim: int, gain: float, key: chex.PRNGKey):
        super().__init__(input_dim, output_dim, key=key)
        self.weight = orthogonal_init(self.weight, gain, key)
        self.bias = 0.0 * jnp.ones_like(self.bias)


class Agent(eqx.Module):
    """The agent class. This class contains the critic and actor networks."""

    critic: eqx.Module
    actor: eqx.Module
    

    def __init__(self, env: Environment, key: chex.PRNGKey):
        # TODO: Refactor this agent to be more general.

        obs_shape = env.observation_space.shape

        action_space_size = env.action_space.size

        keys = jrandom.split(key, 6)

        self.critic = eqx.nn.Sequential(
            [
                QRLinear(
                    jnp.array(obs_shape).prod(),
                    128,
                    jnp.sqrt(2),
                    key=keys[0],
                ),
                eqx.nn.Lambda(jnn.tanh),
                QRLinear(128,64, jnp.sqrt(2), key=keys[1]),
                eqx.nn.Lambda(jnn.tanh),
                QRLinear(64, 1, jnp.array([1]), key=keys[2]),
            ]
        )

        self.actor = eqx.nn.Sequential(
            [
                QRLinear(
                    jnp.array(obs_shape).prod(),
                    128,
                    jnp.sqrt(2),
                    key=keys[3],
                ),
                eqx.nn.Lambda(jnn.tanh),
                QRLinear(128, 64, jnp.sqrt(2), key=keys[4]),
                eqx.nn.Lambda(jnn.tanh),
                QRLinear(64, action_space_size, jnp.array([0.01]), key=keys[5]),
            ]
        )

    @eqx.filter_jit
    def __call__(
        self, x: chex.ArrayDevice
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        """Forward pass of the agent. Returns the value and logits.

        Args:
            x (chex.ArrayDevice): The input to the agent. Should be of the shape (batch_size, obs_shape)

        Returns:
            _type_: _description_
        """
        value = self.critic(x)
        logits = self.actor(x)

        split_logits = jnp.split(logits, [11])

        return split_logits, value 


class AgentCNN(eqx.Module):
    """The agent class. This class contains the critic and actor networks."""

    critic: eqx.Module
    actor: eqx.Module
    

    def __init__(self, env: Environment, key: chex.PRNGKey):
        # TODO: Refactor this agent to be more general.

        obs_shape = env.observation_space.shape

        action_space_size = env.action_space.size

        keys = jrandom.split(key, 6)

        self.critic = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(1, 1, kernel_size=1, key=keys[0]),
                # eqx.nn.AvgPool2d(kernel_size=1),
                eqx.nn.Lambda(jnn.tanh),
                eqx.nn.Lambda(jnp.ravel),
                #     jnp.array(obs_shape).prod(),
                #     256,
                #     jnp.sqrt(2),
                #     key=keys[3],
                # ),
                # eqx.nn.Lambda(jnn.tanh),
                QRLinear(44, 128, jnp.sqrt(2), key=keys[4]),
                eqx.nn.Lambda(jnn.tanh),
                QRLinear(128, 1, jnp.array([1.0]), key=keys[5]),
            ]
        )

        self.actor = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(1, 1, kernel_size=1, key=keys[3]),
                # eqx.nn.AvgPool2d(kernel_size=1),
                eqx.nn.Lambda(jnn.tanh),
                eqx.nn.Lambda(jnp.ravel),
                #     jnp.array(obs_shape).prod(),
                #     256,
                #     jnp.sqrt(2),
                #     key=keys[3],
                # ),
                # eqx.nn.Lambda(jnn.tanh),
                QRLinear(44, 128, jnp.sqrt(2), key=keys[4]),
                eqx.nn.Lambda(jnn.tanh),
                QRLinear(128, action_space_size, jnp.array([0.01]), key=keys[5]),
            ]
        )

    @eqx.filter_jit
    def __call__(
        self, x: chex.ArrayDevice
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        """Forward pass of the agent. Returns the value and logits.

        Args:
            x (chex.ArrayDevice): The input to the agent. Should be of the shape (batch_size, obs_shape)

        Returns:
            _type_: _description_
        """
        x = x[None,...]
        value = self.critic(x)
        logits = self.actor(x)

        split_logits = jnp.split(logits, [11])

        return split_logits, value 
