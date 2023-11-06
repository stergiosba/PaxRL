import jax.random as jrandom
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
import chex
from jax import jit
from pax import make
from typing import List
from tensorflow_probability.substrates import jax as tfp

class A2CNetMultidiscrete(eqx.Module):
    critic: List
    actor: List

    def __init__(self, input_dim: int, output_dim: int, key: chex.PRNGKey):

        keys = jrandom.split(key, 6)
        self.critic = [
            eqx.nn.Linear(input_dim, 64, key=keys[0]),
            jnn.relu,
            eqx.nn.Linear(64, 32, key=keys[1]),
            jnn.relu,
            eqx.nn.Linear(32, 1, key=keys[2]),
        ]

        self.actor = [
            eqx.nn.Linear(input_dim, 64, key=keys[3]),
            jnn.relu,
            eqx.nn.Linear(64, 32, key=keys[4]),
            jnn.relu,
            eqx.nn.Linear(32, output_dim, key=keys[5]),
        ]

    def __call__(self, x, key=None):
        x_v = x
        x_a = x
        for layer in self.critic:
            x_v = layer(x_v)

        for layer in self.actor:
            x_a = layer(x_a)

        value = x_v
        pi = tfp.distributions.Categorical(logits=x_a)
        #pi = jrandom.categorical(key, logits=x_a)
        #log_prob = jnn.softmax(x_a)
        return pi, value

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


class A2CNet(eqx.Module):
    critic: List
    actor: List

    def __init__(self, input_dim: int, output_dim: int, key: chex.PRNGKey):
        # with open(f"{config_name}.toml", mode="rb") as model_file:
        # env_params = tomli.load(model_file)["critic_network"]
        #    crit_params = tomli.load(model_file)["critic_network"]

        keys = jrandom.split(key, 6)
        self.critic = [
            eqx.nn.Linear(input_dim, 64, key=keys[0]),
            jnn.relu,
            eqx.nn.Linear(64, 64, key=keys[1]),
            jnn.relu,
            eqx.nn.Linear(64, 1, key=keys[2]),
        ]

        self.actor = [
            eqx.nn.Linear(input_dim, 64, key=keys[3]),
            jnn.relu,
            eqx.nn.Linear(64, 64, key=keys[4]),
            jnn.relu,
            eqx.nn.Linear(64, output_dim, key=keys[5]),
        ]

    def __call__(self, x, key=None):
        x_v = x
        x_a = x
        for layer in self.critic:
            x_v = layer(x_v)

        for layer in self.actor:
            x_a = layer(x_a)

        value = x_v
        pi = tfp.distributions.Categorical(logits=x_a)
        #pi = jrandom.categorical(key, logits=x_a)
        #log_prob = jnn.softmax(x_a)
        return value, pi

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"

# TODO: Check if faster later and this would also drop the need for tensorflow_probability
@jit
def categorical_entropy(logits):
    @jit
    def _mul_exp(x, logp):
        """Returns `x * exp(logp)` with zero output if `exp(logp)==0`.

        Args:
            x: A `Tensor`.
            logp: A `Tensor`.

        Returns:
            `x * exp(logp)` with zero output and zero gradient if `exp(logp)==0`,
            even if `x` is NaN or infinite.
        """
        p = jnp.exp(logp)
        # If p==0, the gradient with respect to logp is zero,
        # so we can replace the possibly non-finite `x` with zero.
        x = jnp.where(jnp.equal(p, 0), jnp.zeros_like(x), x)
        return x * p

    log_probs = jnn.log_softmax(logits)
    return -jnp.sum(_mul_exp(log_probs, log_probs), axis=-1)
