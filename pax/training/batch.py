import chex
import equinox as eqx
import jax.numpy as jnp
from jax import jit
from jax.debug import print as dprint
from typing import Callable, Tuple, Dict
from functools import partial

# should this be eqx.Module?
class BatchManager(object):
    def __init__(
        self,
        train_config: Dict,
        action_space: chex.ArrayDevice,
        state_shape,
    ):
        self.num_envs = train_config["num_train_envs"]
        self.n_steps = train_config["n_steps"]
        self.discount = train_config["discount"]
        self.gae_lambda = train_config["gae_lambda"]
        self.action_space = action_space
        self.buffer_size = self.num_envs * self.n_steps
        self.state_shape = state_shape
        # self.state_shape = 2

        self.reset()

    @partial(jit, static_argnums=0)
    def reset(self):
        return {
            "states": jnp.empty(
                (self.n_steps, self.num_envs, self.state_shape),
                dtype=jnp.float32,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, self.action_space.n_axes),
            ),
            "rewards": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "values_old": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "_step": 0,
        }

    @partial(jit, static_argnums=0)
    def append(self, buffer, state, action, reward, done, log_pi, value):
        return {
            "states": buffer["states"].at[buffer["_step"]].set(state),
            "actions": buffer["actions"].at[buffer["_step"]].set(action),
            "rewards": buffer["rewards"].at[buffer["_step"]].set(reward.squeeze()),
            "dones": buffer["dones"].at[buffer["_step"]].set(done.squeeze()),
            "log_pis_old": buffer["log_pis_old"].at[buffer["_step"]].set(log_pi),
            "values_old": buffer["values_old"].at[buffer["_step"]].set(value),
            "_step": (buffer["_step"] + 1) % self.n_steps,
        }

    @partial(jit, static_argnums=0)
    def get(self, buffer):
        gae, target = self.calculate_gae(
            value=buffer["values_old"],
            reward=buffer["rewards"],
            done=buffer["dones"],
        )
        batch = (
            buffer["states"][:-1],
            buffer["actions"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
        )
        return batch

    # Apparently this is faster than the one provided by Google with scan.
    @partial(jit, static_argnums=0)
    def calculate_gae(
        self, value: chex.ArrayDevice, reward: chex.ArrayDevice, done: chex.ArrayDevice
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
