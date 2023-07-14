#%%
import equinox as eqx
import optax
import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple
from collections import defaultdict
import flax
import numpy as np
import tqdm

#should this be eqx.Module
class BatchManager(object):
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
        action_size,
        state_space,
    ):
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.num_envs = num_envs
        self.action_size = action_size
        self.buffer_size = num_envs * n_steps

        try:
            temp = state_space.shape[0]
            self.state_shape = state_space.shape
        except Exception:
            self.state_shape = [state_space]

        self.reset()

    @eqx.filter_jit
    def reset(self):
        return {
            "states": jnp.empty(
                (self.n_steps, self.num_envs, *self.state_shape),
                dtype=jnp.float32,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, *self.action_size),
            ),
            "rewards": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_steps, self.num_envs), dtype=jnp.float32
            ),
            "_step": 0,
        }

    @eqx.filter_jit
    def append(self, buffer, state, action, reward, done, log_pi, value):
        return {
                "states":  buffer["states"].at[buffer["_step"]].set(state),
                "actions": buffer["actions"].at[buffer["_step"]].set(action),
                "rewards": buffer["rewards"].at[buffer["_step"]].set(reward.squeeze()),
                "dones": buffer["dones"].at[buffer["_step"]].set(done.squeeze()),
                "log_pis_old": buffer["log_pis_old"].at[buffer["_step"]].set(log_pi),
                "values_old": buffer["values_old"].at[buffer["_step"]].set(value),
                "_step": (buffer["_step"] + 1) % self.n_steps,
            }

    @eqx.filter_jit
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

    @eqx.filter_jit
    def calculate_gae(
        self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
# %%
