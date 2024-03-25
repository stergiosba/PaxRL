import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Union
from pax.core.environment import Environment, EnvParams
from pax.core.state import EnvState
from .wrapper import Wrapper

class LogEnvState(eqx.Module):
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int
    reward_stats: jnp.ndarray


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: Environment):
        super().__init__(env)

    # @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        obs, env_state = self.env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0, jnp.zeros(1))
        return obs, state

    # @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float]
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward_array, done, info  = self.env.step(
            key, state.env_state, action
        )

        reward = reward_array.sum()
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        new_reward_stats = state.reward_stats + reward_array
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
            reward_stats = new_reward_stats* (1 - done)
        )


        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["reward_stats"] = state.reward_stats
        return obs, state, reward, done, info