import chex
import equinox as eqx
import jax.numpy as jnp
from typing import Tuple
from pax.core.state import EnvState
from .wrapper import Wrapper

class ObservationWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key:chex.PRNGKey, params=None) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        obs, state = self.env.reset(key, params)
        return self.observation_map(obs), state

    def step(self, key, state, action: chex.ArrayDevice, params=None) -> Tuple[chex.ArrayDevice, chex.ArrayDevice, chex.ArrayDevice]:
        obs, state, reward_array, done, info = self.env.step(key, state, action)
        # Need reward_array.sum() here as reward_array is an array to plot every different reward component
        return self.observation_map(obs), state, reward_array.sum(), done, info

    def observation_map(self, obs: chex.ArrayDevice) -> chex.ArrayDevice:
        raise NotImplementedError
        
class FlattenObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation_map(self, obs: chex.ArrayDevice) -> chex.ArrayDevice:
        return obs.flatten()
    
class NormalizeVecObsEnvState(eqx.Module):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: EnvState

class NormalizeObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self.env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )