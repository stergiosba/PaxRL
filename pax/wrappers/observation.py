import chex
from typing import Tuple
# from pax.environment import 
from .wrapper import Wrapper

class ObservationWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key:chex.PRNGKey) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        obs, state = self.env.reset(key)
        return self.observation_map(obs), state

    def step(self, key, state, action: chex.ArrayDevice, extra_in) -> Tuple[chex.ArrayDevice, chex.ArrayDevice, chex.ArrayDevice]:
        obs, state, reward, done = self.env.step(key, state, action, extra_in)
        return self.observation_map(obs), state, reward, done

    def observation_map(self, obs: chex.ArrayDevice) -> chex.ArrayDevice:
        raise NotImplementedError
    

class NormalizeObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation_map(self, obs: chex.ArrayDevice) -> chex.ArrayDevice:
        obs_mean = obs.mean()
        obs_std = obs.std()
        return (obs - obs_mean) / obs_std
    
class FlattenObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation_map(self, obs: chex.ArrayDevice) -> chex.ArrayDevice:
        return obs.flatten()