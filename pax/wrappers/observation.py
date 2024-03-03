import chex
import equinox as eqx
from typing import Tuple
# from pax.environment import
import jax.numpy as jnp
from .wrapper import Wrapper

class ObservationWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key:chex.PRNGKey, *args) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        obs, state = self.env.reset(key)
        return self.observation_map(obs, *args), state

    def step(self, key, state, action: chex.ArrayDevice, extra_in, *args) -> Tuple[chex.ArrayDevice, chex.ArrayDevice, chex.ArrayDevice]:
        obs, state, reward, done = self.env.step(key, state, action, extra_in)
        return self.observation_map(obs, *args), state, reward, done

    def observation_map(self, obs: chex.ArrayDevice, *args) -> chex.ArrayDevice:
        raise NotImplementedError
    
class NormalizerRM(eqx.Module):
    mean: chex.ArrayDevice
    var: chex.ArrayDevice
    count: chex.ArrayDevice
    epsilon: chex.ArrayDevice

    def __init__(self, mean, var, count, epsilon):
        self.mean = mean
        self.var = var
        self.count = count
        self.epsilon = epsilon   

class NormalizeObservationWrapper(ObservationWrapper):

    normalizer: NormalizerRM
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        mean = jnp.zeros(jnp.array(env.observation_space.shape).prod())
        var = jnp.ones(jnp.array(env.observation_space.shape).prod())
        count = epsilon

        self.normalizer = NormalizerRM(mean, var, count, epsilon)
    
    def update_statistics(self, batch: chex.ArrayDevice, normalizer):
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = batch.size

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        delta = batch_mean - normalizer.mean
        tot_count = normalizer.count + batch_count
        new_mean = normalizer.mean + delta * batch_count / tot_count
        M2 = normalizer.var * normalizer.count + batch_var * batch_count + delta*delta * normalizer.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        new_normalizer = NormalizerRM(new_mean, new_var, new_count, self.normalizer.epsilon)

        return new_normalizer

    def observation_map(self, obs: chex.ArrayDevice, normalizer=None, *args) -> chex.ArrayDevice:
        if normalizer is None:
            normalizer = self.normalizer
        normalizer = self.update_statistics(obs, normalizer)
        return (obs - normalizer.mean) / jnp.sqrt(normalizer.var+self.normalizer.epsilon)
    
class FlattenObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation_map(self, obs: chex.ArrayDevice, *args) -> chex.ArrayDevice:
        return obs.flatten()