import chex
import equinox as eqx
import jax.random as jrandom
import jax.numpy as jnp
from typing import Dict, Tuple, Callable
from jax import lax, tree_util as jtu
from pax.core.spaces import Space
from pax.core.state import EnvState
from jax.debug import print as dprint


class EnvParams(eqx.Module):
    settings: Dict
    scenario: Dict
    action_space: Dict
    observation_space: Dict


class EnvRewards(eqx.Module):
    """The environment rewards. Essentially a set of dictionaries that can pass via jit.

    `Args`:
        - `r_max_interaction (Callable)`: The reward for the maximum interaction.
        - `r_leader (Callable)`: The reward for the leader.
        - `r_target (Callable)`: The reward for the target.
        - `r_angle (Callable)`: The reward for the angle.
    """

    func: Dict[str, Callable]
    scales: Dict[str, float]

    def __init__(self):
        self.func = {}
        self.scales = {}

    def register(self, reward_func: Callable, scale: float = 1.0):
        self.func[reward_func.__name__] = reward_func
        self.scales[reward_func.__name__] = scale

    def apply(self, prev_state: EnvState, action: chex.ArrayDevice, state: EnvState):
        rew = [
            self.func[name](state, action) * self.scales[name]
            for name in self.func.keys()
        ]
        return jnp.array(rew)

    def total(self, prev_state: EnvState, action: chex.ArrayDevice, state: EnvState):
        return self.apply(prev_state, action, state).sum()

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


class Environment(eqx.Module):
    """The main `Pax` class.
    It encapsulates an environment with arbitrary dynamics. It is a subclass of `equinox.Module` making it jittable.
    An environment can be partially or fully observable.
    """

    params: EnvParams
    action_space: Space
    observation_space: Space
    rewards: EnvRewards

    @eqx.filter_jit
    def step(
        self, key: chex.PRNGKey, state: EnvState, action: chex.ArrayDevice
    ) -> Tuple[chex.ArrayDevice, EnvState, chex.ArrayDevice, chex.ArrayDevice]:
        """Steps the environment. This is to be used externally after implementing the reset_env function for a specific environment.

        Args:
            - `state (EnvState)`: State of the environment
            - `action (chex.ArrayDevice)`: Action to be taken
            - `params (EnvParams)`: Environment parameters
            - `key (chex.PRNGKey)`: Random key

        Returns:
            - `obs (chex.ArrayDevice)`: The new observable part of the state.
            - `state (EnvState)`: The new full state of the environment.
            - `reward (chex.ArrayDevice)`: The reward of taking action in old state and landing in the new state.
            - `done (chex.ArrayDevice)`: Done flag.
        """
        key_step, key_reset = jrandom.split(key)
        obs_step, state_step, reward, done = self.step_env(key_step, state, action)

        # Automatic reset
        obs_reset, state_reset = self.reset_env(key_reset)
        # TODO: Figure out why done needs to be indexed at 0
        state = jtu.tree_map(
            lambda x, y: lax.select(done[0], x, y), state_reset, state_step
        )
        obs = lax.select(done[0], obs_reset, obs_step)

        return obs, state, reward, done

    @eqx.filter_jit
    def reset(self, key: chex.PRNGKey):
        """Resets the environment. This is to be used externally after implementing the reset_env function for a specific environment.

        Args:
            key (chex.PRNGKey): A random key

        Returns:
            Calls the reset_env function for a specific environment.
        """
        return self.reset_env(key)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.ArrayDevice,
    ):
        """Steps a specific environment (To be implemented on a per-environment basis)

        Args:
            state (EnvState): State of the environment
            action (chex.ArrayDevice): Action to be taken
            params (EnvParams): Environment parameters
            key (chex.PRNGKey): Random key

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def reset_env(self, key: chex.PRNGKey):
        """Resets a specific environment (To be implemented on a per-environment basis)

        Args:
            key (chex.PRNGKey): A random key

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Returns the name of the environment

        Returns:
            name (str): The name of the environment
        """
        return self.__class__.__name__

    @property
    def version(self) -> str:
        """Returns the version of the environment

        Returns:
            version (str): The version of the environment
        """
        return NotImplementedError
