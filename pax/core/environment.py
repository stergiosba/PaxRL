import chex
import equinox as eqx
import jax.random as jrandom
from typing import Dict
from jax import lax, tree_util as jtu
from pax.core.spaces import *
from pax.core.state import EnvState
from jax.debug import print as dprint


class EnvParams(eqx.Module):
    settings: Dict
    scenario: Dict
    action_space: Dict
    observation_space: Dict


class Environment(eqx.Module):
    """The main `Pax` class.
    It encapsulates an environment with arbitrary dynamics. It is a subclass of `equinox.Module` making it jittable.
    An environment can be partially or fully observable.
    """

    @eqx.filter_jit
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.ArrayDevice,
        extra_in: chex.ArrayDevice = None,
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
        obs_step, state_step, reward, done = self.step_env(
            key_step, state, action, extra_in
        )

        # Automatic reset
        obs_reset, state_reset = self.reset_env(key_reset)

        state = jtu.tree_map(
            lambda x, y: lax.select(done[0], x, y), state_reset, state_step
        )
        obs = lax.select(done[0], obs_reset, obs_step)

        return obs, state, reward, done

    @eqx.filter_jit
    def reset(self, key):
        """Resets the environment. This is to be used externally after implementing the reset_env function for a specific environment.

        Args:
            key (_type_): A random key

        Returns:
            Calls the reset_env function for a specific environment.
        """
        return self.reset_env(key)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: chex.ArrayDevice,
        extra_in: chex.ArrayDevice=None,
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

    def reset_env(self, key):
        """Resets a specific environment (To be implemented on a per-environment basis)

        Args:
            key (_type_): A random key

        Raises:
            NotImplementedError: Not implemented
        """
        raise NotImplementedError
