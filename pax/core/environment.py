import chex
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
import jax.numpy.linalg as la
from typing import Sequence, Union, Tuple, Sequence, Any, Dict
from jax import jit, vmap, lax, devices
from pax.core.spaces import *

# from pax.scenarios.script_inter import script
from jax.debug import print as dprint  # type: ignore
from pax.core.state import EnvState
from pax.utils.bezier import BezierCurve3


class EnvParams(eqx.Module):
    settings: Dict
    scenario: Dict
    action_space: Dict
    observation_space: Dict
    critic_network: Dict


class Environment(eqx.Module):
    """The main `Pax` class.
        It encapsulates an environment with arbitrary dynamics.
        An environment can be partially or fully observable.

    `Attributes`:
        - `n_agents (int)`: The total number of agents (actors) in the environment.
        - `n_scripted (int)`: The total number of scripted entities in the environment.
        - `action_space (Union[Discrete, Box])`: The action space allowed to the agents.
        - `observation_space (Union[Discrete, Box])`: The observation space of the environment.
        - `params (Dict)`: Parameters given to environment from the TOML file.

    """

    n_agents: int
    n_scripted: int
    action_space: Union[Discrete, MultiDiscrete, Box]
    observation_space: Union[Discrete, MultiDiscrete, Box]
    params: EnvParams

    def __init__(self, params):
        """
        Args:
           `params (EnvParams)`: Parameters given to environment from the TOML file.
        """
        self.params = EnvParams(**params)
        self.n_agents = self.params.scenario["n_agents"]
        self.n_scripted = self.params.scenario["n_scripted_entities"]

        action_range = self.params.action_space.values()
        self.action_space = MultiDiscrete(action_range)  # type: ignore

        o_dtype = jnp.float32
        o_low, o_high = self.params.observation_space.values()
        o_shape = (self.n_agents + self.n_scripted, 2)
        self.observation_space = Box(o_low, o_high, o_shape, o_dtype)  # type: ignore

    @eqx.filter_jit
    def get_obs(self, state: EnvState) -> Sequence[chex.Array]:
        """Applies observation function to state.

        `Args`:
            - `state (EnvState)`: The full state of the environment.

        Returns:
            - `observation (chex.Array)`: The observable part of the state.
        """

        return jnp.array(state.X)

    @eqx.filter_jit
    def reset(self, key: chex.PRNGKey) -> Tuple[Sequence[chex.Array], EnvState]:
        """Resets the environment.

        Args:
            -`key (chex.PRNGKey)`: The random key for the reset.

        Returns:
            - `Observations (chex.Array)`: The initial observation of the environment based on the observation function get_obs
            - `State (EnvState)`: The initial full state of the environment. Used internally.
        """

        init_X_scripted = jrandom.uniform(
            key, minval=445, maxval=565, shape=(self.n_scripted, 2)
        )
        """
        init_X_scripted = jnp.array([500,500])+jrandom.rayleigh(
            key, scale=100, shape=(self.n_scripted, 2)
        )
        """

        init_X_dot_scripted = 5 * jrandom.uniform(
            key, minval=-1, maxval=1, shape=(self.n_scripted, 2)
        )

        init_X_agents = jnp.array([100, 700]) + jrandom.cauchy(
            key, shape=(self.n_agents, 2)
        )

        init_X_dot_agents = jrandom.uniform(
            key, minval=-1, maxval=1, shape=(self.n_agents, 2)
        ) * jnp.sqrt(400)

        # leader = jrandom.randint(key, shape=(), minval=0, maxval=self.n_scripted)
        leader = jrandom.randint(key, shape=(), minval=0, maxval=0)
        final_goal = jrandom.uniform(key, minval=20, maxval=50, shape=(2,))
        init_leader = init_X_scripted[leader]

        A = init_leader  # (init_leader+final_goal)/2

        P = jnp.array(
            [
                A,
                jnp.array([0, 500]) + (2 * A + final_goal) / 3,
                (A + 2 * final_goal) / 3,
                final_goal,
            ]
        )

        leader_path_curve = BezierCurve3(P)

        state = EnvState(
            X=jnp.concatenate([init_X_scripted, init_X_agents]),
            X_dot=jnp.concatenate([init_X_dot_scripted, init_X_dot_agents]),
            leader=leader,
            curve=leader_path_curve,
            t=0.0,
        )
        return (self.get_obs(state), state)  # type: ignore

    @eqx.filter_jit
    def step(
        self, state: EnvState, action: chex.Array
    ) -> Tuple[Sequence[chex.Array], EnvState, chex.Array, chex.Array]:
        """Performs one step in the environment

        `Args`:
            - `state (EnvState)`: State of the environment
            - `action (chex.Array)`: The joint action fro the agents and scripted entities.

        Returns:
            - `environment_step Tuple[Sequence[chex.Array], EnvState, chex.Array, chex.Array])`: A step in the environment.
        """

        dt = self.params.settings["dt"]

        X_dot = state.X_dot + dt / 6 * action

        X_dot = jnp.clip(X_dot, a_min=-20, a_max=20)

        X = state.X + dt * X_dot  # / la.norm(X_dot, axis=1)[:, None]
        X = jnp.clip(X, a_min=0, a_max=800)

        state = EnvState(X, X_dot, state.leader, state.curve, state.t + 1)  # type: ignore

        obs = self.get_obs(state)

        reward = lax.select(state.t <= 500, jnp.array([-1.0]), jnp.array([-2.0]))

        # The state.curve(1.0) is the final goal
        norm_e = la.norm(state.curve.eval(1.0) - X[state.leader])

        done = lax.select(norm_e < 5, jnp.array([1.0]), jnp.array([0.0]))
        # dprint("{x}",x=done)
        return (obs, state, reward, done)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
