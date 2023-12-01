from jax import lax, jit, nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.numpy.linalg as la
from jax.debug import print as dprint  # type: ignore
from pax.core.state import EnvState
from pax.utils.bezier import BezierCurve3
from pax.core.environment import Environment, EnvParams
from pax.core.spaces import *
from typing import Any, Dict, Sequence, Tuple, Union


class Proberenv(Environment):

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
    action_space: SeparateGrid
    observation_space: Box
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
        self.action_space = SeparateGrid(action_range)
            
        o_dtype = jnp.float32
        o_low, o_high = self.params.observation_space.values()
        o_shape = (self.n_agents + self.n_scripted, 2)
        self.observation_space = Box(o_low, o_high, o_shape, o_dtype)

    @eqx.filter_jit
    def get_obs(self, state: EnvState) -> Sequence[chex.Array]:
        """Applies observation function to state.

        `Args`:
            - `state (EnvState)`: The full state of the environment.

        Returns:
            - `observation (chex.Array)`: The observable part of the state.
        """

        return jnp.array(state.X).reshape(-1)

    @eqx.filter_jit
    def reset_env(self, key: chex.PRNGKey) -> Tuple[Sequence[chex.Array], EnvState]:
        """Resets the environment.

        Args:
            -`key (chex.PRNGKey)`: The random key for the reset.

        Returns:
            - `Observations (chex.Array)`: The initial observation of the environment based on the observation function get_obs
            - `State (EnvState)`: The initial full state of the environment. Used internally.
        """

        init_X_scripted = jrandom.uniform(
            key,
            minval=jnp.array([75, 600]),
            maxval=jnp.array([125, 700]),
            shape=(self.n_scripted, 2),
        )

        # init_X_dot_scripted = 5 * jrandom.uniform(
        #     key, minval=-0, maxval=0, shape=(self.n_scripted, 2)
        # )
        init_X_dot_scripted = jnp.zeros((self.n_scripted, 2))

        # init_X_agents = jnp.array([[100, 600]])
        # + jrandom.cauchy(
        #     key, shape=(self.n_agents, 2)
        # )
        init_X_agents = jrandom.uniform(
            key,
            minval=jnp.array([100, 600]),
            maxval=jnp.array([101, 601]),
            shape=(self.n_agents, 2),
        )

        init_X_dot_agents = (
            jrandom.uniform(key, minval=-0, maxval=0, shape=(self.n_agents, 2))
            * 10
            * jnp.sqrt(2)
        )

        # leader = jrandom.randint(key, shape=(), minval=0, maxval=self.n_scripted)0)
        leader = 0

        final_goal = jrandom.uniform(
            key, minval=jnp.array([300, 50]), maxval=jnp.array([750, 125]), shape=(2,)
        )
        init_leader = init_X_scripted[leader]

        p0 = (4 * init_leader + 1 * final_goal) / 5

        P = jnp.array(
            [
                p0,
                (2 * p0 + final_goal) / 3,
                (p0 + 2 * final_goal) / 3,
                final_goal,
            ]
        )

        leader_path_curve = BezierCurve3(P)

        state = EnvState(
            X=jnp.concatenate([init_X_scripted, init_X_agents]),
            X_prev=jnp.concatenate([init_X_scripted, init_X_agents]),
            X_dot=jnp.concatenate([init_X_dot_scripted, init_X_dot_agents]),
            X_dot_prev=jnp.concatenate([init_X_dot_scripted, init_X_dot_agents]),
            B=jnp.zeros(shape=self.n_scripted),
            leader=leader,
            curve=leader_path_curve,
            t=0,
        )
        return (self.get_obs(state), state)  # type: ignore

    @eqx.filter_jit
    def step_env(
        self, key, state: EnvState, action: chex.ArrayDevice, extra_in: Any
    ) -> Tuple[
        Sequence[chex.ArrayDevice], EnvState, chex.ArrayDevice, chex.ArrayDevice
    ]:
        """Performs one step in the environment

        `Args`:
            - `key (chex.PRNGKey)`: A jax random key for the step. Currently unused (present for consistency)
            - `state (EnvState)`: State of the environment
            - `action (chex.Array)`: The joint action fro the agents and scripted entities.

        Returns:
            - `environment_step Tuple[Sequence[chex.Array], EnvState, chex.Array, chex.Array])`: A step in the environment.
        """

        @jit
        def r_max_interaction(B, leader):
            P = jnn.softmax(B)
            return 5 * P[leader] - 1

        @jit
        def r_leader(X, leader):
            return -la.norm(X[-1] - X[leader])/(800*jnp.sqrt(2))
        
        @jit
        def r_angle(X_dot_prev, X_dot):
            # calculate the angle of the velocity before and after the step
            angle_prev = jnp.arctan2(X_dot_prev[1], X_dot_prev[0])
            angle = jnp.arctan2(X_dot[1], X_dot[0])
            # calculate the difference of the angles
            angle_diff = jnp.abs(angle_prev - angle)
            # calculate the reward based on the difference of the angles
            return -angle_diff
        
        @jit 
        def r_distance(X):
            return -la.norm(X[-1] - jnp.array([500,500]))/(800*jnp.sqrt(2))

        acc = action
        new_B = extra_in[0]
        dt = self.params.settings["dt"]
        X_dot_prev = state.X_dot
        X_dot = state.X_dot + dt / 6 * acc  # - 0.01*state.X_dot

        X_dot = jnp.clip(X_dot, a_min=-10 * jnp.sqrt(2), a_max=10 * jnp.sqrt(2))
        X_prev = state.X
        X = state.X + dt * X_dot
        X = jnp.clip(X, a_min=self.observation_space.low, a_max=self.observation_space.high)

        state = EnvState(X, X_prev, X_dot, X_dot_prev, new_B, state.leader, state.curve, state.t + 1)  # type: ignore

        obs = self.get_obs(state)
        #dprint("{x}", x = r_max_interaction(state.B, state.leader))
        #reward = 0.1 * r_max_interaction(state.B, state.leader)# + 0.1*r_angle(X_dot_prev, X_dot)
        reward = 0.1 * r_distance(X)
        done = self.is_terminal(state)

        return (obs, state, jnp.array([reward]), done)

    def is_terminal(self, state):
        # The state.curve(1.0) is the final goal
        norm_e = la.norm(state.curve.eval(1.0) - state.X[state.leader])

        done_1 = lax.select(norm_e < 5, jnp.array([1.0]), jnp.array([0.0]))

        done_time = (state.t > self.params.scenario["episode_size"])

        done = jnp.logical_or(done_1, done_time)
        return done

    @property
    def name(self):
        return "ProberEnv-v0"

    def __repr__(self):
        return f"{self.name}: {str(self.__dict__)}"
