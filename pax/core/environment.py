import chex
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
import jax.numpy.linalg as la
from typing import Sequence, Union, Tuple, Sequence, Any, Dict
from jax import jit, vmap, lax, devices, debug
from pax.core.spaces import *
from pax.core.script_inter import script
from jax.debug import print as dprint  #type: ignore

class EnvState(eqx.Module):
    """The environment state (Multiple Agents)

    `Args`:
        - `X (chex.Array)`: Position of every Agents.
        - `X_dot (chex.Array)`: Velocity of every Agent.
        - `leader (int)`: The id of the leader agent.
        - `goal (chex.Array)`: The location of the goal.
    """
    X: chex.Array
    X_dot: chex.Array
    leader: chex.Array
    goal: chex.Array
    
    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
     
    
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
    
    # TODO: Refactor in the future so that the environment is separate from the scenario
    n_agents: int
    n_scripted: int
    action_space: Union[Discrete, MultiDiscrete, Box]
    observation_space: Union[Discrete, MultiDiscrete, Box]
    params: Dict

    def __init__(
            self,
            params):
        """
        Args:
           `params (Dict)`: Parameters given to environment from the TOML file.
           `params (Dict)`: Parameters given to environment from the TOML file.
        """
        self.params = params
        self.n_agents = self.params["scenario"]["n_agents"]
        self.n_scripted = self.params["scenario"]["n_scripted_entities"]
        self.create_spaces()

    def create_spaces(self):
        action_range = self.params["action_space"].values()
        self.action_space = MultiDiscrete(action_range) #type: ignore

        o_dtype = jnp.float32
        o_low, o_high, o_shape = self.params["observation_space"].values()
        self.observation_space = Box(o_low, o_high, o_shape, o_dtype) #type: ignore

    
    @eqx.filter_jit
    def get_obs(
        self,
        state: EnvState) -> Sequence[chex.Array]:
        """Applies observation function to state.

        `Args`:
            - `state (EnvState)`: The full state of the environment.

        Returns:
            - `observation (chex.Array)`: The observable part of the state.
        """

        return jnp.array([state.X, state.X_dot]), state.leader, state.goal

    @eqx.filter_jit
    def reset(
            self,
            key: chex.PRNGKey) -> Tuple[Sequence[chex.Array], EnvState]:
        """Resets the environment.

        Args:
            -`key (chex.PRNGKey)`: The random key for the reset.

        Returns:
            - `Observations (chex.Array)`: The initial observation of the environment based on the observation function get_obs
            - `State (EnvState)`: The initial full state of the environment. Used internally.
        """        

        init_X_scripted = jrandom.uniform(
            key, minval=645, maxval=765, \
            shape=(self.n_scripted,2)
            )
        
        init_X_dot_scripted = jrandom.uniform(
            key, minval=-1, maxval=1, \
            shape=(self.n_scripted,2)
            )
        
        init_X_agents = jrandom.uniform(
            key, minval=400, maxval=300, \
            shape=(self.n_agents,2)
            )
        
        init_X_dot_agents = jrandom.uniform(
            key, minval=-1, maxval=1, \
            shape=(self.n_agents,2)
            )

        leader = jrandom.randint(key, shape=(), minval=0, maxval=self.n_scripted-1)

        goal = jrandom.uniform(
            key, minval=0, maxval=800, \
            shape=(2,)
            )


        state = EnvState(
            X=jnp.concatenate([init_X_scripted,
                            init_X_agents]),
            X_dot=jnp.concatenate([init_X_dot_scripted,
                            init_X_dot_agents]),
            leader = leader,
            goal = goal
        )
        return (self.get_obs(state), state) #type: ignore
    
    @eqx.filter_jit
    def batch_step(
            self,
            state,
            action):
        return vmap(self.step, in_axes=(0, 0))(state, action)
    
    def batch_rollout(
            self,
            keys: jrandom.PRNGKeyArray) -> Sequence[chex.Array]:
        """Produces a batch rollout of an environment. Vectorized mapping over an array of keys.

        Args:
            keys (jrandom.KeyArray): A matrix of PRNG keys to map over.

        Returns:
            (chex.Array): The batched rollouts for as many environment as the number of keys.
        """
        batch_rollout = vmap(self.single_rollout, in_axes=(0))
        return batch_rollout(keys)
    
    @eqx.filter_jit
    def single_rollout(
            self,
            key: chex.PRNGKey) -> Tuple[chex.Array,chex.Array,chex.Array,
                                        chex.Array,chex.Array,chex.Array]:#, policy_params):
        """
        Efficient rollout of a singe environment episode with lax.scan.

        Args:
            `self` (Environment): The environment instance
            `key` (chex.PRNGKey): The PRNG key that determines the rollout of the episode

        Returns:
            - `obs` (chex.Array): The observations collected from the episode.
            - `action` (chex.Array): The actions collected from the episode.
            - `reward` (chex.Array): The rewards collected from the episode.
            - `next_obs` (chex.Array): The next observation,
            - `done` (chex.Array): Done flag,
            - `cum_return` (chex.Array): Summation of the rewards collected over the episode.
        """
        
        # Reset the environment
        (obs, state) = self.reset(key)

        def policy_step(
                state_input,
                _):
            """lax.scan compatible step transition in jax env."""
            obs, state, key, cum_reward, valid_mask = state_input

            key, act_key, obs_key = jrandom.split(key, 3)
            #if self.model_forward is not None:
            #    action = self.model_forward(policy_params, obs, rng_net)
            #else:
            scripted_action, leader = script(state)
            #joint_action = self.action_space.sample(act_key, samples=self.n_agents+self.n_scripted)
            action = scripted_action
            # Step the environment as normally.
            (next_obs, next_state, reward, done) = self.step(state, action)
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)

            carry = [
                next_obs,
                next_state,
                key,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = lax.scan(
            policy_step,
            [
                obs,
                state,
                key,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.params["settings"]['episode_size'],
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return
    
    def step(
            self,
            state: EnvState,
            action: chex.Array) -> Tuple[Sequence[chex.Array], EnvState, chex.Array, chex.Array]:
        """Performs one step in the environment

        `Args`:
            - `state (EnvState)`: State of the environment
            - `action (chex.Array)`: The joint action fro the agents and scripted entities.

        Returns:
            - `environment_step Tuple[Sequence[chex.Array], EnvState, chex.Array, chex.Array])`: A step in the environment.
        """
        X_ddot = action
        dt=1/60
        X_dot = state.X_dot + dt * X_ddot
        X = state.X + 60*dt*X_dot/la.norm(X_dot, axis=1)[:,None]
        X = jnp.clip(X,a_min=0,a_max=800)

        state = EnvState(X, X_dot, 1, state.goal) #type: ignore

        obs = self.get_obs(state)
        reward = jnp.array([1.0])
        
        norm_e = la.norm(state.goal-X[1])

        done = lax.select(norm_e<5, jnp.array([1.0]), jnp.array([0.0]))

        return (obs, state, reward, done)
    
    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"