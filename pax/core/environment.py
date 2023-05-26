import chex
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
import jax.numpy.linalg as la
from typing import Sequence, Union, Tuple, Sequence, Any, Dict
from jax import jit, vmap, lax, devices, debug
from pax.core.space import Discrete, Box
from pax.core.state import EntityState, AgentState, EnvState
from pax.core.actors import ScriptedEntity, Agent
from pax.core.script import script
import logging
from pax.core.paxlog import Paxlogger
            
    
class Environment(eqx.Module):
    """Environment class

    Args:
        n_agents 
        action_space (Union[Discrete, Box]): Action space allowed to the agents.
        observation_space (Union[Discrete, Box]): Observation space.
        params (Dict): Parameters given to environment form the TOML file.
    """
    params: Dict
    n_agents: int
    n_scripted_entities: int
    action_space: Union[Discrete, Box]
    observation_space: Union[Discrete, Box]
    logger: Any

    def __init__(self, params):
        self.params = params
        self.n_agents = self.params["settings"]["n_agents"]
        self.n_scripted_entities = self.params["settings"]["n_scripted_entities"]
        self.create_spaces()
        self.logger = logging.getLogger("PAX: ")
        self.logger.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(Paxlogger())

        self.logger.addHandler(ch)
        self.logger.debug("Started environment")

    def create_spaces(self):
        actions = jnp.array(self.params["action_space"]["actions"])
        self.action_space = Discrete(actions)

        o_dtype = jnp.float32
        o_low, o_high, o_shape = self.params["observation_space"].values()
        self.observation_space = Box(o_low, o_high, tuple(o_shape), o_dtype)
    
    @eqx.filter_jit
    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.X, state.X_dot]), state.leader

    @eqx.filter_jit
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Resets the environment."""
        init_X_scripted_entities = jnp.array([525,775])+50*jrandom.normal(
            key, shape=(self.n_scripted_entities,2)
            )
        init_X_dot_scripted_entities = jrandom.uniform(
            key, minval=-1, maxval=1, \
            shape=(self.n_scripted_entities,2)
            )
        
        init_X_agents = jrandom.uniform(
            key, minval=150, maxval=250, \
            shape=(self.n_agents,2)
            )
        init_X_dot_agents = jrandom.uniform(
            key, minval=-1, maxval=1, \
            shape=(self.n_agents,2)
            )
        
        leader = jrandom.randint(key, shape=(), minval=0, maxval=self.n_scripted_entities-1)
        
        state = EnvState(
            X=jnp.concatenate([init_X_scripted_entities,
                            init_X_agents]),
            X_dot=jnp.concatenate([init_X_dot_scripted_entities,
                            init_X_dot_agents]),
            leader = leader,
            t=0
        )
        
        return (self.get_obs(state), state)
    
    def batch_rollout(self, keys):
        batch_rollout = vmap(self.single_rollout, in_axes=(0))
        return batch_rollout(keys)
    
    @eqx.filter_jit(donate='all')
    def single_rollout(self, key):#, policy_params):
        """Rollout an environment episode with lax.scan."""
        # Reset the environment
        (obs, state) = self.reset(key)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, key, cum_reward, valid_mask = state_input

            key, act_key, obs_key = jrandom.split(key, 3)
            #if self.model_forward is not None:
            #    action = self.model_forward(policy_params, obs, rng_net)
            #else:
            scripted_action = script(state, self.n_scripted_entities)
            #scripted_action = jnp.zeros([self.n_scripted_entities,2])
            #debug.print("Script.py:{t}:{X}\n---", t=state.t, X=scripted_action)
            joint_action = self.action_space.sample(act_key, samples=self.n_agents)
            action = jnp.concatenate([scripted_action,joint_action])
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
            self.params["settings"]["episode_size"],
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        
        return obs, action, reward, next_obs, done, cum_return
    
    
    @eqx.filter_jit
    def step(self, state:EnvState, action):
        X_ddot = action
        dt=1/60
        X_dot = state.X_dot + dt * X_ddot
        X = state.X + 50*dt*X_dot/la.norm(X_dot, axis=1)[:,jnp.newaxis]
        t = state.t+dt
        #debug.print("{t}:{X}\n---", t=t, X=X_ddot)
        #debug.print("{t}:{X}\n---", t=t, X=X_ddot)
        state = EnvState(X, X_dot ,state.leader, t)
        obs = self.get_obs(state)
        reward = jnp.array([1.0]) 
        terminated = jnp.array([0.0])

        return (obs, state, reward, terminated)