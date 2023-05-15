import chex
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
import jax.numpy.linalg as la
from typing import Sequence, Union, Tuple, Sequence, Any, Dict
from jax import jit, lax, devices
from space import Discrete, Box

class EntityState(eqx.Module):
    """Entity state in the environment.

    Args:
        pos (chex.Array[2]): Position of the entity
        vel (chex.Array[2]): Velocity of the entity
    """    
    pos: chex.Array = jnp.zeros(2)
    vel: chex.Array = jnp.zeros(2) 

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"

class AgentState(EntityState):
    """ State of agents (including communication and internal/mental state)

    Args:
        EntityState (_type_): _description_
    """
    com: chex.Array = jnp.empty(1)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"

class Action(eqx.Module):
    """ Action of the agent

    Args:
        a (chex.Array): physical action),
        c (chex.Array): communication action
    """    
    a: chex.Array = jnp.empty(1)
    c: chex.Array = jnp.empty(1)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
    
class Entity(eqx.Module): 
    """ Properties and state of physical world entity

    Args:
        id (Union[chex.Scalar,int]): The identification number or the entity
        size (Union[chex.Scalar,float32]): Entity size
        movable (Union[chex.Scalar,int]): Flag to denote if entity can be moved or be pushed
        collide (Union[chex.Scalar,int]): Flag to denote if entity responds to collisions
        density (Union[chex.Scalar,float32]): Material density (affects mass)
        color (Union[chex.Array[4],Tuple]): The color of the agent
        max_speed (Union[chex.Scalar,float32]): The maximum allowed speed of entity
        accel (chex.Array[2]): The acceleration of the entity
        state (Union[EntityState, None]): The state of the entity (position/velocity)
        init_mass (Union[chex.Scalar,float32]): The initial mass of the entity (might change)
    """
    id: Union[chex.Scalar,int] = 0
    size: Union[chex.Scalar,jnp.float32] = 1
    movable: Union[chex.Scalar,int] = 0
    collide: Union[chex.Scalar,int] = 0
    density: Union[chex.Scalar,jnp.float32] = 1
    color: Union[chex.Array,Tuple] = (0, 0, 0, 255)
    max_speed: Union[chex.Scalar,jnp.float32] = 1
    accel: chex.Array = jnp.zeros(2)
    state: Union[EntityState, None] = None
    init_mass: Union[chex.Scalar,jnp.float32] = 1

    @property
    def mass(self):
        return self.init_mass
    
    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
    
class Agent(Entity):
    """ Properties and state of physical world entity
    Args:
        id (Union[chex.Scalar,int]): The identification number or the entity
        size (Union[chex.Scalar,float32]): Entity size
        movable (Union[chex.Scalar,int]): Flag to denote if entity can be moved or be pushed
        collide (Union[chex.Scalar,int]): Flag to denote if entity responds to collisions
        silent (Union[chex.Scalar,int]): Flag to denote if entity can communicate
        observant: Union[chex.Scalar,jnp.bool_]: Flag to denote if entity observes the world
        density (Union[chex.Scalar,float32]): Material density (affects mass)
        color (Union[chex.Array[4],Tuple]): The color of the agent
        max_speed (Union[chex.Scalar,float32]): The maximum allowed speed of entity
        accel (chex.Array[2]): The acceleration of the entity
        state (Union[EntityState, None]): The state of the entity (position/velocity)
        init_mass (Union[chex.Scalar,float32]): The initial mass of the entity (might change)
    """
    movable: Union[chex.Scalar,int] = True
    silent: Union[chex.Scalar,int] = False
    observant: Union[chex.Scalar,jnp.bool_] = True
    state: Union[EntityState, AgentState] = AgentState()
    action: Action = Action()
    color: Union[chex.Array,Tuple] = (255, 0, 0, 155)
    action_callback: Any = lambda: ()
        
    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


class EnvState(eqx.Module):
    """The environment state (Multiple Agents)

    Args:
        X (chex.Array): Position of every Agents
        X_dot (chex.Array): Velocity of every Agent
    """
    X: chex.Array
    X_dot: chex.Array
    
    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
    
class Environment(eqx.Module):
    """Environment class

    Args:
        n_agents 
        action_space (Union[Discrete, Box]): Action space allowed to the agents.
        observation_space (Union[Discrete, Box]): Observation space.
        params (Dict): Parameters given to environment form the TOML file.
    """
    n_agents: Union[chex.Scalar,int]
    action_space: Union[Discrete, Box]
    observation_space: Union[Discrete, Box]
    params: Dict

    def __init__(self, params):
        self.params = params
        self.n_agents = self.params["settings"]["n_agents"]
        self.create_spaces()

    def create_spaces(self):
        actions = jnp.array(self.params["action_space"]["actions"])
        self.action_space = Discrete(actions)

        o_dtype = jnp.float32
        o_low, o_high, o_shape = self.params["observation_space"].values()
        self.observation_space = Box(o_low, o_high, tuple(o_shape), o_dtype)
    
    @eqx.filter_jit
    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.X, state.X_dot])

    @eqx.filter_jit
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        init_X = jrandom.uniform(key, minval=250, maxval=650, shape=(self.n_agents,2))
        init_X_dot = jrandom.uniform(key, minval=-1, maxval=1, shape=(self.n_agents,2))
        state = EnvState(
            X=init_X,
            X_dot=init_X_dot
        )
        
        return (self.get_obs(state), state)
    
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
            action = self.action_space.sample(act_key,samples=self.n_agents)
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
    def step(self, state:EnvState, joint_action):
        X_ddot = joint_action
        dt=1/60
        X_dot = state.X_dot + dt * X_ddot
        X = state.X + dt * 50*X_dot/la.norm(X_dot)
        state = EnvState(X, X_dot)
        obs = self.get_obs(state)
        reward = jnp.array([1.0])
        terminated = jnp.array([0.0])

        return (obs, state, reward, terminated)
    
    
    def render(self, O):
        import pyglet as pg
        import numpy as np

        O = np.array(O)

        window = pg.window.Window(800,800, caption="HORC")
        batch = pg.graphics.Batch()
        Agents = []
        for i in range(123):
            agent = pg.shapes.Circle(x=O[i,0,0,0],y=O[i,0,0,1],radius=10,color=(255,0,0,155),batch=batch)
            Agents.append(agent)
            
        probing_agent = pg.shapes.Star(x=O[-1,0,0,0],y=O[-1,0,0,1],num_spikes=5, inner_radius=10,outer_radius=5,color=(0,0,255,255),batch=batch)
        Agents.append(probing_agent)
        window.simulationClock = pg.clock

        t=[0]
        @window.event
        def on_key_press(symbol, mods):
            if symbol==pg.window.key.Q:
                window.on_close()
                pg.app.exit()
            if symbol==pg.window.key.R:
                t[0]=0

        @window.event
        def on_draw():
            window.clear()
            batch.draw()

        def update(dt):
            window.clear()
            # update our circle's position
            for i, agent in enumerate(Agents):
                agent.position = O[t[0],0,i]
            batch.draw()
            pg.image.get_buffer_manager().get_color_buffer().save(f'saved/screenshot_frame_{t[0]}.png')
            t[0] += 1


        window.simulationClock.schedule_interval(update, 1/60)

        #window.simulationClock.schedule_interval(loop, 1)

        pg.app.run()