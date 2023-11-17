import chex
import jax.numpy as jnp
import equinox as eqx
from typing import Union, Tuple, Any
from pax.core.state import EntityState, AgentState
from pax.core.action import Action


class Entity(eqx.Module):
    """Properties and state of physical world entity

    Args:
        id (chex.Scalar): The identification number or the entity
        size (chex.Scalar): Entity size
        movable (chex.Scalar): Flag to denote if entity can be moved or be pushed
        collide (chex.Scalar): Flag to denote if entity responds to collisions
        density (chex.Scalar): Material density (affects mass)
        color (Union[chex.Array[4],Tuple]): The color of the agent
        max_speed (chex.Scalar): The maximum allowed speed of entity
        accel (chex.Array[2]): The acceleration of the entity
        state (Union[EntityState, None]): The state of the entity (position/velocity)
        init_mass (chex.Scalar): The initial mass of the entity (might change)
    """

    id: Union[chex.Scalar, int] = 0
    size: Union[chex.Scalar, jnp.float32] = 10
    movable: Union[chex.Scalar, int] = 0
    collide: Union[chex.Scalar, int] = 0
    density: Union[chex.Scalar, jnp.float32] = 1
    color: Union[chex.Array, Tuple] = (0, 0, 0, 255)
    max_speed: Union[chex.Scalar, jnp.float32] = 1
    accel: chex.Array = jnp.zeros(2)
    state: Union[EntityState, None] = None
    init_mass: Union[chex.Scalar, jnp.float32] = 1

    @property
    def mass(self):
        return self.init_mass

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


class ScriptedEntity(Entity):
    """Properties and state of physical world entity

    Args:
        id (chex.Scalar): The identification number or the entity
        size (chex.Scalar): Entity size
        movable (chex.Scalar): Flag to denote if entity can be moved or be pushed
        collide (chex.Scalar): Flag to denote if entity responds to collisions
        density (chex.Scalar): Material density (affects mass)
        color (Union[chex.Array[4],Tuple]): The color of the agent
        max_speed (chex.Scalar): The maximum allowed speed of entity
        accel (chex.Array[2]): The acceleration of the entity
        state (Union[EntityState, None]): The state of the entity (position/velocity)
        init_mass (chex.Scalar): The initial mass of the entity (might change)
    """

    movable: chex.Scalar= True
    silent: chex.Scalar = False
    observant: Union[chex.Scalar, jnp.bool_] = True
    state: EntityState = EntityState()
    action: Action = Action()
    color: Union[chex.Array, Tuple] = (255, 0, 0, 155)
    action_callback: Any = lambda: ()

    @property
    def mass(self):
        return self.init_mass

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


class Agent(Entity):
    """Properties and state of physical world entity
    Args:
        id (chex.Scalar): The identification number or the entity
        size (chex.Scalar): Entity size
        movable (chex.Scalar): Flag to denote if entity can be moved or be pushed
        collide (chex.Scalar): Flag to denote if entity responds to collisions
        silent (chex.Scalar): Flag to denote if entity can communicate
        observant: Union[chex.Scalar,jnp.bool_]: Flag to denote if entity observes the world
        density (chex.Scalar): Material density (affects mass)
        color (Union[chex.Array[4],Tuple]): The color of the agent
        max_speed (chex.Scalar): The maximum allowed speed of entity
        accel (chex.Array[2]): The acceleration of the entity
        state (Union[EntityState, None]): The state of the entity (position/velocity)
        init_mass (chex.Scalar): The initial mass of the entity (might change)
    """

    movable: chex.Scalar = True
    silent: chex.Scalar = False
    observant: Union[chex.Scalar, jnp.bool_] = True
    state: Union[EntityState, AgentState] = AgentState()
    action: Action = Action()
    color: Union[chex.Array, Tuple] = (0, 0, 255, 155)
    action_callback: Any = lambda: ()

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
