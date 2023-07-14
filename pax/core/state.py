import chex
import jax.numpy as jnp
import equinox as eqx

class EntityState(eqx.Module):
    """Physical state of an entity in the environment.

    `Args`:
        - `pos (chex.Array)`: Position of the entity
        - `vel (chex.Array)`: Velocity of the entity
    """    
    pos: chex.Array = jnp.zeros(2)
    vel: chex.Array = jnp.zeros(2) 

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"

class AgentState(EntityState):
    """ State of agents (inherits from physical state and adds communication)

    `Args`:
        - `com (chex.Array)`: Communication state   
    """
    com: chex.Array = jnp.empty(1)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
    
