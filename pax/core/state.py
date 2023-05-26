import chex
import jax.numpy as jnp
import equinox as eqx

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
    """ State of agents (including communication and internal/physical state)

    Args:
        EntityState (_type_): Entity state
        
    """
    com: chex.Array = jnp.empty(1)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
    
    
class EnvState(eqx.Module):
    """The environment state (Multiple Agents)

    Args:
        X (chex.Array): Position of every Agents
        X_dot (chex.Array): Velocity of every Agent
        leader (int): The leader identification
        t (float): Time step
    """
    X: chex.Array
    X_dot: chex.Array
    leader: int
    t: float
    
    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"