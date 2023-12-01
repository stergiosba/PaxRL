import chex
import jax.numpy as jnp
import equinox as eqx
from pax.utils.bezier import BezierCurve3


class EnvState(eqx.Module):
    """The environment state (Multiple Agents)

    `Args`:
        - `X (chex.Array)`: Position of every Agents.
        - `X_dot (chex.Array)`: Velocity of every Agent.
        - `leader (int)`: The id of the leader agent.
        - `goal (chex.Array)`: The location of the goal.
    """

    X: chex.ArrayDevice
    X_prev: chex.ArrayDevice
    X_dot: chex.ArrayDevice
    X_dot_prev: chex.ArrayDevice
    B: chex.ArrayDevice
    leader: chex.ArrayDevice
    curve: BezierCurve3
    t: int

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"