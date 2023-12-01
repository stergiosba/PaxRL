import chex
import jax.numpy as jnp
import equinox as eqx


class Action(eqx.Module):
    """Action of the agent

    Args:
        a (chex.Array): physical action),
        c (chex.Array): communication action
    """

    a: chex.Array = jnp.empty(1)
    c: chex.Array = jnp.empty(1)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
