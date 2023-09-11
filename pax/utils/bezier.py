import chex
import jax.numpy as jnp
import equinox as eqx
from jax import jit, vmap
from typing import Union


class BezierCurve3(eqx.Module):
    points: chex.ArrayDevice

    def __init__(self, points):
        self.points = points

    @jit
    def eval(self, s: chex.Scalar) -> chex.ArrayDevice:
        return (
            jnp.array(
                [(1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * (1 - s) * s**2, s**3]
            )
            @ self.points
        )

    @jit
    def veval(self, s: chex.ArrayDevice) -> chex.ArrayDevice:
        return vmap(self.eval)(s)
