import chex
import jax.numpy as jnp
import equinox as eqx
from jax import jit, vmap
from typing import Union


class BezierCurve3(eqx.Module):
    points: chex.Array

    def __init__(self, points):
        self.points = points

    def __call__(self, s: Union[chex.Scalar, jnp.float32, chex.Array]):
        @jit
        def evaluate(s):
            return (
                jnp.array(
                    [(1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * (1 - s) * s**2, s**3]
                )
                @ self.points
            )

        if isinstance(s, (chex.Scalar, jnp.float32)):
            return evaluate(s)
        elif isinstance(s, chex.Array):
            return vmap(evaluate)(s)
