import chex
import jax.numpy as jnp
from jax import jit, vmap
import equinox as eqx


class BezierCurve3(eqx.Module):
    points: jnp.ndarray

    def __init__(self, points):
        self.points = points

    @jit
    def __call__(self, s):
        @jit
        def evaluate(s):
            return (
                jnp.array(
                    [(1 - s) ** 3, 3 * s * (1 - s) ** 2, 3 * (1 - s) * s**2, s**3]
                )
                @ self.points
            )

        return vmap(evaluate)(s)


class BezierCurve5(eqx.Module):
    points: jnp.ndarray

    def __init__(self, points: chex.Array) -> chex.Array:
        self.points = points

    @jit
    def __call__(self, s: jnp.float32) -> chex.Array:
        @jit
        def evaluate(s):
            return (
                jnp.array(
                    [
                        (1 - s) ** 5,
                        5 * s * (1 - s) ** 4,
                        10 * (1 - s) ** 2 * s**3,
                        10 * (1 - s) * s**4,
                        s**5,
                    ]
                )
                @ self.points
            )

        return vmap(evaluate)(s)
