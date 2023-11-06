import chex
import jax.random as jrandom
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import TypeVar, Optional, Tuple, Union, Sequence, Any, Dict
from collections import OrderedDict


class Space(eqx.Module):
    """
    PyTree implementaion of superclass Space based on equinox modules (dataclass)
    """

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def contains(self, x: chex.Scalar) -> chex.Scalar:
        raise NotImplementedError


class Discrete(Space):
    """
    Minimal jittable class for discrete spaces.

    `Attributes`:
        - `categories (chex.Array)`: Discrete categories.
        - `mapping: (chex.Scalar)`: Possible mapping #TODO
        - `dtype: (Union[jnp.float32, jnp.int32])`: Datatype of the action.
    """

    categories: chex.ArrayDevice
    dtype: Union[jnp.float32, jnp.int32]
    mapping: chex.Scalar

    def __init__(self, act_range: Sequence, dtype=jnp.float32, mapping=0):
        """A discrete space of possible actions.

        `Args`:
            - act_range (int): Range of possible actions.
            - dtype (_type_, optional): Data format. Defaults to jnp.float32.
            - mapping (int, optional): _description_. Defaults to 0.
        """
        self.categories = jnp.linspace(*act_range)
        self.dtype = dtype
        self.mapping = mapping

    @eqx.filter_jit
    def sample_old(self, key: chex.PRNGKey, shape: Tuple) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jrandom.choice(key, self.categories, shape=shape)

    @eqx.filter_jit
    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jrandom.choice(key, self.categories, shape=self.shape)

    @property
    def size(self) -> int:
        return self.categories.size

    @property
    def shape(self) -> Tuple:
        return self.categories.shape

    def __repr__(self):
        return f"{__class__.__name__}({self.categories}, {self.dtype})"


class MultiDiscrete(Space):
    """
    Minimal jittable class for multi discrete spaces.

    `Attributes`:
        - `categories (chex.Array)`: Discrete categories of the space.
        - `mapping: (chex.Scalar)`: Possible mapping #TODO
        - `dtype: (Union[jnp.float32, jnp.int32])`: Datatype of the action.
    """

    categories: chex.ArrayDevice
    mapping: chex.Scalar
    dtype: Union[jnp.float32, jnp.int32]

    def __init__(self, act_range: Sequence, dtype=jnp.float32, mapping=0):
        """

        `Args`:
            - dimension (int): _description_
            - dtype (_type_, optional): _description_. Defaults to jnp.float32.
            - mapping (int, optional): _description_. Defaults to 0.
        """
        low, high, step = act_range

        X, Y = jnp.mgrid[low : high + 0.1 : step, low : high + 0.1 : step]
        self.categories = jnp.vstack((X.flatten(), Y.flatten())).T
        self.dtype = dtype
        self.mapping = mapping

    @eqx.filter_jit
    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jrandom.choice(key, self.categories, shape=self.shape)

    def contains(self, x: int) -> chex.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond

    @property
    def shape(self) -> Tuple:
        return self.categories.shape

    @property
    def size(self) -> int:
        return self.categories.shape[0]

    def __repr__(self):
        return f"{__class__.__name__}({self.categories.shape}, {self.categories}, {self.dtype})"


class Box(Space):
    """Jittable space for a possibly unbounded Box in R^n

    Args:
        Space [Inheritance]: Abstact class Space

    Returns:
        chex.Array: Uniform sample
    """

    low: Union[chex.Scalar, chex.Array]
    high: Union[chex.Scalar, chex.Array]
    shape: Tuple
    dtype: Union[jnp.float32, jnp.int32]

    @eqx.filter_jit
    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Generates a single random sample inside the Box.
        Args:
            `key (chex.PRNGKey)` : A PRNGKey used for Jax randomness.
        Returns:
            `sample (chex.Array`): Uniform sample from the Box
        """

        return jrandom.uniform(
            key, shape=self.shape, dtype=self.dtype, minval=self.low, maxval=self.high
        )

    @eqx.filter_jit
    def contains(self, x: int) -> chex.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond

    @property
    def size(self):
        return jnp.array(self.shape).prod()

    def __repr__(self):
        return (
            f"{__class__.__name__}({self.low}, {self.high}, {self.shape}, {self.dtype})"
        )
