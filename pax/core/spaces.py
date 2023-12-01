import chex
import jax.random as jrandom
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import Callable, Optional, Tuple, Union, Sequence, Any, Dict
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

    n: int
    start: int
    dtype: Union[jnp.float32, jnp.int32]
    mapping: chex.Scalar

    def __init__(self, n: int, start: int = 0, dtype=jnp.float32, mapping=0):
        """A discrete space of possible actions.

        `Args`:
            - `act_range (int)`: Range of possible actions.
            - `dtype (_type_, optional)`: Data format. Defaults to jnp.float32.
            - `mapping (int, optional)`: _description_. Defaults to 0.
        """
        self.n = n
        self.start = start
        self.dtype = dtype
        self.mapping = mapping

    @eqx.filter_jit
    def sample(self, key: chex.PRNGKey, shape=(1,)) -> chex.Array:
        """Sample random action uniformly from set of categorical choices.

        Args:
            - `key (chex.PRNGKey)`: Random key for sampling.
            - `shape (tuple, optional)`: The shape of the sample. Defaults to (1,).

        Returns:
            - `sample` chex.Array: A random sample from the discrete space.
        """
        return jrandom.randint(key, shape=shape, minval=self.start, maxval=self.n)

    @property
    def size(self) -> int:
        return self.n

    @property
    def shape(self) -> Tuple:
        return self.n

    def __repr__(self):
        return f"{__class__.__name__}({self.start}, {self.n}, {self.dtype})"


class SeparateGrid(Space):
    """
    Minimal jittable class for seperate grid spaces. Essentially a 2D MultiDiscrete.

    `Attributes`:
        - `categories (chex.Array)`: Discrete categories of the space.
        - `dtype: (Union[jnp.float32, jnp.int32])`: Datatype of the action.
    """

    axis_1: chex.ArrayDevice
    axis_2: chex.ArrayDevice
    dtype: Union[jnp.float32, jnp.int32]

    def __init__(self, act_range: Sequence, dtype=jnp.float32):
        """A 2D Grid of possible actions.

        Args:
            act_range (Sequence): The range of possible actions for both axis.
            dtype (type, optional): The data format. Defaults to jnp.float32.
        """
        if len(act_range) == 2:
            low, high = act_range
            step = 1
        else:
            low, high, step = act_range

        self.axis_1 = jnp.mgrid[low : high + 0.1 : step]
        self.axis_2 = jnp.mgrid[low : high + 0.1 : step]
        self.dtype = dtype

    @eqx.filter_jit
    def sample(
        self, key: chex.PRNGKey, shape: chex.ArrayDevice = (1,)
    ) -> chex.ArrayDevice:
        """Sample random action uniformly from set of categorical choices."""
        key_1, key_2 = jrandom.split(key)
        axis_1_sample = jrandom.choice(key_1, self.axis_1, shape=shape)
        axis_2_sample = jrandom.choice(key_2, self.axis_2, shape=shape)
        return jnp.vstack((axis_1_sample, axis_2_sample)).T

    def contains(self, x: int) -> chex.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond

    def map_action(self, action_idces: chex.ArrayDevice) -> chex.ArrayDevice:
        return jnp.array((self.axis_1[action_idces[0]], self.axis_2[action_idces[1]])).T

    @property
    def size(self) -> int:
        """Returns the size of the space.

        Returns:
            - `size (int)`: The size of the space.
        """
        return self.axis_1.shape[0] * self.axis_2.shape[0]

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the space.

        Returns:
            - `shape (Tuple)`: The shape of the space.
        """
        return (self.axis_1.shape[0], self.axis_2.shape[0])
    
    @property
    def n_axes(self) -> int:
        """Returns the number of distinct axes.

        Returns:
            - `n_axes (int)`: The number of distinct axes.
        """
        return self.axis_1.shape[0] + self.axis_2.shape[0]

    def __repr__(self):
        return f"{__class__.__name__} (self.{self.axis_1},{self.axis_2}, {self.dtype})"


class Grid(Space):
    """
    Minimal jittable class for grid spaces.

    `Attributes`:
        - `categories (chex.Array)`: Discrete categories of the space.
        - `dtype: (Union[jnp.float32, jnp.int32])`: Datatype of the action.
    """

    categories: chex.ArrayDevice
    dtype: Union[jnp.float32, jnp.int32]

    def __init__(self, act_range: Sequence, dtype=jnp.float32):
        """A 2D Grid of possible actions.

        Args:
            act_range (Sequence): The range of possible actions for both axis.
            dtype (type, optional): The data format. Defaults to jnp.float32.
        """
        if len(act_range) == 2:
            low, high = act_range
            step = 1
        else:
            low, high, step = act_range

        axis_1, axis_2 = jnp.mgrid[low : high + 0.1 : step, low : high + 0.1 : step]
        self.categories = jnp.vstack((axis_1.flatten(), axis_2.flatten())).T
        self.dtype = dtype

    @eqx.filter_jit
    def sample(
        self, key: chex.PRNGKey, shape: chex.ArrayDevice = (1,)
    ) -> chex.ArrayDevice:
        """Sample random action uniformly from set of categorical choices."""
        return jrandom.choice(key, self.categories, shape=shape)

    def contains(self, x: int) -> chex.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond

    def map_action(self, action_number):
        return self.categories[action_number]

    @property
    def shape(self) -> Tuple:
        return self.categories.shape

    @property
    def size(self) -> int:
        return jnp.array(self.categories.shape).prod()

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
