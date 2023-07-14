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
        - `actions (chex.Array)`: Discrete actions that can take lists.
        - `mapping: (chex.Scalar)`: Possible mapping #TODO
        - `dtype: (Union[jnp.float32, jnp.int32])`: Datatype of the action.
    """
    actions: chex.Array
    mapping: chex.Scalar
    dtype: Union[jnp.float32, jnp.int32]

    def __init__(self, act_range:Sequence, dtype=jnp.float32, mapping=0):
        """tt

        `Args`:
            - dimension (int): _description_
            - dtype (_type_, optional): _description_. Defaults to jnp.float32.
            - mapping (int, optional): _description_. Defaults to 0.
        """
        self.actions = jnp.linspace(*act_range)
        self.mapping = mapping
        self.dtype = dtype

    @eqx.filter_jit
    def sample(self, key: chex.PRNGKey, shape: Tuple) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jrandom.choice(key, self.actions, shape=shape)

    @property
    def size(self) -> int:
        return self.actions.size
    
    @property
    def shape(self) -> Tuple:
        return self.actions.shape
    
    def __repr__(self):
        return f"{__class__.__name__}({self.actions}, {self.dtype})"

class MultiDiscrete(Space):
    """
    Minimal jittable class for multi discrete spaces.

    `Attributes`:
        - `actions (chex.Array)`: Discrete actions that can take lists.
        - `mapping: (chex.Scalar)`: Possible mapping #TODO
        - `dtype: (Union[jnp.float32, jnp.int32])`: Datatype of the action.
    """
    actions: chex.Array
    mapping: chex.Scalar
    dtype: Union[jnp.float32, jnp.int32]

    def __init__(self, act_range:Sequence, dtype=jnp.float32, mapping=0):
        """

        `Args`:
            - dimension (int): _description_
            - dtype (_type_, optional): _description_. Defaults to jnp.float32.
            - mapping (int, optional): _description_. Defaults to 0.
        """
        low, high, step = act_range
        
        X,Y = jnp.mgrid[low:high+0.1:step, low:high+0.1:step]
        self.actions = jnp.vstack((X.flatten(), Y.flatten())).T

        self.mapping = mapping
        self.dtype = dtype

    @eqx.filter_jit
    def sample(self, key: chex.PRNGKey, shape: Tuple) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jrandom.choice(key, self.actions, shape=shape)
    
    def contains(self, x: int) -> chex.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond
    
    @property
    def shape(self) -> Tuple:
        return self.actions.shape

    @property
    def size(self) -> int:
        return self.actions.shape[0]
    
    def __repr__(self):
        return f"{__class__.__name__}({self.actions}, {self.dtype})"


class Box(Space):
    """ Jittable space for a possibly unbounded Box in R^n

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
    def sample(self, key:chex.PRNGKey) -> chex.Array:
        """Generates a single random sample inside the Box.
        Args:
            `key (chex.PRNGKey)` : A PRNGKey used for Jax randomness.
        Returns:
            `sample (chex.Array`): Uniform sample from the Box
        """
        
        return jrandom.uniform(
            key, shape=self.shape, dtype=self.dtype, minval=self.low, maxval=self.high)
    
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
        return f"{__class__.__name__}({self.low}, {self.high}, {self.shape}, {self.dtype})"