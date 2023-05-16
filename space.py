from typing import Tuple, Union, Sequence, Any, Dict
from collections import OrderedDict
import chex
import jax.random as jrandom
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import jit
from timeit import timeit


class Space(eqx.Module):
    """
    PyTree implementaion of superclass Space
    """

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def contains(self, x: chex.Scalar) -> chex.Scalar:
        raise NotImplementedError


class Discrete(Space):
    """
    Minimal jittable class for discrete gymnax spaces.
    TODO: For now this is a 1d space. Make composable for multi-discrete.
    """
    actions: chex.Array
    mapping: chex.Scalar
    dtype: jnp.float32

    def __init__(self, actions: chex.Array, mapping=0):
        self.actions = actions
        self.mapping = mapping
        self.dtype = jnp.float32

    @eqx.filter_jit
    def sample(self, key: chex.PRNGKey, samples: int=1) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jrandom.choice(key, self.actions, shape=(samples,))
    
    def contains(self, x: int) -> chex.Array:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond

    def __repr__(self):
        return f"Discrete({self.actions}, {self.dtype})"


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
    dtype: jnp.float32
    
    @eqx.filter_jit
    def sample(self, key:chex.PRNGKey) -> chex.Array:
        r"""Generates a single random sample inside the Box.
        Args:
            chex.PRNGKey key: A PRNGKey used for Jax randomness.
        Returns:
            chex.Array: Uniform sample from the Box
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
    
    def __repr__(self):
        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"