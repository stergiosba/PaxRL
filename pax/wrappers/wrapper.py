from typing import Any
import equinox as eqx
from pax import environment


class Wrapper(eqx.Module):
    env: environment.Environment

    def __init__(self, env: environment.Environment):
        self.env = env

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, action, *args, **kwargs):
        raise NotImplementedError
    
    def __getattr__(self, name):
        if name in self.env.__dict__:
            return getattr(self.env, name)
        else:
            return getattr(self.env, name)