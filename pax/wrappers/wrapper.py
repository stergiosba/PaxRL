from typing import Any
from pax import environment


class Wrapper:
    def __init__(self, env: environment.Environment):
        self.env = env

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
    
    def __getattr__(self, name):
        return getattr(self.env, name)