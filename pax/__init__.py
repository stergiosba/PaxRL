import tomli
import pax.core.state as state
import pax.core.action as action
import pax.core.actors as actors
import pax.core.spaces as spaces
from typing import Dict, Tuple
from pax.core.environment import Environment


__version__ = "0.1.0"


def make(env_name: str)-> Tuple[Environment, Dict]:

    with open(f"{env_name}.toml",mode="rb") as config_file:
        config = tomli.load(config_file)
        
    return (Environment(config), config) #type: ignore

