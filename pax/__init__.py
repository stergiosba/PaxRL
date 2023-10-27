import tomli
import pax.core.state as state
import pax.core.action as action
import pax.core.actors as actors
import pax.core.spaces as spaces
from pax.utils.read_toml import read_config
from typing import Dict, Tuple
from pax.core.environment import Environment


__version__ = "0.2.0"

def make(env_name:str = "env_cfg")-> Environment:

    config = read_config(env_name)        
    return Environment(config)

