import tomli
from typing import Dict, Tuple
from pax.core.environment import Environment


def make(env_name: str)-> Tuple[Environment, Dict]:

    with open(f"{env_name}.toml",mode="rb") as config_file:
        config = tomli.load(config_file)
        
    return (Environment(config), config) #type: ignore
