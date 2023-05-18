from pax.core.environment import Environment
import tomli

def make(env_name)-> Environment:
    with open(env_name+".toml",mode="rb") as tomlfile:
        env_params = tomli.load(tomlfile)

    return Environment(env_params)