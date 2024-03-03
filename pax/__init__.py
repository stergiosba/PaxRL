import pax.core.state as state
import pax.core.action as action
import pax.core.spaces as spaces
import pax.core.environment as environment
import pax.training as training
import pax.wrappers as wrappers
from pax.utils.read_toml import read_config
from typing import Dict, Tuple
from pax.environments import Prober, Target


__version__ = "0.2.0"


def make(env_name: str, train: bool = False) -> environment.Environment:
    """Create an environment from a TOML file."""

    cfg = read_config(env_name)
    config = cfg["environment"]
    train_config = cfg["ppo"]

    if env_name == "Target-v0":
        env = Target(config)

    elif env_name == "Prober-v0":
        env = Prober(config)

    if train:
        return env, train_config
    else:
        return env
