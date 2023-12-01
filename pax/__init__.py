import pax.core.state as state
import pax.core.action as action
import pax.core.spaces as spaces
import pax.core.environment as environment
import pax.training as training
import pax.render.render as render
from pax.utils.read_toml import read_config
from typing import Dict, Tuple
from pax.scenarios.prober import Proberenv


__version__ = "0.2.0"


def make(env_name: str = "prob_env") -> Proberenv:

    config = read_config(env_name)
    return Proberenv(config)
