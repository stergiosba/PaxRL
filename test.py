#%%
#%load_ext autoreload
#%autoreload 2
import importlib
from typing import Tuple
import core
core = importlib.reload(core)
from core import Environment
import jax.random as jrandom
import tomli
import time
import jax


def make(env_name)-> Environment:
    with open(env_name+".toml",mode="rb") as tomlfile:
        env_params = tomli.load(tomlfile)

    return Environment(env_params)

env = make("prob_env")

key = jrandom.PRNGKey(env.params["settings"]["seed"])
s = time.time()
O, A, R, _, _, cr = env.single_rollout(key)
print(time.time()-s)
#env.show_results(O,A,R)
#print(O)

# %%
env.render(O)
# %%
