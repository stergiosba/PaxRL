import os
import time
import jax.random as jrandom
import jax.numpy.linalg as la
import numpy as np
import pax.training as paxt
import pandas as pd
from pax.training.models import Agent
from pax import make

def prober_test(env_name="Prober-v0"):
    env, train_config = make(env_name, train=True)

    key_input = jrandom.PRNGKey(env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)

    model = Agent(env, key_model)
    trainer = paxt.Trainer(env)

    s = time.time()

    num_total_epochs, log_steps, log_return = trainer(model, key, train_config)


    df = pd.DataFrame({"log_steps": log_steps, "log_return": log_return})
    # create a folder for th save if it does not exist
    folder_name = "log_experiments"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    if not os.path.exists(f"{folder_name}/{env.name}"):
        os.mkdir(f"{folder_name}/{env.name}")
    df.index.rename("epoch", inplace=True)
    df.to_csv(f"{folder_name}/{env.name}/ppo_{time.strftime('%m%d%Y_%H%M%S', time.gmtime())}.csv")
    
    print(f"Time for trainer: {time.time()-s}")


def target_test(env_name="Target-v0"):
    env, train_config = make(env_name, train=True)

    key_input = jrandom.PRNGKey(env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)

    model = Agent(env, key_model)
    trainer = paxt.Trainer(env)

    s = time.time()

    num_total_epochs, log_steps, log_return = trainer(model, key, train_config)
    print(f"Time for trainer: {time.time()-s}")



def selector(env):
    if env == "target":
        target_test()
    elif env == "prober":
        prober_test()

__all__ = ["selector"]