import os
import time
import equinox as eqx
import jax.random as jrandom
import jax.numpy.linalg as la
import numpy as np
import pax.training as paxt
import pandas as pd
from pax.training.models import Agent, AgentCNN
from pax import make
from pax.wrappers import NormalizeObservationWrapper, FlattenObservationWrapper, LogWrapper

def prober_test(env_name="Prober-v0", save=False):
    env, train_config = make(env_name, train=True)

    wrapped_env = LogWrapper(FlattenObservationWrapper(env))

    key_input = jrandom.PRNGKey(wrapped_env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)

    model = Agent(wrapped_env, key_model)
    trainer = paxt.Trainer(wrapped_env)

    s = time.time()

    # num_total_epochs, log_steps, log_return = trainer(model, key, train_config)
    runner_state ,_ = eqx.filter_jit(trainer)(model, key, train_config)


    trainer.save("ppo_agent_final.eqx", {}, runner_state[0].params)

    # if save:
    #     df = pd.DataFrame({"log_steps": log_steps, "log_return": log_return})
    #     # create a folder for th save if it does not exist
    #     folder_name = "log_experiments"
    #     if not os.path.exists(folder_name):
    #         os.mkdir(folder_name)
    #     if not os.path.exists(f"{folder_name}/{env.name}"):
    #         os.mkdir(f"{folder_name}/{env.name}")
    #     df.index.rename("epoch", inplace=True)
    #     df.to_csv(f"{folder_name}/{env.name}/ppo_{time.strftime('%m%d%Y_%H%M%S', time.gmtime())}.csv")
    
    print(f"Time for trainer: {time.time()-s}")


def target_test(env_name="Target-v0"):
    env, train_config = make(env_name, train=True)

    # env =

    key_input = jrandom.PRNGKey(env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)

    model = Agent(env, key_model)
    _, static = eqx.partition(model, eqx.is_array)
    trainer = paxt.Trainer(env)
    
    s = time.time()

    runner_state, _ = trainer(model, key, train_config)

    model = eqx.combine(runner_state[0].params, static)
    trainer.save(f"ppo_agent_final.eqx", {}, model)
    print(f"Time for trainer: {time.time()-s}")



def selector(env):
    if env == "target":
        target_test()
    elif env == "prober":
        prober_test()

__all__ = ["selector"]