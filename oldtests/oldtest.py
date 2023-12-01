import os
import time
import jax
import jax.random as jrandom
import jax.numpy as jnp
import jax.numpy.linalg as la
import numpy as np
import pax.training as tpax
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pax.training.models import Agent
from pax import make
from pax.render.render_old import render, matplotlib_render

matplotlib.use("Qt5Agg")


def test_1(console_args):
    env_name = "prob_env"
    env = make(env_name)

    key_input = jrandom.PRNGKey(env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)

    model = Agent(env, key_model)
    trainer = tpax.Trainer(env)

    s = time.time()
    if console_args.profile == "y":
        with jax.profiler.trace("/tmp/tensorboard"):
            # Run the operations to be profiled
            batch = trainer(model, key, "train_cfg")
            # (obs, state, act, batch_reward) = trainer(key, "train_cfg")
    else:
        num_total_epochs, log_steps, log_return = trainer(model, key, "train_cfg")




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

    # if console_args.render in ["human", "Human", "h", "H"]:
    #     render(env, state, record=env.params.settings["record"])

    # elif console_args.render in ["matplotlib", "m", "M", "mat"]:
    #     matplotlib_render(env, state)


TESTS = {"1": test_1}


def run(console_args):
    s = time.time()
    # TESTS[console_args.test](console_args)
    TESTS["1"](console_args)
    print(f"Total time:{time.time()-s}")
