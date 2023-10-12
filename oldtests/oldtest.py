import jax.random as jrandom
import jax
import jax.numpy as jnp
import time
import tomli
from pax import make
from pax.models.mlp import A2CNet
from pax.managers.trainer import Trainer
from pax.render.render import render, matplotlib_render


def test_1(console_args):
    env_name = "prob_env"
    env, config = make(env_name)

    key_input = jrandom.PRNGKey(config["settings"]["seed"])
    key, key_model = jrandom.split(key_input)
    trainer = Trainer(env, key_model)

    s = time.time()
    (obs, act, batch_reward) = trainer(key)
    print(f"Time for trainer: {time.time()-s}")

    if console_args.render in ["human", "Human", "h", "H"]:
        render(env, obs, record=False)

    elif console_args.render in ["matplotlibe", "m", "M", "mat"]:
        matplotlib_render(env, obs, record=False)


TESTS = {"1": test_1}

def run(console_args):
    s = time.time()
    #TESTS[console_args.test](console_args)
    TESTS["1"](console_args)
    print(f"Total time:{time.time()-s}")
