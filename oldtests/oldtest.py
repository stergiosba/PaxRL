import jax.random as jrandom
import jax
import jax.numpy as jnp
import time
from pax import make
from pax.models.mlp import A2CNet
from pax.managers.trainer import Trainer
from pax.render.render import render, matplotlib_render


def test_1(console_args):
    env_name = "prob_env"
    env = make(env_name)

    key_input = jrandom.PRNGKey(env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)
    trainer = Trainer(env, key_model, "train_cfg")

    s = time.time()
    if console_args.profile == "y":
        with jax.profiler.trace("/tmp/tensorboard"):
            # Run the operations to be profiled
            (obs, state, act, batch_reward) = trainer(key)
            obs.block_until_ready()
    else:
        (obs, state, act, batch_reward) = trainer(key)

    print(f"Time for trainer: {time.time()-s}")

    if console_args.render in ["human", "Human", "h", "H"]:
        render(env, state, record=env.params.settings["record"])

    elif console_args.render in ["matplotlib", "m", "M", "mat"]:
        matplotlib_render(env, state)


TESTS = {"1": test_1}

def run(console_args):
    s = time.time()
    #TESTS[console_args.test](console_args)
    TESTS["1"](console_args)
    print(f"Total time:{time.time()-s}")
