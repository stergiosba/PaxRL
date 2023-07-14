import jax.random as jrandom
import jax
import jax.numpy as jnp
import time
import tomli
from pax import make
from pax.models.mlp import A2CNet
from pax.managers.trainer import Trainer
from pax.render.render import render


def test_1(console_args):
    s = time.time()
    env = make("prob_env")
    
    key = jrandom.PRNGKey(env.params["settings"]["seed"])
    keys = jrandom.split(key, env.params["parallel"]["n_env"])

    jax_batch_rollout = jax.jit(env.batch_rollout, backend=console_args.device)
    if console_args.profile:
        with jax.profiler.trace(console_args.profile, create_perfetto_link=True):
            O, A, R, _, _, cr = jax_batch_rollout(keys)
    else:
        O, A, R, _, _, cr = jax_batch_rollout(keys)
    #O.block_until_ready()
    print(time.time()-s)
    #print(O)
    render(env, O, console_args.render, 0)


def test_2(console_args):
    env_name = "prob_env"
    env, config = make(env_name)

    key = jrandom.PRNGKey(config["settings"]["seed"])

    trainer = Trainer(env, key)

    keys = jrandom.split(key, config["settings"]["n_env"])
    s = time.time()
    (obs, act, batch_reward) = trainer(keys)
    print(f"Time for trainer: {time.time()-s}")
    print(obs[0].shape)


TESTS = {"1": test_1, "2":test_2}

def run_test(console_args):
    s = time.time()
    TESTS[console_args.test](console_args)
    print(f"Total time:{time.time()-s}")