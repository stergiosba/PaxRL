import jax.random as jrandom
import jax
import time
from pax import make
from pax.render.render import render


def run_test_1(console_args):
    env = make("prob_env")
    
    key = jrandom.PRNGKey(env.params["settings"]["seed"])
    keys = jrandom.split(key, 1)
    s = time.time()
    O, A, R, _, _, cr = env.batch_rollout(keys)
    print(time.time()-s)

    render(env, O, 0, console_args.render)