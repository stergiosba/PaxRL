import jax.random as jrandom
import jax
import time
from pax import make
from pax.render.render import render

def run_test_1(console_args):
    env = make("prob_env")
    
    key = jrandom.PRNGKey(env.params["settings"]["seed"])
    keys = jrandom.split(key, env.params["parallel"]["n_env"])

    s = time.time()
    jax_batch_rollout = jax.jit(env.batch_rollout, backend=console_args.device)
    if console_args.profile:
        with jax.profiler.trace(console_args.profile, create_perfetto_link=True):
            O, A, R, _, _, cr = jax_batch_rollout(keys)
    else:
        O, A, R, _, _, cr = jax_batch_rollout(keys)
    #O.block_until_ready()
    print(time.time()-s)

    render(env, O, console_args.render, 0)
