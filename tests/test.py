import jax.random as jrandom
import time
from pax import make
from pax.render.render import render

def run_test_1():
    
    env = make("prob_env")

    key = jrandom.PRNGKey(env.params["settings"]["seed"])
    s = time.time()
    O, A, R, _, _, cr = env.single_rollout(key)
    print(time.time()-s)

    render(env, O)
