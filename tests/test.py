import jax.random as jrandom
import jax
import time
from pax import make
from pax.render.render import render

def run_test_1():
    
    env = make("prob_env")
    
    key = jrandom.PRNGKey(env.params["settings"]["seed"])
    keys = jrandom.split(key, 1000)
    s = time.time()
    O, A, R, _, _, cr = env.batch_rollout(keys)
    print(time.time()-s)
    render(env, O, 340)