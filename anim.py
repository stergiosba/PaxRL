#%%
import jax.nn as jnn
import jax.random as jrandom
import jax.numpy as jnp

key = jrandom.key(1)

logits = jnp.array([.00001, 59, .00001])
# %%
samples = 5999
sums=0
for _ in range(samples):
    s = jrandom.categorical(key,logits=logits)
    key, _ = jrandom.split(key, 2)
    if s==0 or s==2:
        sums+=1
# %%
