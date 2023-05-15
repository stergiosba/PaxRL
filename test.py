#%%
#%load_ext autoreload
#%autoreload 2
import importlib
from typing import Tuple
import core
core = importlib.reload(core)
from core import Environment
import jax.random as jrandom
import tomli
import time

def make(env_name)-> Environment:
    with open(env_name+".toml",mode="rb") as tomlfile:
        env_params = tomli.load(tomlfile)

    return Environment(env_params)

env = make("prob_env")

key = jrandom.PRNGKey(env.params["settings"]["seed"])
s = time.time()
O, A, R, _, _, cr = env.single_rollout(key)
print(time.time()-s)
#env.show_results(O,A,R)
#print(O)

# %%
import pyglet as pg
import numpy as np

O = np.array(O)

window = pg.window.Window(800,800, caption="HORC")
batch = pg.graphics.Batch()
Agents = []
for i in range(123):
    agent = pg.shapes.Circle(x=O[i,0,0,0],y=O[i,0,0,1],radius=10,color=(255,0,0,155),batch=batch)
    Agents.append(agent)
    
probing_agent = pg.shapes.Star(x=O[-1,0,0,0],y=O[-1,0,0,1],num_spikes=5, inner_radius=10,outer_radius=5,color=(0,0,255,255),batch=batch)
Agents.append(probing_agent)
window.simulationClock = pg.clock

t=[0]
@window.event
def on_key_press(symbol, mods):
    if symbol==pg.window.key.Q:
        window.on_close()
    if symbol==pg.window.key.R:
        t[0]=0

@window.event
def on_draw():
    window.clear()
    batch.draw()

def update(dt):
    window.clear()
    # update our circle's position
    for i, agent in enumerate(Agents):
        agent.position = O[t[0],0,i]
    batch.draw()
    pg.image.get_buffer_manager().get_color_buffer().save(f'saved/screenshot_frame_{t[0]}.png')
    t[0] += 1


window.simulationClock.schedule_interval(update, 1/16)

#window.simulationClock.schedule_interval(loop, 1)

pg.app.run()

# %%
