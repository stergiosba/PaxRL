import json
import fire
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pax
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
from tensorflow_probability.substrates import jax as tfp
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def make_model(env_name, key):
    env, _ = pax.make(env_name)
    return env, pax.training.Agent(env, key)

def load(filename, model):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        return eqx.tree_deserialise_leaves(f, model)
    
    
def use_model(model, env, obs, key, action_mapping=True):
    value, split_logits = model(obs)
    multi_pi = tfp.distributions.Categorical(logits=split_logits)
    action = multi_pi.sample(seed=key)

    if action_mapping:
        action = env.action_space.map_action(action)
    
    action = action[None, :]
    return value, action

key = jr.PRNGKey(0)

env, model = make_model("Prober-v0", key)
agent = load("ppo_agent_9.eqx", model)
obs, state = env.reset(key)
obs = np.array(obs)

print(state)

# n = env.params.scenario["n_scripted_entities"]
# zero_action = np.concatenate([[20*np.ones(n)],[np.zeros(n)]]).transpose()


# O = []
# while True:
#     value, action = use_model(agent, env, obs, key)
#     total_action = np.concatenate([zero_action, action], axis=0)
#     obs, state, reward, done = env.step(key, state, total_action, np.empty([1,n]))
#     obs = np.array(obs)
#     key,_ = jr.split(key)
#     O.append(obs)
#     if done:
#         obs, state = env.reset(key)
#         break


# O = np.array(O)
# numframes = O.shape[0]
# O = O.reshape(numframes,6,2)

# def matplotlib_render(obs, numframes=1000, record=False):

#     fig, ax = plt.subplots()
#     ax.set_xlim(0, 800)  # Set x-axis limits
#     ax.set_ylim(0, 800)  # Set y-axis limits

#     # Generate initial data for the scatter plot

#     x_data = obs[0, :, 0]
#     y_data = obs[0, :, 1]
    
#     scatter = ax.scatter(x_data, y_data, s=80)  # Initial scatter plot

#     # Define the update function for the animation
#     def update(frame):
#         #  Update the x and y coordinates of the scatter plot with new values
#         scatter.set_offsets(obs[frame, :, :])

#         return scatter,

#     # Create the animation
#     ani = FuncAnimation(fig, update, frames=numframes, interval=20, blit=True, repeat=False)

#     # Show the animation
#     plt.show()
    
# matplotlib_render(O, numframes)