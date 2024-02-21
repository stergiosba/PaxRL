import json
import fire
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pax
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tensorflow_probability.substrates import jax as tfp
import plotly.figure_factory as ff

def make_model(env_name, key):
    env, _ = pax.make(env_name)
    return env, pax.training.Agent(env, key)

def load(filename, model):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        return eqx.tree_deserialise_leaves(f, model)
    
    
def use_model(model, obs, key, action_mapping=True):
    value, split_logits = model(obs)
    multi_pi = tfp.distributions.Categorical(logits=split_logits)
    action = multi_pi.sample(seed=key)

    if action_mapping:
        action = env.action_space.map_action(action)
    
    action = action[None, :]
    return value, action

key = jr.PRNGKey(0)

env, model = make_model("Prober-v0", key)
obs, state = env.reset(key)
obs = np.array(obs)


# agent_1 = load("ppo_agent_1.eqx", model)
# agent_2 = load("ppo_agent_4.eqx", model)
agent = load("ppo_agent_fixed_withprev.eqx", model)


# action = env.action_space.sample(key)


x_range = np.linspace(0, 800, 20)
y_range = np.linspace(0, 800, 20)

V = np.zeros([len(x_range), len(y_range)])

for i,x in enumerate(x_range):
    O = obs
    O[-2] = x
    for j, y in enumerate(y_range):
        O[-1] = y
        value, action = use_model(agent, O, key)

        V[j,i] = value[0]
        


# XX,YY = np.meshgrid(np.arange(0, 800, 20), np.arange(0, 800, 20))

# fig = ff.create_quiver(XX, YY, A_x, A_y)
# fig.show()
# print(A_x)
# print(A_y)
import plotly.express as px
# fig = make_subplots(rows=1, cols=2)

img = px.imshow(V, x=x_range, y=y_range)
img.update_layout(title="Critic Network Evaluation", xaxis_title="X", yaxis_title="Y")
img.show()

# img = px.imshow(A_x, x=x_range, y=y_range)
# img.update_layout(title="Critic Network Evaluation", xaxis_title="X", yaxis_title="Y")
# img.show()

# img = px.imshow(A_y, x=x_range, y=y_range)
# img.update_layout(title="Critic Network Evaluation", xaxis_title="X", yaxis_title="Y")
# img.show()


zero_action = jnp.zeros([env.params.scenario["n_scripted_entities"],2])
total_action = jnp.concatenate([zero_action, action], axis=0)

# print(state)
# # print(total_action)

obs, state, reward, done = env.step(key, state, total_action, jnp.empty([1,env.params.scenario["n_scripted_entities"]]))
value, action = use_model(agent, obs, key)
# # # print(state)
returns = jnp.array([0.0, value[0]], dtype=jnp.float32)
env.render(state, returns)