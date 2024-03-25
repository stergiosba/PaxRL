import json
import chex
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
from jax.debug import print as dprint

from matplotlib.animation import FuncAnimation

def make_model(env_name, key):
    env = pax.make(env_name)
    return env, pax.training.Agent(env, key)

def load(filename, model):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        return eqx.tree_deserialise_leaves(f, model)
    


key = jr.PRNGKey(3726)

env, model_base = make_model("Prober-v0", key)

env = pax.wrappers.FlattenObservationWrapper(env)

agent = load("ppo_agent_final.eqx", model_base)

agent = jax.jit(jax.vmap(agent, in_axes=0))


@jax.jit
def distance_tensor_jax(X) -> chex.ArrayDevice:
    """Parallelized calculation of euclidean distances between all pairs of swarm agents
        for all environment instances in one go.

    Args:
        `X (chex.Array)`: Swarm agents position tensor of shape: (n_envs,n_agents,2).

    Returns:
        `Distance_matrix (chex.Array)`: Distance tensor of shape: (n_envs,n_agents,n_agents)
    """
    return jnp.einsum("ijkl->ijk", (X[:, None, ...] - X[..., None, :]) ** 2)


@eqx.filter_jit
def batch_reset(env, key_reset: chex.PRNGKey, num_envs: int):
    return jax.vmap(env.reset, in_axes=(0))(
        jax.random.split(key_reset, num_envs)
    )

@eqx.filter_jit
def batch_step(env, key, state, action):
    return jax.vmap(env.step, in_axes=(0, 0, 0))(
        key, state, action
    )


@eqx.filter_jit
def batch_evaluate(env, model, num_envs, key_input):
    
    def policy(model, obs, key):
        split_logits, value  = model(obs)
        multi_pi = tfp.distributions.Categorical(logits=split_logits)
        action = multi_pi.sample(seed=key)
        return action.T, value
    
    """Rollout an episode with lax.scan."""
    # Reset the environments
    key_rst, key_ep = jax.random.split(key_input)
    obs, state = batch_reset(env, key_rst, num_envs)

    def policy_step(state_input, _):
        """lax.scan compatible step transition in jax env."""
        obs, state, key, cum_reward, valid_mask = state_input
        key, key_step, key_net = jax.random.split(key, 3)

        # Get the action from the model
        action, value = policy(model, obs, key_net)      

        key_step = jax.random.split(key_step, num_envs)
        next_o, next_s, reward, done, info = batch_step(env,
            key_step, state, action
        )

        dist = distance_tensor_jax(next_s.X)

        new_cum_reward = cum_reward + reward * valid_mask
        new_valid_mask = valid_mask * (1 - done)
        carry = [
            next_o,
            next_s,
            key,
            new_cum_reward,
            new_valid_mask,
        ]

        y = [next_o, next_s, action, new_valid_mask, done, reward, info, dist]
        return carry, y

    # Scan over episode step loop
    carry_out, scan_out = jax.lax.scan(
        policy_step,
        [
            obs,
            state,
            key_ep,
            jnp.array(num_envs * [0.0]),
            jnp.array(num_envs * [1.0]),
        ],
        (),
        1024,
    )
    obs, state, action, _, done, reward, info, dist = scan_out
    cum_return = carry_out[-2].squeeze()
    # return carry_out[0], jnp.mean(cum_return)
    # return obs, state, action, jnp.mean(cum_return), done, reward
    return state, jnp.mean(cum_return), jnp.std(cum_return), done, info, dist


env_nums = 100
state, return_mean, return_std, done, info, dist = batch_evaluate(env, agent, env_nums, key)

env_id = 21
env.render(state, env_id)

episode_lengths = jnp.where(done.T==1)[1]

fig, ax = plt.subplots()
plt.plot(jnp.sqrt(dist[:-1,env_id,-1,:-1]))
plt.show()

# data = jnp.sqrt(dist[-2,:,-1,:-1])
data = jnp.sqrt(dist[episode_lengths-1,jnp.arange(env_nums),-1,:-1])


# Define colors for each dataset
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
x_ticks = np.arange(env_nums)
n_agents = 5
# Plotting
plt.figure(figsize=(10, 5))  # Adjust figure size as needed
for i in range(n_agents):
    plt.scatter(x_ticks, data[:, i], color=colors[i], label=f'Value {i+1}')

count = 0
for x in x_ticks:
    highlight_index = state.leader[0, x]
    plt.scatter(x, data[x, highlight_index], color='gold', marker='*', s=200)
    if all(data[x, highlight_index] <= data[x, i] for i in range(n_agents) if i != highlight_index):
        count += 1
plt.axhline(y=60, color='black', linestyle='--')

print(f"Golden star is under all points at {100*count/env_nums}% of the runs")


# Adding labels and legend
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Plot with 100 Ticks on X-axis and 6 Values as Dots on Y-axis')
plt.legend()

# Show plot
plt.show()
