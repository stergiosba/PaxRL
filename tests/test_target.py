import os
import time
import optax
import jax
import jax.random as jrandom
import pax.training as paxt
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pax import make
from pax.training.models import Agent

matplotlib.use("Qt5Agg")

def test_target_env(env_name="Target-v0"):
    env,_ = make(env_name)

    key_input = jrandom.PRNGKey(env.params.settings["seed"])
    key, key_model = jrandom.split(key_input)

    model = Agent(env, key_model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.scale_by_adam(eps=1e-5)
    )
    train_state = paxt.TrainState(model=model, optimizer=optimizer)

    s = time.time()
    map_action  = jax.jit(jax.vmap(env.action_space.map_action, in_axes=0))
    rollout_manager = paxt.RolloutManager(env, map_action)
    show_state, show_actions, log_return, rewards = rollout_manager.batch_evaluate_analysis(key, train_state, 1)

    # fig, ax = plt.subplots()
    # ax.plot(rewards)
    # ax.set_xlabel("time")
    # ax.set_ylabel("reward")
    # ax.set_title("Reward over time")
    # ax.grid()
    # plt.show()

    env.render(show_state, show_actions, log_return)

if __name__ == "__main__":
    test_target_env()