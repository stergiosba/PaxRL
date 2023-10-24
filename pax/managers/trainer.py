from typing import Any
from pax import make
from pax.managers.rollout import RolloutManager
from pax.managers.batch import BatchManager
from pax.models.mlp import A2CNet
import jax.random as jrandom
import jax.numpy as jnp
import tomli
from tqdm import tqdm
import time


class Trainer:
    def __init__(self, env, key, train_cfg_mame="train_cfg"):
        self.env = env
        model = A2CNet(self.env.observation_space.size, self.env.action_space.size, key)
        self.r_manager = RolloutManager(model, env)

        if ".toml" in train_cfg_mame:
            with open(f"{train_cfg_mame}", mode="rb") as config_file:
                train_config = tomli.load(config_file)
        else:
            with open(f"{train_cfg_mame}.toml", mode="rb") as config_file:
                train_config = tomli.load(config_file)

        self.train_cfg = train_config
        self.b_manager = BatchManager(
            train_config["params"],
            self.r_manager.action_size,
            self.r_manager.observation_space.size,
        )

    def __call__(self, key):
        
        batch = self.b_manager.reset()

        rng, rng_step, rng_reset, rng_eval, rng_update = jrandom.split(key, 5)

        # Reset a batch of environments
        #obs, state, act, batch_reward = self.r_manager.batch_reset(
        #    rng_reset, self.env.params.settings["n_env"]
        #)
        #obs, state = self.batch_reset(jax.random.split(rng_reset, self.env.params.settings["n_env"]))
        
        total_steps = 0
        log_steps, log_return = [], []
        T = tqdm(range(self.train_cfg["params"]["n_steps"]), colour="#FFA500", desc="PPO", leave=True)
        
        for step in T:
            total_steps += self.env.params.settings["n_env"]
            # Run a rollout for a batch of environments

            obs, state, act, batch_reward = self.r_manager.batch_evaluate(
                key, self.env.params.settings["n_env"])
            
        print(total_steps)
        return (obs, state, act, batch_reward)
