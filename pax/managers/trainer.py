from typing import Any
from pax import make
from pax.managers.rollout import RolloutManager
from pax.managers.batch import BatchManager
from pax.models.mlp import A2CNet
import jax.numpy as jnp


class Trainer:
    def __init__(self, env, key):
        self.env = env
        model = A2CNet(self.env.observation_space.size, self.env.action_space.size, key)
        self.r_manager = RolloutManager(model, env)
        """
        b_manager = BatchManager(discount, 
                                 gae_lambda, 
                                 n_steps, 
                                 num_envs, 
                                 action_size,
                                 state_space)
        """

    def __call__(self, key):
        # Run a rollout for a batch of environments
        obs, state, act, batch_reward = self.r_manager.batch_evaluate(
            key, self.env.params.settings["n_env"]
        )

        return (obs, state, act, batch_reward)
