from typing import Any
import jax.random as jrandom
import jax.numpy as jnp
import time
import equinox as eqx
import chex
from pax.training.rollout import RolloutManager
from pax.training.batch import BatchManager
from pax.training.models import A2CNet
from pax.utils.read_toml import read_config
from jax import jit
from tqdm import tqdm
from typing import Callable




class Trainer:
    def __init__(self, env, key):
        self.env = env
        self.model = A2CNet(self.env.observation_space.size, self.env.action_space.size, key)


    def __call__(self, key, train_config="train_cfg"):
        @jit
        def get_transition(obs, state, batch, key):
            # Run the policy once
            action, log_pi, value, new_key = r_manager.select_action_ppo(obs, key)

            # Split the key into two
            new_key, key_step = jrandom.split(new_key)

            # Split the key_step into as many keys as there are environments
            #batch_keys = jrandom.split(key_step, self.env.params.settings["n_env"])

            # Run a step for a batch of environments TODO POSSIBLE WRONG
            next_obs, next_state, reward, done = r_manager.batch_step(
                state, action
            )

            # Add the rollout to the buffer
            batch = b_manager.append(
                batch, obs, action, reward, done, log_pi, value
            )
            return next_obs, next_state, batch, new_key
        
        train_cfg = read_config("train_cfg")

        r_manager = RolloutManager(self.model, self.env)

        b_manager = BatchManager(
            train_cfg["ppo_params"],
            r_manager.action_size,
            r_manager.observation_space.size,
        )

        # Reset the batch buffer
        batch = b_manager.reset()

        # Split the key as necessary
        rng, rng_step, rng_reset, rng_eval, rng_update = jrandom.split(key, 5)

        # Reset a batch of environments
        obs, state = r_manager.batch_reset(rng_reset, self.env.params.settings["n_env"])
            
        total_steps = 0
        log_steps, log_return = [], []
        T = tqdm(
            range(train_cfg["ppo_params"]["n_steps"]),
            colour="#FFA500",
            desc="PPO",
            leave=True,
        )

        for step in T:
            # Increment the step counter
            total_steps += self.env.params.settings["n_env"]

            # Get a transition
            obs, state, batch, rng_step = get_transition(obs, state, batch, rng_step)
            # Run a rollout for a batch of environments

            # obs, state, act, batch_reward, done, log_prob, value = self.r_manager.batch_evaluate(
            #    key, self.env.params.settings["n_env"])

            obs, batch_reward = r_manager.batch_evaluate(
                key, self.env.params.settings["n_env"]
            )

            # self.b_manager.append(batch, obs, act, batch_reward, done, log_prob, value)

        return obs, batch_reward
        # return (obs, state, act, batch_reward)


def loss_actor_and_critic(
    model: eqx.Module,
    apply_fn: Callable[..., Any],
    obs: chex.ArrayDevice,
    target: chex.ArrayDevice,
    value_old: chex.ArrayDevice,
    log_pi_old: chex.ArrayDevice,
    gae: chex.ArrayDevice,
    action: chex.ArrayDevice,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> chex.ArrayDevice:

    pi, value_pred = model(obs)
    value_pred = value_pred[:, 0]

    # TODO: Figure out why training without 0 breaks categorical model
    # And why with 0 breaks gaussian model pi
    log_prob = pi.log_prob(action[..., -1])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps
    )
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae_mean = gae.mean()
    # Normalize the advantage estimate
    gae = (gae - gae_mean) / (gae.std() + 1e-8)

    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = (
        loss_actor + critic_coeff * value_loss - entropy_coeff * entropy
    )

    return total_loss, (
        value_loss,
        loss_actor,
        entropy,
        value_pred.mean(),
        target.mean(),
        gae_mean,
    )