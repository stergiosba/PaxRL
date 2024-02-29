import jax
import json
import chex
import optax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from collections import defaultdict
from typing import NamedTuple
from pax.training import RolloutManager, BatchManager, TrainState
from pax.core.environment import EnvState
from pax.utils.read_toml import read_config
from pax.environments import scripted_act
from jax import jit, vmap
from tqdm import tqdm
from typing import Tuple
from jax.debug import print as dprint, breakpoint as brk
from tensorflow_probability.substrates import jax as tfp
import plotly.express as px
from collections import deque


class Trainer:
    def __init__(self, env):
        self.env = env
        self.map_action = jit(vmap(env.action_space.map_action, in_axes=0))
        
    def save(self, filename, hyperparams, model):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)

    def __call__(self, model, key, train_config):
        @eqx.filter_jit
        def get_transition(
            train_state: TrainState,
            obs: chex.ArrayDevice,
            state: EnvState,
            batch,
            key: chex.PRNGKey,
        ):
            action_unmapped, log_pi, value, new_key = rollout_manager.select_action(
                train_state.model, obs, key
            )
            action_agent = self.map_action(action_unmapped)
            action, extra_in = scripted_act(state, self.env.params)
            action = action.at[:, -1].set(action_agent)

            # action_unmapped, log_pi, value, new_key = rollout_manager.select_action(train_state.model, obs, key)
            # action_agent = self.map_action(action_unmapped)
            # action = action_agent

            new_key, key_step = jrandom.split(key)
            b_key = jrandom.split(key_step, train_config["num_train_envs"])
            # # Automatic env resetting in gymnax step!
            next_obs, next_state, reward, done = rollout_manager.batch_step(
                b_key, state, action, extra_in
            )

            batch = batch_manager.append(
                batch, obs, action_unmapped, reward, done, log_pi, value.flatten()
            )
            return train_state, next_obs, next_state, batch, new_key

        num_total_epochs = int(
            train_config["num_train_steps"] // train_config["num_train_envs"] + 1
        )

        num_steps_warm_up = int(
            train_config["num_train_steps"] * train_config["lr_warmup"]
        )
        schedule_fn = optax.linear_schedule(
            init_value=-float(train_config["lr_begin"]),
            end_value=-float(train_config["lr_end"]),
            transition_steps=num_steps_warm_up,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(train_config["max_grad_norm"]),
            optax.scale_by_adam(eps=1e-5),
            optax.scale_by_schedule(schedule_fn),
        )

        params, static = eqx.partition(model, eqx.is_array)

        train_state = TrainState(params=params, optimizer=optimizer)

        rollout_manager = RolloutManager(self.env, self.map_action)
        batch_manager = BatchManager(
            train_config=train_config,
            action_space=self.env.action_space,
            state_shape=self.env.observation_space.size,
        )
        # Reset the batch buffer
        batch = batch_manager.reset()
        # Split the key as necessary
        rng, rng_step, rng_reset, rng_eval, rng_update = jrandom.split(key, 5)

        # Run the initialization step of the environments at this point
        # the observation has shape (n_env, n_agents, 2)
        obs, state = rollout_manager.batch_reset(
            rng_reset, train_config["num_train_envs"]
        )
 
        total_steps = 0
        log_steps, log_return = [], []
        logg_return = deque(maxlen=num_total_epochs)

        def update():
        
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                model = eqx.combine(train_state.params, static)
                value, split_logits = model(last_obs)
                multi_pi = tfp.distributions.Categorical(logits=split_logits)
                action = multi_pi.sample(seed=key)
                log_prob = multi_pi.log_prob(action)
                # return action.T, log_prob.sum(0), value.flatten(), key

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, train_config["num_train_envs"])
                obs, state, reward, done = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action)


                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )

                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, train_config["n_steps"]
            )

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray