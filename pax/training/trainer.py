import jax
import json
import chex
import optax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from typing import NamedTuple, Dict
from pax.training import TrainState
from pax.core.environment import EnvState
from jax import jit, vmap
from jax.debug import print as dprint, breakpoint as brk
from tensorflow_probability.substrates import jax as tfp
import plotly.express as px
import pandas as pd


class Trainer:
    def __init__(self, env):
        self.env = env

    def save(self, filename, hyperparams, model):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)

    def __call__(self, model, key, train_config):

        n_total_updates = int(
            train_config["n_train_steps"]
            // train_config["n_steps"]
            // train_config["n_train_envs"]
        )

        n_steps_warm_up = int(train_config["n_train_steps"] * train_config["lr_warmup"])
        schedule_fn = optax.linear_schedule(
            init_value=-float(train_config["lr_begin"]),
            end_value=-float(train_config["lr_end"]),
            transition_steps=n_steps_warm_up,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(train_config["max_grad_norm"]),
            optax.scale_by_adam(eps=1e-5),
            optax.scale_by_schedule(schedule_fn),
        )

        params, static = eqx.partition(model, eqx.is_array)

        train_state = TrainState(params=params, optimizer=optimizer)

        # Split the key as necessary
        rng, rng_reset = jrandom.split(key, 2)

        # Run the initialization step of the environments at this point
        # the observation has shape (n_env, n_agents, 2)

        obs, state = jax.vmap(self.env.reset, in_axes=(0))(
            jax.random.split(rng_reset, train_config["n_train_envs"])
        )

        def update(runner_state, unused):

            def _get_transition(runner_state, unused):
                train_state, state, last_obs, rng = runner_state
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                split_logits, value = vmodel(last_obs)
                multi_pi = tfp.distributions.Categorical(logits=split_logits)
                action = multi_pi.sample(seed=key)
                log_prob = multi_pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, train_config["n_train_envs"])
                obs, state, reward, done, info = jax.vmap(self.env.step, in_axes=(0, 0, 0))(
                    rng_step, state, action.T
                )
                # dprint("{x}",x=reward)
                transition = Transition(
                    done, action.T, value.flatten(), reward, log_prob.T, last_obs, info
                )
                runner_state = (train_state, state, obs, rng)
                return runner_state, transition

            model = eqx.combine(runner_state[0].params, static)
            vmodel = vmap(model, in_axes=(0))
            runner_state, traj_batch = jax.lax.scan(
                _get_transition, runner_state, None, train_config["n_steps"]
            )

            train_state, state, last_obs, rng = runner_state
            _, last_val = vmodel(last_obs)            

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + train_config["discount"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + train_config["discount"] * train_config["gae_lambda"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch, last_val.flatten())

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        model = eqx.combine(params, static)
                        vmodel = vmap(model, in_axes=(0))
                        # RERUN NETWORK
                        split_logits, value = vmodel(traj_batch.obs)
                        multi_pi = tfp.distributions.Categorical(logits=split_logits)
                        log_prob = multi_pi.log_prob(traj_batch.action.T).sum(axis=0)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-train_config["clip_eps"], train_config["clip_eps"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob.sum(axis=1))
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - train_config["clip_eps"],
                                1.0 + train_config["clip_eps"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        # Calculate the entropy loss
                        # H(X, Y) = H(X) + H(Y) for entropy of independent random variables X and Y.
                        # entropy = pi.entropy().mean()
                        entropy = multi_pi.entropy().sum(axis=0).mean()

                        total_loss = (
                            loss_actor
                            + train_config["critic_coeff"] * value_loss
                            - train_config["entropy_coeff"] * entropy
                        )

                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = eqx.filter_value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    # dprint("{x}", x=grads.actor.layers[0].weight)
                    train_state = train_state.apply_gradients(grads=grads)

                    # So the gradients are zero because I am not using the model in the loss function.
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Batching and Shuffling
                # Batch size must be equal to number of steps * number of envs"
                minibatch_size = (
                    train_config["n_train_envs"]
                    * train_config["n_steps"]
                    // train_config["n_minibatch"]
                )
                batch_size = train_config["n_minibatch"] * minibatch_size
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [train_config["n_minibatch"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, train_config["epoch_ppo"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]


            def callback(info):
                # This is like a filter that takes only the values from info["returned_episode_returns"] where info["returned_episode"] is True
                return_values = info["returned_episode_returns"][info["returned_episode"]]
                timesteps = info["timestep"][info["returned_episode"]] * train_config["n_train_envs"]

                
                if len(return_values)!=0:
                    R_mean.append(np.mean(return_values))
                    R_std.append(np.std(return_values))
                    np.savetxt("mean.csv", R_mean, delimiter=",\n", header="Mean")
                    np.savetxt("std.csv", R_std, delimiter=",\n", header="Std")

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                    
                    # print(info['reward_stats'][info["returned_episode"]])
                    print("---------")

                # if info["rewards_stats"].sum(axis=0).mean(axis=0)[2] > 30:
                #     print(info)


            jax.debug.callback(callback, metric)
            runner_state = (train_state, state, last_obs, rng)
            return runner_state, metric

        # statistics = [pd.DataFrame(columns=["timestep", "returns", "std"])]
        R_mean = []
        R_std =[]
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, state, obs, _rng)
        sub_runs = 4
        # for i in range(sub_runs):
        runner_state, metric = jax.lax.scan(update, runner_state, None, n_total_updates//sub_runs)
        runner_state, metric = jax.lax.scan(update, runner_state, None, n_total_updates//sub_runs)
        runner_state, metric = jax.lax.scan(update, runner_state, None, n_total_updates//sub_runs)
        runner_state, metric = jax.lax.scan(update, runner_state, None, n_total_updates//sub_runs)
        return runner_state, metric


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict
