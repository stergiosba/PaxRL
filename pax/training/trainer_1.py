import jax
import json
import chex
import optax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from collections import defaultdict
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

        train_state = TrainState(model=model, optimizer=optimizer)

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
        T = tqdm(
            range(1, num_total_epochs + 1),
            colour="#950dFF",
            desc="PaxRL",
            leave=True,
        )

        for step in T:
            # Get a transition
            train_state, obs, state, batch, rng_step = get_transition(
                train_state, obs, state, batch, rng_step
            )
            # Increment the step counter
            total_steps += train_config["num_train_envs"]

            if step % (train_config["n_steps"] + 1) == 0:
                metric_dict, train_state, rng_update = update(
                    train_state,
                    batch_manager.get(batch),
                    train_config,
                    rng_update,
                )
                batch = batch_manager.reset()

            if (step + 1) % train_config["evaluate_every_epochs"] == 0:
                rng, rng_eval = jrandom.split(rng)
                show_state, reward, rewards = rollout_manager.batch_evaluate(
                    rng_eval,
                    train_state,
                    train_config["num_test_rollouts"],
                )
                log_steps.append(total_steps)
                log_return.append(rewards)
                logg_return.append(rewards)
                np.savetxt("data.csv", logg_return, delimiter=",\n", header="Reward")
                T.set_description(f"Rewards: {rewards}")
                T.refresh()
                
                if (step + 1) % train_config["checkpoint_every_epochs"] == 0:
                    print(f"Saving model at {total_steps} steps")
                    self.save(f"ppo_agent_{total_steps}.eqx", {}, train_state.model)

                if (step + 1) % train_config["render_every_epochs"] == 0:
                    df = train_state.model.actor.layers[0].weight.T@train_state.model.actor.layers[0].weight
                    df2 = train_state.model.critic.layers[0].weight.T@train_state.model.critic.layers[0].weight
                    fig = px.imshow(df)
                    # fig2 = px.imshow(df2)
                    fig.show()
                    # fig2.show()
                    print(f"Rendering Performance after {total_steps} steps")
                    self.env.render(show_state, log_return)

                # if mle_log is not None:
                #     mle_log.update(
                #         {"num_steps": total_steps},
                #         {"return": rewards},
                #         model=jnp.array([-1]),
                #         save=True,
                #     )
        return num_total_epochs, log_steps, log_return


@jit
def flatten_dims(x):
    
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


@eqx.filter_jit
def loss_actor_and_critic(
    model,
    obs: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    action: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> jnp.ndarray:
    # value_pred, logits = jax.vmap(model)(obs)
    value_pred, split_logits = jax.vmap(model)(obs)
    # pi = tfp.distributions.Categorical(logits=logits)
    multi_pi = tfp.distributions.Categorical(logits=split_logits)
    value_pred = value_pred[:, 0]
    # TODO: (FROM GYMNAX) Figure out why training without 0 breaks categorical model
    # And why with 0 breaks gaussian model pi
    # log_prob = pi.log_prob(action[..., -1])

    # Calculate the actor loss
    log_prob = multi_pi.log_prob(action.T).sum(axis=1)
    ratio = jnp.exp(log_prob - log_pi_old)
    gae_mean = gae.mean()
    gae = (gae - gae_mean) / (gae.std() + 1e-8)
    loss_actor_unclipped = ratio * gae
    loss_actor_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor_unclipped, loss_actor_clipped)
    loss_actor = loss_actor.mean()

    # Calculate the critic loss
    value_pred_clipped = value_old + (value_pred - value_old).clip(-clip_eps, clip_eps)
    loss_value_clipped = jnp.square(value_pred_clipped - target)
    loss_value_unclipped = jnp.square(value_pred - target)
    loss_value = 0.5 * jnp.maximum(loss_value_clipped, loss_value_unclipped).mean()

    # Calculate the entropy loss
    # H(X, Y) = H(X) + H(Y) for entropy of independent random variables X and Y.
    # entropy = pi.entropy().mean()
    entropy = multi_pi.entropy().sum(axis=0).mean()

    # Calculate the total loss
    total_loss = loss_actor + critic_coeff * loss_value - entropy_coeff * entropy

    # brk()
    return total_loss, (
        loss_value,
        loss_actor,
        entropy,
        value_pred.mean(),
        target.mean(),
        gae_mean,
    )


def update(
    train_state: TrainState,
    batch: Tuple,
    train_config,
    rng: chex.PRNGKey,
):
    """Perform multiple epochs of updates with multiple updates."""
    obs, action, log_pi_old, value, target, gae = batch
    size_batch = train_config["num_train_envs"] * train_config["n_steps"]
    size_minibatch = size_batch // train_config["n_minibatch"]
    idxes = jnp.arange(train_config["num_train_envs"] * train_config["n_steps"])
    avg_metrics_dict = defaultdict(int)

    for _ in range(train_config["epoch_ppo"]):
        idxes = jrandom.permutation(rng, idxes)
        idxes_list = [
            idxes[start : start + size_minibatch]
            for start in jnp.arange(0, size_batch, size_minibatch)
        ]
        train_state, total_loss = update_epoch(
            train_state,
            idxes_list,
            flatten_dims(obs),
            flatten_dims(action),
            flatten_dims(log_pi_old),
            flatten_dims(value),
            jnp.array(flatten_dims(target)),
            jnp.array(flatten_dims(gae)),
            train_config["clip_eps"],
            train_config["entropy_coeff"],
            train_config["critic_coeff"],
        )

        total_loss, (
            value_loss,
            loss_actor,
            entropy,
            value_pred,
            target_val,
            gae_val,
        ) = total_loss

        avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        avg_metrics_dict["entropy"] += np.asarray(entropy)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        avg_metrics_dict["target"] += np.asarray(target_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)
        rng, _ = jrandom.split(rng)

    for k, v in avg_metrics_dict.items():
        avg_metrics_dict[k] = v / (train_config["epoch_ppo"])

    return avg_metrics_dict, train_state, rng


@eqx.filter_jit
def update_epoch(
    train_state: TrainState,
    idxes: jnp.ndarray,
    obs,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
):
    for idx in idxes:
        grad_fn = eqx.filter_value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.model,
            obs=obs[idx],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            # action=action[idx].reshape(-1, 1),
            action=jnp.expand_dims(action[idx], -1),
            clip_eps=clip_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )

        train_state = train_state.apply_gradients(grads)

    return train_state, total_loss
