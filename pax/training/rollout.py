import jax
import chex
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Tuple
from pax.core.environment import Environment
from pax.scenarios.rax import scripted_act
from pax.core.state import EnvState
from tensorflow_probability.substrates import jax as tfp
from functools import partial
from jax.debug import print as dprint


@partial(jax.jit, static_argnums=0)
def policy(
    model: eqx.Module,
    obs: jnp.ndarray,
    key: chex.PRNGKey,
):
    value, split_logits = jax.vmap(model)(obs)
    #dprint("{x}", x=split_logits)
    multi_pi = [tfp.distributions.Categorical(logits=logits) for logits in split_logits]
    return value, multi_pi


# should this be eqx.Module?
class RolloutManager(object):
    def __init__(self, env: Environment, map_action: Callable):
        # Setup functionalities for vectorized batch rollout
        self.env = env
        self.map_action = map_action

    @eqx.filter_jit
    def select_action(
        self,
        model,
        obs: jnp.ndarray,
        key: chex.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, chex.PRNGKey]:
        value, multi_pi = policy(model, obs, key)
        action = jnp.vstack([pi.sample(seed=key) for pi in multi_pi])
        log_prob = jnp.vstack([pi.log_prob(a) for a, pi in zip(action, multi_pi)])
        return action.T, log_prob.sum(0), value[:, 0], key
    

    @eqx.filter_jit
    def select_action2(
        self,
        model,
        obs: jnp.ndarray,
        key: chex.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, chex.PRNGKey]:
        return jnp.ones(16)

    @eqx.filter_jit
    def batch_reset(self, key_reset: chex.PRNGKey, num_envs: int):
        return jax.vmap(self.env.reset, in_axes=(0))(
            jax.random.split(key_reset, num_envs)
        )

    @eqx.filter_jit
    def batch_step(self, key, state: EnvState, action, extra_out):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            key, state, action, extra_out
        )

    @eqx.filter_jit
    def batch_evaluate(self, key_input: chex.PRNGKey, train_state, num_envs: int):
        """Rollout an episode with lax.scan."""
        # Reset the environments
        key_rst, key_ep = jax.random.split(key_input)
        obs, state = self.batch_reset(key_rst, num_envs)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jax.random.split(key, 3)

            action, extra_out = scripted_act(state, self.env.params)
            action_agent, _, _, key_net = self.select_action(
                train_state.model, obs, key_net
            )

            action_agent = self.map_action(action_agent)
            #dprint("{x}", x=action_agent)
            action = action.at[:, -1].set(action_agent)

            key_step = jax.random.split(key_step, num_envs)
            next_o, next_s, reward, done = self.batch_step(
                key_step, state, action, extra_out
            )

            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_o,
                next_s,
                key,
                new_cum_reward,
                new_valid_mask,
            ]
            # TODO Log prob and value should be used but that is a future problem.
            # We only return the action of the prober (last element).
            y = [next_o, next_s, action[:, -1], new_valid_mask, done, reward]
            # y = [new_valid_mask]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                key_ep,
                jnp.array(num_envs * [0.0])[:, None],
                jnp.array(num_envs * [1.0])[:, None],
            ],
            (),
            self.env.params.scenario["episode_size"],
        )
        obs, state, action, _, done, reward = scan_out
        cum_return = carry_out[-2].squeeze()
        # return carry_out[0], jnp.mean(cum_return)
        # return obs, state, action, jnp.mean(cum_return), done, reward
        return state, jnp.mean(cum_return)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
