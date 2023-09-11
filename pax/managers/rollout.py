import chex
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, Tuple
from jax.debug import print as dprint  # type: ignore
from pax.core.environment import Environment
from pax.scenarios.rax import script
from pax.core.state import EnvState


# should this be eqx
class RolloutManager(object):
    def __init__(self, model, env: Environment):
        # Setup functionalities for vectorized batch rollout
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_size = self.env.action_space.size
        self.model = model
        self.select_action = self.select_action_ppo

    @eqx.filter_jit
    def random_action(self, num_envs, key):
        return self.env.action_space.sample(key, shape=(num_envs, self.env.n_agents))

    @eqx.filter_jit
    def select_action_ppo(
        self, obs: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.PRNGKey]:
        value, pi, log_prob = policy(self.model, obs, key)
        action = pi
        return action, log_prob, value[:, 0], key

    @eqx.filter_jit
    def batch_reset(self, keys: chex.PRNGKey):
        return jax.vmap(self.env.reset, in_axes=(0))(keys)

    @eqx.filter_jit
    def batch_step(self, state: EnvState, action):
        return jax.vmap(self.env.step, in_axes=(0, 0))(state, action)

    def batch_evaluate_loopy(self, key_input: chex.PRNGKey, num_envs: int):
        key_rst, key_ep = jax.random.split(key_input)
        O = []
        obs, state = self.batch_reset(jax.random.split(key_rst, num_envs))
        O.append(obs[0])
        cum_re = jnp.array(num_envs * [0.0])[:, None]
        valid_mask = jnp.array(num_envs * [1.0])[:, None]

        m = self.env.params["scenario"]["episode_size"]

        for episode_step in range(m):
            key, key_step, key_net = jax.random.split(key_ep, 3)
            # action, _, _, key = self.select_action(obs, key_net)
            action = script(state)
            # action  = self.random_action(num_envs, key_net)
            obs, state, reward, done = self.batch_step(
                state,
                action,
            )
            cum_re = cum_re + reward * valid_mask
            valid_mask = valid_mask * (1 - done)
            O.append(obs[0])

        return jnp.array(O), action, jnp.mean(cum_re)

    @eqx.filter_jit
    def batch_evaluate(self, key_input: chex.PRNGKey, num_envs: int):
        """Rollout an episode with lax.scan."""
        # Reset the environments
        key_rst, key_ep = jax.random.split(key_input)
        obs, state = self.batch_reset(jax.random.split(key_rst, num_envs))

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jax.random.split(key, 3)
            # action, _, _, key = self.select_action(obs, key_net)
            action = script(state)
            # action  = self.random_action(num_envs, key_net)
            next_o, next_s, reward, done = self.batch_step(
                state,
                action,
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
            y = [next_o, action, new_valid_mask]
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
            self.env.params["scenario"]["episode_size"],
        )
        obs, action, _ = scan_out
        cum_return = carry_out[-2].squeeze()
        return obs, action, jnp.mean(cum_return)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


@eqx.filter_jit()
def policy(model: Callable[..., Any], obs: chex.Array, key: chex.PRNGKey):
    value, pi, log_prob = model(obs, key)
    return value, pi, log_prob
