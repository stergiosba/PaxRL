import chex
import jax
import jax.numpy as jnp
import equinox as eqx
import pax.training as tpax
from typing import Any, Callable, Tuple
from jax.debug import print as dprint
from pax.core.environment import Environment
from pax.scenarios.rax import scripted_act
from pax.core.state import EnvState


# should this be eqx
class RolloutManager(object):
    def __init__(self, env: Environment, map_action: Callable):
        # Setup functionalities for vectorized batch rollout
        self.env = env
        self.map_action = map_action

    @eqx.filter_jit
    def batch_reset(self, key_reset: chex.PRNGKey, num_envs: int):
        return jax.vmap(self.env.reset, in_axes=(0))(
            jax.random.split(key_reset, num_envs)
        )

    @eqx.filter_jit
    def batch_step(self, key, state: EnvState, action, extra_out):
        return jax.vmap(self.env.step, in_axes=(None, 0, 0, None))(key, state, action, extra_out)

    # Slower than batch_evaluate but maybe useful in the future for other applications
    def batch_evaluate_loopy(self, key_input: chex.PRNGKey, num_envs: int):
        key_rst, key_ep = jax.random.split(key_input)
        O = []
        S = []
        obs, state = self.batch_reset(jax.random.split(key_rst, num_envs))
        O.append(obs)
        S.append(state)
        cum_re = jnp.array(num_envs * [0.0])[:, None]
        valid_mask = jnp.array(num_envs * [1.0])[:, None]

        for episode_step in range(self.env.params.scenario["episode_size"]):
            key, key_step, key_net = jax.random.split(key_ep, 3)
            # action, _, _, key = self.select_action(obs, key_net)
            action = script(state, self.env.params)
            # action  = self.random_action(num_envs, key_net)
            obs, state, reward, done = self.batch_step(
                state,
                action,
            )
            cum_re = cum_re + reward * valid_mask
            valid_mask = valid_mask * (1 - done)
            O.append(obs)
            S.append(state)
        return O, S, action, jnp.mean(cum_re)

    @eqx.filter_jit
    def batch_evaluate(self, key_input: chex.PRNGKey, model, num_envs: int):
        """Rollout an episode with lax.scan."""
        # Reset the environments
        key_rst, key_ep = jax.random.split(key_input)
        obs, state = self.batch_reset(key_rst, num_envs)
        
        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, key, cum_reward, valid_mask = state_input
            key, key_step, key_net = jax.random.split(key, 3)

            action, extra_out = scripted_act(state, self.env.params)
            #dprint("{x}", x=action)

            #_, pi = model.get_action_and_value(obs)
            #prober_action = self.map_action(pi.sample(seed=key_net))
            #action = action.at[:, -1].set(prober_action)

            key_step  = jax.random.split(key_step, num_envs)
            next_o, next_s, reward, done = self.batch_step(
                key_step,
                state,
                action,
                extra_out
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
            y = [next_o, next_s, action[:,-1], new_valid_mask, done, reward]
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
        #return obs, state, action, jnp.mean(cum_return), done, reward
        return state, jnp.mean(cum_return)

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"
