from typing import Any
from pax import make
from pax.managers.rollout import RolloutManager
from pax.managers.batch import BatchManager
from pax.models.mlp import A2CNet
import jax.numpy as jnp


class Trainer():

    def __init__(self, env, key):
        self.env = env
        input_dim = self.env.observation_space.size*(self.env.n_agents+self.env.n_scripted)
        output_dim_actor = self.env.action_space.size
        
        model = A2CNet(input_dim, output_dim_actor, key)
        self.r_manager = RolloutManager(model, env)
        '''
        b_manager = BatchManager(discount, 
                                 gae_lambda, 
                                 n_steps, 
                                 num_envs, 
                                 action_size,
                                 state_space)
        '''

    def __call__(self, keys):
        obs, state = self.r_manager.batch_reset(keys)

        obs,act,batch_reward = self.r_manager.batch_evaluate(keys[0], self.env.params["settings"]["n_env"],)
        return (obs,act,batch_reward)