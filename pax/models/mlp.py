import jax.random as jrandom
import jax.nn as jnn
import equinox as eqx
import chex
import tomli
from pax import make
from typing import List


class PolicyMLP(eqx.Module):
    actor: List

    def __init__(self, input_dim: int, output_dim: int, key: chex.PRNGKey):
        # with open(f"{config_name}.toml", mode="rb") as model_file:
        # env_params = tomli.load(model_file)["critic_network"]
        #    crit_params = tomli.load(model_file)["critic_network"]

        keys = jrandom.split(key, 3)
        self.actor = [
            eqx.nn.Linear(input_dim, 64, key=keys[0]),
            jnn.relu,
            eqx.nn.Linear(64, 32, key=keys[1]),
            jnn.relu,
            eqx.nn.Linear(32, output_dim, key=keys[2]),
        ]

        """
        self.layers.append(eqx.nn.Linear(input_dim, crit_params["hidden_layers"][0], key=key))

        for i, (layer, activation) in enumerate(zip(crit_params["hidden_layers"], crit_params["activations"])):
            self.layers.append(eqx.nn.Linear(layer[0], layer[1], key=keys[i]))
            self.layers.append(ACTIVATIONS[activation])
        
        """

    def __call__(self, x, key):
        x_a = x
        for layer in self.actor:
            x_a = layer(x_a)

        pi = jrandom.categorical(key, logits=x_a)
        log_prob = jnn.softmax(x_a)
        return pi, log_prob


class A2CNet(eqx.Module):
    critic: List
    actor: List

    def __init__(self, input_dim: int, output_dim: int, key: chex.PRNGKey):
        # with open(f"{config_name}.toml", mode="rb") as model_file:
        # env_params = tomli.load(model_file)["critic_network"]
        #    crit_params = tomli.load(model_file)["critic_network"]

        keys = jrandom.split(key, 6)
        self.critic = [
            eqx.nn.Linear(input_dim, 64, key=keys[0]),
            jnn.relu,
            eqx.nn.Linear(64, 32, key=keys[1]),
            jnn.relu,
            eqx.nn.Linear(32, 1, key=keys[2]),
        ]

        self.actor = [
            eqx.nn.Linear(input_dim, 64, key=keys[3]),
            jnn.relu,
            eqx.nn.Linear(64, 32, key=keys[4]),
            jnn.relu,
            eqx.nn.Linear(32, output_dim, key=keys[5]),
        ]
        """
        self.layers.append(eqx.nn.Linear(input_dim, crit_params["hidden_layers"][0], key=key))

        for i, (layer, activation) in enumerate(zip(crit_params["hidden_layers"], crit_params["activations"])):
            self.layers.append(eqx.nn.Linear(layer[0], layer[1], key=keys[i]))
            self.layers.append(ACTIVATIONS[activation])
        
        """

    def __call__(self, x, key):
        x_v = x
        x_a = x
        for layer in self.critic:
            x_v = layer(x_v)

        for layer in self.actor:
            x_a = layer(x_a)

        v = x_v
        pi = jrandom.categorical(key, logits=x_a)
        log_prob = jnn.softmax(x_a)
        return v, pi, log_prob

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"