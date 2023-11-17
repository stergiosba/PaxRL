import pax
import pytest
import jax.numpy as jnp
from pax.core.spaces import Box, MultiDiscrete


class TestBasic:
    @pytest.fixture
    def env_config(self):
        return pax.make("tests/test_env")

    def test_version(self):
        assert pax.__version__ == "0.2.0"

    def test_make(self, env_config):
        # Arrange
        env, config = env_config
        # Act
        action_shape = [
            config["action_space"]["low"],
            config["action_space"]["high"],
            config["action_space"]["step"],
        ]

        # Assert
        assert env.observation_space == Box(
            config["observation_space"]["low"],
            config["observation_space"]["high"],
            config["observation_space"]["shape"],
            dtype=jnp.float32,
        )
        assert env.action_space == MultiDiscrete(action_shape, dtype=jnp.float32)

        assert env.params["settings"]["seed"] == config["settings"]["seed"]
        assert env.params["settings"]["n_env"] == config["settings"]["n_env"]
        assert env.params["scenario"]["n_agents"] == config["scenario"]["n_agents"]
        assert (
            env.params["scenario"]["n_scripted_entities"]
            == config["scenario"]["n_scripted_entities"]
        )
        assert (
            env.params["scenario"]["episode_size"] == config["scenario"]["episode_size"]
        )

        # Cleanup
