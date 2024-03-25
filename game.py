import pygame
import sys
import random
import numpy as np
import json
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pax
import plotly.express as px
from tensorflow_probability.substrates import jax as tfp


def make_model(env_name, key):
    env = pax.make(env_name)
    return env, pax.training.Agent(env, key)

def load(filename, model):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        return eqx.tree_deserialise_leaves(f, model)
    
    
def use_model(model, env, obs, key, action_mapping=True):
    value, split_logits = model(obs)
    multi_pi = tfp.distributions.Categorical(logits=split_logits)
    action = multi_pi.sample(seed=key)

    if action_mapping:
        action = env.action_space.map_action(action)
    
    action = action[None, :]
    return value, action

key = jr.PRNGKey(0)
env, model = make_model("Prober-v0", key)
model = load("ppo_agent_final.eqx", model)


# Initialize Pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Box Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Player attributes
player_size = 20
player_speed = 5
player_position = np.array([WIDTH // 2, HEIGHT // 2])

# Agent attributes
num_agents = 5
agent_size = 20
agent_speed = 3
# agent_positions= np.random.rand(num_agents, 2) * np.array([WIDTH - agent_size, HEIGHT - agent_size])
mean_position = np.array([300,600])
agent_positions = [mean_position + np.random.randint(-50, 50, size=(2,)) for _ in range(num_agents)]


# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get pressed keys
    keys = pygame.key.get_pressed()

    # Update player position based on pressed keys
    player_movement = np.array([0, 0])
    if keys[pygame.K_LEFT]:
        player_movement[0] -= player_speed
    if keys[pygame.K_RIGHT]:
        player_movement[0] += player_speed
    if keys[pygame.K_UP]:
        player_movement[1] -= player_speed
    if keys[pygame.K_DOWN]:
        player_movement[1] += player_speed

    obs = np.vstack([agent_positions, player_position]).flatten()
    value, action = use_model(model, env, obs, key)
    print(value)
    
    # player_movement = np.array([0, 0])
    # if action[0,0]>0:
    #     player_movement[0] -= player_speed
    # if action[0,0]<0:
    #     player_movement[0] += player_speed
    # if action[0,1]>0:
    #     player_movement[1] -= player_speed
    # if action[0,1]<0:
    #     player_movement[1] += player_speed

    screen.fill(WHITE)
    
    # Update player position
    player_position = np.maximum(0, np.minimum(player_position + player_movement, [WIDTH - player_size, HEIGHT - player_size]))

    # Move agents randomly
    # agent_positions += np.random.randint(-agent_speed, agent_speed+1, size=(num_agents, 2))
    agent_positions += np.ones([num_agents,2])

    # Keep agents within the screen boundaries
    agent_positions = np.maximum(0, np.minimum(agent_positions, [WIDTH - agent_size, HEIGHT - agent_size]))

    # Draw the player
    pygame.draw.rect(screen, BLACK, (*player_position, player_size, player_size))

    # Draw the agents
    for agent_pos in agent_positions:
        pygame.draw.rect(screen, RED, (*agent_pos, agent_size, agent_size))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
