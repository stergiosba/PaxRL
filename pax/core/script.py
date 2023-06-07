import chex
import jax.numpy as jnp
import equinox as eqx
import jax.debug as debug
import jax.numpy.linalg as la
import numpy as np
from typing import Tuple
from jax import jit, lax
from pax.core.state import EntityState, AgentState, EnvState


@jit
def distances_matrix_jax(X:chex.Array) -> chex.Array:
    """Calculates the euclidean distances between all pairs of swarm agents.

    Args:
        X (chex.Array): Swarm agents position matrix [shape: (n,2)]

    Returns:
        Distance_matrix chex.Array: Distance matrix
    """    
    return jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)


@jit
def leader_neighbors(
        leader_distances:chex.Array,
        leader_radius: float) -> chex.Array:
    """Calculates the neighborhood for the leader agent as a set.
    Args:
        leader_distances (chex.Array): Row of distance matrix, containing euclidean distances between the leader and each swarm agent.
        leader_radius (float): Influence radius for the leader agent (typically taken as a scaled version of the simple agent's radius)

    Returns:
        leader_neighbors_matrix (chex.Array): Boolean vector of leader's neighbors.\n
    """
    return ((leader_distances>0)&(leader_distances<=(leader_radius)**2))


@jit
def neighbors(
        distance_matrix: chex.Array,
        agent_radius: float,
        leader_neighbors_count: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Calculates the neighborhood for each agent as a set and the cardinality of each set.
    Args:
        distance_matrix (chex.Array): Distance matrix, contains euclidean distances between all pairs of swarm agents.
        agent_radius (float): Influence radius for the simple agents

    Returns:
        neighbors_matrix (chex.Array): Boolean array of neighbors.\n
        neighbors_count (chex.Array): Row-wise sum of neighbors_matrix, i.e. cardinality of each neighborhood. 
    """
    neighbors_matrix = ((distance_matrix>0)&(distance_matrix<=agent_radius**2))
    neighbors_count = jnp.sum(neighbors_matrix, axis=0)#-leader_neighbors_count
    
    return neighbors_matrix, neighbors_count


@jit
def cohesion_steer(
        X: chex.Array,
        leader_id: int,
        leader_factor: float, 
        neighbors_matrix: chex.Array,
        leader_neighbors_count: chex.Array,
        total_count: chex.Array) -> chex.Array:
    """ Part of Jax Reynold's Boids Algorithm (Rax)
        Calculates the cohesion steering based on local interactions with simple agents and with the leader using Reynold's Dynamics
    
    Args:
        X (chex.Array): Swarm agents position matrix [shape: (n,2)]
        leader_id (int): The ID of the leader.
        leader_factor (float): Quantifies the strength of the leader agent.
        neighbors_matrix (chex.Array): Boolean array of neighbors.
        leader_neighbors_count (chex.Array): _description_
        total_count (chex.Array): _description_

    Returns:
        chex.Array: _description_
    """
    # Small epsilon to avoid division by zero
    eps = 10**-16
    
    # Influence from simple neighbors
    neighbors_influence = jnp.sum(neighbors_matrix[:,:,None]*X, axis=-2)
    
    # Leader influence
    leader_influence = leader_factor*leader_neighbors_count[:,None]*X[leader_id]
    
    # Cohesion steer calculation
    cohesion = (neighbors_influence+leader_influence) \
                /(total_count[:,None]+eps) \
                -X
                
    # Leader whitening
    cohesion = cohesion.at[leader_id].set(jnp.array([0,0]))
                
    # Return normalized cohesion
    return cohesion/la.norm(cohesion+eps, axis=1)[:,None]


@jit
def alignment_steer(
        X_dot: chex.Array, 
        leader_id: int,
        leader_factor: float, 
        neighbors_matrix: chex.Array,
        leader_neighbors_count: chex.Array,
        total_count: chex.Array) -> chex.Array:
    # Small epsilon to avoid division by zero
    eps = 10**-16
    
    # Influence from simple neighbors
    neighbors_influence = jnp.sum(neighbors_matrix[:,:,None]*X_dot, axis=-2)
    
    # Leader influence
    leader_influence = leader_factor*leader_neighbors_count[:,None]*X_dot[leader_id]
    
    # Alignment steer calculation
    alignment = (neighbors_influence+leader_influence) \
                /(total_count[:,None]+eps)
                
    # Leader whitening
    alignment = alignment.at[leader_id].set(jnp.array([0,0]))
    
    return alignment/la.norm(alignment+eps, axis=1)[:,None]


@jit 
def separation_steer(X, dist_mat, neighbors_matrix):
    # Small epsilon to avoid division by zero
    eps = 10**-16
    n = len(X)
    
    # The main diagonal of a distance matrix is 0 since d(1,1) is the distance of agent 1 from itself
    # Therefore when we divide by the distances we would get 1/0. Notice that adding the identity matrix
    # will not change the calculations for separation since 
    adj_dist_mat = dist_mat+jnp.eye(n)
    
    # Separation steer calculation
    separation = jnp.sum(neighbors_matrix/adj_dist_mat,axis=1)[:,None]*X - ((neighbors_matrix/adj_dist_mat)[:,None]@X).reshape(n,2)
    
    return separation/la.norm(separation+eps, axis=1)[:,None]


@jit
def reynolds_jax(leader, X, X_dot):
    n = len(X)
    distance_matrix = distances_matrix_jax(X)
    agent_radius = 30
    leader_radius = 4*agent_radius
    
    leader_factor=5

    # Calculate if i'th agent is in the leader's neighborhood 
    leader_neighbors_count = leader_neighbors(distance_matrix[leader], leader_radius)
    
    # Calculate every swarm agent's neighborhood
    neighbors_matrix, neighbors_count = neighbors(distance_matrix, agent_radius, leader_neighbors_count)
    
    #leader = jnp.argmax(neighbors_count, axis=0)
    
    total_count = neighbors_count+leader_factor*leader_neighbors_count
    # Cohesion steer calculations
    cohesion = cohesion_steer(X, leader, leader_factor, neighbors_matrix, leader_neighbors_count, total_count)
    
    # Alignment steer calculations
    alignment = alignment_steer(X_dot, leader, leader_factor, neighbors_matrix, leader_neighbors_count, total_count)
    
    # Separation steer calculations
    separation =  separation_steer(X, distance_matrix, neighbors_matrix)
    neighbors_mask = (jnp.any(neighbors_matrix,axis=0) + leader_neighbors_count)[:,None]
    
    return neighbors_mask*(4*(cohesion-X_dot)+ 20*(alignment-X_dot) + 15*(separation-X_dot)), leader


@eqx.filter_jit
def script(state: EnvState, n_scripted_entities: int) -> chex.Array:
    """Calculates the scripted action for the swarm agents.

    Args:
        state (EnvState): State of the environment.
        n_scripted_entities (int): How many scripted agents are in the environment 

    Returns:
        chex.Array: _description_
    """    
    X_scripted = state.X[:n_scripted_entities]
    X_dot_scripted = state.X_dot[:n_scripted_entities]
    
    return reynolds_jax(state.leader, X_scripted, X_dot_scripted)