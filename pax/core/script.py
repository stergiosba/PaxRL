import chex
import jax.numpy as jnp
import equinox as eqx
from jax.debug import print as dprint
import jax.numpy.linalg as la
import numpy as np
from typing import Tuple
from jax import jit, lax
from pax.core.state import EntityState, AgentState, EnvState


@jit
def distance_matrix_jax(X:chex.Array) -> chex.Array:
    """Calculates the euclidean distances between all pairs of swarm agents.

    Args:
        `X (chex.Array)`: Swarm agents position matrix of shape: (n,2).

    Returns:
        `Distance_matrix (chex.Array)`: Distance matrix of shape:(n,n)
    """  
    
    return jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)


@jit
def leader_neighbors(
        leader_distances:chex.Array,
        leader_radius: float) -> chex.Array:
    """Calculates the neighborhood for the leader agent as a set.
    Args:
        - `leader_distances (chex.Array)`: Row of distance matrix, containing euclidean distances between the leader and each swarm agent.
        - `leader_radius (float)`: Influence radius for the leader agent (typically taken as a scaled version of the simple agent's radius)

    Returns:
        - `leader_neighbors_matrix (chex.Array)`: Boolean vector of leader's neighbors.
    """
    
    return ((leader_distances>0)&(leader_distances<=(leader_radius)**2)).astype(int)


@jit
def neighbors(
        distance_matrix: chex.Array,
        agent_radius: float) -> Tuple[chex.Array, chex.Array]:
    """Calculates the neighborhood for each agent as a set and the cardinality of each set.
    Args:
        - `distance_matrix (chex.Array)`: Distance matrix, contains euclidean distances between all pairs of swarm agents.
        - `agent_radius (float)`: Influence radius for the simple agents.

    Returns:
        - `neighbors_matrix (chex.Array)`: Boolean array of neighbors.
        - `neighbors_count (chex.Array)`: Row-wise sum of neighbors_matrix, i.e. cardinality of each neighborhood. 
    """
    
    neighbors_matrix = ((distance_matrix>0)&(distance_matrix<=agent_radius**2)).astype(int) 
    neighbors_count = jnp.sum(neighbors_matrix, axis=0)
    
    return neighbors_matrix, neighbors_count

@jit
def neighbors_influence(
    A: chex.Array,
    neighbors_matrix: chex.Array) -> chex.Array:
    """Calculates the influence of simple agents.

    Args:
        A (chex.Array): Swarm agents position or velocity matrix of shape: (n,2).
        neighbors_matrix (chex.Array): _description_
        max_speed (float): _description_

    Returns:
        - `chex.Array`: _description_
    """
        
    return jnp.sum(neighbors_matrix[:,:,None]*A, axis=-2)

    
@jit
def cohesion_steer(
        X: chex.Array,
        X_dot: chex.Array,
        leader_id: int,
        leader_str: float, 
        neighbors_matrix: chex.Array,
        leader_neighbors_count: chex.Array,
        total_count: chex.Array) -> chex.Array:
    """ Cohesion calculations of `Rax`: Leader modified Reynolds flocking model in Jax.
        Calculates the cohesion steering based on local interactions with simple agents and with the leader using Reynold's Dynamics
    
    Args:
        - `X (chex.Array)`: Swarm agents position matrix of shape: (n,2).
        - `leader_id (int)`: The id of the leader.
        - `leader_str (float)`: The strength of the leader agent.
        - `neighbors_matrix (chex.Array)`: Boolean array of neighbors.
        - `leader_neighbors_count (chex.Array)`: _description_
        - `total_count (chex.Array)`: The total number of neighbors for each agent.

    Returns:
        - `cohesion (chex.Array)`: The cohesion force exerted to each agent.
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-16
    
    # Influence from simple neighbors
    neighbors_inf = neighbors_influence(X, neighbors_matrix)
    
    # Leader influence
    leader_inf = leader_str*leader_neighbors_count[:,None]*X[leader_id]
    #debug.print("lead_inf={x}",x=leader_inf)
    # Cohesion steer calculation
    cohesion = (neighbors_inf+leader_inf)/(total_count[:,None]+eps)-X
    
    max_speed = 2
    cohesion = max_speed*(cohesion/la.norm(cohesion+eps, axis=1)[:,None]) - X_dot

    # Leader whitening (i.e. the leader is not affected by anyone.)
    #cohesion = cohesion.at[leader_id].set(jnp.array([0,0]))
    
    max_force = 20
    return max_force*(cohesion/la.norm(cohesion+eps, axis=1)[:,None])


@jit
def alignment_steer(
        X_dot: chex.Array, 
        leader_id: int,
        leader_factor: float, 
        neighbors_matrix: chex.Array,
        leader_neighbors_count: chex.Array,
        total_count: chex.Array) -> chex.Array:
    """Alignment calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X_dot (chex.Array)`: _description_
        - `leader_id (int)`: _description_
        - `leader_factor (float)`: _description_
        - `neighbors_matrix (chex.Array)`: _description_
        - `leader_neighbors_count (chex.Array)`: _description_
        - `total_count (chex.Array)`: _description_

    Returns:
        - `chex.Array`: _description_
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-16
    
    # Influence from simple neighbors
    neighbors_inf = neighbors_influence(X_dot, neighbors_matrix)
    
    # Leader influence
    leader_inf = leader_factor*leader_neighbors_count[:,None]*X_dot[leader_id]
    # Alignment steer calculation
    
    alignment = (neighbors_inf+leader_inf)/(total_count[:,None]+eps)
    
    max_speed = 2
    alignment = max_speed*(alignment/la.norm(alignment+eps, axis=1)[:,None]) - X_dot

    # Leader whitening (i.e. the leader is not affected by anyone.)
    #alignment = alignment.at[leader_id].set(jnp.array([0,0]))
    
    max_force = 20
    return max_force*(alignment/la.norm(alignment+eps, axis=1)[:,None])


@jit
def separation_steer(
        X: chex.Array,
        X_dot: chex.Array, 
        dist_mat: chex.Array,
        neighbors_matrix: chex.Array) -> chex.Array:
    """Separation calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X (chex.Array)`: _description_
        - `X_dot (chex.Array)`: _description_
        - `dist_mat (chex.Array)`: _description_
        - `neighbors_matrix (chex.Array)`: _description_

    Returns:
        - `chex.Array`: _description_
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-16
    n = len(X)
    
    # The main diagonal of a distance matrix is 0 since d(1,1) is the distance of agent 1 from itself
    # Therefore when we divide by the distances we would get 1/0. Notice that adding the identity matrix
    # will not change the separation calculations for other distance pairs as we only add one in the diagonal elements.
    adj_dist_mat = dist_mat+jnp.eye(n)
    
    scaled_neighbors_matrix = neighbors_matrix/adj_dist_mat
    
    # Separation steer calculation
    max_speed = 2
    separation = jnp.sum(scaled_neighbors_matrix,axis=1)[:,None]*X - scaled_neighbors_matrix@X
    separation = max_speed*(separation/(la.norm(separation+eps, axis=1)[:,None])) - X_dot
    
    max_force = 20
    return max_force*(separation/(la.norm(separation+eps, axis=1)[:,None]))


@jit
def reynolds_jax(leader: int, X: chex.Array, X_dot: chex.Array) -> Tuple[chex.Array, int]:
    """`Rax`: Leader modified Reynolds flocking model in Jax.
    
    Note: Currently the id of the leader is returned as in the future we could use this function
    for dynamic leader allocation based on some metric (e.x. agent with most neighbors).

    Args:
        - `leader (int)`: The id number of the leader.
        - `X (chex.Array)`: Swarm agents position matrix of shape: (n,2).
        - `X_dot (chex.Array)`: Swarm agents velocity matrix of shape: (n,2).

    Returns:
        - `steer (chex.Array)`: The steer vector of swarm agents shape=(n,2).
        - `leader (int)`: The id number of the leader.
    """
    
    distance_matrix = distance_matrix_jax(X)
    agent_radius = 30
    leader_radius = 4*agent_radius
    
    # Leader strength
    leader_str = 50

    # Calculate the neighborhood of the leader that uses different radius.
    leader_neighbors_count = leader_neighbors(distance_matrix[leader], leader_radius)
    
    # Calculate the neighborhood of every swarm agent.
    neighbors_matrix, neighbors_count = neighbors(distance_matrix, agent_radius)
    
    # For dynamic leader allocation. Later work.
    #leader = jnp.argmax(neighbors_count, axis=0)
    
    # Calculate the total number of neighbors for each agent.
    total_count = neighbors_count+leader_str*leader_neighbors_count

    # Cohesion steer calculation.
    cohesion = cohesion_steer(X, X_dot, leader, leader_str, neighbors_matrix, leader_neighbors_count, total_count)
    
    # Alignment steer calculation.
    alignment = alignment_steer(X_dot, leader, leader_str, neighbors_matrix, leader_neighbors_count, total_count)
    
    # Separation steer calculation.
    separation =  separation_steer(X, X_dot, distance_matrix, neighbors_matrix)
    
    # Performs neighbors masking.
    total_mask = (total_count>0)[:,None]
    neighbors_mask = (neighbors_count>0)[:,None]
    w_c = 0.4
    w_a = 1
    w_s = 2

    return (total_mask*(w_c*cohesion+ w_a*alignment) + neighbors_mask*w_s*separation), leader


@eqx.filter_jit
def script(state: EnvState, n_scripted: int) -> chex.Array:
    """Calculates the scripted action for the swarm agents.

    `Args`:
        - `state` (EnvState): State of the environment.
        - `n_scripted` (int): How many scripted agents are in the environment 

    `Returns`:
        - `steer` (chex.Array): The steer vector of swarm agents shape=(n,2).
    """
    
    X_scripted = state.X[:n_scripted]
    X_dot_scripted = state.X_dot[:n_scripted]
    return reynolds_jax(state.leader, X_scripted, X_dot_scripted)