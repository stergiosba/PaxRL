import chex
import jax.numpy.linalg as la
from jax import jit, numpy as jnp, nn as jnn
from typing import Tuple
from jax.debug import print as dprint
#from pax.core.environment import EnvState

@jit
def distance_tensor_jax(X:chex.Array) -> chex.Array:
    """Parallelized calculation of euclidean distances between all pairs of swarm agents
        for all environment instances in one go.

    Args:
        `X (chex.Array)`: Swarm agents position tensor of shape: (n_envs,n_agents,2).

    Returns:
        `Distance_matrix (chex.Array)`: Distance tensor of shape: (n_envs,n_agents,n_agents)
    """  
    
    return jnp.einsum('ijkl->ijk',(X[:,None, ...]-X[...,None,:]) **2)

@jit
def leader_neighbors(
        distance_tensor:chex.Array,
        leader:chex.Array,
        leader_radius: float) -> chex.Array:
    """Calculates the neighborhood for the leader agent as a set for every parallel environment.
    Args:
        - `leader_distances (chex.Array)`: Slice of distance tensor, distances between leader and each swarm agent for every
            parallel environment.
        - `leader_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

    Returns:
        - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
    """
    n_env, n, _ = distance_tensor.shape
    leader_OH = jnn.one_hot(leader, n)
    leader_distances = (leader_OH@distance_tensor)[jnp.arange(n_env), 0]
    leader_neighbors_matrix = ((leader_distances>0)&(leader_distances<=(leader_radius)**2)).astype(int)
    leader_neighbors_matrix = leader_neighbors_matrix.at[:,-1].set(jnp.array(0))

    return leader_neighbors_matrix

@jit
def prober_neighbors(
        distance_tensor:chex.Array,
        prober_radius: float) -> chex.Array:
    """Calculates the neighborhood for the leader agent as a set for every parallel environment.
    Args:
        - `leader_distances (chex.Array)`: Slice of distance tensor, distances between leader and each swarm agent for every
            parallel environment.
        - `prober_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

    Returns:
        - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
    """
    n_env, n, _ = distance_tensor.shape
    prober_OH = jnn.one_hot((n-1)*jnp.ones(n_env), n)
    prober_distances = (prober_OH@distance_tensor)[jnp.arange(n_env), 0]
    prober_neighbors_matrix = ((prober_distances>0)&(prober_distances<=(prober_radius)**2)).astype(int)

    return prober_neighbors_matrix


@jit
def neighbors(
        distance_tensor: chex.Array,
        agent_radius: float) -> Tuple[chex.Array, chex.Array]:
    """Calculates the neighborhood for each agent as a set and the cardinality of each set.
    Args:
        - `distance_tensor (chex.Array)`: Distance tensor, euclidean distances between all pairs of swarm agents for all 
            parallel environments.
        - `agent_radius (float)`: Influence radius for the simple agents. Same for all environments.

    Returns:
        - `neighbors_tensor (chex.Array)`: Tensor of neighbors for every agent and every environment.
        - `neighbors_count (chex.Array)`: Agent-wise sum of neighbors_tensor, i.e. cardinality of each neighborhood. 
    """
    _, n, _ = distance_tensor.shape
    neighbors_tensor = ((distance_tensor>0)&(distance_tensor<=agent_radius**2)).astype(int)

    # Its faster to do prober whitening here and use jax.jit rather than breaking the array
    # and doing eqx.filter_jax.
    neighbors_tensor = neighbors_tensor.at[...,-1].set(jnp.zeros(n, dtype=int))
    neighbors_tensor = neighbors_tensor.at[:,-1].set(jnp.zeros(n, dtype=int))
    
    # Calculating all neighbors counts, Summing over the agents axis: 1.
    neighbors_count = jnp.sum(neighbors_tensor, axis=1)

    return neighbors_tensor, neighbors_count

@jit
def total_influence(
    X,
    X_dot,
    neighbors_matrix,
    leader_neighbors_matrix,
    leader,
    leader_str):
    """Calculates the influence of simple agents and then of the leader agent.

    Args:
        - `X (chex.Array)`: Swarm agents position matrix of shape: (n,2).
        - `X_dot (chex.Array)`: Swarm agents velocity matrix of shape: (n,2).
        - `neighbors_matrix (chex.Array)`: Array of neighbors for every agent.
        - `leader_neighbors_matrix (chex.Array): _description_
        - `leader (chex.Array): _description_

    Returns:
        - `chex.Array`: _description_
    """
    K = jnp.vstack([[X],[X_dot]])

    neight_inf = neighbors_matrix@K
    leader_inf = leader_str*leader_neighbors_matrix[:,None]*(K[:,leader][:,None])

    return neight_inf+leader_inf

@jit
def mixed_steer(
        X: chex.Array,
        X_dot: chex.Array,
        leader:int,
        total_inf: chex.Array,
        total_count: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """ Cohesion and alignment calculations of `Rax`: Leader modified Reynolds flocking model in Jax.
        Calculates the cohesion steering based on local interactions with simple agents and with the leader using Reynold's Dynamics
    
    Args:
        - `X (chex.Array)`: Swarm agents position matrix of shape: (n,2).
        - `X_dot (chex.Array)`: Swarm agents velocity matrix of shape: (n,2).
        - `total_inf (chex.Array)`: The total influence calculated for each agent
        - `total_count (chex.Array)`: The total number of neighbors for each agent.

    Returns:
        - `cohesion (chex.Array)`: The cohesion force exerted to each agent.
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-7

    # Cohesion steer calculation
    cohesion = (total_inf[0])/(total_count[:,None]+eps)-X

    alignment = (total_inf[1])/(total_count[:,None]+eps)
    
    max_speed = 20
    cohesion = max_speed*(cohesion/la.norm(cohesion+eps, axis=1)[:,None]) - X_dot
    alignment = max_speed*(alignment/la.norm(alignment+eps, axis=1)[:,None]) - X_dot

    # Leader whitening (i.e. the leader is not affected by anyone.)
    cohesion = cohesion.at[leader].set(jnp.array([0,0]))
    alignment = alignment.at[leader].set(jnp.array([0,0]))
    
    max_force = 40
    return (max_force*(cohesion/la.norm(cohesion+eps, axis=1)[:,None]),
            max_force*(alignment/la.norm(alignment+eps, axis=1)[:,None]))

@jit
def separation_steer(
        X: chex.Array,
        X_dot: chex.Array, 
        corr_dist_mat: chex.Array,
        neighbors_matrix: chex.Array) -> chex.Array:
    """Separation calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X (chex.Array)`: _description_
        - `X_dot (chex.Array)`: _description_
        - `corr_dist_mat (chex.Array)`: _description_
        - `neighbors_matrix (chex.Array)`: _description_

    Returns:
        - `chex.Array`: _description_
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-7
    
    scaled_neighbors_matrix = neighbors_matrix/corr_dist_mat
    
    # Separation steer calculation
    max_speed = 20
    separation = jnp.einsum('ij,jk->jk',scaled_neighbors_matrix,X) - scaled_neighbors_matrix@X
    separation = max_speed*(separation/(la.norm(separation+eps, axis=1)[:,None])) - X_dot

    max_force = 40
    separation = max_force*(separation/(la.norm(separation+eps, axis=1)[:,None]))
    return separation

@jit
def interaction_steer(
        X: chex.Array,
        corr_dist_mat: chex.Array) -> chex.Array:
    """Separation calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X (chex.Array)`: _description_
        - `X_dot (chex.Array)`: _description_
        - `corr_dist_mat (chex.Array)`: _description_
        - `neighbors_matrix (chex.Array)`: _description_

    Returns:
        - `chex.Array`: _description_
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-7

    max_force = 75
    
    interaction = (X-X[-1])/(corr_dist_mat[:,-1][:,None])

    return max_force*(interaction/(la.norm(interaction+eps, axis=1)[:,None]))

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
    
    # The distance tensor, contains distance matrices for each separate environment. We calculate everything in parallel
    # for all environments.
    T_d = distance_tensor_jax(X)
    
    # The main diagonal of a distance matrix is 0 since d(1,1) is the distance of agent 1 from itself
    # Therefore when we divide by the distances we would get 1/0. Adding the ones matrix scaled down to eps
    # makes the algorithm stable. The identity matrix could also be used but there are cases where two agents
    # occupy the same space and then it blows up ot NaN. In reality the agents need some form of real collision
    # avoidance but this would make the simulation run slower. Also this happens because agents are in box and they
    # do not bounce of the wall.
    n_env, n, _ = X.shape
    eps = 10**-7
    T_d_corr = T_d + eps*jnp.ones([n_env,n,n])

    agent_radius = 30
    ldr_radius = 4*agent_radius
    prb_radius = 3*agent_radius
    
    # Leader strength
    ldr_str = 50

    # Calculate the neighborhood of the leader.
    T_ldr = leader_neighbors(T_d, leader, ldr_radius)
    
    # Calculate the neighborhood of every swarm agent.
    T_nbr, C_nbr = neighbors(T_d, agent_radius)

    # Calculate the neighborhood of the prober.
    T_prb = prober_neighbors(T_d[-1], prb_radius)
    
    # For dynamic leader allocation. Later work.
    #leader = jnp.argmax(neighbors_count, axis=0)
    
    # Calculate the total number of neighbors for each agent.
    total_count = C_nbr+ldr_str*T_ldr

    # Influence from simple neighbors and leader
    total_inf = total_influence(X, X_dot, T_nbr, T_ldr, leader, ldr_str)

    # Cohesion and  Alignment steer calculation.
    cohesion, alignment = mixed_steer(X, X_dot, leader, total_inf, total_count)

    # Separation steer calculation.
    separation =  separation_steer(X, X_dot, T_d_corr, T_nbr)

    # Prober interaction.
    interaction = interaction_steer(X, T_d_corr)

    # Performs neighbors masking.
    total_mask = (total_count>0)[:,None]
    neighbors_mask = (C_nbr>0)[:,None]
    probed_mask = T_prb[:,None]
    w_c = 0.4
    w_a = 1
    w_s = 1.2

    return (total_mask*(w_c*cohesion+ w_a*alignment) + neighbors_mask*w_s*separation + 0*probed_mask*interaction), leader


@jit
def script(state, *args) -> Tuple[chex.Array, int]:
    """Calculates the scripted action for the swarm agents.

    `Args`:
        - `state` (EnvState): State of the environment.

    `Returns`:
        - `steer` (chex.Array): The steer vector of swarm agents shape=(n,2).
    """
    S, leader = reynolds_jax(state.leader, state.X, state.X_dot)

    e = state.goal-state.X[leader]
    Kp = 0
    u = Kp*e
    #dprint("{x}",x=S[leader])
    S = S.at[leader].set(S[leader]+u)
    #dprint("{x}\n---",x=la.norm(state.X_dot[leader]))
    #S = S.at[-1].set(jnp.zeros(2))

    return S, leader