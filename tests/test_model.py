#%%
import chex
import jax.numpy as jnp
import equinox as eqx
import jax.debug as debug
import jax.numpy.linalg as la
import numpy as np
from typing import Tuple
from jax import jit, lax, random
from scipy.spatial import distance_matrix


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
def neighbors(
        distance_matrix: chex.Array,
        agent_radius: float) -> Tuple[chex.Array, chex.Array]:
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

#@jit
def alignment_steer(
        X_dot: chex.Array, 
        neighbors_matrix: chex.Array,
        total_count: chex.Array) -> chex.Array:
    # Small epsilon to avoid division by zero
    eps = 10**-16
    
    # Influence from simple neighbors
    max_speed = 1
    neighbors_influence = jnp.sum(neighbors_matrix[:,:,None]*X_dot, axis=-2)
    neighbors_influence = max_speed*(neighbors_influence/la.norm(neighbors_influence, axis=1)[:,None])

    # Alignment steer calculation
    alignment = (neighbors_influence)/(total_count[:,None])-V
                
    max_force = 1
    
    return max_force*alignment/la.norm(alignment+eps, axis=1)[:,None]

def align(aid, velocities, neighbors, leader_id):
    neighbors_aid = neighbors[aid]
    sum = np.sum(neighbors_aid[:,None]*velocities,axis=0)

    max_speed = 2
    max_force = 40
    if jnp.sum(neighbors_aid):
        sum=(sum/la.norm(sum))*max_speed
        steer=(sum/jnp.sum(neighbors_aid))-velocities[aid,:]  #steering velocity based on the above calculations
        steer=(steer/la.norm(steer))*max_force;   #normalising acceleration/force
    return steer

n=5
key = random.PRNGKey(1)
key2 = random.PRNGKey(2)
A = random.uniform(key,shape = (n,2))
V = random.uniform(key2,shape = (n,2))
agent_radius=51

D1 = distance_matrix(A,A)
D2 = distances_matrix_jax(A)

N1 = ((D1>0)&(D1<=agent_radius))
N2,c2 = neighbors(D2**0.5, agent_radius)
assert((N1==N2).all())
# %%

for i in range(n):
    steer = align(i,V,N1,0)
    print(steer)

alignment_steer(V,N2,c2)


# %%
