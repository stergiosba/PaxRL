#%%  
from jax.debug import print as dprint
import jax.numpy.linalg as la
import jax.numpy as jnp
from jax import jit, lax, random, vmap
import timeit
import jax


@jit
def distance_matrix_jax(X):
    """Calculates the euclidean distances between all pairs of swarm agents.

    Args:
        `X ()`: Swarm agents position matrix of shape: (n,2).

    Returns:
        `Distance_matrix ()`: Distance matrix of shape:(n,n)
    """  
    
    return jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

@jit
def neighbors(
        distance_matrix,
        agent_radius: float):
    """Calculates the neighborhood for each agent as a set and the cardinality of each set.
    Args:
        - `distance_matrix ()`: Distance matrix, contains euclidean distances between all pairs of swarm agents.
        - `agent_radius (float)`: Influence radius for the simple agents.

    Returns:
        - `neighbors_matrix ()`: Boolean array of neighbors.
        - `neighbors_count ()`: Row-wise sum of neighbors_matrix, i.e. cardinality of each neighborhood. 
    """
    
    neighbors_matrix = ((distance_matrix>0)&(distance_matrix<=agent_radius**2)).astype(int) 
    neighbors_count = jnp.sum(neighbors_matrix, axis=0)
    
    return neighbors_matrix, neighbors_count

@jit
def neighbors_influence(
    A,
    neighbors_matrix):
    """Calculates the influence of simple agents.

    Args:
        A (): Swarm agents position or velocity matrix of shape: (n,2).
        neighbors_matrix (): _description_
        max_speed (float): _description_

    Returns:
        - ``: _description_
    """
        
    return jnp.sum(neighbors_matrix[:,:,None]*A, axis=-2)

    
@jit
def cohesion_steer(
        X,
        X_dot,
        neighbors_matrix,
        total_count):
    """ Cohesion calculations of `Rax`: Leader modified Reynolds flocking model in Jax.
        Calculates the cohesion steering based on local interactions with simple agents and with the leader using Reynold's Dynamics
    
    Args:
        - `X ()`: Swarm agents position matrix of shape: (n,2).
        - `leader_id (int)`: The id of the leader.
        - `leader_str (float)`: The strength of the leader agent.
        - `neighbors_matrix ()`: Boolean array of neighbors.
        - `leader_neighbors_count ()`: _description_
        - `total_count ()`: The total number of neighbors for each agent.

    Returns:
        - `cohesion ()`: The cohesion force exerted to each agent.
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-16
    
    # Influence from simple neighbors
    neighbors_inf = neighbors_influence(X, neighbors_matrix)
    
    # Leader influence

    #debug.print("lead_inf={x}",x=leader_inf)
    # Cohesion steer calculation
    cohesion = (neighbors_inf)/(total_count[:,None]+eps)-X
    
    max_speed = 10
    cohesion = max_speed*(cohesion/la.norm(cohesion+eps, axis=1)[:,None]) - X_dot

    # Leader whitening (i.e. the leader is not affected by anyone.)
    #cohesion = cohesion.at[leader_id].set(jnp.array([0,0]))
    
    max_force = 20
    return max_force*(cohesion/la.norm(cohesion+eps, axis=1)[:,None])


@jit
def alignment_steer(
        X_dot, 
        neighbors_matrix,
        total_count):
    """Alignment calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X_dot ()`: _description_
        - `leader_id (int)`: _description_
        - `leader_factor (float)`: _description_
        - `neighbors_matrix ()`: _description_
        - `leader_neighbors_count ()`: _description_
        - `total_count ()`: _description_

    Returns:
        - ``: _description_
    """
    
    # Small epsilon to avoid division by zero
    eps = 10**-16
    
    # Influence from simple neighbors
    neighbors_inf = neighbors_influence(X_dot, neighbors_matrix)
    
    # Alignment steer calculation
    
    alignment = (neighbors_inf)/(total_count[:,None]+eps)
    
    max_speed = 10
    alignment = max_speed*(alignment/la.norm(alignment+eps, axis=1)[:,None]) - X_dot

    # Leader whitening (i.e. the leader is not affected by anyone.)
    #alignment = alignment.at[leader_id].set(jnp.array([0,0]))
    
    max_force = 20
    return max_force*(alignment/la.norm(alignment+eps, axis=1)[:,None])



@jit
def separation_steer(
        X,
        X_dot, 
        dist_mat,
        neighbors_matrix):
    """Separation calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X ()`: _description_
        - `X_dot ()`: _description_
        - `dist_mat ()`: _description_
        - `neighbors_matrix ()`: _description_

    Returns:
        - ``: _description_
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
    max_speed = 10
    separation = jnp.sum(scaled_neighbors_matrix,axis=1)[:,None]*X - scaled_neighbors_matrix@X
    separation = max_speed*(separation/(la.norm(separation+eps, axis=1)[:,None])) - X_dot
    
    max_force = 20
    return max_force*(separation/(la.norm(separation+eps, axis=1)[:,None]))


@jit
def reynolds_jax(X, X_dot):
    """`Rax`: Leader modified Reynolds flocking model in Jax.
    
    Note: Currently the id of the leader is returned as in the future we could use this function
    for dynamic leader allocation based on some metric (e.x. agent with most neighbors).

    Args:
        - `leader (int)`: The id number of the leader.
        - `X ()`: Swarm agents position matrix of shape: (n,2).
        - `X_dot ()`: Swarm agents velocity matrix of shape: (n,2).

    Returns:
        - `steer ()`: The steer vector of swarm agents shape=(n,2).
        - `leader (int)`: The id number of the leader.
    """
    
    distance_matrix = distance_matrix_jax(X)
    agent_radius = 30
    
    # Calculate the neighborhood of every swarm agent.
    neighbors_matrix, neighbors_count = neighbors(distance_matrix, agent_radius)
    
    # For dynamic leader allocation. Later work.
    #leader = jnp.argmax(neighbors_count, axis=0)
    
    # Calculate the total number of neighbors for each agent.
    total_count = neighbors_count

    # Cohesion steer calculation.
    cohesion = cohesion_steer(X, X_dot, neighbors_matrix, total_count)
    
    # Alignment steer calculation.
    alignment = alignment_steer(X_dot, neighbors_matrix, total_count)
    
    # Separation steer calculation.
    separation =  separation_steer(X, X_dot, distance_matrix, neighbors_matrix)
    
    w_c = 0.2
    w_a = 1
    w_s = 1

    return w_c*cohesion+ w_a*alignment+w_s*separation

def test(X,X_dot):
    for i in range(1000):
        d = reynolds_jax(X, X_dot).block_until_ready()
    
    X = X+d
    return d

# %%
n=256
m=3
key = random.PRNGKey(n)
key_v = random.PRNGKey(n*m-32)
X = random.uniform(key,shape=(n,2))*1
X_dot = random.uniform(key_v,shape=(n,2))*1

reynolds_jax.lower(X,X_dot).compile()
#%%
with jax.profiler.trace("/tmp/jax-trace4", create_perfetto_link=True):
    for i in range(m):
        x = test(X, X_dot)
x = x+1

#%%
m=3
TT=[]
for n in range(1,1601,100):
    key = random.PRNGKey(n)
    key_v = random.PRNGKey(n*m-32)
    X = jnp.array(random.uniform(key,shape=(n,2)))*800
    X_dot = jnp.array(random.uniform(key_v,shape=(n,2)))*3
    #x = test(X,X_dot).block_until_ready()
    #x=single_rollout(key,X,X_dot).block_until_ready()
    #t=(timeit.timeit("single_rollout(key,X,X_dot).block_until_ready()", setup="from __main__ import single_rollout,key,X,X_dot",number=m)/m)
    t=(timeit.timeit("test(X,X_dot)", setup="from __main__ import test,X,X_dot",number=m)/m)
    #TT.append(t)
    print(f"{n}, {t}")

with jax.profiler.trace("/tmp/jax-trace4", create_perfetto_link=True):
    for i in range(m):
        x = test(X, X_dot)
# %%
