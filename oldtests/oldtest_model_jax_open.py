#%% 
#%env XLA_PYTHON_CLIENT_PREALLOCATE=false 
%env unset XLA_PYTHON_CLIENT_PREALLOCATE 
from jax.debug import print as dprint
import jax.numpy.linalg as la
import jax.numpy as jnp
from jax import jit, lax, random, vmap
import timeit
import jax

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
    
    distance_matrix = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    agent_radius = 30
    
    # Calculate the neighborhood of every swarm agent.
    neighbors_matrix = ((distance_matrix>0)&(distance_matrix<=agent_radius**2))
    neighbors_count = jnp.sum(neighbors_matrix, axis=0)
    
    # For dynamic leader allocation. Later work.
    #leader = jnp.argmax(neighbors_count, axis=0)
    
    # Calculate the total number of neighbors for each agent.
    total_count = neighbors_count

    # Cohesion steer calculation.
    K = jnp.vstack([[X], [X_dot]])
    neighbors_inf = neighbors_matrix@K

    # Leader influence
    eps = 10**-16
    max_force = 20
    max_speed = 10

    # Cohesion steer calculation
    cohesion = (neighbors_inf[0])/(total_count[:,None]+eps)-X
    
    cohesion = max_speed*(cohesion/la.norm(cohesion+eps, axis=1)[:,None]) - X_dot

    cohesion = max_force*(cohesion/la.norm(cohesion+eps, axis=1)[:,None])
    
    alignment = (neighbors_inf[1])/(total_count[:,None]+eps)
    
    alignment = max_speed*(alignment/la.norm(alignment+eps, axis=1)[:,None]) - X_dot

    alignment = max_force*(alignment/la.norm(alignment+eps, axis=1)[:,None])
    
    # Separation steer calculation.
    n = len(X)
    adj_dist_mat = distance_matrix+jnp.eye(n)
    
    scaled_neighbors_matrix = neighbors_matrix/adj_dist_mat
    
    separation = jnp.sum(scaled_neighbors_matrix,axis=1)[:,None]*X - scaled_neighbors_matrix@X
    
    separation = max_speed*(separation/(la.norm(separation+eps, axis=1)[:,None])) - X_dot
    
    separation =  max_force*(separation/(la.norm(separation+eps, axis=1)[:,None]))
    
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
n=512
m=3
key = random.PRNGKey(n)
key_v = random.PRNGKey(n*m-32)
X = random.uniform(key,shape=(n,2))*1
X_dot = random.uniform(key_v,shape=(n,2))*1

#%%
with jax.profiler.trace("/tmp/jax-trace4", create_perfetto_link=True):
    for i in range(m):
        x = test(X, X_dot)
x = x+1

#%%
m=3
TT=[]
for n in range(1,501,10):
    key = random.PRNGKey(n)
    key_v = random.PRNGKey(n*m-32)
    X = jnp.array(random.uniform(key,shape=(n,2)))*800
    X_dot = jnp.array(random.uniform(key_v,shape=(n,2)))*3
    x = test(X,X_dot)
    #x=single_rollout(key,X,X_dot).block_until_ready()
    #t=(timeit.timeit("single_rollout(key,X,X_dot).block_until_ready()", setup="from __main__ import single_rollout,key,X,X_dot",number=m)/m)
    x+=1
    t=(timeit.timeit("test(X,X_dot)", setup="from __main__ import test,X,X_dot",number=m)/m)
    #TT.append(t)
    print(f"{n}, {t}")




# %%
with jax.profiler.trace("/tmp/jax-trace4", create_perfetto_link=True):
    for i in range(m):
        x = test(X, X_dot)
# %%
