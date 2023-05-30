import jax.numpy as jnp
import equinox as eqx
import jax.debug as debug
import jax.numpy.linalg as la
import numpy as np
from jax import jit, lax

@jit
def nearest_neighbors_jax(t, leader, X, X_dot):
    distances = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    neighborhood_radius = 40

    column_indices = jnp.where(((distances>0)&(distances<=neighborhood_radius**2)),size=len(X)*(len(X)-1))[1]
    queries = X[column_indices]

    # Calculate every swarm agent's neighborhood
    I = ((distances>0)&(distances<=neighborhood_radius**2)).astype(int)
    
    # Calculate leader's neighborhood
    L = ((distances>0)&(distances<=(2*neighborhood_radius)**2)).astype(int)[leader]
    neighbors_count = jnp.sum(I, axis=0)
    S = neighbors_count+L
    
    # Small epsilon to avoid division by zero
    eps = 10**-16
    # Cohesion steer calculations
    cohesion = (jnp.sum(I[:,:,None]*X, axis=-2)+L[:,None]*X[leader])/(S[:,None]+eps)-X
    normed_cohesion = cohesion/la.norm(cohesion+eps, axis=1)[:,None]
    
    # Alignment steer calculations
    alignment = (jnp.mean(I[:,:,None]*X_dot, axis=-2)+L[:,None]*X_dot[leader])/(S[:,None]+eps)
    normed_alignment = alignment/la.norm(alignment+eps, axis=1)[:,None]
    
    
    steer = 20*(normed_cohesion-X_dot) + 20*(normed_alignment-X_dot)
    #debug.print("t:{t} - A={J}\n",t=t, J=I@X)    
    

    return steer,queries,neighbors_count,column_indices

@eqx.filter_jit
def script(state, n_scripted_entities):
    X_scripted = state.X[:n_scripted_entities]
    X_dot_scripted = state.X_dot[:n_scripted_entities]
    steer,queries,neighbor_counts,column_indices = nearest_neighbors_jax(state.t, state.leader, X_scripted, X_dot_scripted)

    return steer


def cohesion(aid, self, positions, velocities):
    n_agents = len(positions)
    neighbor_dist = 30.0
    steer = np.zeros(2)

    #storing distance of object with other boid, last index has distance 0 i.e. distance with the object itself
    d = pdist(np.append([self],positions,axis=0))
    d = d[0:n_agents]

    neighbors = np.where((d > 0) & (d <neighbor_dist))[0]
    ave = np.mean(positions[neighbors],axis=0)
    max_speed = 40
    max_force = 2
    desired = ave - self
    desired = desired/la.norm(desired)*max_speed
    steer = desired - self
    steer = steer/la.norm(steer)*max_force

    return steer

def align(aid, self, positions, velocities):
    n_agents = len(positions)
    neighbor_dist = 30
    l_index = 0
    leader_position = np.zeros(2)

    d = pdist(np.append([self], positions,axis=0))
    
    neighbors = np.where((d > 0) & (d <neighbor_dist))
    sum = np.sum(velocities[neighbors],axis=0)

    #ave = normalize([np.ma.average(velocities[np.where((d > 0) & (d <neighbor_dist))],axis=0)])*40
    max_speed = 40
    max_force = 2
    if len(neighbors):
        sum/=la.norm(sum)*max_speed
        steer=(sum/len(neighbors))-velocities[aid,:]  #steering velocity based on the above calculations
        steer/=la.norm(steer)*max_force;   #normalising acceleration/force
    return steer

def seperate(aid, self, positions, velocities):
    n_agents = len(positions)
    desired_separation = 30.0    #neighbourhood
    steer = np.zeros(2)

    #storing distance of object with other boid, last index has distance 0 i.e. distance with the object itself
    d = pdist(np.append([self],positions,axis=0))
    d = d[0:n_agents]    
    neighbors = np.where((d > 0) & (d <desired_separation))[0]

    if len(neighbors):
        D = normalize(positions[aid]-positions[neighbors,:])/np.transpose([d[neighbors]])
        steer = np.mean(D,axis=0)
        
        max_speed = 40
        max_force = 2
        if la.norm(steer)>0:
            steer = steer/la.norm(steer)*max_speed
            steer -=velocities[aid,:]
            steer = steer/la.norm(steer)*max_force

    return steer