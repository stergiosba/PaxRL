import jax.numpy as jnp
import equinox as eqx
import jax.debug as debug
import jax.numpy.linalg as la
import numpy as np
from jax import jit, lax


@jit
def distances_matrix_jax(X):
    return jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)

@jit
def neighbors(distance_matrix, agent_radius):
    neighbors_matrix = ((distance_matrix>0)&(distance_matrix<=agent_radius**2))
    neighbors_count = jnp.sum(neighbors_matrix, axis=0)
    return neighbors_matrix, neighbors_count

@jit
def leader_neighbors(leader_distances, leader_radius):
    return ((leader_distances>0)&(leader_distances<=(2*leader_radius)**2))

@jit
def cohesion_steer(X, leader, neighbors_matrix, leader_neighbors_count, total_count):
    # Small epsilon to avoid division by zero
    eps = 10**-16
    # Cohesion steer calculations
    cohesion = (jnp.sum(neighbors_matrix[:,:,None]*X, axis=-2)+leader_neighbors_count[:,None]*X[leader])/(total_count[:,None]+eps)-X
    
    return cohesion/la.norm(cohesion+eps, axis=1)[:,None]

@jit
def alignment_steer(X_dot, leader, neighbors_matrix, leader_neighbors_count, total_count):
    # Small epsilon to avoid division by zero
    eps = 10**-16
    # Alignment steer calculations
    alignment = (jnp.sum(neighbors_matrix[:,:,None]*X_dot, axis=-2)+leader_neighbors_count[:,None]*X_dot[leader])/(total_count[:,None]+eps)
    
    return alignment/la.norm(alignment+eps, axis=1)[:,None]
    
@jit 
def separation_steer(X, distance_matrix, neighbors_matrix):
    # Small epsilon to avoid division by zero
    eps = 10**-16
    n = len(X)
    corrected_distance_matrix = distance_matrix+jnp.eye(n)
    separation =  jnp.sum(neighbors_matrix/corrected_distance_matrix,axis=1)[:,None]*X - ((neighbors_matrix/corrected_distance_matrix)[:,None]@X).reshape(n,2)
    
    return separation/la.norm(separation+eps, axis=1)[:,None]

#@jit(backend="cpu")
def reynolds_jax(t, leader, X, X_dot):
    n = len(X)
    distance_matrix = distances_matrix_jax(X)
    agent_radius = 40
    leader_radius = 2*agent_radius

    # Calculate every swarm agent's neighborhood
    neighbors_matrix, neighbors_count = neighbors(distance_matrix, agent_radius)
    
    # Calculate if i'th agent is in the leader's neighborhood 
    leader_neighbors_count = leader_neighbors(distance_matrix[leader], leader_radius)
    
    total_count = neighbors_count+leader_neighbors_count
    
    # Cohesion steer calculations
    cohesion = cohesion_steer(X, leader, neighbors_matrix, leader_neighbors_count, total_count)
    
    # Alignment steer calculations
    alignment = alignment_steer(X_dot, leader, neighbors_matrix, leader_neighbors_count, total_count)
    
    # Separation steer calculations
    separation =  separation_steer(X, distance_matrix, neighbors_matrix)

    return 20*(cohesion-X_dot) + 10*(alignment-X_dot) + 25*(separation-X_dot)

@eqx.filter_jit
def script(state, n_scripted_entities):
    X_scripted = state.X[:n_scripted_entities]
    X_dot_scripted = state.X_dot[:n_scripted_entities]
    steer = reynolds_jax(state.t, state.leader, X_scripted, X_dot_scripted)

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