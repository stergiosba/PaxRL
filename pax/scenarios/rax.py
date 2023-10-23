import chex
import jax.numpy.linalg as la
from jax import jit, numpy as jnp, nn as jnn
from typing import Tuple, Dict
from jax.debug import print as dprint
from pax.core.state import EnvState
from pax.core.environment import EnvParams


def distance_tensor_jax(X: chex.ArrayDevice) -> chex.ArrayDevice:
    """Parallelized calculation of euclidean distances between all pairs of swarm agents
        for all environment instances in one go.

    Args:
        `X (chex.Array)`: Swarm agents position tensor of shape: (n_envs,n_agents,2).

    Returns:
        `Distance_matrix (chex.Array)`: Distance tensor of shape: (n_envs,n_agents,n_agents)
    """
    return jnp.einsum("ijkl->ijk", (X[:, None, ...] - X[..., None, :]) ** 2)


def leader_neighbors(
    distance_tensor: chex.ArrayDevice, leader: chex.ArrayDevice, leader_radius: float
) -> chex.ArrayDevice:
    """Calculates the neighborhood for the leader agent as a set for every parallel environment.
    Args:
        - `leader_distances (chex.Array)`: Slice of distance tensor, distances between leader and each swarm agent for every
            parallel environment.
        - `leader_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

    Returns:
        - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
    """
    n_env, _, _ = distance_tensor.shape
    leader_distances = distance_tensor[jnp.arange(n_env), leader, :]
    leader_neighbors_matrix = (leader_distances > 0) & (
        leader_distances <= (leader_radius) ** 2
    )

    return leader_neighbors_matrix


def prober_neighbors(
    distance_tensor: chex.ArrayDevice, prober_radius: float
) -> chex.ArrayDevice:
    """Calculates the neighborhood for the leader agent as a set for every parallel environment.
    Args:
        - `leader_distances (chex.Array)`: Slice of distance tensor, distances between leader and each swarm agent for every
            parallel environment.
        - `prober_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

    Returns:
        - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
    """
    prober_distances = distance_tensor[:, -1, :]
    prober_neighbors_matrix = (prober_distances > 0) & (
        prober_distances <= (prober_radius) ** 2
    )

    return prober_neighbors_matrix


def neighbors(
    distance_tensor: chex.ArrayDevice, agent_radius: float
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    """Calculates the neighborhood for each agent as a set and the cardinality of each set.
    Args:
        - `distance_tensor (chex.ArrayDevice)`: Distance tensor, euclidean distances between all pairs of swarm agents for all
            parallel environments.
        - `agent_radius (float)`: Influence radius for the simple agents. Same for all environments.

    Returns:
        - `neighbors_tensor (chex.ArrayDevice)`: Tensor of neighbors for every agent and every environment.
        - `neighbors_count (chex.ArrayDevice)`: Agent-wise sum of neighbors_tensor, i.e. cardinality of each neighborhood.
    """
    neighbors_tensor = (distance_tensor > 0) & (distance_tensor <= agent_radius**2)

    # Calculating all neighbors counts, Summing over the agents axis: 1.
    neighbors_count = jnp.sum(neighbors_tensor, axis=1)

    return neighbors_tensor, neighbors_count


def total_influence(
    X: chex.ArrayDevice,
    X_dot: chex.ArrayDevice,
    neighbors_matrix: chex.ArrayDevice,
    leader_neighbors_matrix: chex.ArrayDevice,
    leader: chex.ArrayDevice,
    leader_str: float,
):
    """Calculates the influence of simple agents and then of the leader agent.

    Args:
        - `X (chex.ArrayDevice)`: Swarm agents position matrix of shape: (n,2).
        - `X_dot (chex.ArrayDevice)`: Swarm agents velocity matrix of shape: (n,2).
        - `neighbors_matrix (chex.ArrayDevice)`: Array of neighbors for every agent.
        - `leader_neighbors_matrix (chex.ArrayDevice): _description_
        - `leader (chex.ArrayDevice): _description_

    Returns:
        - `chex.ArrayDevice`: _description_
    """
    n_env, _, _ = X.shape
    K = jnp.vstack([[X], [X_dot]])  # type: ignore

    neight_inf = neighbors_matrix @ K
    leader_inf = leader_str * jnp.einsum(
        "ij,kil->kijl", leader_neighbors_matrix, K[:, jnp.arange(n_env), leader, :]
    )

    total_inf = neight_inf + leader_inf
    total_inf = total_inf.at[..., -1, :].set(jnp.array(0))
    return total_inf


def mixed_steer(
    X: chex.ArrayDevice,
    X_dot: chex.ArrayDevice,
    leader: chex.ArrayDevice,
    total_infl: chex.ArrayDevice,
    total_neighbors: chex.ArrayDevice,
    max_speed: float = 20,
    max_force: float = 40,
    eps: float = 10**-8,
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    """Cohesion and alignment calculations of `Rax`: Leader modified Reynolds flocking model in Jax.
        Calculates the cohesion steering based on local interactions with simple agents and with the leader using Reynold's Dynamics

    Args:
        - `X (chex.ArrayDevice)`: Swarm agents position matrix of shape: (n,2).
        - `X_dot (chex.ArrayDevice)`: Swarm agents velocity matrix of shape: (n,2).
        - `total_infl (chex.ArrayDevice)`: The total influence calculated for each agent
        - `total_neighbors (chex.ArrayDevice)`: The total number of neighbors for each agent.
        - `max_speed (float)`: Maximum speed of the agents. Defaults to 20.
        - `max_force (float)`: Maximum force of the agents. Defaults to 40.
        - `eps (float)`: Small epsilon to avoid division by zero. Defaults to 10**-8.

    Returns:
        - `mixed_steer (Tuple[chex.ArrayDevice, chex.ArrayDevice])`: The cohesion and alignment force exerted to each agent.
    """
    n_env, _, _ = X.shape

    T = total_infl / (total_neighbors[..., None] + eps)
    # Cohesion steer calculation
    coh = T[0] - X
    coh = max_speed * (coh / (la.norm(coh + eps, axis=-1)[..., None]))  # - X_dot

    # Alignment steer calculations
    alig = T[1]
    alig = max_speed * (alig / (la.norm(alig + eps, axis=-1)[..., None])) - X_dot

    coh = max_force * (coh / la.norm(coh + eps, axis=-1)[..., None])
    alig = max_force * (alig / la.norm(alig + eps, axis=-1)[..., None])

    # Leader whitening (i.e. cohesion and alignment of the leader are set to 0.)
    coh = coh.at[jnp.arange(n_env), leader, :].set(jnp.array([0, 0]))
    alig = alig.at[jnp.arange(n_env), leader, :].set(jnp.array([0, 0]))

    return coh, alig


def separation_steer(
    X: chex.Array,
    X_dot: chex.Array,
    corr_dist_mat: chex.Array,
    neighbors_matrix: chex.Array,
    max_speed: float = 20,
    max_force: float = 40,
    eps: float = 10**-8,
) -> chex.Array:
    """Separation calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X (chex.Array)`: _description_
        - `X_dot (chex.Array)`: _description_
        - `corr_dist_mat (chex.Array)`: _description_
        - `neighbors_matrix (chex.Array)`: _description_
        - `max_speed (float)`: Maximum speed of the agents. Defaults to 20.
        - `max_force (float)`: Maximum force of the agents. Defaults to 40.
        - `eps (float)`: Small epsilon to avoid division by zero. Defaults to 10**-8.

    Returns:
        - `chex.Array`: _description_
    """
    # TODO: We need a better seperation function.
    scaled_neighbors_matrix = jnp.nan_to_num(neighbors_matrix / corr_dist_mat)

    # Separation steer calculation
    sep = (
        jnp.einsum("ijk,ijl->ijl", scaled_neighbors_matrix, X)
        - scaled_neighbors_matrix @ X
    )
    sep = max_speed * (sep / (la.norm(sep + eps, axis=-1)[..., None])) - X_dot
    sep = max_force * (sep / (la.norm(sep + eps, axis=-1)[..., None]))

    return sep


def interaction_steer(
    X: chex.ArrayDevice,
    corr_dist_mat: chex.ArrayDevice,
    max_force: float = 300,
    eps: float = 10**-8,
) -> chex.ArrayDevice:
    """Separation calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `Diff (chex.ArrayDevice)`: _description_
        - `corr_dist_mat (chex.ArrayDevice)`: _description_
        - `max_force (float)`: Maximum force of the agents. Defaults to 300.
        - `eps (float)`: Small epsilon to avoid division by zero. Defaults to 10**-8.

    Returns:
        chex.ArrayDevice: _description_
    """
    n_env, _, _ = X.shape
    Diff = X - X[jnp.arange(n_env), None, -1]
    interaction = jnp.nan_to_num(Diff / (corr_dist_mat[..., -1][..., None]))
    return max_force * (interaction / (la.norm(interaction + eps, axis=-1)[..., None]))


@jit
def reynolds_dynamics(
    leader: chex.ArrayDevice, X: chex.ArrayDevice, X_dot: chex.ArrayDevice, params: EnvParams
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    """`Rax`: Leader modified Reynolds flocking model in Jax.

    Note: Currently the id of the leader is returned as in the future we could use this function
    for dynamic leader allocation based on some metric (e.x. agent with most neighbors).

    Args:
        - `leader (chex.ArrayDevice)`: The id number of the leader.
        - `X (chex.ArrayDevice)`: Swarm agents position matrix of shape: (n,2).
        - `X_dot (chex.ArrayDevice)`: Swarm agents velocity matrix of shape: (n,2).

    Returns:
        - `steer (chex.ArrayDevice)`: The steer vector of swarm agents shape=(n,2).
        - `leader (chex.ArrayDevice)`: The id number of the leader for every environment.
    """

    # Reading parameters from the params dictionary.
    agent_radius = params.scenario["agent_radius"]
    ldr_radius = params.scenario["ldr_radius"]
    prb_radius = params.scenario["prb_radius"]
    ldr_str = params.scenario["ldr_str"]
    w_c = params.scenario["w_c"]
    w_a = params.scenario["w_a"]
    w_s = params.scenario["w_s"]


    # The distance tensor, contains distance matrices for each separate environment. We calculate everything in parallel
    # for all environments.
    T_d = distance_tensor_jax(X)

    # The main diagonal of a distance matrix is 0 since d(i,i) is the distance of agent i from itself.
    # Therefore if we divide by the distances we would get 1/0. Adding ones matrix scaled down to eps
    # makes the algorithm stable. The identity matrix could also be used but there are cases where two agents
    # Occupy the same space because after sometime the agents reach the walls of the box and they do not bounce off of it
    # This cause NaNs.
    # In reality the agents need some form of real collision avoidance but this would make the simulation run slower.
    # eps = 10**-7
    # T_d_corr = T_d + eps*jnp.ones([n_env, n, n])

    # Calculate the neighborhood of the leader.
    T_ldr = leader_neighbors(T_d, leader, ldr_radius)

    # Calculate the neighborhood of every swarm agent.
    T_nbr, C_nbr = neighbors(T_d, agent_radius)

    # Calculate the neighborhood of the prober.
    T_prb = prober_neighbors(T_d, prb_radius)

    # For dynamic leader allocation. Later work.
    # leader = jnp.argmax(neighbors_count, axis=0)

    # Calculate the total number of neighbors for each agent.
    total_count = C_nbr + ldr_str * T_ldr

    # Influence from simple neighbors and leader
    total_infl = total_influence(X, X_dot, T_nbr, T_ldr, leader, ldr_str)

    # Cohesion and  Alignment steer calculation.
    cohesion, alignment = mixed_steer(X, X_dot, leader, total_infl, total_count)

    # Separation steer calculation.
    separation = separation_steer(X, X_dot, T_d, T_nbr)

    # Prober interaction.
    interaction = interaction_steer(X, T_d)
    # Performs neighbors masking.
    total_mask = total_count > 0
    #neighbors_mask = C_nbr > 0
    prober_mask = T_prb > 0

    steer = jnp.einsum("ijm,ij->ijm", (w_c * cohesion + w_a * alignment), total_mask)

    steer += jnp.einsum("ijm,ij->ijm", (w_s * separation), total_mask)

    # This sets the influences on the prober to 0.
    steer = steer.at[..., -1, :].set(jnp.array([0, 0]))

    steer = steer + jnp.einsum("ijm,ij->ijm", interaction, prober_mask)
    return steer


@jit
def swarm_leader(X, leader):
    n_env, _, _ = X.shape
    return X[jnp.arange(n_env), leader]

@jit
def closest_agent(X):
    n_env, _, _ = X.shape
    closest = la.norm(X - X[jnp.arange(n_env), None, -1], axis=-1)

    return closest

@jit
def swarm_center(X):
    n_env, _, _ = X.shape
    return jnp.mean(X[jnp.arange(n_env), 0:-2], axis=1)

@jit 
def reynolds_dynamics_rk4(state, params):
    dt = params.settings["dt"]
    
    S1 = reynolds_dynamics(
        state.leader,
        state.X,
        state.X_dot,
        params
    )
    S2 = reynolds_dynamics(
        state.leader,
        state.X + 0.5 * dt * state.X_dot,
        state.X_dot + dt / 2 * S1,
        params
    )
    S3 = reynolds_dynamics(
        state.leader,
        state.X + 0.5 * dt * state.X_dot,
        state.X_dot + 0.5 * dt * S2,
        params
    )
    S4 = reynolds_dynamics(
        state.leader,
        state.X + dt * state.X_dot,
        state.X_dot + dt * S3,
        params
    )
    
    return S1+2*(S2+S3)+S4

@jit
def script(state: EnvState, params: EnvParams, *args) -> Tuple[chex.ArrayDevice, int]:
    """Calculates the scripted action for the swarm agents.

    `Args`:
        - `state` (EnvState): State of the environment.

    `Returns`:
        - `steer` (chex.ArrayDevice): The steer vector of swarm agents shape=(n,2).
    """

    X = state.X
    leader = state.leader
    time = state.t[0]
    n_env, _, _ = X.shape
    ff = params.scenario["episode_size"] - 1

    steer = reynolds_dynamics_rk4(state, params)

    e_leader = state.curve.eval(time / ff) - X[jnp.arange(n_env), leader]
    u_leader = params.scenario["Kp_l"] * e_leader
    steer = steer.at[jnp.arange(n_env), leader].set(
        steer[jnp.arange(n_env), leader] + u_leader
    )

    e_prob = swarm_center(X) - X[jnp.arange(n_env), -1]
    u_prob = params.scenario["Kp_p"] * e_prob
    steer = steer.at[jnp.arange(n_env), -1].set(steer[jnp.arange(n_env), -1] + u_prob)
    

    return steer
