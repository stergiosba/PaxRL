import chex
import jax.numpy.linalg as la
from jax import jit, numpy as jnp, nn as jnn
from typing import Tuple
from jax.debug import print as dprint
from pax.core.state import EnvState


@jit
def distance_tensor_jax(X: chex.ArrayDevice) -> chex.ArrayDevice:
    """Parallelized calculation of euclidean distances between all pairs of swarm agents
        for all environment instances in one go.

    Args:
        `X (chex.Array)`: Swarm agents position tensor of shape: (n_envs,n_agents,2).

    Returns:
        `Distance_matrix (chex.Array)`: Distance tensor of shape: (n_envs,n_agents,n_agents)
    """
    diff_tensor = X[:, None, ...] - X[..., None, :]
    return jnp.einsum("ijkl->ijk", diff_tensor**2)


@jit
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


@jit
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
    prober_distances = distance_tensor[..., -1]
    prober_neighbors_matrix = (prober_distances > 0) & (
        prober_distances <= (prober_radius) ** 2
    )

    return prober_neighbors_matrix


@jit
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


@jit
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


@jit
def mixed_steer(
    X: chex.ArrayDevice,
    X_dot: chex.ArrayDevice,
    leader: chex.ArrayDevice,
    total_infl: chex.ArrayDevice,
    total_neighbors: chex.ArrayDevice,
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    """Cohesion and alignment calculations of `Rax`: Leader modified Reynolds flocking model in Jax.
        Calculates the cohesion steering based on local interactions with simple agents and with the leader using Reynold's Dynamics

    Args:
        - `X (chex.ArrayDevice)`: Swarm agents position matrix of shape: (n,2).
        - `X_dot (chex.ArrayDevice)`: Swarm agents velocity matrix of shape: (n,2).
        - `total_infl (chex.ArrayDevice)`: The total influence calculated for each agent
        - `total_neighbors (chex.ArrayDevice)`: The total number of neighbors for each agent.

    Returns:
        - `mixed_steer (Tuple[chex.ArrayDevice, chex.ArrayDevice])`: The cohesion and alignment force exerted to each agent.
    """

    # Small epsilon to avoid division by zero
    eps = 10**-8
    n_env, _, _ = X.shape

    T = total_infl / (total_neighbors[..., None] + eps)
    max_speed = 20
    # Cohesion steer calculation
    coh = T[0] - X
    coh = max_speed * (coh / (la.norm(coh + eps, axis=-1)[..., None])) - X_dot

    # Alignment steer calculations
    alig = T[1]
    alig = max_speed * (alig / (la.norm(alig + eps, axis=-1)[..., None])) - X_dot

    max_force = 40
    coh = max_force * (coh / la.norm(coh + eps, axis=-1)[..., None])
    alig = max_force * (alig / la.norm(alig + eps, axis=-1)[..., None])

    # Leader whitening (i.e. cohesion and alignment of the leader are set to 0.)
    coh = coh.at[jnp.arange(n_env), leader, :].set(jnp.array([0, 0]))
    alig = alig.at[jnp.arange(n_env), leader, :].set(jnp.array([0, 0]))

    return coh, alig


@jit
def separation_steer(
    X: chex.Array,
    X_dot: chex.Array,
    corr_dist_mat: chex.Array,
    neighbors_matrix: chex.Array,
) -> chex.Array:
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
    eps = 10**-8

    scaled_neighbors_matrix = jnp.nan_to_num(neighbors_matrix / corr_dist_mat)

    # Separation steer calculation
    max_speed = 20
    sep = (
        jnp.einsum("ijk,ijl->ijl", scaled_neighbors_matrix, X)
        - scaled_neighbors_matrix @ X
    )
    sep = max_speed * (sep / (la.norm(sep + eps, axis=-1)[..., None])) - X_dot

    max_force = 40
    sep = max_force * (sep / (la.norm(sep + eps, axis=-1)[..., None]))

    return sep


@jit
def interaction_steer(
    Diff: chex.ArrayDevice, corr_dist_mat: chex.ArrayDevice
) -> chex.ArrayDevice:
    """Separation calculations of `Rax`: Leader modified Reynolds flocking model in Jax.

    Args:
        - `X (chex.Array)`: _description_
        - `X_dot (chex.Array)`: _description_
        - `corr_dist_mat (chex.ArrayDevice)`: _description_
        - `neighbors_matrix (chex.ArrayDevice)`: _description_

    Returns:
        - `interaction_steer (chex.ArrayDevice)`: _description_
    """

    # Small epsilon to avoid division by zero
    eps = 10**-8

    max_force = 500

    interaction = jnp.nan_to_num(Diff / (corr_dist_mat[..., -1][..., None]))

    return max_force * (interaction / (la.norm(interaction + eps, axis=-1)[..., None]))


@jit
def reynolds_jax(
    leader: chex.ArrayDevice, X: chex.ArrayDevice, X_dot: chex.ArrayDevice
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

    # The distance tensor, contains distance matrices for each separate environment. We calculate everything in parallel
    # for all environments.
    T_d = distance_tensor_jax(X)

    # The main diagonal of a distance matrix is 0 since d(i,i) is the distance of agent i from itself.
    # Therefore if we divide by the distances we would get 1/0. Adding ones matrix scaled down to eps
    # makes the algorithm stable. The identity matrix could also be used but there are cases where two agents
    # Occupy the same space because after sometime the agents reach the walls of the box and they do not bounce off of it
    # This cause NaNs.
    # In reality the agents need some form of real collision avoidance but this would make the simulation run slower.
    # n_env, n, _ = X.shape
    # eps = 10**-7
    # T_d_corr = T_d + eps*jnp.ones([n_env, n, n])

    agent_radius = 30
    ldr_radius = 4 * agent_radius
    prb_radius = 3 * agent_radius

    # Leader strength
    ldr_str = 50

    # Calculate the neighborhood of the leader.
    T_ldr = leader_neighbors(T_d, leader, ldr_radius)

    # Calculate the neighborhood of every swarm agent.
    T_nbr, C_nbr = neighbors(T_d, agent_radius)

    # Calculate the neighborhood of the prober.
    T_prb = prober_neighbors(T_d[-1], prb_radius)

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
    interaction = interaction_steer(X - X[:, -1], T_d)

    # Performs neighbors masking.
    total_mask = total_count > 0
    neighbors_mask = C_nbr > 0
    prober_mask = (T_prb > 0)[None, :]
    w_c = 0
    w_a = 0
    w_s = 1

    steer = jnp.einsum(
        "ijm,ij->ijm", (w_c * cohesion + w_a * alignment), total_mask
    ) + jnp.einsum("ijm,ij->ijm", (w_s * separation), total_mask)

    # This sets the influences on the prober to 0.
    #steer = steer.at[..., -1, :].set(jnp.array([0, 0]))

    #steer = steer + jnp.einsum("ijm,ij->ijm", interaction, prober_mask)
    return steer

@jit
def swarm_center(X):
    return jnp.mean(X, axis=1)


@jit
def script(state: EnvState, *args) -> Tuple[chex.ArrayDevice, int]:
    """Calculates the scripted action for the swarm agents.

    `Args`:
        - `state` (EnvState): State of the environment.

    `Returns`:
        - `steer` (chex.ArrayDevice): The steer vector of swarm agents shape=(n,2).
    """
    dt = 1/30
    S1 = reynolds_jax(state.leader, state.X, state.X_dot)
    S2 = reynolds_jax(state.leader, state.X+dt/2*state.X_dot, state.X_dot+dt/2*S1)
    S3 = reynolds_jax(state.leader, state.X+dt/2*state.X_dot, state.X_dot+dt/2*S2)
    S4 = reynolds_jax(state.leader, state.X+dt*state.X_dot, state.X_dot+dt*S3)
    n_env, _, _ = state.X.shape
    ff = 2000 - 1

    # dprint("s=  {x}",x=S[jnp.arange(n_env), state.leader])
    e = state.curve.eval(state.t[0] / ff) - state.X[jnp.arange(n_env), state.leader]

    e_com = swarm_center(state.X) - state.X[jnp.arange(n_env), -1]

    # e = state.curve.eval(1) - state.X[jnp.arange(n_env), state.leader]
    Kp = 0
    u = Kp * e
    up = 5 * e_com
    # dprint("u= {x}",x=u)
    S = S1+2*(S2+S3)+S4
    S = S.at[jnp.arange(n_env), state.leader].set(
        S[jnp.arange(n_env), state.leader] + u
    )

    S = S.at[jnp.arange(n_env), -1].set(
        S[jnp.arange(n_env), -1] + up
    )
    # dprint("s+u= {x}",x=S[jnp.arange(n_env), state.leader])
    return S
