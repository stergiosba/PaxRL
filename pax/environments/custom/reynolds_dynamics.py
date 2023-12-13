import chex
import jax.numpy.linalg as la
from jax import jit, numpy as jnp, nn as jnn
from typing import Tuple, Dict, Callable
from jax.debug import print as dprint
from pax.environments.custom.prober import EnvState
from pax.core.environment import EnvParams
from functools import partial


@jit
def reynolds_dynamics(
    leader: chex.ArrayDevice,
    X: chex.ArrayDevice,
    X_dot: chex.ArrayDevice,
    B,
    time,
    params: EnvParams,
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

    def distance_tensor_jax() -> chex.ArrayDevice:
        """Parallelized calculation of euclidean distances between all pairs of swarm agents
            for all environment instances in one go.

        Args:
            `X (chex.Array)`: Swarm agents position tensor of shape: (n_envs,n_agents,2).

        Returns:
            `Distance_matrix (chex.Array)`: Distance tensor of shape: (n_envs,n_agents,n_agents)
        """
        return jnp.einsum("ijkl->ijk", (X[:, None, ...] - X[..., None, :]) ** 2)

    def leader_neighbors(leader_radius: float) -> chex.ArrayDevice:
        """Calculates the neighborhood for the leader agent as a set for every parallel environment.
        Args:
            - `leader_distances (chex.Array)`: Slice of distance tensor, distances between leader and each swarm agent for every
                parallel environment.
            - `leader_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

        Returns:
            - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
        """
        leader_distances = T_d[jnp.arange(n_env), leader, :]
        leader_neighbors_matrix = (leader_distances > 0) & (
            leader_distances <= (leader_radius) ** 2
        )

        return leader_neighbors_matrix

    def neighbors(agent_radius: float) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        """Calculates the neighborhood for each agent as a set and the cardinality of each set.
        Args:
            - `distance_tensor (chex.ArrayDevice)`: Distance tensor, euclidean distances between all pairs of swarm agents for all
                parallel environments.
            - `agent_radius (float)`: Influence radius for the simple agents. Same for all environments.

        Returns:
            - `neighbors_tensor (chex.ArrayDevice)`: Tensor of neighbors for every agent and every environment.
            - `neighbors_count (chex.ArrayDevice)`: Agent-wise sum of neighbors_tensor, i.e. cardinality of each neighborhood.
        """
        neighbors_tensor = (T_d > 0) & (T_d <= agent_radius ** 2)

        # Calculating all neighbors counts, Summing over the agents axis: 1.
        neighbors_count = jnp.sum(neighbors_tensor, axis=1)

        return neighbors_tensor, neighbors_count

    def prober_neighbors(prober_radius: float) -> chex.ArrayDevice:
        """Calculates the neighborhood for the leader agent as a set for every parallel environment.
        Args:
            - `leader_distances (chex.Array)`: Slice of distance tensor, distances between leader and each swarm agent for every
                parallel environment.
            - `prober_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

        Returns:
            - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
        """
        prober_distances = T_d[:, -1, :]
        prober_neighbors_matrix = (prober_distances > 0) & (
            prober_distances <= (prober_radius) ** 2
        )

        return prober_neighbors_matrix

    def total_influence():
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
        K = jnp.vstack([[X], [X_dot]])

        # Remove the leader from the normal neighborhood of each agent for the cohesion an alignment calculations
        # The leader has his own influence calculated as leader_inf.
        neighbors_tensor = T_nbr.at[jnp.arange(n_env), leader, :].set(False)

        neight_inf = neighbors_tensor @ K
        leader_inf = ldr_str * jnp.einsum(
            "ij,kil->kijl", T_ldr, K[:, jnp.arange(n_env), leader, :]
        )

        total_inf = neight_inf + leader_inf
        # total_inf = total_inf.at[..., -1, :].set(jnp.array(0))
        return total_inf

    def mixed_steer(
        max_speed: float = 20,
        max_force: float = 40,
        eps: float = 10 ** -8,
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        """Cohesion and alignment calculations for `Rax`: Leader modified Reynolds flocking model in Jax.
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

        T = total_infl / (total_neighbors[..., None] + eps)

        # Cohesion steer calculation
        coh = T[0] - X
        coh = max_speed * (coh / (la.norm(coh + eps, axis=-1)[..., None])) - X_dot
        coh = max_force * (coh / la.norm(coh + eps, axis=-1)[..., None])

        # Alignment steer calculations
        alig = T[1]
        alig = max_speed * (alig / (la.norm(alig + eps, axis=-1)[..., None])) - X_dot
        alig = max_force * (alig / la.norm(alig + eps, axis=-1)[..., None])

        # Leader whitening (i.e. cohesion and alignment of the leader are set to 0.)
        coh = coh.at[jnp.arange(n_env), leader, :].set(jnp.array([0, 0]))
        alig = alig.at[jnp.arange(n_env), leader, :].set(jnp.array([0, 0]))

        return coh, alig

    def separation_steer(
        max_speed: float = 20,
        max_force: float = 40,
        eps: float = 10 ** -8,
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
        scaled_neighbors_matrix = jnp.nan_to_num(T_nbr / T_d)

        # Separation steer calculation
        sep = (
            jnp.einsum("ijk,ijl->ijl", scaled_neighbors_matrix, X)
            - scaled_neighbors_matrix @ X
        )
        sep = max_speed * (sep / (la.norm(sep + eps, axis=-1)[..., None])) - X_dot
        sep = max_force * (sep / (la.norm(sep + eps, axis=-1)[..., None]))

        return sep

    def interaction_steer(
        max_force: float = 60, eps: float = 10 ** -8
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

        Diff = X - X[jnp.arange(n_env), None, -1]
        interaction = jnp.nan_to_num(Diff / (T_d[..., -1][..., None]))
        return max_force * (
            interaction / (la.norm(interaction + eps, axis=-1)[..., None])
        )

    # Reading parameters from the params dictionary.
    agent_radius = params.scenario["agent_radius"]
    ldr_radius = params.scenario["ldr_radius"]
    prb_radius = params.scenario["prb_radius"]
    ldr_str = params.scenario["ldr_str"]
    w_c = params.scenario["w_c"]
    w_a = params.scenario["w_a"]
    w_s = params.scenario["w_s"]
    n_env, _, _ = X.shape

    # The distance tensor, contains distance matrices for each separate environment. We calculate everything in parallel
    # for all environments.
    T_d = distance_tensor_jax()

    # The main diagonal of a distance matrix is 0 since d(i,i) is the distance of agent i from itself.
    # Therefore if we divide by the distances we would get 1/0. Adding ones matrix scaled down to eps
    # makes the algorithm stable. The identity matrix could also be used but there are cases where two agents
    # Occupy the same space because after sometime the agents reach the walls of the box and they do not bounce off of it
    # This cause NaNs.
    # In reality the agents need some form of real collision avoidance but this would make the simulation run slower.
    # eps = 10**-7
    # T_d_corr = T_d + eps*jnp.ones([n_env, n, n])

    # Calculate the neighborhood of the leader.
    T_ldr = leader_neighbors(ldr_radius)

    # Calculate he neighborhood of every swarm agent.
    T_nbr, C_nbr = neighbors(agent_radius)

    # Calculate the neighborhood of the prober.
    T_prb = prober_neighbors(prb_radius)

    # For dynamic leader allocation. Later work.
    # leader = jnp.argmax(neighbors_count, axis=0)

    # Calculate the total number of neighbors for each agent.
    total_neighbors = C_nbr + ldr_str * T_ldr

    # Influence from simple neighbors and leader
    total_infl = total_influence()

    # Cohesion and  Alignment steer calculation.
    cohesion, alignment = mixed_steer()
    # Separation steer calculation.
    separation = separation_steer()

    # Prober interaction.
    interaction = interaction_steer()

    # Performs neighbors masking.
    total_mask = total_neighbors > 0
    prober_mask = T_prb > 0
    B += prober_mask[:, :-1]

    steer = jnp.einsum(
        "ijm,ij->ijm", (w_c * cohesion + w_a * alignment + w_s * separation), total_mask
    )

    # This sets the influences on the prober to 0.
    steer = steer.at[..., -1, :].set(jnp.array([0, 0]))

    steer += jnp.einsum("ijm,ij->ijm", interaction, prober_mask)
    return steer, (B)


@partial(jit, static_argnums=(0))
def rk4_integration(dynamics_fun: Callable, state, params):
    dt = params.settings["dt"]

    s1, extra_out = dynamics_fun(
        state.leader, state.X, state.X_dot, state.B, state.time[0], params
    )
    s2, extra_out = dynamics_fun(
        state.leader,
        state.X + 0.5 * dt * state.X_dot,
        state.X_dot + dt / 2 * s1,
        state.B,
        state.time[0],
        params,
    )
    s3, extra_out = dynamics_fun(
        state.leader,
        state.X + 0.5 * dt * state.X_dot,
        state.X_dot + 0.5 * dt * s2,
        state.B,
        state.time[0],
        params,
    )
    s4, extra_out = dynamics_fun(
        state.leader,
        state.X + dt * state.X_dot,
        state.X_dot + dt * s3,
        state.B,
        state.time[0],
        params,
    )

    s = s1 + 2 * (s2 + s3) + s4
    return s, extra_out


@jit
def scripted_act(
    state: EnvState, params: EnvParams, *args
) -> Tuple[chex.ArrayDevice, int]:
    """Calculates the scripted action for the swarm members.

    `Args`:
        - `state` (EnvState): State of the environment.

    `Returns`:
        - `steer` (chex.ArrayDevice): The steer vector of swarm members shape=(n_env, n, 2).
    """

    @jit
    def closest_member() -> chex.ArrayDevice:
        """Tracks the closest member of the swarm to the prober.

        Returns:
            -`Position (chex.ArrayDevice)`: The position of the closest member of the swarm to the prober.
        """
        closest = la.norm(X - X[jnp.arange(n_env), None, -1], axis=-1)[:, :-1]

        return X[jnp.arange(n_env), jnp.argmin(closest, axis=1)]

    @jit
    def specific_member(memb_id: int) -> chex.ArrayDevice:
        """Tracks a specific member of the swarm in every environment.

        Args:
            memb_id (int): The ID of the member to track.

        Returns:
            - `Position (chex.ArrayDevice)`: The position of the tracked member in every environment.
        """
        return X[jnp.arange(n_env), memb_id]

    @jit
    def swarm_center() -> chex.ArrayDevice:
        """Tracks the center (average) of the swarm.

        Returns:
            - `Position (chex.ArrayDevice)`: The average position of the swarm members.
        """
        return jnp.mean(X[jnp.arange(n_env), 0:-1], axis=1)

    @jit
    def swarm_leader() -> chex.ArrayDevice:
        """Tracks the leader of the swarm in every environment. The leader can and will be different for every environment.

        Returns:
            `Position (chex.ArrayDevice)` : The position of the leader in every environment.
        """
        return X[jnp.arange(n_env), leader]

    X = state.X
    leader = state.leader
    time = state.time[0]
    # dprint("{x}", x=X.shape)
    n_env, _, _ = X.shape

    # TODO: Actually fix this so that the leader goes to the points that it should but no faster pace.
    ff = 0.66 * (params.scenario["episode_size"] - 1)

    steer, extra_out = rk4_integration(reynolds_dynamics, state, params)
    # dprint("{x}", x=steer)
    e_leader = state.curve.eval(time / ff) - X[jnp.arange(n_env), leader]
    u_leader = params.scenario["Kp_l"] * e_leader
    steer = steer.at[jnp.arange(n_env), leader].add(u_leader)

    e_prob = swarm_leader() - X[jnp.arange(n_env), -1]
    u_prob = params.scenario["Kp_p"] * e_prob
    steer = steer.at[jnp.arange(n_env), -1].add(u_prob)

    return steer, extra_out
