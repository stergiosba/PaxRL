#%%
import chex
import jax.numpy.linalg as la
from jax import jit, numpy as jnp, nn as jnn
from typing import Tuple, Dict, Callable
from jax.debug import print as dprint
from pax.core.environment import EnvParams
from functools import partial
from pax.core.state import EnvState


@jit
def reynolds_dynamics(leader, X, X_dot, B, time, params):
    def interaction_steer(T_d,
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

        Diff = X - X[-1]
        interaction = jnp.nan_to_num(Diff / (T_d[-1][:, None])**0.5)

        return max_force * (
            interaction / (la.norm(interaction + eps, axis=-1)[:, None])
        )

    
    
    def swarm_reynolds(
        leader: chex.ArrayDevice,
        X: chex.ArrayDevice,
        X_dot: chex.ArrayDevice,
        time,
        params,
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        """`Rax`: Leader modified Reynolds flocking model in Jax.

        Note: Currently we do not return the ID of the leader but in the future we could use this function
        for dynamic leader allocation based on some metric (i.e. agent with most neighbors).

        Args:
            - `leader (chex.ArrayDevice)`: The id number of the leader.
            - `X (chex.ArrayDevice)`: Swarm agents position matrix of shape: (n,2).
            - `X_dot (chex.ArrayDevice)`: Swarm agents velocity matrix of shape: (n,2).

        Returns:
            - `steer (chex.ArrayDevice)`: The steer vector of swarm agents shape=(n,2).
            - `leader (chex.ArrayDevice)`: The id number of the leader for every environment.
        """

        def distance_matrix_jax() -> chex.ArrayDevice:
            """Parallelized calculation of euclidean distances between all pairs of swarm agents
                for all environment instances in one go.

            Args:
                `X (chex.Array)`: Swarm agents position matrix of shape: (n_agents,2).

            Returns:
                `Distance_matrix (chex.Array)`: Distance matrix of shape: (n_agents,n_agents)
            """
            return jnp.einsum("ijk->ij", (X[:, None] - X[None, :]) ** 2)

        def leader_neighbors(leader_radius: float) -> chex.ArrayDevice:
            """Calculates the neighborhood for the leader agent as a set for every parallel environment.
            Args:
                - `leader_distances (chex.Array)`: Slice of distance matrix, distances between leader and each swarm agent for every
                    parallel environment.
                - `leader_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

            Returns:
                - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
            """
            leader_distances = T_d[leader]
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
            neighbors_matrix = (T_d > 0) & (T_d <= 20 ** 2)
            neighbors_matrix = neighbors_matrix.at[:,-1].set(False)

            masked_leader_neighbors_matrix = neighbors_matrix.at[:, leader].set(False)

            # Calculating all neighbors counts, Summing over the agents axis: 1.
            neighbors_count = jnp.sum(neighbors_matrix, axis=1)
            masked_leader_neighbors_count = jnp.sum(masked_leader_neighbors_matrix, axis=1)

            collision_matrix = (T_d > 0) & (T_d <= 40 ** 2)

            return neighbors_matrix, neighbors_count, masked_leader_neighbors_matrix, masked_leader_neighbors_count, collision_matrix

        def prober_neighbors(prober_radius: float) -> chex.ArrayDevice:
            """Calculates the neighborhood for the leader agent as a set for every parallel environment.
            Args:
                - `leader_distances (chex.Array)`: Slice of distance tensor, distances between leader and each swarm agent for every
                    parallel environment.
                - `prober_radius (float)`: Influence radius for the leader agent (typically a scaled multiple of simple agent's radius)

            Returns:
                - `leader_neighbors_matrix (chex.Array)`: Boolean matrix of leader's neighbors for each environment.
            """
            prober_distances = T_d[-1]
            prober_neighbors_matrix = (prober_distances > 0) & (
                prober_distances <= (prober_radius) ** 2
            )

            return prober_neighbors_matrix

        def total_influence():
            """Calculates the influence of simple agents and then of the leader agent.

            Args:
                - `X (chex.ArrayDevice)`: Swarm agents position matrix of shape: (n_agents, 2).
                - `X_dot (chex.ArrayDevice)`: Swarm agents velocity matrix of shape: (n_agents, 2).
                - `neighbors_matrix (chex.ArrayDevice)`: Array of neighbors for every agent.
                - `leader_neighbors_matrix (chex.ArrayDevice): The matrix of leader neighbors.
                - `leader (chex.ArrayDevice): The ID of the leader

            Returns:
                - `chex.ArrayDevice`: _description_
            """
            K = jnp.concat([X[None,:],X_dot[None,:]])

            neight_inf = T_nbr_noleader @ K
            leader_inf = ldr_str * T_ldr[:, None] * (K[:, leader][:, None])

            total_inf = neight_inf + leader_inf

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

            T = total_infl / (total_neighbors[:, None] + eps)
            # Cohesion steer calculation
            coh = T[0] - X
            coh = max_speed * (coh / (la.norm(coh + eps, axis=-1)[:, None])) #- X_dot
            coh = max_force * (coh / la.norm(coh + eps, axis=-1)[:, None])

            # Alignment steer calculations
            alig = T[1]
            alig = max_speed * (alig / (la.norm(alig + eps, axis=-1)[:, None])) #- X_dot
            alig = max_force * (alig / la.norm(alig + eps, axis=-1)[:, None])

            # Leader whitening (i.e. cohesion and alignment of the leader are set to 0.)
            # coh = coh.at[leader, :].set(jnp.array([0, 0]))
            # alig = alig.at[leader, :].set(jnp.array([0, 0]))

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
            scaled_neighbors_matrix = jnp.nan_to_num(collision_matrix / T_d)

            # Separation steer calculation
            sep = (
                scaled_neighbors_matrix.sum(axis=1)[:,None]*X - scaled_neighbors_matrix @ X
            )
            sep = max_speed * (sep / (la.norm(sep + eps, axis=-1)[:, None])) #- X_dot
            sep = max_force * (sep / (la.norm(sep + eps, axis=-1)[:, None]))

            return sep

        # Reading parameters from the params dictionary.
        agent_radius = params.scenario["agent_radius"]
        ldr_radius = params.scenario["ldr_radius"]
        prb_radius = params.scenario["prb_radius"]
        ldr_str = params.scenario["ldr_str"]
        w_c = params.scenario["w_c"]
        w_a = params.scenario["w_a"]
        w_s = params.scenario["w_s"]
        max_speed = params.scenario["max_speed"]
        max_force = params.scenario["max_force"]


        # The distance tensor, contains distance matrices for each separate environment. We calculate everything in parallel
        # for all environments.
        T_d = distance_matrix_jax()

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
        T_nbr, C_nbr, T_nbr_noleader, C_nbr_noleader, collision_matrix = neighbors(agent_radius)

        # Calculate the neighborhood of the prober.
        T_prb = prober_neighbors(prb_radius)

        # Calculate the total number of neighbors for each agent.
        total_neighbors = C_nbr_noleader + ldr_str * T_ldr

        # Influence from simple neighbors and leader
        total_infl = total_influence()

        # Cohesion and  Alignment steer calculation.
        cohesion, alignment = mixed_steer(max_speed, max_force)

        # Separation steer calculation.
        separation = separation_steer(max_speed, max_force)

        steer = w_c * cohesion + w_a * alignment + w_s * separation

        steer = steer.at[leader].set(jnp.array([0, 0]))

        return steer, T_d, T_prb

        
    steer, T_d, T_prb = swarm_reynolds(leader, X, X_dot, time, params)

    # Prober interaction.
    prb_str = params.scenario["prb_str"]
    interaction = interaction_steer(T_d, prb_str)
    prober_mask = T_prb > 0
    B += prober_mask[:-1]
    
    # This sets the influences on the prober to 0.
    steer = steer.at[-1, :].set(jnp.array([0, 0]))

    steer += jnp.einsum("ij,i->ij", interaction, prober_mask)
    
    return steer, B

@partial(jit, static_argnums=(0))
def rk4_integration(dynamics_fun: Callable, state, params):
    dt = params.settings["dt"]

    s1, extra_out = dynamics_fun(
        state.leader, state.X, state.X_dot, state.interactions, state.timestep, params
    )
    s2, extra_out = dynamics_fun(
        state.leader,
        state.X + 0.5 * dt * state.X_dot,
        state.X_dot + dt / 2 * s1,
        state.interactions,
        state.timestep + 0.5 * dt,
        params,
    )
    s3, extra_out = dynamics_fun(
        state.leader,
        state.X + 0.5 * dt * state.X_dot,
        state.X_dot + 0.5 * dt * s2,
        state.interactions,
        state.timestep + 0.5 * dt,
        params,
    )
    s4, extra_out = dynamics_fun(
        state.leader,
        state.X + dt * state.X_dot,
        state.X_dot + dt * s3,
        state.interactions,
        state.timestep+ dt,
        params,
    )

    s = s1 + 2 * (s2 + s3) + s4
    return s, extra_out


@jit
def scripted_act(
    state, params, *args
) -> Tuple[chex.ArrayDevice, int]:
    """Calculates the scripted action for the swarm members.

    `Args`:
        - `state` (EnvState): State of the environment.

    `Returns`:
        - `steer` (chex.ArrayDevice): The steer vector of swarm members shape=(n_env, n, 2).
    """


    X = state.X
    leader = state.leader
    time = state.timestep

    T = (params.scenario["episode_size"] - 1)

    steer, extra_out = rk4_integration(reynolds_dynamics, state, params)

    e_leader = state.curve.eval(time) - X[leader]
    new_integral = state.integral + e_leader

    u_leader = params.scenario["Kp_l"] * e_leader + params.scenario["Ki_l"] * new_integral * params.settings["dt"]
    u_leader = jnp.clip(u_leader, -params.scenario["max_force"], params.scenario["max_force"])
    steer = steer.at[leader].add(u_leader)

    extra_out = (extra_out, new_integral)


    # e_prob = X[leader] - X[-1]
    # u_prob = 2 * e_prob
    # steer = steer.at[-1].add(u_prob)
    # dprint("{x}", x = u_leader)

    # steer = jnp.clip(steer, -1*params.scenario["max_force"], 1*params.scenario["max_force"])
    return steer, extra_out
