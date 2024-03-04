import chex
import equinox as eqx
import numpy as np
import jax.numpy.linalg as la
from jax import jit, vmap
from jax import lax, jit, nn as jnn, numpy as jnp, random as jrandom
from pax.utils.bezier import BezierCurve3
from pax.core.environment import Environment, EnvParams, EnvRewards
from pax.core.spaces import SeparateGrid, Box
from typing import Any, Dict, Sequence, Tuple, Union, Callable
from .reynolds_dynamics import scripted_act
from jax.debug import print as dprint, breakpoint as brk


@jit
def r_max_interaction(B, leader):
    P = jnn.softmax(B)
    return len(B) * P[leader] - 1


@jit
def r_leader(state, action):
    return la.norm(state.X[-1] - state.X[state.leader]) / (800 * jnp.sqrt(2))


@jit
def r_target(state):
    return -la.norm(state.curve.eval(1.0) - state.X[-1]) / (800 * jnp.sqrt(2))


@jit
def r_angle(X_dot_prev, X_dot):
    # calculate the angle of the velocity before and after the step
    diff = jnp.abs(X_dot - X_dot_prev)
    angle_prev = jnp.arctan2(diff[:, 0], diff[:, 1])
    # angle = jnp.arctan2(X_dot[1], X_dot[0])
    # calculate the difference of the angles
    # angle_diff = jnp.abs(angle_prev - angle)
    # calculate the reward based on the difference of the angles
    return jnp.sum(angle_prev)


@jit
def r_acceleration(state, action):
    return la.norm(action**2)


@jit
def r_leader_simple(state, action):
    return la.norm(state.X[-1] - state.X[state.leader]) < 25


class EnvState(eqx.Module):
    """The environment state (Multiple Agents)

    `Args`:
        - `X (chex.Array)`: Position of every Agents.
        - `X_dot (chex.Array)`: Velocity of every Agent.
        - `interactions (chex.Array)`: The interaction of every agent with the scripted entities.
        - `leader (int)`: The id of the leader agent.
        - `goal (chex.Array)`: The location of the goal.
    """

    X: chex.ArrayDevice
    X_dot: chex.ArrayDevice
    interactions: chex.ArrayDevice
    leader: chex.ArrayDevice
    curve: BezierCurve3
    time: int

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


class Prober(Environment):
    """The main `Prober` environment class.
        Reynolds swarm dynamics with probing interactions.
        The environment can be partially or fully observable.

    `Attributes`:
        - `n_agents (int)`: The total number of agents (actors) in the environment.
        - `n_scripted (int)`: The total number of scripted entities in the environment.
        - `action_space (Union[Discrete, Box])`: The action space allowed to the agents.
        - `observation_space (Union[Discrete, Box])`: The observation space of the environment.

    """

    n_agents: int
    n_scripted: int
    action_space: SeparateGrid
    observation_space: Box
    map_action: Callable

    @property
    def version(self):
        return "0.3"

    def __init__(self, params: Dict):
        """
        Args:
           `params (EnvParams)`: Parameters given to environment from the TOML file.
        """
        self.params = EnvParams(**params)
        self.n_agents = self.params.scenario["n_agents"]
        self.n_scripted = self.params.scenario["n_scripted_entities"]

        action_range = self.params.action_space.values()
        self.action_space = SeparateGrid(action_range)
        # self.action_space = Grid(action_range)
        self.map_action = self.action_space.map_action

        obs_dtype = jnp.float32
        obs_lower, obs_upper = self.params.observation_space["limits"]

        if self.params.observation_space["use_previous_state"]:
            obs_shape = (2 * (self.n_agents + self.n_scripted), 2)
        else:
            obs_shape = (self.n_agents + self.n_scripted, 2)

        self.observation_space = Box(obs_lower, obs_upper, obs_shape, obs_dtype)

        self.rewards = EnvRewards()
        self.rewards.register(r_leader_simple, 1.0)
        # self.rewards.register(r_acceleration, -0.000005)

    @eqx.filter_jit
    def get_obs(
        self,
        prev_state: EnvState,
        action: chex.ArrayDevice = None,
        new_state: EnvState = None,
    ) -> Sequence[chex.ArrayDevice]:
        """Applies observation function to state.

        `Args`:
            - `state (EnvState)`: The full state of the environment.

        Returns:
            - `observation (chex.Array)`: The observable part of the state.
        """

        return new_state.X

    @eqx.filter_jit
    def reset_env(
        self, key: chex.PRNGKey
    ) -> Tuple[Sequence[chex.ArrayDevice], EnvState]:
        """Resets the environment.

        Args:
            -`key (chex.PRNGKey)`: The random key for the reset.

        Returns:
            - `Observations (chex.Array)`: The initial observation of the environment based on the observation function get_obs
            - `State (EnvState)`: The initial full state of the environment. Used internally.
        """

        # x_swarm, y_swarm = jrandom.uniform(
        #     key, minval=jnp.array([50, 50]), maxval=jnp.array([750, 750]), shape=(2,)
        # )
        # th = jnp.arange(0, self.n_scripted) * 2 * jnp.pi / (self.n_scripted)
        # init_X_scripted = jnp.vstack(
        #     [x_swarm + 30 * jnp.cos(th), y_swarm + 30 * jnp.sin(th)]
        # ).T


        init_X_scripted = jrandom.uniform(
            key,
            minval=jnp.array([75, 75]),
            maxval=jnp.array([125, 125]),
            shape=(self.n_scripted, 2),
        )

        # init_X_dot_scripted = 5 * jrandom.uniform(
        #     key, minval=-0, maxval=0, shape=(self.n_scripted, 2)
        # )
        init_X_dot_scripted = jnp.zeros((self.n_scripted, 2))

        # init_X_agents = jnp.array([[100, 600]])
        # + jrandom.cauchy(
        #     key, shape=(self.n_agents, 2)
        # )
        init_X_agents = jrandom.uniform(
            key,
            minval=jnp.array([475, 475]),
            maxval=jnp.array([525, 525]),
            shape=(self.n_agents, 2),
        )

        init_X_dot_agents = (
            jrandom.uniform(key, minval=-1, maxval=1, shape=(self.n_agents, 2))
            * 10
            * jnp.sqrt(2)
        )

        leader = jrandom.randint(key, shape=(), minval=0, maxval=self.n_scripted)
        # leader = 3

        # final_goal = jrandom.uniform(
        #     key, minval=jnp.array([100, 100]), maxval=jnp.array([700, 700]), shape=(2,)
        # )

        final_goal = jrandom.uniform(
            key, minval=jnp.array([650, 50]), maxval=jnp.array([700, 125]), shape=(2,)
        )
        init_leader = init_X_scripted[leader]

        p0 = (4 * init_leader + 1 * final_goal) / 5

        P = jnp.array(
            [
                p0,
                (2 * p0 + final_goal) / 3,
                (p0 + 2 * final_goal) / 3,
                final_goal,
            ]
        )

        leader_path_curve = BezierCurve3(P)

        init_state = EnvState(
            X=jnp.concatenate([init_X_scripted, init_X_agents]),
            X_dot=jnp.concatenate([init_X_dot_scripted, init_X_dot_agents]),
            interactions=jnp.zeros(shape=self.n_scripted),
            leader=leader,
            curve=leader_path_curve,
            time=0,
        )
        return (self.get_obs(init_state, None, init_state), init_state)  # type: ignore

    @eqx.filter_jit
    def step_env(
        self, key, state: EnvState, action: chex.ArrayDevice
    ) -> Tuple[
        Sequence[chex.ArrayDevice], EnvState, chex.ArrayDevice, chex.ArrayDevice
    ]:
        """Performs one step in the environment

        `Args`:
            - `key (chex.PRNGKey)`: A jax random key for the step. Currently unused (present for consistency)
            - `state (EnvState)`: State of the environment
            - `action (chex.Array)`: The joint action fro the agents and scripted entities.

        Returns:
            - `environment_step Tuple[Sequence[chex.Array], EnvState, chex.Array, chex.Array])`: A step in the environment.
        """
        # Saving the previous state.
        prev_state = state
        total_action = (
            10
            * jnp.sqrt(2)
            * jnp.array(
                [
                    jnp.ones(
                        [
                            self.n_agents + self.n_scripted,
                        ]
                    ),
                    -jnp.ones(
                        [
                            self.n_agents + self.n_scripted,
                        ]
                    ),
                ]
            ).T
        )

        # brk()
        action = self.map_action(action)
        # action, extra_out = scripted_act(state, self.env.params)

        total_action = total_action.at[-1, :].set(action)
        # scripted_action = scripted_act(state, self.params)
        acc = total_action
        new_interactions = jnp.zeros(shape=self.n_scripted)
        dt = self.params.settings["dt"]
        # Applying the action to the state and getting the new state. Clipping the velocity to a maximum of 10 * sqrt(2) per axis.
        X_dot = jnp.clip(
            state.X_dot + dt / 6 * acc, a_min=-10 * jnp.sqrt(2), a_max=10 * jnp.sqrt(2)
        )

        X = state.X + dt * X_dot
        X = jnp.clip(
            X, a_min=self.observation_space.low, a_max=self.observation_space.high
        )

        new_state = EnvState(X, X_dot, new_interactions, state.leader, state.curve, state.time + 1)  # type: ignore

        # Using the transition from the previous state (s) to the new state (s') using action (a).
        # s, a -> s' : And then obtaining the observations and rewards.
        obs = self.get_obs(prev_state, action, new_state)
        reward = self.rewards.total(prev_state, action, new_state)
        done = self.is_terminal(new_state)

        return (obs, new_state, jnp.array([reward]), done)

    def is_terminal(self, state):
        # The state.curve(1.0) is the final goal
        norm_e = la.norm(state.curve.eval(1.0) - state.X[state.leader])

        done_1 = lax.select(norm_e < 50, jnp.array([1.0]), jnp.array([0.0]))

        done_2 = state.time == self.params.scenario["episode_size"]

        done_3 = lax.select(
            jnp.any(state.X[-1] <= self.observation_space.low),
            jnp.array([1.0]),
            jnp.array([0.0]),
        )
        done_4 = lax.select(
            jnp.any(state.X[-1] >= self.observation_space.high),
            jnp.array([1.0]),
            jnp.array([0.0]),
        )
        done_12 = jnp.logical_or(done_1, done_2)
        done_34 = jnp.logical_or(done_3, done_4)
        done = jnp.logical_or(done_12, done_34)

        return done

    def render(self, state, log_returns):
        try:
            import imgui
            import glfw
            import OpenGL.GL as gl
            from imgui.integrations.glfw import GlfwRenderer
        except ImportError:
            raise ImportError(
                f"Pax: Please install `imgui` and `glfw` to render the {self.name} environment."
            )

        def impl_glfw_init(window_name=f"PAX: {self.name}", width=1280, height=800):
            if not glfw.init():
                print("Could not initialize OpenGL context")
                exit(1)

            # OS X supports only forward-compatible core profiles from 3.2
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

            # Create a windowed mode window and its OpenGL context
            window = glfw.create_window(
                int(width), int(height), window_name, None, None
            )
            glfw.make_context_current(window)

            if not window:
                glfw.terminate()
                print("Could not initialize Window")
                exit(1)

            return window

        def create_circles(P, L, G, t):
            draw_list = imgui.get_window_draw_list()
            f = t % P.shape[0]
            for i in range(P.shape[2] - 1):
                if i == L[f, 0]:
                    draw_list.add_circle_filled(
                        P[f, 0, i, 0],
                        P[f, 0, i, 1],
                        10,
                        imgui.get_color_u32_rgba(0, 1, 0, 1),
                    )
                    draw_list.add_circle_filled(
                        P[f, 0, i, 0],
                        P[f, 0, i, 1],
                        25,
                        imgui.get_color_u32_rgba(0, 1, 0, 0.2),
                    )
                else:
                    draw_list.add_circle_filled(
                        P[f, 0, i, 0],
                        P[f, 0, i, 1],
                        10,
                        imgui.get_color_u32_rgba(0, 0, 1, 1),
                    )

            draw_list.add_circle_filled(
                P[f, 0, i + 1, 0],
                P[f, 0, i + 1, 1],
                10,
                imgui.get_color_u32_rgba(1, 0, 0, 1),
            )

            draw_list.add_circle_filled(
                G[f, 0, 0], G[f, 0, 1], 10, imgui.get_color_u32_rgba(1, 1, 0, 1)
            )

            draw_list.add_line(
                P[f, 0, L[f, 0], 0],
                P[f, 0, L[f, 0], 1],
                P[f, 0, -1, 0],
                P[f, 0, -1, 1],
                imgui.get_color_u32_rgba(1, 1, 1, 1),
                2,
            )
            return draw_list

        global render_time_step
        # start = False

        backgroundColor = (0, 0, 0, 1)
        window = impl_glfw_init()
        gl.glClearColor(*backgroundColor)
        imgui.create_context()
        impl = GlfwRenderer(window)

        render_time_step = 0

        def show(self, state, log_returns, auto_start=True):
            record = self.params.settings["record"]
            global render_time_step

            P = state.X
            if isinstance(P, chex.Array):
                P = np.array(P)

            if len(P.shape) == 3:
                P = P[None, :]

            if len(P.shape) == 2:
                P = P[None, None, :]

            L = state.leader
            if isinstance(L, chex.Array) or isinstance(L, int):
                L = np.array(L)

            if len(L.shape) == 0:
                L = L.reshape(1, 1)

            G = state.curve.eval(1.0)
            if isinstance(G, chex.Array):
                G = np.array(G)

            if len(G.shape) == 2:
                G = G[None, :]

            if len(G.shape) == 1:
                G = G[None, None, :]

            while not glfw.window_should_close(window):
                glfw.poll_events()
                impl.process_inputs()
                imgui.new_frame()
                imgui.set_next_window_size(800, 800)
                imgui.set_next_window_position(0, 0)

                if auto_start:
                    with imgui.begin(
                        "Prober",
                        False,
                        flags=imgui.WINDOW_NO_RESIZE
                        | imgui.WINDOW_NO_COLLAPSE
                        | imgui.WINDOW_NO_MOVE
                        | imgui.WINDOW_NO_TITLE_BAR,
                    ):
                        draw_list = create_circles(P, L, G, render_time_step)
                    render_time_step += 1

                # with imgui.begin("Reward Panel"):
                #     for i, r in enumerate(log_returns):
                #         imgui.text(f"Epoch {i*1000}: Reward:{r}")

                imgui.set_next_window_size(200, 200)
                imgui.set_next_window_position(800, 0)
                with imgui.begin("Control Panel"):
                    x = imgui.button("Start")

                if x:
                    auto_start = True

                imgui.set_next_window_size(200, 200)
                imgui.set_next_window_position(800, 200)
                imgui.plot_lines(
                    "Returns",
                    np.array(log_returns),
                    overlay_text="Rewards",
                    # offset by one item every milisecond, plot values
                    # buffer its end wraps around
                    # values_offset=int(time() * 100) % L[-1],
                    # 0=autoscale => (0, 50) = (autoscale width, 50px height)
                    graph_size=(0, 100),
                )

                imgui.render()

                gl.glClearColor(*backgroundColor)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)

                impl.render(imgui.get_draw_data())
                glfw.swap_buffers(window)

                if render_time_step == self.params.scenario["episode_size"]:
                    glfw.set_window_should_close(window, True)

            impl.shutdown()
            glfw.terminate()

        show(self, state, log_returns)
