import chex
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.numpy.linalg as la
from jax.debug import print as dprint  # type: ignore
from pax.core.environment import Environment, EnvParams
from pax.core.spaces import *
from typing import Sequence, Tuple, Dict
from jax import lax, jit


class EnvState(eqx.Module):
    """The environment state for the TargetEnv.

    `Args`:
        - `X (chex.Array)`: Position of every Agents.
        - `X_dot (chex.Array)`: Velocity of every Agent.
        - `goal (chex.Array)`: The location of the goal.
        - `time (int)`: The current time step.
    """

    X: chex.ArrayDevice
    X_dot: chex.ArrayDevice
    goal: chex.ArrayDevice
    time: int

    def __repr__(self):
        return f"{__class__.__name__}: {str(self.__dict__)}"


class Target(Environment):

    """The main `Pax` class.
        It encapsulates an environment with arbitrary dynamics.
        An environment can be partially or fully observable.

    `Attributes`:
        - `n_agents (int)`: The total number of agents (actors) in the environment.
        - `n_scripted (int)`: The total number of scripted entities in the environment.
        - `action_space (Union[Discrete, Box])`: The action space allowed to the agents.
        - `observation_space (Union[Discrete, Box])`: The observation space of the environment.
        - `params (Dict)`: Parameters given to environment from the TOML file.

    """

    n_agents: int
    action_space: SeparateGrid
    observation_space: Box

    @property
    def version(self):
        return "0.1"

    def __init__(self, params: Dict):
        """
        Args:
           `params (Dict)`: Parameters given to environment from the TOML file.
        """
        self.params = EnvParams(**params)
        self.n_agents = self.params.scenario["n_agents"]

        action_range = self.params.action_space.values()
        self.action_space = SeparateGrid(action_range)

        o_dtype = jnp.float32
        o_low, o_high = self.params.observation_space.values()
        o_shape = (self.n_agents, 2)
        self.observation_space = Box(o_low, o_high, o_shape, o_dtype)

    @eqx.filter_jit
    def get_obs(self, state: EnvState) -> Sequence[chex.Array]:
        """Applies observation function to state.

        `Args`:
            - `state (EnvState)`: The full state of the environment.

        Returns:
            - `observation (chex.Array)`: The observable part of the state.
        """
        
        return jnp.array(state.X).reshape(-1)

    @eqx.filter_jit
    def reset_env(self, key: chex.PRNGKey) -> Tuple[Sequence[chex.Array], EnvState]:
        """Resets the environment.

        Args:
            -`key (chex.PRNGKey)`: The random key for the reset.

        Returns:
            - `Observations (chex.Array)`: The initial observation of the environment based on the observation function get_obs
            - `State (EnvState)`: The initial full state of the environment. Used internally.
        """

        init_X = jrandom.uniform(
            key,
            minval=jnp.array([400, 400]),
            maxval=jnp.array([601, 601]),
            shape=(self.n_agents, 2),
        )

        init_X_dot = (
            jrandom.uniform(key, minval=-0, maxval=0, shape=(self.n_agents, 2))
            * 10
            * jnp.sqrt(2)
        )

        goal = jrandom.uniform(
            key, minval=jnp.array([300, 50]), maxval=jnp.array([750, 125]), shape=(2,)
        )

        state = EnvState(
            X=init_X,
            X_dot=init_X_dot,
            goal = goal,
            time=0,
        )
        return (self.get_obs(state), state)  # type: ignore

    @eqx.filter_jit
    def step_env(
        self, key, state: EnvState, action: chex.ArrayDevice, extra_in=None
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

        @jit
        def r_distance():
            return la.norm(state.X - state.goal)

        acc = action
        dt = self.params.settings["dt"]
        X_dot = state.X_dot + dt / 6 * acc  # - 0.01*state.X_dot

        X_dot = jnp.clip(X_dot, a_min=-10 * jnp.sqrt(2), a_max=10 * jnp.sqrt(2))
        X = state.X + dt * X_dot
        X = jnp.clip(
            X, a_min=self.observation_space.low, a_max=self.observation_space.high
        )

        state = EnvState(X, X_dot, state.goal, state.time + 1)  # type: ignore

        obs = self.get_obs(state)
        reward = -0.01 + 10 * (r_distance() < 50)
        done = self.is_terminal(state)

        return (obs, state, jnp.array([reward]), done)

    def is_terminal(self, state):
        norm_e = la.norm(state.goal - state.X)

        done_1 = lax.select(norm_e < 50, jnp.array([1.0]), jnp.array([0.0]))

        # create a done that checks if the agent hit a wall
        done_2 = lax.select(
            jnp.any(state.X <= self.observation_space.low),
            jnp.array([1.0]),
            jnp.array([0.0]),
        )
        done_3 = lax.select(
            jnp.any(state.X >= self.observation_space.high),
            jnp.array([1.0]),
            jnp.array([0.0]),
        )

        done_4 = state.time == self.params.scenario["episode_size"]
        done_12 = jnp.logical_or(done_1, done_2)
        done_34 = jnp.logical_or(done_3, done_4)
        done = jnp.logical_or(done_12, done_34)
        return done


    def __repr__(self):
        return f"{self.name}: {str(self.__dict__)}"
    

    def render(self, state, actions, log_returns, auto_start=True):

        try: 
            import imgui
            import glfw
            import OpenGL.GL as gl
            from imgui.integrations.glfw import GlfwRenderer
        except ImportError:
            raise ImportError(f"Pax: Please install `imgui` and `glfw` to render the {self.name} environment.")

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
            window = glfw.create_window(int(width), int(height), window_name, None, None)
            glfw.make_context_current(window)

            if not window:
                glfw.terminate()
                print("Could not initialize Window")
                exit(1)

            return window


        def create_circles(A, P, G, t):
            draw_list = imgui.get_window_draw_list()
            f = t % P.shape[0]

            draw_list.add_circle_filled(P[f,0,0,0], P[f,0,0,1], 10, imgui.get_color_u32_rgba(1,0,0,1))

            draw_list.add_circle_filled(G[f,0,0], G[f,0,1], 10, imgui.get_color_u32_rgba(1,1,0,1))

            draw_list.add_line(P[f,0,0,0], P[f,0,0,1], P[f,0,0,0], P[f,0,0,1], imgui.get_color_u32_rgba(1,1,0,1), 3)
            return draw_list

        global time

        backgroundColor = (0, 0, 0, 1)
        window = impl_glfw_init()
        gl.glClearColor(*backgroundColor)
        imgui.create_context()
        impl = GlfwRenderer(window)

        time = 0
        record = self.params.settings["record"]

        A = actions
        if isinstance(A, chex.Array):
            A = np.array(A)
        # A = A/la.norm(A, axis=-1)[:,None]

        P = state.X
        if isinstance(P, chex.Array):
            P = np.array(P)

        G = state.goal
        if isinstance(G, chex.Array):
            G = np.array(G)

        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()
            imgui.new_frame()
            imgui.set_next_window_size(800, 800)
            imgui.set_next_window_position(0, 0)
        
            if auto_start:
                with imgui.begin("Prober", False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR):
                    draw_list = create_circles(A, P, G, time)
                time+=1

            # with imgui.begin("Reward Panel"):
            #     for i, r in enumerate(log_returns):
            #         imgui.text(f"Epoch {i*1000}: Reward:{r}")


            imgui.set_next_window_size(200, 200)
            imgui.set_next_window_position(800, 0)
            with imgui.begin("Control Panel"):
                x = imgui.button("Start")

            if x:
                auto_start = True
            
            imgui.render()

            gl.glClearColor(*backgroundColor)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
            
            if time == self.params.scenario["episode_size"]:
                glfw.set_window_should_close(window, True)

        impl.shutdown()
        glfw.terminate()



