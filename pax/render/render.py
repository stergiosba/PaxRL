import imgui
import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import jax.numpy as jnp
import jax.random as jrandom
import chex
import numpy as np

def impl_glfw_init(window_name="Probing Environment", width=1280, height=800):
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


def create_circles(P, t):
    draw_list = imgui.get_window_draw_list()
    f = t % P.shape[0]
    for i in range(P.shape[2]-1):
        draw_list.add_circle_filled(P[f,0,i,0], P[f,0,i,1], 10, imgui.get_color_u32_rgba(1,1,0,1))

    draw_list.add_circle_filled(P[f,0,i+1,0], P[f,0,i+1,1], 10, imgui.get_color_u32_rgba(1,0,0,1))
    return draw_list

class Renderer(object):
    def __init__(self):
        super().__init__()
        self.backgroundColor = (0, 0, 0, 1)
        self.window = impl_glfw_init()
        gl.glClearColor(*self.backgroundColor)
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)

        self.time = 0

    def show(self, env, state, log_returns, record=False):
        P = state.X
        if isinstance(P, chex.Array):
            P = np.array(P)

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            imgui.new_frame()
            imgui.set_next_window_size(800, 800)
            imgui.set_next_window_position(0, 0)
            
            with imgui.begin("Prober", False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR):
                draw_list = create_circles(P, self.time)
            self.time+=1

            with imgui.begin("Reward Panel"):
                for i, r in enumerate(log_returns):
                    imgui.text(f"Epoch {i*1000}: Reward:{r}")

            imgui.render()

            gl.glClearColor(*self.backgroundColor)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
            
            if self.time == env.params.scenario["episode_size"]:
                glfw.set_window_should_close(self.window, True)

        self.impl.shutdown()
        glfw.terminate()


