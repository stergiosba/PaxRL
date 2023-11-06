import chex
import os
import pyglet as pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pax.render.utils import ScriptedEntity, AgentEntity
from scipy.stats import multivariate_normal

ScriptedEntities = []
Agents = []
Goal = []


def reset(env, P, L, GG, batch, env_id=0):
    for i in range(env.n_scripted):
        if i == L[0, env_id]:
            scripted_entity = ScriptedEntity(
                id=i,
                x=P[0, env_id, i, 0],
                y=P[0, env_id, i, 1],
                radius=10,
                neighborhood_radius=120,
                color=(255, 128, 0, 255),
                batch=batch,
            )
        else:
            scripted_entity = ScriptedEntity(
                id=i,
                x=P[0, env_id, i, 0],
                y=P[0, env_id, i, 1],
                radius=10,
                neighborhood_radius=30,
                color=(50, 50, 255, 255),
                batch=batch,
            )

        ScriptedEntities.append(scripted_entity)

    for i in range(env.n_scripted, env.n_scripted + env.n_agents):
        agent = AgentEntity(
            x=P[0, env_id, i, 0],
            y=P[0, env_id, i, 1],
            radius=10,
            neighborhood_radius=60,
            color=(180, 0, 10, 255),
            batch=batch,
        )
        Agents.append(agent)

    n_p = 10
    for i in range(n_p - 1):
        G = GG.eval(i / n_p)
        pos = pg.shapes.Star(
            x=G[0, env_id, 0],
            y=G[0, env_id, 1],
            outer_radius=2,
            inner_radius=2,
            num_spikes=5,
            color=(155, 0, 0, 255),
            batch=batch,
        )
        Goal.append(pos)
    G = GG.eval(1.0)
    goal = pg.shapes.Star(
        x=G[0, env_id, 0],
        y=G[0, env_id, 1],
        outer_radius=5,
        inner_radius=10,
        num_spikes=5,
        color=(200, 200, 0, 255),
        batch=batch,
    )
    Goal.append(goal)


def render(env, state, record=False):
    # Check for instance of observations array, if jax.array cast it to numpy for fast access.
    P = state.X
    V = state.X_dot
    L = state.leader
    GG = state.curve

    if isinstance(P, chex.Array):
        P = np.array(P)

    if isinstance(V, chex.Array):
        V = np.array(V)

    if isinstance(L, chex.Array):
        L = np.array(L)

    if record:
        os.makedirs("record", exist_ok=True)

    window = pg.window.Window(800, 800, caption="Swarm simulation")
    fps = pg.window.FPSDisplay(window=window)
    batch = pg.graphics.Batch()
    font_size = 28
    time_prefix = f"Time:"
    time_label = pg.text.Label(
        f"{time_prefix} 0",
        font_name="Arial",
        font_size=font_size,
        bold=True,
        color=(30, 30, 30, 105),
        x=window.width // 2,
        y=font_size,
        anchor_x="center",
        anchor_y="center",
        batch=batch
    )
    env_prefix = f"Env:"
    env_label = pg.text.Label(
        f"{env_prefix} 1",
        font_name="Arial",
        font_size=font_size,
        bold=True,
        color=(30, 30, 30, 105),
        x=window.width // 2+300,
        y=font_size,
        anchor_x="center",
        anchor_y="center",
        batch=batch
    )
    window.simulationClock = pg.clock

    # Created as lists so that they are global
    t = [0]
    env_id = [0]
    reset(env, P, L, GG, batch, env_id[0])

    @window.event
    def on_key_press(symbol, mods):
        global ScriptedEntities, Agents, Goal

        if symbol == pg.window.key.Q:
            window.on_close()
            pg.app.exit()

        if symbol == pg.window.key.R:
            t[0] = 0

        if symbol == pg.window.key.UP:
            ScriptedEntities = []
            Agents = []
            Goal = []

            env_id[0] += 1
            if env_id[0] >= env.params.settings["n_env"] - 1:
                env_id[0] = env.params.settings["n_env"] - 1
            reset(env, P, L, GG, batch, env_id[0])

        if symbol == pg.window.key.DOWN:
            ScriptedEntities = []
            Agents = []
            Goal = []

            env_id[0] -= 1
            if env_id[0] <= 0:
                env_id[0] = 0
            reset(env, P, L, GG, batch, env_id[0])

        if symbol == pg.window.key.P:
            window.simulationClock.unschedule(update)

        if symbol == pg.window.key.S:
            window.simulationClock.schedule_interval(update, 1 / 60)

    @window.event
    def on_draw():
        pg.gl.glClearColor(1, 1, 1, 1)
        window.clear()
        batch.draw()
        fps.draw()

    def update(dt):
        window.clear()

        for i in range(env.n_scripted):
            ScriptedEntities[i].update(P[t[0], env_id[0], i], V[t[0], env_id[0], i])

        for i in range(env.n_agents):
            Agents[i].update(
                P[t[0], env_id[0], env.n_scripted + i],
                V[t[0], env_id[0], env.n_scripted + i],
            )

        batch.draw()
        time_label.text = f"{time_prefix} {t[0]+1} [{100*(t[0]+1)/env.params.scenario['episode_size']:0.1f}%]"
        env_label.text = f"{env_prefix} {env_id[0]+1}/{env.params.settings['n_env']}"
        pg.gl.glClearColor(1, 1, 1, 1)

        if record:
            pg.image.get_buffer_manager().get_color_buffer().save(
                f"record/frame_{t[0]}.png"
            )
        if t[0] == env.params.scenario["episode_size"] - 1:
            t[0] == env.params.scenario["episode_size"] - 1
        else:
            t[0] += 1

        # t[0] %= env.params.scenario["episode_size"]

    pg.app.run()


def gauss2d(mu, sigma, to_plot=False):
    w, h = 100, 100

    std = [np.sqrt(sigma[0]), np.sqrt(sigma[1])]
    x = np.linspace(mu[0] - 3 * std[0], mu[0] + 3 * std[0], w)
    y = np.linspace(mu[1] - 3 * std[1], mu[1] + 3 * std[1], h)

    x, y = np.meshgrid(x, y)

    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(w, h, order="F")

    if to_plot:
        plt.contourf(x, y, z.T)
        plt.show()

    return plt.contourf(x, y, z.T)


def matplotlib_render(env, obs, record=False):
    P = obs[0]
    L = obs[1]
    GG = obs[2]

    if isinstance(P, chex.Array):
        P = np.array(P)

    if isinstance(L, chex.Array):
        L = np.array(L)

    if isinstance(GG, chex.Array):
        GG = np.array(GG)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 800)  # Set x-axis limits
    ax.set_ylim(0, 800)  # Set y-axis limits

    # Generate initial data for the scatter plot

    n_env = 0
    print(P[:, n_env, 0, :, 0])

    x_data = P[0, n_env, 0, :, 0]
    y_data = P[0, n_env, 0, :, 1]

    mean = [5, 5]  # Mean of the distribution (centered at x=5, y=5)
    covariance = [75.0, 90.0]  # Covariance matrix

    # Create a grid of points for the 2D normal distribution
    x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    pos = np.dstack((x, y))

    # Calculate the 2D normal distribution values at the grid points
    rv = multivariate_normal(mean, [[2.0, 0.3], [0.3, 0.5]])
    z = rv.pdf(pos)

    # z = gauss2d(MU, SIGMA, True)
    scatter = ax.scatter(x_data, y_data)  # Initial scatter plot

    # Plot the 2D normal distribution as a contour plot
    contour = ax.contourf(x, y, z, cmap="viridis", alpha=0.5)

    # Define the update function for the animation
    def update(frame):
        #    # Update the x and y coordinates of the scatter plot with new random values
        scatter.set_offsets(P[frame, n_env, 0, :, :])

        # Update the contour plot (2D normal distribution)
        z = rv.pdf(pos + frame * 0.1)  # Shift the distribution over time
        contour = ax.contourf(x, y, z, cmap="viridis", alpha=0.5)

        return scatter, contour

    # Create the animation
    ani = FuncAnimation(fig, update, frames=2000, interval=20, blit=True)

    # Show the animation
    plt.show()
