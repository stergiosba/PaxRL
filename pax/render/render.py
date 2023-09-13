import chex
import pyglet as pg
import numpy as np
from pax.render.utils import ScriptedEntity, AgentEntity

ScriptedEntities = []
Agents = []
Goal = []


def reset(env, P, L, GG, batch, env_id=0):
    for i in range(env.n_scripted):
        if i == L[0, env_id]:
            scripted_entity = ScriptedEntity(
                id=i,
                x=P[0, env_id, 0, i, 0],
                y=P[0, env_id, 0, i, 1],
                radius=10,
                neighborhood_radius=120,
                color=(255, 128, 0, 255),
                batch=batch,
            )
        else:
            scripted_entity = ScriptedEntity(
                id=i,
                x=P[0, env_id, 0, i, 0],
                y=P[0, env_id, 0, i, 1],
                radius=10,
                neighborhood_radius=30,
                color=(50, 50, 255, 255),
                batch=batch,
            )

        ScriptedEntities.append(scripted_entity)

    for i in range(env.n_scripted, env.n_scripted + env.n_agents):
        agent = AgentEntity(
            x=P[0, env_id, 0, i, 0],
            y=P[0, env_id, 0, i, 1],
            radius=8,
            neighborhood_radius=90,
            color=(0, 0, 0, 255),
            batch=batch,
        )
        Agents.append(agent)

    n_p = 10
    for i in range(n_p-1):
        G = GG.eval(i/n_p)
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
    goal = pg.shapes.Star(x=G[0,env_id,0],y=G[0,env_id,1], \
                        outer_radius=5, inner_radius = 10, num_spikes=5,color=(200,200,0,255), batch=batch)
    Goal.append(goal)


def render(env, obs, record=False):
    # Check for instance of observations array, if jax.array cast it to numpy for fast access.
    P = obs[0]
    L = obs[1]
    GG = obs[2]

    if isinstance(P, chex.Array):
        P = np.array(P)

    if isinstance(L, chex.Array):
        L = np.array(L)

    if isinstance(GG, chex.Array):
        GG = np.array(GG)


    window = pg.window.Window(800, 800, caption="Swarm simulation")
    fps = pg.window.FPSDisplay(window=window)
    batch = pg.graphics.Batch()
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
            if env_id[0] >= env.params["settings"]["n_env"] - 1:
                env_id[0] = env.params["settings"]["n_env"] - 1
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
            ScriptedEntities[i].update(
                P[t[0], env_id[0], 0, i], P[t[0], env_id[0], 1, i]
            )

        for i in range(env.n_agents):
            Agents[i].update(
                P[t[0], env_id[0], 0, env.n_scripted + i],
                P[t[0], env_id[0], 1, env.n_scripted + i],
            )

        if record:
            pg.image.get_buffer_manager().get_color_buffer().save(
                f"saved/screenshot_frame_{t[0]}.png"
            )
        t[0] += 1

        t[0] %= env.params["scenario"]["episode_size"]

    pg.app.run()
