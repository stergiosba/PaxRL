import chex
import pyglet as pg
import numpy as np


def render(env, O, id, save=0):
    # Check for instance of observations array, if jax.array cast it to numpy for fast access.
    P = O[0][id]
    L = O[1][id][0]
    if isinstance(P, chex.Array):
        P = np.array(P)
        
    if isinstance(L, chex.Array):
        L = np.array(L)
    
        
    window = pg.window.Window(1400,800, caption="HORC")
    batch = pg.graphics.Batch()
    fps = pg.window.FPSDisplay(window=window)
    ScriptedEntities = []
    for i in range(env.n_scripted_entities):
        if i==L:
            scripted_entity = pg.shapes.Circle(x=P[0,0,i,0],y=P[0,0,i,1],\
            radius=10,color=(0,255,255,155), batch=batch)
        else:
            scripted_entity = pg.shapes.Circle(x=P[0,0,i,0],y=P[0,0,i,1],\
                radius=10,color=(255,0,0,155), batch=batch)
        ScriptedEntities.append(scripted_entity)
        
    Agents = []
    for i in range(env.n_scripted_entities, env.n_scripted_entities+env.n_agents):
        agent = pg.shapes.Star(x=P[0,0,i,0],y=P[0,0,i,1], \
            outer_radius=5, inner_radius = 10, num_spikes=5,color=(0,0,255,155), batch=batch)
        Agents.append(agent)
        
    window.simulationClock = pg.clock

    t=[0]
    @window.event
    def on_key_press(symbol, mods):
        if symbol==pg.window.key.Q:
            window.on_close()
            pg.app.exit()
        if symbol==pg.window.key.R:
            t[0]=0

    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        fps.draw()

    def update(dt):
        window.clear()
        for i in range(env.n_scripted_entities):
            ScriptedEntities[i].position = P[t[0],0,i]
            
        for i in range(env.n_agents):
            Agents[i].position = P[t[0],0,env.n_scripted_entities+i]
        batch.draw()
        pg.gl.glClearColor(1, 1, 1, 1)
        if save:
            pg.image.get_buffer_manager().get_color_buffer().save(f'saved/screenshot_frame_{t[0]}.png')
        t[0] += 1
        
        t[0]%=env.params["settings"]['episode_size']


    window.simulationClock.schedule_interval(update, 1/60)

    pg.app.run()