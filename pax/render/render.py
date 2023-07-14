import chex
import pyglet as pg
import numpy as np
from pax.render.utils import VisualEntity

def reset(env, P, L, G, batch, env_id=0):
    ScriptedEntities = []
    for i in range(env.n_scripted):
        if i==L[env_id,0]:
            scripted_entity = VisualEntity(id=i, x=P[env_id,0,0,i,0],y=P[env_id,0,0,i,1],\
                radius=10, neighborhood_radius=120, color=(255,128,0,155), batch=batch)
        else:
            scripted_entity = VisualEntity(id=i, x=P[env_id,0,0,i,0],y=P[env_id,0,0,i,1],\
                radius=10, neighborhood_radius=30, color=(0,0,255,155), batch=batch)
        
        ScriptedEntities.append(scripted_entity)
        
    Agents = []
    for i in range(env.n_scripted, env.n_scripted+env.n_agents):
        agent = VisualEntity(id=i, x=P[env_id,0,0,i,0],y=P[env_id,0,0,i,1],\
                radius=10, neighborhood_radius=90, color=(0,0,0,155), batch=batch)
        Agents.append(agent)

    goal_boss = pg.shapes.Star(x=G[env_id,0,0],y=G[env_id,0,1], \
                outer_radius=4, inner_radius = 20, num_spikes=5,color=(0,0,0,255), batch=batch)
    goal = pg.shapes.Star(x=G[env_id,0,0],y=G[env_id,0,1], \
                        outer_radius=5, inner_radius = 10, num_spikes=5,color=(200,200,0,255), batch=batch)
    
    return ScriptedEntities, Agents, goal

def render(env, O, render_mode=None, save=0):
    if not render_mode:
        pass
    elif render_mode in ['h','H',"human","Human"]:
        # Check for instance of observations array, if jax.array cast it to numpy for fast access.
        P = O[0]
        L = O[1]
        G = O[2]
        if isinstance(P, chex.Array):
            P = np.array(P)
            
        if isinstance(L, chex.Array):
            L = np.array(L)

        if isinstance(G, chex.Array):
            G = np.array(G)
        
        window = pg.window.Window(800,800, caption="HORC")
        fps = pg.window.FPSDisplay(window=window)
        batch = pg.graphics.Batch()
        window.simulationClock = pg.clock
        
        t=[0]
        env_id = [0]
        ScriptedEntities, Agents, goal  = reset(env, P, L, G, batch, env_id[0])
        
        @window.event
        def on_key_press(symbol, mods):
            if symbol==pg.window.key.Q:
                window.on_close()
                pg.app.exit()
                
            if symbol==pg.window.key.R:
                t[0] = 0
            
            if symbol==pg.window.key.UP:
                batch = pg.graphics.Batch()
                env_id[0]+=1
                if env_id[0] > env.params["parallel"]["n_env"]-1:
                    env_id[0] = env.params["parallel"]["n_env"]-1
                ScriptedEntities, Agents, goal = reset(env, P, L, G, batch, env_id[0])
                
            if symbol==pg.window.key.DOWN:
                batch = pg.graphics.Batch()
                env_id[0]-=1
                if env_id[0] < 0:
                    env_id[0] = 0
                ScriptedEntities, Agents, goal = reset(env, P, L, G, batch, env_id[0])
                
            if symbol==pg.window.key.P:
                window.simulationClock.unschedule(update)
                
            if symbol==pg.window.key.S:
                window.simulationClock.schedule_interval(update, 1/60)

        @window.event
        def on_draw():
            pg.gl.glClearColor(1, 1, 1, 1)
            window.clear()
            batch.draw()
            fps.draw()

        def update(dt):
            window.clear()
            for i in range(env.n_scripted):
                ScriptedEntities[i].update(P[env_id[0],t[0],0,i], P[env_id[0],t[0],1,i])
                
            for i in range(env.n_agents):
                Agents[i].update(P[env_id[0],t[0],0,env.n_scripted+i], P[env_id[0],t[0],1,env.n_scripted+i])
            
            if save:
                pg.image.get_buffer_manager().get_color_buffer().save(f'saved/screenshot_frame_{t[0]}.png')
            t[0] += 1
            
            t[0]%=env.params["settings"]['episode_size']

        #window.simulationClock.schedule_interval(update, 1/60)

        pg.app.run()