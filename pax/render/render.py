import chex
import pyglet as pg
import numpy as np


def render(env, O, env_id, render_mode=None, save=0):
    if not render_mode:
        pass
    elif render_mode in ['h','H',"human","Human"]:
        # Check for instance of observations array, if jax.array cast it to numpy for fast access.
        P = O[0][env_id]
        L = O[1][env_id][0]
        if isinstance(P, chex.Array):
            P = np.array(P)
            
        if isinstance(L, chex.Array):
            L = np.array(L)
        
            
        window = pg.window.Window(800,800, caption="HORC")
        batch = pg.graphics.Batch()
        fps = pg.window.FPSDisplay(window=window)
        ScriptedEntities = []
        virtuals = []
        agents_virtuals = []
        Labels = []
        
        for i in range(env.n_scripted_entities):
            if i==L:
                scripted_entity = pg.shapes.Circle(x=P[0,0,i,0],y=P[0,0,i,1],\
                radius=10,color=(0,255,255,155), batch=batch)
                entity_neighborhood = pg.shapes.Circle(x=P[0,0,i,0],y=P[0,0,i,1],\
                radius=80,color=(0,255,255,45), batch=batch)
                line1 = pg.shapes.Line(P[0,0,i,0], P[0,0,i,1], P[0,0,i,0]+150*P[0,1,i,0], P[0,0,i,1]+150*P[0,1,i,1], 2, color = (0, 0, 0, 255), batch = batch)

            else:
                scripted_entity = pg.shapes.Circle(x=P[0,0,i,0],y=P[0,0,i,1],\
                    radius=10,color=(255,0,0,155), batch=batch)

                entity_neighborhood = pg.shapes.Circle(x=P[0,0,i,0],y=P[0,0,i,1],\
                radius=40,color=(255,0,0,45), batch=batch)
            label = pg.text.Label(str(i),
                        font_name='Times New Roman',
                        font_size=12,
                        x=P[0,0,i,0], y=P[0,0,i,1],
                        color=(0,0,0,255),
                        anchor_x='center', anchor_y='center', batch=batch)
            ScriptedEntities.append(scripted_entity)
            virtuals.append(entity_neighborhood)
            Labels.append(label)
            
        Agents = []
        for i in range(env.n_scripted_entities, env.n_scripted_entities+env.n_agents):
            agent = pg.shapes.Star(x=P[0,0,i,0],y=P[0,0,i,1], \
                outer_radius=5, inner_radius = 10, num_spikes=5,color=(0,0,255,155), batch=batch)
            agent_radius = pg.shapes.Circle(x=P[0,0,i,0],y=P[0,0,i,1],\
            radius=120,color=(0,0,255,45), batch=batch)
            Agents.append(agent)
            agents_virtuals.append(agent_radius)
            
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
                virtuals[i].position = P[t[0],0,i]
                Labels[i].position = (P[t[0],0,i][0],P[t[0],0,i][1],0)
            line1.position = P[t[0],0,L]
                
            for i in range(env.n_agents):
                Agents[i].position = P[t[0],0,env.n_scripted_entities+i]
                agents_virtuals[i].position = P[t[0],0,env.n_scripted_entities+i]
            batch.draw()
            pg.gl.glClearColor(1, 1, 1, 1)
            if save:
                pg.image.get_buffer_manager().get_color_buffer().save(f'saved/screenshot_frame_{t[0]}.png')
            t[0] += 1
            
            t[0]%=env.params["settings"]['episode_size']


        window.simulationClock.schedule_interval(update, 1/60)

        pg.app.run()