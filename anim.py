# %%
import pyglet as pg

frames = []

for i in range(1,239):
    frames.append(pg.resource.image(f"saved/screenshot_frame_{i}.png"))

window = pg.window.Window(800,800, caption="HORC")
ani = pg.image.Animation.from_image_sequence(frames, duration=1/16, loop=True)
sprite = pg.sprite.Sprite(ani)

@window.event
def on_key_press(symbol, mods):
    if symbol==pg.window.key.Q:
        window.on_close()

@window.event
def on_draw():
    window.clear()
    sprite.draw()

pg.app.run()
