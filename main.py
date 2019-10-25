import pyglet
from pyglet.window import key

window = pyglet.window.Window()

label = pyglet.text.Label(text="Eat my ass!",
                          font_name="Arial",
                          font_size=32,
                          x=window.width // 2,
                          y=window.height // 2,
                          anchor_x="center", anchor_y="center")


@window.event
def on_draw():
    window.clear()
    label.draw()


@window.event
def on_key_press(symbol, modifier):
    if symbol == key.LEFT:
        print("User pressed left arrow key")


pyglet.app.run()
