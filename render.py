import pyglet
from pyglet.window import key

window = pyglet.window.Window()

offset = 100
dx = 10
dy = 10




def move_square():
    batch = pyglet.graphics.Batch()
    x1 = offset - dx
    x2 = offset + dx
    y1 = offset - dy
    y2 = offset + dy
    batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", [x2, y2, x1, y2, x1, y1, x2, y1]))
    batch.draw()


@window.event
def on_draw():
    window.clear()
    move_square()

@window.event
def on_key_press(symbol, modifiers):
    global offset
    if symbol == key.RIGHT:
        print("pressed right")
        offset += 20
        window.clear()
        move_square()

print(window.width)
print(window.height)

pyglet.app.run()
