import pyglet
import torch
from pyglet.window import key
from pyglet import clock
from models import ActorCritic

from env_vectorized import Snake

ai_play = True
players = 1
eps = 0.
config = pyglet.gl.Config(double_buffer=True)
window = pyglet.window.Window(config=config)

board_size = 40
offset = 20
width = window.width - offset
height = window.height - offset

board_unit = min(width // board_size, height // board_size)
x1_board = window.width // 2 - (board_size // 2) * board_unit
x2_board = x1_board + board_size * board_unit
y1_board = window.height // 2 - (board_size // 2) * board_unit
y2_board = y1_board + board_size * board_unit

env = Snake(2, board_size=board_size, terminal_step=100)
model = ActorCritic().eval()
model.load_state_dict(torch.load("weights.pt"))
state, invalid = env.reset()
body_coord = torch.nonzero(state[0, 3])
ids = []
dist, val = model(state[0:1], invalid[0:1])
ai_action = dist.sample().item()
q_values = dist.probs.tolist()[0]
batch = pyglet.graphics.Batch()


def draw_board(batch):
    batch.add(2 * 4, pyglet.gl.GL_LINES, None, ("v2i", (x1_board, y1_board, x2_board, y1_board,
                                                        x2_board, y1_board, x2_board, y2_board,
                                                        x2_board, y2_board, x1_board, y2_board,
                                                        x1_board, y2_board, x1_board, y1_board)))


draw_board(batch)


def draw_apple():
    global state, ids, batch
    apple_coord = torch.nonzero(state[0, 0])
    apple_coord = game_to_window_boxes(apple_coord)
    apple_id = batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", apple_coord))
    ids.append(apple_id)


def draw_player():
    global state, ids, batch
    player_coord = torch.nonzero(state[0, 3])
    head_coord = torch.nonzero(state[0, 1])
    tail_coord = torch.nonzero(state[0, 5])
    snake_len = len(player_coord)
    head_coord = game_to_window_boxes(head_coord)
    player_coord = game_to_window_boxes(player_coord)
    tail_coord = game_to_window_boxes(tail_coord)
    head_id = batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", head_coord), ('c3B', [0, 128, 0]*4))
    player_id = batch.add(4 * snake_len, pyglet.gl.GL_QUADS, None, ("v2i", player_coord))
    tail_id = batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", tail_coord), ('c3B', [192, 192, 192]*4))
    apple_steps_id = pyglet.text.Label("{}".format(env.steps_since_apple[0].item()),
                                       font_size=15,
                                       x=0, y=30,
                                       anchor_x="left", anchor_y="bottom",
                                       batch=batch)
    value_text_id = pyglet.text.Label("{:.3f}".format(val.item()),
                                      font_size=15,
                                      x=0, y=0,
                                      anchor_x="left", anchor_y="bottom",
                                      batch=batch)
    probs_text_id = pyglet.text.Label(("{:.2f} "*4).format(*q_values),
                                      font_size=10,
                                      x=0, y=window.height,
                                      anchor_x="left", anchor_y="top",
                                      batch=batch)
    ids.extend((player_id, probs_text_id, apple_steps_id, head_id, value_text_id, tail_id))


def game_to_window_boxes(coords):
    new_coords = []
    for coord in coords:
        coord = [coord[1], board_size - coord[0] - 1]
        # coord[1] = board_size - coord[1] - 1
        x1 = coord[0] * board_unit + x1_board
        x2 = x1 + board_unit
        y1 = coord[1] * board_unit + y1_board
        y2 = y1 + board_unit
        new_coords.extend([x2, y2, x1, y2, x1, y1, x2, y1])
    return new_coords

def ai_take_action(dt=None):
    global state, invalid, ai_action, q_values, val
    if ai_play:
        state, invalid, _, done = env.step(torch.tensor([ai_action]*2))
        if done[0] == 1:
            model.load_state_dict(torch.load("weights.pt"))
            print("model reloaded")
    with torch.no_grad():
        dist, val = model(state[0:1], invalid[0:1])
        ai_action = dist.sample().item()
        probs = dist.probs.tolist()[0]


@window.event
def on_draw():
    global ids
    for i in range(len(ids)):
        id = ids.pop()
        id.delete()
    window.clear()
    draw_apple()
    draw_player()
    batch.draw()


@window.event
def on_key_press(symbol, modifiers):
    if not ai_play:
        global state, invalid, ai_action
        if symbol == key.RIGHT:
            action = 0
        elif symbol == key.DOWN:
            action = 1
        elif symbol == key.LEFT:
            action = 2
        elif symbol == key.UP:
            action = 3
        else:
            action = ai_action
        action = torch.tensor([action]*2)
        ai_take_action()
        state, invalid, reward, done = env.step(action)
        print(reward)
        print(done)


clock.schedule_interval(ai_take_action, 0.1)
pyglet.app.run()
