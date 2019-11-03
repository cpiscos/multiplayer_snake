import pyglet
import torch
from pyglet import clock
from pyglet.window import key

from dqn_utils import eps_greedy
from env_vectorized_np import Snake
from models import NoisyIQN, NoisyIQN1
import numpy as np

ai_play = True
players = 1
eps = 0
config = pyglet.gl.Config(double_buffer=True)
window = pyglet.window.Window(config=config)

board_size = 39
offset = 20
tau_samples = 100
weights_name = "weights.pt"
width = window.width - offset
height = window.height - offset

board_unit = min(width // board_size, height // board_size)
x1_board = window.width // 2 - (board_size // 2 + 5) * board_unit
x2_board = x1_board + board_size * board_unit
y1_board = window.height // 2 - (board_size // 2) * board_unit
y2_board = y1_board + board_size * board_unit

min_val = -1
max_val = 1
env = Snake(1, board_size=board_size, terminal_step=200)
model = NoisyIQN1(test=False).eval()
model.load_state_dict(torch.load(weights_name, map_location='cpu'))
state, invalid = env.reset()
ids = []
with torch.no_grad():
    q_values, quantiles = model(torch.FloatTensor(state), n_tau_samples=tau_samples)

ai_action = eps_greedy(q_values.mean(1), torch.BoolTensor(invalid), eps=eps)
batch = pyglet.graphics.Batch()


def draw_board(batch):
    batch.add(2 * 4, pyglet.gl.GL_LINES, None, ("v2i", (x1_board, y1_board, x2_board, y1_board,
                                                        x2_board, y1_board, x2_board, y2_board,
                                                        x2_board, y2_board, x1_board, y2_board,
                                                        x1_board, y2_board, x1_board, y1_board)))


draw_board(batch)


def draw_apple():
    global state, ids, batch
    apple_coord = np.stack(np.nonzero(state[0, 0])).T
    apple_coord = game_to_window_boxes(apple_coord)
    apple_id = batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", apple_coord), ('c3B', [196, 196, 0]*4))
    ids.append(apple_id)


def draw_graphs():
    global q_values, quantiles, batch, ids, invalid, min_val, max_val
    min_val, max_val = 0.9*min_val + 0.1*q_values.min(), 0.9*max_val + 0.1*q_values.max()
    # min_val, max_val = torch.tensor(-1.0).float(), torch.tensor(2.0).float()
    graph_width = (window.width - board_unit) - (x2_board + board_unit)
    graph_height = (board_size // 4) * 0.9 * board_unit
    graph_width_scale = graph_width / (max_val - min_val)
    new_widths = (q_values - min_val) * graph_width_scale
    new_widths = new_widths[0]
    new_heights = quantiles * graph_height
    sorted_idx = new_widths.argsort(0)

    base_lines = [x2_board + board_unit, y1_board, window.width - board_unit, y1_board]
    for i in range(3):
        new_line = base_lines[:4]
        new_line[1] += (board_size // 4) * (i + 1) * board_unit
        new_line[3] += (board_size // 4) * (i + 1) * board_unit
        base_lines.extend(new_line)
    new_widths += base_lines[0]
    sorted_widths = []
    sorted_heights = []
    valid_idx = np.stack((~invalid[0]).nonzero())[0]
    for i in valid_idx:
        sorted_widths.append(new_widths[sorted_idx[:, i], i])
        sorted_heights.append(new_heights[sorted_idx[:, i]] + base_lines[int(1 + 4 * i)])
    sorted_widths = torch.stack(sorted_widths).clamp(min=base_lines[0], max=base_lines[2])
    sorted_heights = torch.stack(sorted_heights)

    coords = []
    for i in range(sorted_widths.shape[1] - 1):
        coord_block = torch.stack((sorted_widths[:, i:i + 2], sorted_heights[:, i:i + 2]), 2).reshape(sorted_widths.shape[0], -1)
        coords.append(coord_block)
    coords = torch.cat(coords, -1).reshape(-1).tolist()
    graphs_id = batch.add(2 * 4, pyglet.gl.GL_LINES, None, ('v2i', base_lines))
    graphs_lines_id = batch.add(len(coords) // 2, pyglet.gl.GL_LINES, None, ('v2f', coords))
    min_val_id = pyglet.text.Label("{:.2f}".format(min_val.item()),
                                   font_size=10,
                                   x=base_lines[0], y=base_lines[1] - 1,
                                   anchor_x="center", anchor_y="top",
                                   batch=batch)
    mean_val_id = pyglet.text.Label("{:.2f}".format(((max_val - min_val) / 2 + min_val).item()),
                                    font_size=10,
                                    x=(base_lines[2]-base_lines[0]) // 2 + base_lines[0], y=base_lines[1] - 1,
                                    anchor_x="center", anchor_y="top",
                                    batch=batch)
    max_val_id = pyglet.text.Label("{:.2f}".format(max_val.item()),
                                   font_size=10,
                                   x=base_lines[2], y=base_lines[1] - 1,
                                   anchor_x="center", anchor_y="top",
                                   batch=batch)
    ids.extend((graphs_id, graphs_lines_id, min_val_id, mean_val_id, max_val_id))


def draw_player():
    global state, ids, batch
    player_coord = np.stack(np.nonzero(state[0, 3].astype(int) - state[0, 1] - state[0, 5])).T
    head_coord = np.stack(np.nonzero(state[0, 1])).T
    tail_coord = np.stack(np.nonzero(state[0, 5])).T
    snake_len = len(player_coord)
    head_coord = game_to_window_boxes(head_coord)
    player_coord = game_to_window_boxes(player_coord)
    tail_coord = game_to_window_boxes(tail_coord)
    player_id = batch.add(4 * snake_len, pyglet.gl.GL_QUADS, None, ("v2i", player_coord))
    head_id = batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", head_coord), ('c3B', [0, 128, 0] * 4))
    tail_id = batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", tail_coord), ('c3B', [192, 192, 192] * 4))
    apple_steps_id = pyglet.text.Label("{}".format(env.steps_since_apple[0].item()),
                                       font_size=15,
                                       x=0, y=30,
                                       anchor_x="left", anchor_y="bottom",
                                       batch=batch)
    # value_text_id = pyglet.text.Label("{:.3f}".format(val.item()),
    #                                   font_size=15,
    #                                   x=0, y=0,
    #                                   anchor_x="left", anchor_y="bottom",
    #                                   batch=batch)
    # probs_text_id = pyglet.text.Label(("{:.2f} "*4).format(*q_values[0].tolist()),
    #                                   font_size=10,
    #                                   x=0, y=window.height,
    #                                   anchor_x="left", anchor_y="top",
    #                                   batch=batch)
    ids.extend((player_id, apple_steps_id, head_id, tail_id))


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
    global state, invalid, ai_action, q_values, quantiles
    if ai_play:
        ai_action = ai_action.cpu().numpy()
        state, invalid, _, done = env.step(ai_action)
        if done[0] == 1:
            model.load_state_dict(torch.load(weights_name, map_location='cpu'))
            print("model reloaded")
    with torch.no_grad():
        q_values, quantiles = model(torch.FloatTensor(state), n_tau_samples=tau_samples)
        ai_action = eps_greedy(q_values.mean(1), torch.BoolTensor(invalid), eps=eps)


@window.event
def on_draw():
    global ids
    for i in range(len(ids)):
        id = ids.pop()
        id.delete()
    window.clear()
    draw_apple()
    draw_player()
    draw_graphs()
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
            model.load_state_dict(torch.load("weights.pt"))
            print("reloaded model")
            action = ai_action
        action = np.array([action])
        state, invalid, reward, done = env.step(action)
        ai_take_action()


clock.schedule_interval(ai_take_action, 0.1)
pyglet.app.run()
