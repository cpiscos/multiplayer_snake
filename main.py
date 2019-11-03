import pyglet
import torch
from pyglet import clock
from pyglet.window import key

from env import TrainEnvSingle
from models import ActorCritic

# players = int(input("Number of players:"))
players = 1
config = pyglet.gl.Config(double_buffer=True)
window = pyglet.window.Window(config=config)


board_size = 40
offset = 20
width = window.width - offset
height = window.height - offset

board_unit = min(width // board_size, height // board_size)
x1_board = window.width // 2 - (board_size // 2 + 1) * board_unit
x2_board = x1_board + (board_size + 1) * board_unit
y1_board = window.height // 2 - (board_size // 2 + 1) * board_unit
y2_board = y1_board + (board_size + 1) * board_unit

print(x1_board, x2_board, y1_board, y2_board)

env = TrainEnvSingle()
game = env.game
model = ActorCritic()
model.load_state_dict(torch.load("weights.pt"))
model = model.eval()
state, invalid = env.reset()
dist, value = model(state, invalid)
q_values = dist.probs.tolist()[0]


def take_action(dt):
    pass


def reload_model(dt):
    global model
    model.load_state_dict(torch.load("weights.pt"))
    print("Reloaded model")


def draw_board(batch):
    batch.add(2 * 4, pyglet.gl.GL_LINES, None, ("v2i", (x1_board, y1_board, x2_board, y1_board,
                                                        x2_board, y1_board, x2_board, y2_board,
                                                        x2_board, y2_board, x1_board, y2_board,
                                                        x1_board, y2_board, x1_board, y1_board)))


def game_to_window_boxes(coords):
    new_coords = []
    for coord in coords:
        x1 = coord[0] * board_unit + x1_board
        x2 = x1 + board_unit
        y1 = coord[1] * board_unit + y1_board
        y2 = y1 + board_unit
        new_coords.extend([x2, y2, x1, y2, x1, y1, x2, y1])
    return new_coords


def draw_apple(game, batch):
    apple_coords = game.apple
    apple_coords = game_to_window_boxes([apple_coords])
    apple_id = batch.add(4, pyglet.gl.GL_QUADS, None, ("v2i", apple_coords))
    return apple_id


def draw_player(game, batch, player):
    global state, invalid, env, value, q_values
    if env.steps_since_apple == 0:
        reload_model(None)
    with torch.no_grad():
        state = torch.FloatTensor(state)[None]
        invalid = torch.BoolTensor(invalid)[None]
        dist, value = model(state, invalid)
        action = dist.sample().item()
        value = value.item()
        probs = dist.probs.tolist()[0]
    player_coords = game.players[player]
    snake_len = len(player_coords)
    player_coords = game_to_window_boxes(player_coords)
    player_id = batch.add(4 * snake_len, pyglet.gl.GL_QUADS, None, ("v2i", player_coords))
    value_text_id = pyglet.text.Label("{:.3f}".format(value),
                                      font_size=15,
                                      x=0, y=0,
                                      anchor_x="left", anchor_y="bottom",
                                      batch=batch)
    probs_text_id = pyglet.text.Label(("{:.2f} "*4).format(*probs),
                                      font_size=10,
                                      x=0, y=window.height,
                                      anchor_x="left", anchor_y="top",
                                      batch=batch)
    apple_steps_id = pyglet.text.Label("{}".format(env.steps_since_apple),
                                      font_size=15,
                                      x=0, y=30,
                                      anchor_x="left", anchor_y="bottom",
                                      batch=batch)

    score_id = pyglet.text.Label("{}".format(len(env.game.players[0]) - 5),
                                       font_size=15,
                                       x=window.width, y=window.height,
                                       anchor_x="right", anchor_y="top",
                                       batch=batch)

    state, reward, invalid, _ = env.action(action)
    ids = [player_id, value_text_id, probs_text_id, apple_steps_id, score_id]
    return ids


batch = pyglet.graphics.Batch()
draw_board(batch)
apple_id = draw_apple(game, batch)

players_ids = [draw_player(game, batch, i) for i in range(players)]
clock.schedule_interval(take_action, 0.1)
# clock.schedule_interval(reload_model, 60)


@window.event
def on_draw():
    global apple_id, players_ids
    apple_id.delete()
    apple_id = draw_apple(game, batch)
    for ids in players_ids:
        for id in ids:
            id.delete()
    players_ids = [draw_player(game, batch, i) for i in range(players)]
    window.clear()
    batch.draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.RIGHT:
        pass
        # take_action(None)
#     elif symbol == key.DOWN:
#         action = 1
#     elif symbol == key.LEFT:
#         action = 2
#     else:
#         action = 3
#     reward, apple_got, _ = game.action(action, 0)
#     if apple_got:
#         apple_id.delete()
#         apple_id = draw_apple(game, batch)
#     for p in player_list:
#         p.delete()
#     player_list = [draw_player(game, batch, i) for i in range(players)]
#     print(reward)

pyglet.app.run()
