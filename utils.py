import numpy as np


def paint_board(coords, board_size, board=None, value=1):
    if isinstance(board, type(None)):
        board = np.zeros(board_size ** 2).astype(np.int8)
    orig_coords = coords
    coords = np.array(coords)
    coords[:, 1] = board_size - 1 - coords[:, 1]
    coords[:, 1] *= board_size
    coords = coords.sum(1)
    if 1600 in coords:
        print(orig_coords)
    board[coords] = value
    return board


def gameboard_to_array(game, player, board_size=40):
    players_coords = game.players[player]
    board = paint_board(players_coords, board_size).reshape(board_size, board_size)
    apple_board = paint_board([game.apple], board_size).reshape(board_size, board_size)
    if game.player_count > 1:
        opponent_coords = []
        for i, p in enumerate(game.players):
            if i != player:
                opponent_coords.extend(p)
        opp_board = paint_board(opponent_coords, board_size).reshape(board_size, board_size)
    else:
        opp_board = np.zeros((board_size, board_size))
    board = np.stack((board, opp_board, apple_board))
    return board


if __name__ == "__main__":
    from snake import Snake

    board_size = 10
    game = Snake(players=1, board_size=board_size)
    board = gameboard_to_array(game, 0, board_size=board_size)
    print(game.players[0])
    print(game.apple)
    print(board)
