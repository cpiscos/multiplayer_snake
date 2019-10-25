import numpy as np
from collections import deque


class Snake:
    # actions: 0 = Right, 1 = Down, 2 = Left, 3 = Up
    def __init__(self, board_size=40, players=1):
        self.board_size = board_size
        self.players = players
        self.players = [[]] * players
        self.facing = []
        self.apple = None

        self.new_apple()
        for i in range(players):
            self.new_snake(i)

    def get_filled_coords(self):
        filled_coords = [self.apple]
        for player in self.players:
            filled_coords.extend(player)
        return filled_coords

    def get_players_coords(self):
        players_coords = []
        for player in self.players:
            players_coords.extend(player)
        return players_coords

    def new_apple(self):
        filled_coords = self.get_filled_coords()
        coord = np.random.randint(0, self.board_size, 2).tolist()
        if coord in filled_coords:
            self.new_apple()
        else:
            self.apple = coord

    def new_snake(self, player, length=3):
        filled_coords = self.get_filled_coords()
        body_length = length - 1
        coords = deque([])
        coord = np.random.randint(body_length, self.board_size - body_length, 2)
        col = np.random.randint(2)
        dir = np.random.choice((1, -1))
        coords.append(coord.tolist())
        clear = True
        for _ in range(body_length):
            if coords[-1] in filled_coords:
                clear = False
                self.new_snake(player)
                break
            coord[col] += dir
            coords.append(coord.tolist())

        if clear:
            self.players[player] = coords
            if col == 0 and dir == 1:
                self.facing.append(2)
            elif col == 0 and dir == -1:
                self.facing.append(0)
            elif col == 1 and dir == 1:
                self.facing.append(1)
            else:
                self.facing.append(3)

    def action(self, action, player):
        if action == ((self.facing[player] + 2) % 4):
            print("invalid action")
        else:
            col = None
            dir = None
            if action == 0:
                col = 0
                dir = 1
            elif action == 1:
                col = 1
                dir = -1
            elif action == 2:
                col = 0
                dir = -1
            elif action == 3:
                col = 1
                dir = 1
            self.facing = action
            player_coords = self.players[player]
            head = player_coords[0].copy()
            head[col] += dir
            player_coords.appendleft(head)
            if head == self.apple:
                self.new_apple()
            else:
                player_coords.pop()
            if not 0 <= head[0] <= self.board_size or not 0 <= head[1] <= self.board_size:
                print("Player crashed")
                self.new_snake(player)
            self.players[player] = player_coords


if __name__ == "__main__":
    game = Snake(players=1)
    print(game.players)
    print(game.facing)
    game.action(0, 0)
    print(game.players)
