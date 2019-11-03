from collections import deque

import numpy as np


class Snake:
    # actions: 0 = Right, 1 = Down, 2 = Left, 3 = Up
    def __init__(self, board_size=40, players=1):
        self.board_size = board_size
        self.crashed_scores = [0]*players
        self.player_count = players
        self.players = [deque([])] * players
        self.facing = [0]*players
        self.apple = None

        self.new_apple()
        for i in range(players):
            self.new_snake(i)

    def get_filled_coords(self):
        filled_coords = [self.apple]
        for player in self.players:
            filled_coords.extend(player)
        return filled_coords

    def new_apple(self):
        filled_coords = self.get_filled_coords()
        coord = np.random.randint(0, self.board_size, 2, dtype=np.int8).tolist()
        if coord in filled_coords:
            self.new_apple()
        else:
            self.apple = coord

    def new_snake(self, player, length=5):
        self.players[player].clear()
        filled_coords = self.get_filled_coords()
        body_length = length - 1
        coords = self.players[player]
        coord = np.random.randint(body_length, self.board_size - body_length, 2, dtype=np.int8)
        col = np.random.randint(2, dtype=np.int8)
        dir = np.random.choice((1, -1)).astype(np.int8)
        coords.append(coord.tolist())
        clear = True
        for _ in range(body_length):
            if coords in filled_coords:
                clear = False
                break
            coord[col] += dir
            coords.append(coord.tolist())

        if clear:
            if col == 0 and dir == 1:
                self.facing[player] = 2
            elif col == 0 and dir == -1:
                self.facing[player] = 0
            elif col == 1 and dir == 1:
                self.facing[player] = 1
            else:
                self.facing[player] = 3
        else:
            self.new_snake(player)

    def action(self, action, player):
        """
        :param action:
        :param player:
        :return: reward, apple taken, invalid move
        """
        reward = 0
        apple_got = False
        invalid = False
        crashed = False
        if action == ((self.facing[player] + 2) % 4):
            print("invalid action")
            invalid = True
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
            self.facing[player] = action
            player_coords = self.players[player]
            head = player_coords[0].copy()
            head[col] += dir
            player_coords.appendleft(head)
            if head == self.apple:
                self.new_apple()
                apple_got = True
                reward += 1
            else:
                player_coords.pop()
            self.players[player] = player_coords
            crashed = False
            if not 0 <= head[0] < self.board_size or not 0 <= head[1] < self.board_size or head in list(
                    player_coords)[1:]:
                crashed = True
            for i, p in enumerate(self.players):
                if i != player and head in p:
                    self.crashed_scores[i] += 1
                    crashed = True
            if crashed:
                self.new_snake(player)
                reward += -1
            reward += self.crashed_scores[player]
            self.crashed_scores[player] = 0
        return reward, apple_got, invalid, crashed


if __name__ == "__main__":
    game = Snake(players=1)
    print(game.players)
    print(game.facing)
    game.action(0, 0)
    print(game.players)
