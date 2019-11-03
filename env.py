import numpy as np

from snake import Snake
from utils import gameboard_to_array


class TrainEnvSingle:
    def __init__(self):
        self.game = Snake(players=1)
        self.prev_board = np.zeros((2, 40, 40))
        self.steps_since_apple = 0

    def _cat_prev_board(self, board, save=True):
        state = np.concatenate((board, self.prev_board), 0)
        if save:
            self.prev_board = board[:2]
        return state

    def _reset(self):
        self.game.__init__(players=1)
        self.prev_board = np.zeros((2, 40, 40))
        self.steps_since_apple = 0
        return self.reset()

    def reset(self):
        new_board = gameboard_to_array(self.game, 0)
        invalid_actions = self._invalid_actions_mask()
        return self._cat_prev_board(new_board), invalid_actions

    def _invalid_actions_mask(self):
        invalid_action = (self.game.facing[0] + 2) % 4
        invalid_actions = [False] * 4
        invalid_actions[invalid_action] = True
        return invalid_actions

    def action(self, action):
        self.steps_since_apple += 1
        reward, apple_got, invalid, died = self.game.action(action, 0)
        new_board = gameboard_to_array(self.game, 0)
        state = self._cat_prev_board(new_board, not invalid)
        invalid_actions = self._invalid_actions_mask()
        # if self.steps_since_apple == ((len(self.game.players[0]) // 40) + 1) * 200:
        if self.steps_since_apple == 1000:
            died = True
        if died:
            state, _ = self._reset()
            invalid_actions = self._invalid_actions_mask()
        if apple_got:
            self.steps_since_apple = 0
            died = True
            # self.prev_board = np.zeros((2, 40, 40))
            # self.game.new_apple()
        # if self.steps_since_apple == 10:
        #     self.prev_board = np.zeros((2, 40, 40))
        #     self.game.__init__(players=1)
        #     self.steps_since_apple = 0
        return state, reward, invalid_actions, died


class TrainEnvMultiple:
    def __init__(self, players):
        assert players > 1
        self.players = players
        self.game = Snake(players=players)
        self.turn = 0

    def reset(self):
        return gameboard_to_array(self.game, 0)

    def action(self, action):
        player = self.turn // self.players
        reward, _, invalid = self.game.action(action, player)
        if not invalid:
            self.turn += 1
            return gameboard_to_array(self.game, player), reward

