from time import time

import numpy as np
import torch
import torch.nn.functional as F

from filters import SNAKE_FILTERS, MOVEMENT_FILTERS


class Snake:
    def __init__(self, env_size, player_size=1, board_size=40, terminal_step=None):
        self.env_size = env_size
        self.board_size = board_size
        self.player_size = player_size
        self.terminal_step = terminal_step
        if player_size > 1:
            raise NotImplementedError

        self.apple_idx = 0
        self.head_idx = 1
        self.body_idx = self.head_idx + max(player_size, 2)
        self.board = np.zeros((env_size, (1 + 2 * max(player_size, 2)), board_size, board_size), dtype=np.float32)
        self.facing = np.zeros(env_size, dtype=np.uint8)
        self.steps_since_apple = np.zeros(env_size, dtype=np.int32)
        self.generate_snake()
        self.generate_apple()

    def reset(self):
        return self._get_observation(), self._get_invalid_actions()

    def generate_snake(self, env_indices=None):
        if isinstance(env_indices, type(None)):
            env_indices = np.arange(self.env_size)
        coo_indices = (np.arange(len(env_indices)),
                       np.random.randint(1, self.board_size - 2, len(env_indices)),
                       np.random.randint(1, self.board_size - 2, len(env_indices)))
        arr = np.zeros((len(env_indices), self.board_size, self.board_size))
        arr[coo_indices] = 1
        body = F.conv2d(torch.FloatTensor(arr[:, None]), SNAKE_FILTERS, padding=1).numpy()
        facing = np.random.randint(0, 4, len(env_indices))
        body = body[np.arange(len(env_indices)), facing][:, None]
        self.board[env_indices, self.body_idx:self.body_idx + 1] = body
        self.board[env_indices, self.head_idx:self.head_idx + 1] = body == 3
        self.facing[env_indices] = facing

    def generate_apple(self, env_indices=None):
        if isinstance(env_indices, type(None)):
            env_indices = np.arange(self.env_size)
        open_board = self.board[env_indices, 1:].max(1) == 0
        open_board = open_board.reshape(len(env_indices), -1)
        apple_idx_one_dim = torch.multinomial(torch.FloatTensor(open_board), 1).squeeze(-1).numpy()
        apple_board = np.zeros((len(env_indices), open_board.shape[1]))
        apple_board[np.arange(len(env_indices)), apple_idx_one_dim] = 1
        apple_board = apple_board.reshape((len(env_indices), 1, self.board_size, self.board_size))
        self.board[env_indices, :1] = apple_board

    def _get_observation(self):
        obs = self.board
        obs = np.concatenate((obs > 0, obs[:, -2:] == 1), 1)
        return obs.astype(bool)

    def _get_invalid_actions(self):
        invalid_idx = (self.facing + 2) % 4
        invalid = np.zeros((self.env_size, 4), dtype=bool)
        invalid[np.arange(self.env_size), invalid_idx.astype(np.uint8)] = True
        return invalid

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, int):
            actions = np.array([actions])
        actions = actions.astype(np.uint8)
        reward = np.zeros(self.env_size)
        done = np.zeros(self.env_size, dtype=bool)
        heads = self.board[:, self.head_idx:self.head_idx + 1]
        apples = self.board[:, self.apple_idx:self.apple_idx + 1]
        bodies = self.board[:, self.body_idx:self.body_idx + 1]
        head = F.conv2d(torch.FloatTensor(heads), MOVEMENT_FILTERS, padding=1).numpy()
        head = head[np.arange(self.env_size), actions][:, None]
        head_exists_idx = np.nonzero(head)
        self.steps_since_apple += 1
        if len(head_exists_idx[0]) > 0:
            apple_got = apples[head_exists_idx].astype(bool)
            apple_got_env_idx = head_exists_idx[0][apple_got]
            self.steps_since_apple[apple_got_env_idx] = 0

            collision = bodies[head_exists_idx] > 1
            collision_idx = head_exists_idx[0][collision]
            done[collision_idx] = 1
            reward[collision_idx] = -1

            not_apple_got_env_idx = np.ones(self.env_size)
            not_apple_got_env_idx[apple_got_env_idx] = 0
            not_apple_got_env_idx = not_apple_got_env_idx.nonzero()[0]

            bodies_not_apple_got = bodies[not_apple_got_env_idx]
            bodies_not_apple_got[bodies_not_apple_got > 0] -= 1
            bodies[not_apple_got_env_idx] = bodies_not_apple_got

            snake_len = bodies[head_exists_idx[0], 0]
            snake_len = snake_len.max(1).max(1)
            bodies[head_exists_idx] = snake_len + 1

            self.board[:, self.head_idx:self.head_idx + 1] = head
            self.board[:, self.body_idx:self.body_idx + 1] = bodies
            if len(apple_got_env_idx) > 0:
                reward[apple_got_env_idx] = 1
                self.generate_apple(apple_got_env_idx)

        no_head_exists_idx = np.ones(self.env_size)
        no_head_exists_idx[head_exists_idx[0]] = 0
        no_head_exists_idx = no_head_exists_idx.nonzero()[0]
        done[no_head_exists_idx] = 1
        reward[no_head_exists_idx] = -1
        if not isinstance(self.terminal_step, type(None)):
            term_reached_idx = (self.steps_since_apple == self.terminal_step).nonzero()[0]
            done[term_reached_idx] = 1

        done_idx = np.nonzero(done)[0]
        self.facing = actions
        if len(done_idx) > 0:
            self.generate_snake(env_indices=done_idx)
            self.generate_apple(env_indices=done_idx)
            self.steps_since_apple[done_idx] = 0
        return self._get_observation(), self._get_invalid_actions(), reward, done


class MultSnake:
    def __init__(self, env_size, player_size=2, board_size=40, terminal_step=None):
        self.env_size = env_size
        self.board_size = board_size
        self.player_size = player_size
        self.terminal_step = terminal_step
        if player_size < 2:
            raise NotImplementedError

        self.apple_idx = 0
        self.head_idx = 1
        self.body_idx = self.head_idx + player_size
        self.board = np.zeros((env_size, (1 + 2 * player_size), board_size, board_size), dtype=np.float32)
        self.facing = np.zeros((env_size, player_size), dtype=np.uint8)
        self.kills_since_step = np.zeros((env_size, player_size))
        self.steps_since_apple = np.zeros((env_size, player_size), dtype=np.int32)
        self.player = 0
        [self.generate_snake(i) for i in range(self.player_size)]
        self.generate_apple()

    def reset(self):
        return self._get_observation(), self._get_invalid_actions()

    def _get_not_player_indices(self, player):
        not_player = np.arange(self.player_size)
        not_player = np.delete(not_player, player)
        return not_player


    def _get_empty(self, player, env_indices):
        not_player = self._get_not_player_indices(player)
        not_player_board = self.board[:, [0, *(self.body_idx + not_player).tolist()]].max(1)[:, None]
        not_player_board = F.conv2d(torch.FloatTensor(not_player_board), torch.ones(1, 1, 3, 3, dtype=torch.float32)).squeeze(1).numpy()
        not_filled = not_player_board.max(1) == 0
        return not_filled

    def generate_snake(self, player, env_indices=None):
        if isinstance(env_indices, type(None)):
            env_indices = np.arange(self.env_size)
        not_filled = self._get_empty(player, env_indices)
        not_filled = not_filled.reshape(len(env_indices), -1)
        seeds_idx = torch.multinomial(torch.FloatTensor(not_filled), 1).squeeze(1).numpy()
        seeds = np.zeros((len(env_indices), (self.board_size-2)**2))
        seeds[env_indices, seeds_idx] = 1
        body_seeds = np.zeros((len(env_indices), self.board_size, self.board_size))
        body_seeds[:, 1:-1, 1:-1] = seeds.reshape((len(env_indices), self.board_size-2, self.board_size-2))
        body = F.conv2d(torch.FloatTensor(body_seeds[:, None]), SNAKE_FILTERS, padding=1).numpy()
        facing = np.random.randint(0, 4, len(env_indices), dtype=np.uint8)
        body = body[np.arange(len(env_indices)), facing]
        self.board[env_indices, self.body_idx+player] = body
        self.board[env_indices, self.head_idx+player] = body == 3
        self.facing[env_indices, player] = facing

    def generate_apple(self, env_indices=None):
        if isinstance(env_indices, type(None)):
            env_indices = np.arange(self.env_size)
        open_board = self.board[env_indices, 1:].max(1) == 0
        open_board = open_board.reshape(len(env_indices), -1)
        apple_idx_one_dim = torch.multinomial(torch.FloatTensor(open_board), 1).squeeze(-1).numpy()
        apple_board = np.zeros((len(env_indices), open_board.shape[1]))
        apple_board[np.arange(len(env_indices)), apple_idx_one_dim] = 1
        apple_board = apple_board.reshape((len(env_indices), 1, self.board_size, self.board_size))
        self.board[env_indices, :1] = apple_board

    def _get_observation(self):
        apple = self.board[:, self.apple_idx:self.apple_idx+1].astype(bool)
        heads = self.board[:, self.head_idx:self.head_idx+self.player_size].astype(bool)
        bodies = self.board[:, self.body_idx:self.body_idx+self.player_size]
        heads[:, [0, self.player]] = heads[:, [self.player, 0]]
        bodies[:, [0, self.player]] = bodies[:, [self.player, 0]]
        heads = np.stack((heads[:, 0], heads[:, 1:].max(1)), 1)
        bodies = np.stack((bodies[:, 0], bodies[:, 1:].max(1)), 1)
        obs = np.concatenate((apple, heads, bodies.astype(bool), bodies == 1), 1)
        return obs

    def _get_invalid_actions(self):
        invalid_idx = (self.facing[np.arange(self.env_size), self.player] + 2) % 4
        invalid = np.zeros((self.env_size, 4), dtype=bool)
        invalid[np.arange(self.env_size), invalid_idx.astype(np.uint8)] = True
        return invalid

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, int):
            actions = np.array([actions])
        actions = actions.astype(np.uint8)
        reward = np.zeros(self.env_size)
        done = np.zeros(self.env_size, dtype=bool)
        apple = self.board[:, self.apple_idx][:, None]
        heads = self.board[:, self.head_idx+self.player][:, None]
        bodies = self.board[:, self.body_idx+self.player][:, None]
        all_bodies = self.board[:, self.body_idx:self.body_idx+self.player_size]
        head = F.conv2d(torch.FloatTensor(heads), MOVEMENT_FILTERS, padding=1).numpy()
        head = head[np.arange(self.env_size), actions][:, None]
        head_exists_idx = np.nonzero(head)
        not_player_idc = self._get_not_player_indices(self.player)
        self.steps_since_apple += 1
        if len(head_exists_idx[0]) > 0:
            apple_got = apple[head_exists_idx].astype(bool)
            apple_got_env_idx = head_exists_idx[0][apple_got]
            self.steps_since_apple[apple_got_env_idx] = 0

            collision = (head.astype(bool) & all_bodies.astype(bool)).any(2).any(2)
            self.kills_since_step[collision] += 1
            collision_self = collision[:, self.player]
            collision_idx = head_exists_idx[0][collision_self]
            done[collision_idx] = 1
            reward[collision_idx] = -1

            not_apple_got_env_idx = np.ones(self.env_size)
            not_apple_got_env_idx[apple_got_env_idx] = 0
            not_apple_got_env_idx = not_apple_got_env_idx.nonzero()[0]

            bodies_not_apple_got = bodies[not_apple_got_env_idx]
            bodies_not_apple_got[bodies_not_apple_got > 0] -= 1
            bodies[not_apple_got_env_idx] = bodies_not_apple_got

            snake_len = bodies[head_exists_idx[0], 0]
            snake_len = snake_len.max(1).max(1)
            bodies[head_exists_idx] = snake_len + 1

            self.board[:, self.head_idx:self.head_idx + 1] = head
            self.board[:, self.body_idx:self.body_idx + 1] = bodies
            if len(apple_got_env_idx) > 0:
                reward[apple_got_env_idx] = 1
                self.generate_apple(apple_got_env_idx)

        no_head_exists_idx = np.ones(self.env_size)
        no_head_exists_idx[head_exists_idx[0]] = 0
        no_head_exists_idx = no_head_exists_idx.nonzero()[0]
        done[no_head_exists_idx] = 1
        reward[no_head_exists_idx] = -1
        if not isinstance(self.terminal_step, type(None)):
            term_reached_idx = (self.steps_since_apple == self.terminal_step).nonzero()[0]
            done[term_reached_idx] = 1

        done_idx = np.nonzero(done)[0]
        self.facing[:, self.player] = actions
        if len(done_idx) > 0:
            self.generate_snake(self.player, env_indices=done_idx)
            self.steps_since_apple[done_idx] = 0
        return self._get_observation(), self._get_invalid_actions(), reward, done

    class Gridworld:
        def __init__(self, env_size, board_size=5):
            self.board = torch.zeros(env_size, 2, board_size, board_size)


if __name__ == "__main__":
    start = time()
    game = MultSnake(4, player_size=2, board_size=10)
    observation, invalid = game.reset()
    # print(observation)
    print("initialization time:", time() - start)

    start = time()
    observation, invalid, reward, done = game.step(1)
    # print(observation, invalid, reward, done)
    print("action step time:", time() - start)
