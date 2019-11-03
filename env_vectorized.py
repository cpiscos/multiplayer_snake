from time import time

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
        self.board = torch.zeros(env_size, (1 + 2 * max(player_size, 2)), board_size, board_size)
        self.facing = torch.zeros(env_size).long()
        self.steps_since_apple = torch.zeros(env_size).short()
        self.generate_snake()
        self.generate_apple()

    def reset(self):
        return self._get_observation(), self._get_invalid_actions()

    def generate_snake(self, player=None, env_indices=None):
        if type(env_indices) != torch.Tensor:
            env_indices = torch.arange(self.env_size)
        if type(player) != int:
            players = torch.arange(self.player_size)
        else:
            raise NotImplementedError
        coo_indices = torch.stack([torch.arange(len(env_indices)),
                                   torch.randint(1, self.board_size - 1, (len(env_indices),)),
                                   torch.randint(1, self.board_size - 1, (len(env_indices),))])
        body_seeds = torch.sparse_coo_tensor(coo_indices, torch.ones(len(env_indices)),
                                             (len(env_indices), self.board_size, self.board_size)).unsqueeze(1)
        body = F.conv2d(body_seeds.to_dense(), SNAKE_FILTERS, padding=1)
        facing = torch.randint(4, (len(env_indices),))
        body = body[torch.arange(len(env_indices)), facing].unsqueeze(1)
        self.board[env_indices, self.body_idx:self.body_idx + 1] = body
        self.board[env_indices, self.head_idx:self.head_idx + 1] = (body == 3).float()
        self.facing[env_indices] = facing

    def generate_apple(self, env_indices=None):
        if type(env_indices) != torch.Tensor:
            env_indices = torch.arange(self.env_size)
        open_board = self.board[env_indices, 1:].max(1)[0] == 0
        open_board = open_board.float().reshape(len(env_indices), -1)
        apple_idx_one_dim = torch.multinomial(open_board, 1).squeeze(-1)
        apple_board = torch.zeros((len(env_indices), open_board.shape[1]))
        apple_board[torch.arange(len(env_indices)), apple_idx_one_dim] = 1
        apple_board = apple_board.reshape(len(env_indices), 1, self.board_size, self.board_size)
        self.board[env_indices, :1] = apple_board

    def _get_observation(self):
        obs = self.board
        obs = torch.cat((obs, (obs[:, -2:] == 1).float()), 1)
        return (obs > 0).float()

    def _get_invalid_actions(self):
        invalid_idx = (self.facing + 2) % 4
        invalid = torch.zeros((self.env_size, 4), dtype=torch.bool)
        invalid[torch.arange(self.env_size), invalid_idx] = True
        return invalid

    def step(self, actions):
        if type(actions) != torch.Tensor:
            raise RuntimeError("actions must be of type torch.Tensor")
        reward = torch.zeros(self.env_size)
        done = torch.zeros(self.env_size)
        heads = self.board[:, self.head_idx:self.head_idx + 1]
        apples = self.board[:, self.apple_idx:self.apple_idx + 1]
        bodies = self.board[:, self.body_idx:self.body_idx + 1]
        head = F.conv2d(heads, MOVEMENT_FILTERS, padding=1)
        head = head[torch.arange(self.env_size), actions].unsqueeze(1)
        head_exists_idx = torch.nonzero(head).t()
        head_exists_idx = [head_exists_idx[i] for i in range(4)]
        self.steps_since_apple += 1
        if len(head_exists_idx[0]) > 0:
            apple_got = apples[head_exists_idx].bool()
            apple_got_env_idx = head_exists_idx[0][apple_got]
            self.steps_since_apple[apple_got_env_idx] = 0

            collision = bodies[head_exists_idx] > 1
            collision_idx = head_exists_idx[0][collision]
            done[collision_idx] = 1
            reward[collision_idx] = -1

            not_apple_got_env_idx = torch.ones(self.env_size)
            not_apple_got_env_idx[apple_got_env_idx] = 0
            not_apple_got_env_idx = not_apple_got_env_idx.nonzero().squeeze(1)

            bodies_not_apple_got = bodies[not_apple_got_env_idx]
            bodies_not_apple_got[bodies_not_apple_got > 0] -= 1
            bodies[not_apple_got_env_idx] = bodies_not_apple_got

            snake_len = bodies[head_exists_idx[0], 0]
            snake_len = snake_len.max(1)[0].max(1)[0]
            bodies[head_exists_idx] = snake_len + 1

            self.board[:, self.head_idx:self.head_idx + 1] = head
            self.board[:, self.body_idx:self.body_idx + 1] = bodies
            if len(apple_got_env_idx) > 0:
                reward[apple_got_env_idx] = 1
                self.generate_apple(apple_got_env_idx)

        no_head_exists_idx = torch.ones(self.env_size)
        no_head_exists_idx[head_exists_idx[0]] = 0
        no_head_exists_idx = no_head_exists_idx.nonzero().squeeze(1)
        done[no_head_exists_idx] = 1
        reward[no_head_exists_idx] = -1
        if not isinstance(self.terminal_step, type(None)):
            term_reached_idx = (self.steps_since_apple == self.terminal_step).nonzero().squeeze(1)
            done[term_reached_idx] = 1

        done_idx = torch.nonzero(done).squeeze(1)
        # reward = torch.zeros(self.env_size)
        # reward[actions==0] = -1
        self.facing = actions
        if len(done_idx) > 0:
            self.generate_snake(env_indices=done_idx)
            self.generate_apple(env_indices=done_idx)
            self.steps_since_apple[done_idx] = 0
        return self._get_observation(), self._get_invalid_actions(), reward, done.bool()

    class Gridworld:
        def __init__(self, env_size, board_size=5):
            self.board = torch.zeros(env_size, 2, board_size, board_size)




if __name__ == "__main__":
    start = time()
    game = Snake(128, board_size=40)
    obs, invalid = game.reset()
    print("initialization time:", time()-start)

    actions = torch.randint(4, (128,)).long()

    start = time()
    obs, invalid, reward, done = game.step(actions)
    print("action step time:", time()-start)
