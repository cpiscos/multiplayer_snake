from collections import deque, namedtuple

import torch
import numpy as np

Transition = namedtuple("transition", ("state", "action", "reward", "terminal", "next_state", "next_invalid"))


class ReplayBuffer:
    def __init__(self, maxlen=100000):
        self.buffer = deque(maxlen=maxlen)

    def push(self, states, invalids, actions, rewards, next_states, terminals):
        for i in range(states.shape[0]):
            transition = states[i], invalids[i], actions[i], rewards[i], next_states[i], terminals[i]
            self.buffer.append(transition)

    def batch(self, batch_size=64):
        buffer = list(self.buffer)
        idx = torch.randint(len(self.buffer), (batch_size,))
        states, invalids, actions, rewards, next_states, dones = map(torch.stack, zip(*[buffer[i] for i in idx]))
        states = states.float().cuda()
        invalids = invalids.bool().cuda()
        actions = actions.long().cuda()
        rewards = rewards.float().cuda()
        next_states = next_states.float().cuda()
        dones = dones.float().cuda()
        return states, invalids, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, maxlen: int = int(1e6), alpha=0.6):
        self.maxlen = maxlen
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.state_shape = (7, 40, 40)

    def __len__(self):
        return len(self.buffer)

    @staticmethod
    def _to_sparse(x):
        idc = x.nonzero()
        idc = tuple(map(lambda x: x.astype(np.uint8), idc))
        return idc

    def _to_array(self, x):
        array = np.zeros(self.state_shape, dtype=bool)
        array[x] = 1
        return array

    def push(self, states, actions, rewards, next_states, next_invalids, terminals, priorities=None):
        if isinstance(priorities, type(None)):
            if len(self.buffer) == 0:
                priorities = torch.ones(states.shape[0])
            else:
                priorities = torch.tensor(self.priorities).max().repeat(states.shape[0])
        # actions = actions.cpu().numpy().astype(np.uint8)
        # rewards = rewards.cpu().numpy().astype(np.float32)
        # next_invalids = next_invalids.cpu().numpy().astype(bool)
        # terminals = terminals.cpu().numpy().astype(bool)
        # states = states.cpu().numpy()
        # next_states = next_states.cpu().numpy()
        for i in range(states.shape[0]):
            state, action, reward, next_state, next_invalid, terminal = states[i], actions[i], rewards[i], next_states[i], next_invalids[i], terminals[i]
            state = self._to_sparse(state)
            next_state = self._to_sparse(next_state)
            transition = state, action, reward, next_state, next_invalid, terminal
            self.buffer.append(transition)
            self.priorities.append(priorities[i].item())

    def batch(self, batch_size=64, beta=0.4):
        priorities = np.array(self.priorities)

        p = priorities ** self.alpha
        p /= p.sum()

        idx = np.random.choice(len(self.buffer), batch_size, p=p)
        weights = (len(self.buffer) * p[idx]) ** -beta
        weights /= weights.max()

        states, actions, rewards, next_states, next_invalids, terminals = [], [], [], [], [], []
        for i in idx:
            state, action, reward, next_state, next_invalid, terminal = self.buffer[i]
            state = self._to_array(state)
            next_state = self._to_array(next_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_invalids.append(next_invalid)
            terminals.append(terminal)

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        next_invalids = np.stack(next_invalids)
        terminals = np.stack(terminals)

        # states = states.numpy()
        # actions = actions.long().numpy()
        # rewards = rewards.float().numpy()
        # next_states = next_states.numpy()
        # next_invalids = next_invalids.numpy()
        # terminals = terminals.numpy()
        idx = idx.tolist()
        # weights = weights.numpy()

        return states, actions, rewards, next_states, next_invalids, terminals, weights, idx

    def update_priorities(self, idx, priorities):
        for i, priority in zip(idx, priorities):
            self.priorities[i] = priority

    def cut_to_maxlen(self):
        if len(self.buffer) >= self.maxlen:
            del self.buffer[:-self.maxlen]
            del self.priorities[:-self.maxlen]

    def flush(self):
        buffer = self.buffer.copy()
        priorities = self.priorities.copy()
        self.buffer.clear()
        self.priorities.clear()
        return buffer, priorities

    def merge(self, buffer, priorities):
        self.buffer.extend(buffer)
        self.priorities.extend(priorities)

