from collections import deque

import numpy as np
import torch
from torch.multiprocessing import Pipe, Process

from dqn_utils import eps_greedy, calculate_iqn_loss, PrioritizedReplayBuffer


class Collector:
    def __init__(self, env_fn, n_step, gamma, model, target_model):
        self.env = env_fn()
        self.avg_rewards = deque()
        self.n_states = []
        self.n_invalids = []
        self.n_actions = []
        self.n_rewards = []
        self.n_terminals = []
        self.n_step = n_step
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.initialize_n_steps()

    def initialize_n_steps(self):
        state, invalid = self.env.reset()
        self.n_states.append(state)
        self.n_invalids.append(invalid)

        while len(self.n_rewards) < self.n_step:
            self.take_step()

    def take_step(self):
        with torch.no_grad():
            q_vals, _ = self.model(torch.FloatTensor(self.n_states[-1]).cuda(), n_tau_samples=8)
            action = eps_greedy(q_vals.mean(1).cpu(), torch.BoolTensor(self.n_invalids[-1]), eps=0)
        action = action.numpy()

        state, invalid, reward, terminal = self.env.step(action)
        self.n_states.append(state)
        self.n_invalids.append(invalid)
        self.n_actions.append(action)
        self.n_rewards.append(reward)
        self.n_terminals.append(terminal)
        self.avg_rewards.append(reward.mean().item())

    def get_step(self):
        returns = np.zeros(self.n_rewards[0].shape)
        for n in reversed(range(self.n_step)):
            returns[self.n_terminals[n]] = 0
            returns = self.n_rewards[n] + self.gamma * returns
        states = self.n_states[0]
        actions = self.n_actions[0]
        next_states = self.n_states[-1]
        next_invalids = self.n_invalids[-1]
        next_terminals = np.stack(self.n_terminals).any(0)

        del self.n_states[0]
        del self.n_invalids[0]
        del self.n_actions[0]
        del self.n_rewards[0]
        del self.n_terminals[0]
        self.take_step()

        with torch.no_grad():
            states_ = torch.FloatTensor(states).cuda()
            actions_ = torch.LongTensor(actions).cuda()
            returns_ = torch.FloatTensor(returns).cuda()
            next_states_ = torch.FloatTensor(next_states).cuda()
            next_invalids_ = torch.BoolTensor(next_invalids).cuda()
            next_terminals_ = torch.BoolTensor(next_terminals).cuda()
            priorities = calculate_iqn_loss(self.model, self.target_model, states_, actions_, returns_, next_states_,
                                            next_invalids_,
                                            next_terminals_, self.gamma, self.n_step).cpu()
        return states, actions, returns, next_states, next_invalids, next_terminals, priorities

    def get_avg_rewards(self):
        avg_rewards = np.mean(self.avg_rewards)
        self.avg_rewards.clear()
        return avg_rewards


def collector_worker(env_fn, n_step, gamma, model, target_model, queue):
    col = Collector(env_fn, n_step, gamma, model, target_model)
    buffer = PrioritizedReplayBuffer()
    step = 1
    while True:
        buffer.push(*col.get_step())
        step += 1
        if step % 25 == 0:
            queue.put((buffer.flush(), col.get_avg_rewards()), block=True)
            # proc = psutil.Process(os.getpid())
            # print("Collector", proc.memory_info().rss)
            # sys.stdout.flush()


def buffer_worker(remote, parent_remote, buffer_fn, queue):
    parent_remote.close()
    buffer = buffer_fn()
    avg_rewards = deque()

    while True:
        cmd, data = remote.recv()
        if cmd == "batch":
            batch = buffer.batch(*data)
            remote.send(batch)
        elif cmd == "update_priorities":
            buffer.update_priorities(*data)
        elif cmd == "len":
            remote.send(len(buffer))
        elif cmd == "resize":
            buffer.cut_to_maxlen()
        elif cmd == "return_avg":
            avg_reward = np.mean(avg_rewards) if len(avg_rewards) > 0 else None
            avg_rewards.clear()
            remote.send(avg_reward)
        if not queue.empty():
            merge, avg_reward = queue.get()
            avg_rewards.append(avg_reward)
            buffer.merge(*merge)


def worker(remote, parent_remote, env_fn, env_num):
    parent_remote.close()
    env = env_fn(env_num)
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            obs, invalid = env.reset()
            remote.send((obs, invalid))
        elif cmd == "step":
            obs, invalid, reward, done = env.step(data)
            remote.send((obs, invalid, reward, done))


class SubprocWrapper:
    def __init__(self, env_fns, env_num):
        self.num_workers = len(env_fns)
        self.parent_conns, self.remote_conns = zip(*[Pipe() for _ in range(self.num_workers)])
        self.ps = [Process(target=worker, args=(remote_conn, parent_conn, env_fn, env_num))
                   for (remote_conn, parent_conn, env_fn) in zip(self.remote_conns, self.parent_conns, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.remote_conns:
            remote.close()

    def reset(self):
        for pipe in self.parent_conns:
            pipe.send(("reset", None))
        obs, invalid = zip(*[pipe.recv() for pipe in self.parent_conns])
        return torch.cat(obs), torch.cat(invalid)

    def step(self, actions):
        actions = actions.reshape(self.num_workers, -1)
        for pipe, action in zip(self.parent_conns, actions):
            pipe.send(("step", action))
        obs, invalid, reward, done = zip(*[pipe.recv() for pipe in self.parent_conns])
        return torch.cat(obs), torch.cat(invalid), torch.cat(reward), torch.cat(done)
