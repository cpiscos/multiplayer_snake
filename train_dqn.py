from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dqn_utils import PrioritizedReplayBuffer
from dqn_utils.utils import eps_greedy, calculate_iqn_loss
from env_vectorized import Snake
from models import NoisyIQN
from multiprocessing_utils import SubprocWrapper


def make_env(env_num):
    return Snake(env_num, board_size=40, terminal_step=200)


write = True
save = True
load = False
num_steps = 4
env_num = 128
worker_num = 2
gamma = 0.99
n_step = 3

envs_fns = [make_env for _ in range(worker_num)]
envs = SubprocWrapper(envs_fns, env_num)

if write:
    writer = SummaryWriter()
model = NoisyIQN()
model = model.cuda()

buffer = PrioritizedReplayBuffer(int(1e6), alpha=0.2)
# buffer = ReplayBuffer(100000)
opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1.5e-4)

if load:
    model.load_state_dict(torch.load("weights.pt"))

step = 0
n_states = []
n_invalids = []
n_actions = []
n_rewards = []
n_terminals = []

while True:
    time_steps = []
    avg_rewards = []

    if len(n_states) == 0:
        state, invalid = envs.reset()
        n_states.append(state)
        n_invalids.append(invalid)

    for i in range(num_steps):
        start_time = time()
        with torch.no_grad():
            q_vals, _ = model(n_states[-1].cuda(), n_tau_samples=8)
            action = eps_greedy(q_vals.mean(1), n_invalids[-1], eps=0)
        action = action.cpu()

        state, invalid, reward, terminal = envs.step(action)
        n_states.append(state)
        n_invalids.append(invalid)
        n_actions.append(action)
        n_rewards.append(reward)
        n_terminals.append(terminal)
        avg_rewards.append(reward.mean().item())
        if len(n_rewards) == n_step:
            returns_ = torch.zeros(reward.shape)
            for n in reversed(range(n_step)):
                returns_[n_terminals[n]] = 0
                returns_ = n_rewards[n] + gamma * returns_
            states = n_states[0]
            actions = n_actions[0]
            next_states = n_states[-1]
            next_invalids = n_invalids[-1]
            next_terminals = torch.stack(n_terminals).any(0)

            n_states.pop(0)
            n_invalids.pop(0)
            n_actions.pop(0)
            n_rewards.pop(0)
            n_terminals.pop(0)
            buffer.push(states, actions, returns_, next_states, next_invalids, next_terminals)
        time_steps.append(time() - start_time)

    losses = []
    if len(buffer) > 80000:
        for i in range(64):
            states, actions, rewards, next_states, next_invalids, next_terminals, idx, weights = buffer.batch(32, device="cuda")
            rho_loss = calculate_iqn_loss(model, states, actions, rewards, next_states, next_invalids, next_terminals, gamma, n_step)
            final_loss = (rho_loss * weights[:, None, None]).mean()
            priority = rho_loss.detach() + 1e-5

            opt.zero_grad()
            final_loss.backward()
            opt.step()
            buffer.update_priorities(idx, priority.tolist())
            losses.append(final_loss.item())

        if write:
            writer.add_scalar("Loss/loss", np.mean(losses), step)
            writer.add_scalar("Reward/mean_reward", np.mean(avg_rewards), step)
            writer.add_scalar("Verbose/steps_per_sec", env_num * worker_num / np.mean(time_steps), step)
            writer.add_scalar("Verbose/out1_avg_noise_weight", model.out1.weight_sigma.detach().abs().mean().item(), step)
            writer.add_scalar("Verbose/out2_avg_noise_weight", model.out2.weight_sigma.detach().abs().mean().item(), step)
        if save:
            model = model.cpu()
            torch.save(model.state_dict(), 'weights.pt')
            model = model.cuda()
        # import matplotlib.pyplot as plt
        # prios = buffer.priorities[:buffer.step] ** 0.15
        # prios /= prios.sum()
        # plt.plot(prios.sort()[0].numpy())
        # plt.show()
    step += 1
    print(step, len(buffer))
