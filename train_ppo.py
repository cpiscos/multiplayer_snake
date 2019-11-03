from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env_vectorized import Snake
from models import ActorCritic
from multiprocessing_utils import SubprocWrapper
from ppo_utils.utils import compute_gae, ppo_update


def make_env(env_num):
    return Snake(env_num, board_size=40, terminal_step=None)


write = True
save = True
load = False
num_steps = 200
env_num = 128
worker_num = 2

mini_batch_size = 64
ppo_epochs = 4
if write:
    writer = SummaryWriter()
model = ActorCritic()
if load:
    model.load_state_dict(torch.load("weights.pt"))
model = model.cuda()
opt = torch.optim.AdamW(model.parameters())

envs_fns = [make_env for _ in range(worker_num)]
envs = SubprocWrapper(envs_fns, env_num)

step = 0
while True:
    log_probs = []
    values = []
    states = []
    invalids = []
    rewards = []
    actions = []
    terminals = []
    time_steps = []

    state, invalid = envs.reset()
    for _ in range(num_steps):
        start_time = time()
        with torch.no_grad():
            dist, value = model(state.cuda(), invalid.cuda())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.cpu()
        log_probs.append(log_prob.cpu())
        values.append(value.cpu())
        states.append(state)
        invalids.append(invalid)
        state, invalid, reward, terminal = envs.step(action)
        rewards.append(reward)
        actions.append(action)
        terminals.append(terminal.bool())
        time_steps.append(time() - start_time)

    with torch.no_grad():
        _, next_value = model(state.cuda(), invalid.cuda())
    returns = compute_gae(next_value.cpu(), rewards, values, terminals)

    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    states = torch.cat(states)
    actions = torch.cat(actions)
    advantage = returns - values
    invalids = torch.cat(invalids)

    loss, actor_loss, critic_loss, entropy = ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs,
                                                        returns, advantage, invalids, model, opt)
    if write:
        writer.add_scalar("Loss/loss", loss, step)
        writer.add_scalar("Loss/actor_loss", actor_loss, step)
        writer.add_scalar("Loss/critic_loss", critic_loss, step)
        writer.add_scalar("Loss/entropy", entropy, step)
        writer.add_scalar("Reward/mean_reward", torch.stack(rewards).mean().item(), step)
        writer.add_scalar("Verbose/steps_per_sec", env_num * worker_num / np.mean(time_steps), step)
    if save:
        model = model.cpu()
        torch.save(model.state_dict(), 'weights.pt')
        model = model.cuda()
    step += 1
    print(step)
