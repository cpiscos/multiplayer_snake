from collections import deque
from collections import namedtuple
from multiprocessing import Queue, Pipe
from time import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from dqn_utils import PrioritizedReplayBuffer, calculate_iqn_loss
from env_vectorized_np import Snake
from models import NoisyIQN1
from multiprocessing_utils import buffer_worker, collector_worker

Models = namedtuple('models', ('model', 'target'))


def make_env():
    return Snake(2048, board_size=39, terminal_step=400)


def make_buffer():
    return PrioritizedReplayBuffer(int(2e6), alpha=0.5)


if __name__ == "__main__":
    mp.set_start_method('spawn')

    write = True
    load = True
    n_step = 3
    gamma = 0.99
    min_buffer = 100000
    prefetch_batches = 16
    batch_size = 128
    num_collectors = 1

    model = NoisyIQN1().cuda()
    target_model = NoisyIQN1().cuda()
    if load:
        model.load_state_dict(torch.load('weights.pt', map_location='cuda'))
    model.share_memory()
    target_model.share_memory()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3, eps=1.5e-7)
    if write:
        writer = SummaryWriter(comment="-dist")

    q = Queue(maxsize=2)

    collector_process = [mp.Process(target=collector_worker, args=(make_env, n_step, gamma, model, target_model, q)) for
                         _ in range(num_collectors)]
    for p in collector_process:
        p.daemon = True
        p.start()

    parent_end, remote_end = Pipe()
    buffer_process = mp.Process(target=buffer_worker, args=(remote_end, parent_end, make_buffer, q))
    buffer_process.daemon = True
    buffer_process.start()
    remote_end.close()

    step = 0
    losses = deque()
    target_model.load_state_dict(model.state_dict())
    total_prefetched_frames = prefetch_batches * batch_size
    while True:
        start = time()
        parent_end.send(("len", None))
        buffer_len = parent_end.recv()
        beta = min(1.0, 0.4 + 0.6 * (step * prefetch_batches * batch_size / 2e6))
        if buffer_len > min_buffer:
            parent_end.send(("batch", (total_prefetched_frames, beta)))
            batch = parent_end.recv()
            states, actions, returns, next_states, next_invalids, next_terminals, weights, idx = batch
            states = torch.FloatTensor(states).cuda()
            actions = torch.LongTensor(actions).cuda()
            returns = torch.FloatTensor(returns).cuda()
            next_states = torch.FloatTensor(next_states).cuda()
            next_invalids = torch.BoolTensor(next_invalids).cuda()
            next_terminals = torch.BoolTensor(next_terminals).cuda()
            weights = torch.FloatTensor(weights).cuda()
            priorities = []
            for i in range(prefetch_batches):
                states_, actions_, returns_, next_states_, next_invalids_, next_terminals_, weights_ = map(
                    lambda x: x[batch_size * i:batch_size * i + batch_size],
                    (states, actions, returns, next_states, next_invalids, next_terminals, weights))
                loss = calculate_iqn_loss(model, target_model, states_, actions_, returns_, next_states_,
                                          next_invalids_, next_terminals_, gamma, n_step)
                priorities.extend(loss.tolist())
                loss = (loss * weights_).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            parent_end.send(("update_priorities", (idx, priorities)))
            step += 1

            if step % 20 == 0:
                parent_end.send(("return_avg", None))
                avg_reward = parent_end.recv()
                parent_end.send(("resize", None))
                if write:
                    # batches_processed = step * prefetch_batches
                    torch.save(model.state_dict(), "weights.pt")
                    if not isinstance(avg_reward, type(None)):
                        writer.add_scalar("Reward/mean_reward", avg_reward, step)
                    writer.add_scalar("Loss/loss", np.mean(losses), step)
                    writer.add_scalar("Verbose/out1_avg_noise_weight",
                                      model.out1.weight_sigma.detach().abs().mean().item(), step)
                    writer.add_scalar("Verbose/out2_avg_noise_weight",
                                      model.out2.weight_sigma.detach().abs().mean().item(), step)
                    writer.add_scalar("Verbose/steps_per_sec", time() - start, step)
                losses.clear()

            if step % 50 == 0:
                target_model.load_state_dict(model.state_dict())
        print(step, buffer_len, time() - start)
