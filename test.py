from dqn_utils import PrioritizedReplayBuffer
from multiprocessing_utils import Collector
from env_vectorized_np import Snake
from models import NoisyIQN1, NoisyIQN
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import sys
import os
import psutil
from time import time

def make_env():
    return Snake(1024, board_size=39)

if __name__ == "__main__":
    model = NoisyIQN1()
    model1 = NoisyIQN()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model1.parameters() if p.requires_grad))
    model(torch.randn(1, 7, 39, 39))
#     buffer = PrioritizedReplayBuffer()
#     buffer1 = PrioritizedReplayBuffer()
#     start = time()
#     col = Collector(make_env, 3, 0.99, model, model)
# #     print(time()-start)
#     buffer.push(*col.get_step())
#     # buffer1.merge(*buffer.flush())
#     # col.get_avg_rewards()
#
# # def worker(model):
# #     while True:
# #         print(model.weight.data)
# #         sys.stdout.flush()
# #
# # if __name__ == '__main__':
# #     mp.set_start_method('spawn')
# #
# #     model = nn.Linear(10, 10).cuda()
# #     target_model = nn.Linear(10, 10).cuda()
# #     target_model.share_memory()
# #     target_model.load_state_dict(model.state_dict())
# #     p = mp.Process(target=worker, args=(target_model,))
# #     p.daemon = True
# #     p.start()
# #     while True:
# #         model.weight.data.zero_().add_(1)
# #         target_model.load_state_dict(model.state_dict())
#
#
#
# #
# #
#
# # print(model[0].weight.data.zero_())
#
#
# # env = Snake(10)
# # buffer = PrioritizedReplayBuffer()
# # action = torch.zeros(10, dtype=torch.int64)
# # state, invalid, reward, done = env.step(action)
# # buffer.push(state, action, reward, state, invalid, done)
# #
# # states, actions, rewards, next_states, next_invalids, terminals, idx, weights = buffer.batch(2)
