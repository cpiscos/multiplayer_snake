import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from layers import NoisyLinear


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(7, 32, 5, 2), nn.ReLU(),
                                     nn.Conv2d(32, 64, 5, 5), nn.ReLU(),
                                     nn.Conv2d(64, 128, 3), nn.ReLU())

        self.actor = nn.Sequential(nn.Linear(128, 4))

        self.critic = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, mask=None):
        x = self.encoder(x).squeeze(3).squeeze(2)
        logits = self.actor(x)
        if not isinstance(mask, type(None)):
            logits = logits.masked_fill(mask, -100)
        dist = Categorical(logits=logits)
        value = self.critic(x).squeeze(-1)
        return dist, value


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(7, 32, 4, 2), nn.ReLU(),
                                     nn.Conv2d(32, 64, 5, 2), nn.ReLU(),
                                     nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
                                     nn.Conv2d(128, 128, 3), nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                       nn.Linear(64, 4))
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                   nn.Linear(64, 1))

    def forward(self, x, advantage=False, return_value=False):
        x = self.encoder(x).squeeze(3).squeeze(2)
        adv = self.advantage(x)
        if not advantage:
            val = self.value(x)
            q_values = val + adv - adv.mean(1).unsqueeze(1)
            if return_value:
                return q_values, val
            else:
                return q_values
        else:
            return adv


class NoisyIQN1(nn.Module):
    def __init__(self, test=False):
        super().__init__()
        self.test = test
        # self.encoder = nn.Sequential(nn.Conv2d(7, 32, 3, 1), nn.ReLU(),
        #                              nn.Conv2d(32, 64, 5, 4), nn.ReLU(),
        #                              nn.Conv2d(64, 128, 5, 4), nn.ReLU(),
        #                              nn.Conv2d(128, 128, 2), nn.ReLU())
        self.encoder = nn.Sequential(nn.Conv2d(7, 32, 3, 2), nn.ReLU(),
                                     nn.Conv2d(32, 64, 5, 2), nn.ReLU(),
                                     nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
                                     nn.Conv2d(128, 128, 3), nn.ReLU())
        self.tau_encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.out1 = NoisyLinear(128, 64, test=test)
        self.out2 = NoisyLinear(64, 4, test=test)
        self.register_buffer('tau_range', torch.arange(64, dtype=torch.float32))

    def forward(self, x, quantiles=None, n_tau_samples=8):
        if not self.test:
            self.out1.reset_noise()
            self.out2.reset_noise()
        x = self.encoder(x)
        assert x.shape[-2:] == torch.Size([1, 1])
        x = x[:, None, :, 0, 0]
        if isinstance(quantiles, type(None)):
            quantiles = torch.rand(n_tau_samples, device=x.device)
        tau_embedding = torch.cos(self.tau_range * quantiles.unsqueeze(1).repeat((1, 64)) * math.pi)
        tau_encoded = self.tau_encoder(tau_embedding)
        x = x * tau_encoded
        x = torch.relu(self.out1(x))
        x = self.out2(x)
        return x, quantiles


class NoisyIQN(nn.Module):
    def __init__(self, test=False):
        super().__init__()
        self.test = test
        self.encoder = nn.Sequential(nn.Conv2d(7, 32, 4, 2), nn.ReLU(),
                                     nn.Conv2d(32, 64, 5, 2), nn.ReLU(),
                                     nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
                                     nn.Conv2d(128, 128, 3), nn.ReLU())
        self.tau_encoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.out1 = NoisyLinear(128, 64, test=test)
        self.out2 = NoisyLinear(64, 4, test=test)

        self.register_buffer('tau_range', torch.arange(64, dtype=torch.float32))

    def forward(self, x, quantiles=None, n_tau_samples=8):
        if not self.test:
            self.out1.reset_noise()
            self.out2.reset_noise()
        x = self.encoder(x)
        assert x.shape[-2:] == torch.Size([1, 1])
        x = x[:, None, :, 0, 0]
        if isinstance(quantiles, type(None)):
            quantiles = torch.rand(n_tau_samples, device=x.device)
        tau_embedding = torch.cos(self.tau_range * quantiles.unsqueeze(1).repeat((1, 64)) * math.pi)
        tau_encoded = self.tau_encoder(tau_embedding)
        x = x * tau_encoded
        x = torch.relu(self.out1(x))
        x = self.out2(x)
        return x, quantiles


if __name__ == "__main__":
    model = NoisyIQN()
    sample_state = torch.randn(12, 7, 40, 40)
    model(sample_state)
