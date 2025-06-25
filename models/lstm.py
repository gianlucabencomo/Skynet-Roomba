import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

from .helper import *


def atanh(x, eps=1e-6):
    return 0.5 * torch.log(
        (1 + x.clamp(-1 + eps, 1 - eps)) / (1 - x.clamp(-1 + eps, 1 - eps))
    )


class LSTMContinuousActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_width: int = 128,
        n_layers: int = 2,
        normalize_obs: bool = True,
    ):
        super().__init__()
        self.normalize = NormalizeAndClip(obs_dim) if normalize_obs else nn.Identity()

        self.shared = lstm_init(
            nn.LSTM(obs_dim, hidden_width, n_layers, batch_first=True)
        )
        self.actor_mean = layer_init(nn.Linear(hidden_width, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = layer_init(nn.Linear(hidden_width, 1), std=1.0)

    def get_states(self, x, state, done):
        B, T, _ = x.shape
        # normalize input
        x = self.normalize(x.view(B * T, -1)).view(B, T, -1)
        h, c = state
        outputs = []
        for t in range(T):
            done_t = done[:, t].float().view(1, B, 1)
            h = h * (1.0 - done_t)
            c = c * (1.0 - done_t)
            out_t, (h, c) = self.shared(x[:, t].unsqueeze(1), (h, c))
            outputs.append(out_t)

        output = torch.cat(outputs, dim=1)
        output = output.reshape(B * T, -1)
        return output, (h, c)

    def get_value(self, x, state, done):
        hidden, _ = self.get_states(x, state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, state, done, action=None):
        hidden, state = self.get_states(x, state, done)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            raw_action = dist.rsample()  # Reparameterized sample
            action = torch.tanh(raw_action)

            # Log prob with correction for tanh squashing
            log_prob = dist.log_prob(raw_action).sum(-1)
            log_prob -= (
                2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))
            ).sum(-1)
        else:
            # Action passed in is assumed to be already squashed
            raw_action = atanh(action.clamp(-0.999, 0.999))
            log_prob = dist.log_prob(raw_action).sum(-1)
            log_prob -= (
                2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))
            ).sum(-1)

        entropy = dist.entropy().sum(-1)
        value = self.critic(hidden)

        return action, log_prob, entropy, value, state
