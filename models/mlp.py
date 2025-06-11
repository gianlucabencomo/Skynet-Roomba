import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

from .helper import *

from collections import deque

HIDDEN_WIDTHS = [
    # 2-layer MLPs
    [64, 64], [64, 128], [128, 64], [128, 128], [128, 256], [256, 128], [256, 256],
    # 3-layer MLPs
    [64, 64, 64], [64, 64, 128], [64, 128, 64], [128, 64, 64], [64, 128, 128], 
    [128, 128, 64], [64, 128, 256], [256, 128, 64], [128, 128, 128], [128, 128, 256], 
    [128, 256, 128], [256, 128, 128], [128, 256, 256], [256, 256, 128], [256, 256, 256],
]

def atanh(x, eps=1e-6):
    return 0.5 * torch.log((1 + x.clamp(-1 + eps, 1 - eps)) / (1 - x.clamp(-1 + eps, 1 - eps)))

class MlpContinuousActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_hidden_widths: list = HIDDEN_WIDTHS[3],
        critic_hidden_widths: list = HIDDEN_WIDTHS[3],
        normalize_obs: bool = True,
    ):
        super().__init__()
        self.n_actor_layers = len(actor_hidden_widths)
        self.n_critic_layers = len(critic_hidden_widths)
        self.normalize = NormalizeAndClip(obs_dim) if normalize_obs else nn.Identity()
        self.input_dim = obs_dim
        
        # No layer norm in critic
        critic_layers = []
        critic_layers.append(layer_init(nn.Linear(self.input_dim, critic_hidden_widths[0])))
        critic_layers.append(nn.ReLU())
        for i in range(self.n_critic_layers - 1):
            critic_layers.append(layer_init(nn.Linear(critic_hidden_widths[i], critic_hidden_widths[i + 1])))
            critic_layers.append(nn.ReLU())
        critic_layers.append(layer_init(nn.Linear(critic_hidden_widths[-1], 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

        actor_layers = []
        actor_layers.append(layer_init(nn.Linear(self.input_dim, actor_hidden_widths[0])))
        actor_layers.append(nn.LayerNorm(actor_hidden_widths[0]))
        actor_layers.append(nn.ReLU())
        for i in range(self.n_actor_layers - 1):
            actor_layers.append(layer_init(nn.Linear(actor_hidden_widths[i], actor_hidden_widths[i + 1])))
            actor_layers.append(nn.LayerNorm(actor_hidden_widths[i + 1]))
            actor_layers.append(nn.ReLU())
        actor_layers.append(layer_init(nn.Linear(actor_hidden_widths[-1], action_dim), std=0.01))
        self.actor_mean = nn.Sequential(*actor_layers)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
    
    def get_value(self, x):
        x = self.normalize(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.normalize(x)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            raw_action = dist.rsample()  # Reparameterized sample
            action = torch.tanh(raw_action)

            # Log prob with correction for tanh squashing
            log_prob = dist.log_prob(raw_action).sum(-1)
            log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)
        else:
            # Action passed in is assumed to be already squashed
            raw_action = atanh(action.clamp(-0.999, 0.999))
            log_prob = dist.log_prob(raw_action).sum(-1)
            log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)

        entropy = dist.entropy().sum(-1)
        value = self.critic(x)

        return action, log_prob, entropy, value

    def actor_params(self):
        return list(self.actor_mean.parameters()) + [self.actor_logstd]

    def critic_params(self):
        return list(self.critic.parameters())