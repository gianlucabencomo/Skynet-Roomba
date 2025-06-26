import torch
import torch.nn as nn

import numpy as np

def atanh(x, eps=1e-6):
    return 0.5 * torch.log(
        (1 + x.clamp(-1 + eps, 1 - eps)) / (1 - x.clamp(-1 + eps, 1 - eps))
    )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def lstm_init(lstm, std=1.0, bias_const=0.0):
    for name, param in lstm.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, bias_const)
        elif "weight" in name:
            nn.init.orthogonal_(param, std)
    return lstm


class NormalizeAndClip(nn.Module):
    """Normalization and Clip layer for NNs that can be saved with .pt files for simplicity.
    
    Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py."""

    def __init__(self, obs_dim, clip_range=10.0, epsilon=1e-8):
        super().__init__()
        self.obs_dim = obs_dim
        self.clip_range = clip_range
        self.epsilon = epsilon

        self.register_buffer("mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(obs_dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def update(self, x):
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_count = x.size(0)

            old_count = self.count.item()

            self.count.add_(batch_count)
            alpha = batch_count / self.count

            delta = batch_mean - self.mean
            self.mean.add_(delta * alpha)

            m_a = self.var * old_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta.pow(2) * (old_count * batch_count / self.count)
            self.var = M2 / self.count

    def forward(self, x):
        assert x.dim() == 2, f"Expected input tensor with 2 dimensions, got {x.dim()}"
        # update obs stats when model.train()
        if self.training:
            self.update(x)
        # Normalize and then clip
        normalized = (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)
        clipped = torch.clamp(normalized, -self.clip_range, self.clip_range)
        return clipped
