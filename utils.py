import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import random

import copy


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on: {device}")
    return device


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class RunningMeanStd:
    """Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/normalize.py."""

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        x = x.detach()
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


def clone_policy(policy: nn.Module) -> nn.Module:
    policy_clone = copy.deepcopy(policy).cpu().eval()
    for param in policy_clone.parameters():
        param.requires_grad_(False)  # Freeze all parameters
    return policy_clone
