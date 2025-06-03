import numpy as np
import torch
import torch.nn as nn

class ZeroActionAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.space = envs.single_action_space

    def get_action_and_value(self, x, action=None):
        B, _ = x.shape
        action = np.stack([np.zeros_like(self.space.shape) for _ in range(B)])
        action = torch.Tensor(action)
        return action, None, None, None

class RandomActionAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.space = envs.single_action_space

    def get_action_and_value(self, x, action=None):
        B, _ = x.shape
        action = np.stack([self.space.sample() for _ in range(B)])
        action = torch.Tensor(action)
        return action, None, None, None