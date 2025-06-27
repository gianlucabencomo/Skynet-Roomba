import numpy as np
import torch
import torch.nn as nn


class ZeroActionAgent(nn.Module):
    """Agent that outputs zero actions for every input."""
    def __init__(self, envs):
        super().__init__()
        self.space = envs.single_action_space

    def get_action_and_value(self, x, action=None):
        B, _ = x.shape
        action = np.stack([np.zeros_like(self.space.shape) for _ in range(B)])
        action = torch.Tensor(action)
        return action, None, None, None


class RandomActionAgent(nn.Module):
    """Agent that samples uniformly from action space."""
    def __init__(self, envs):
        super().__init__()
        self.space = envs.single_action_space

    def get_action_and_value(self, x, action=None):
        B, _ = x.shape
        action = np.stack([self.space.sample() for _ in range(B)])
        action = torch.Tensor(action)
        return action, None, None, None

class LSTMZeroActionAgent(nn.Module):
    def __init__(self, action_dim, n_layers=2, hidden_size=128, device=None):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.shared = type('', (), {})()
        self.shared.num_layers = n_layers
        self.shared.hidden_size = hidden_size

    def get_action_and_value(self, x, state, done, action=None):
        # Defensive: ensure x is a tensor and batch-major
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device if self.device is not None else None)
        shape = x.shape
        if len(shape) == 3:
            B, T, _ = shape
        elif len(shape) == 2:
            B, T = shape[0], 1
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Input x shape is {shape}, expected [B, T, obs_dim] or [B, obs_dim]")
        action = torch.zeros(B * T, self.action_dim, device=x.device)
        logprob = torch.zeros(B * T, device=x.device)
        entropy = torch.zeros(B * T, device=x.device)
        value = torch.zeros(B * T, 1, device=x.device)
        return action, logprob, entropy, value, state

    def get_value(self, x, state, done):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.device if self.device is not None else None)
        shape = x.shape
        if len(shape) == 3:
            B, T, _ = shape
        elif len(shape) == 2:
            B, T = shape[0], 1
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Input x shape is {shape}, expected [B, T, obs_dim] or [B, obs_dim]")
        return torch.zeros(B * T, 1, device=x.device)

    def actor_params(self):
        return []

    def critic_params(self):
        return []