import numpy as np
import torch
import torch.nn as nn

class HeuristicSumoAgent(nn.Module):
    EDGE_THRESH = 0.2            # tune for your arena radius
    FWD_TORQUE = 0.4              # nominal “full-speed ahead”
    TURN_GAIN  = 4.0              # higher → snappier turns
    SPIN_TORQUE = 0.4             # used when velocity is ~0
    VEL_EPS = 0.05

    def __init__(self, env):
        super().__init__()
        if hasattr(env, "single_action_space"):
            self.space = env.single_action_space
        else:  # raw PettingZoo ParallelEnv
            self.space = env.action_space("maximus")
        self.low  = self.space.low
        self.high = self.space.high

    def get_action_and_value(self, x: torch.Tensor, action=None, deterministic: bool = False):
        """
        x: (B, obs_dim) – may include stacked frames; we only need last 10 dims.
        returns: (action, logprob, entropy, value) – PPO ignores the None’s.
        """
        # detach to NumPy for convenience
        obs_np = x.cpu().numpy()
        B, D = obs_np.shape
        obs_last = obs_np[:, -10:]

        pos      = obs_last[:, 0:2]
        vel      = obs_last[:, 2:4]
        rel_pos  = obs_last[:, 6:8]

        acts = np.zeros((B, 2), dtype=np.float32)

        for i in range(B):
            p  = pos[i]
            v  = vel[i]
            rp = rel_pos[i]

            if np.linalg.norm(p) > self.EDGE_THRESH:
                target_vec = -p
            else:
                target_vec = rp

            # ------------------------------------------------------------------
            # 1. determine current heading (fall back to x-axis if nearly still)
            # ------------------------------------------------------------------
            if np.linalg.norm(v) > self.VEL_EPS:
                heading = v / (np.linalg.norm(v) + 1e-8)
            else:
                heading = np.array([1.0, 0.0], dtype=np.float32)

            tgt_dir = target_vec / (np.linalg.norm(target_vec) + 1e-8)

            # signed 2-D cross product gives “how far left/right to turn”
            cross = heading[0] * tgt_dir[1] - heading[1] * tgt_dir[0]

            # ------------------------------------------------------------------
            # 2. build wheel torques
            # ------------------------------------------------------------------
            if np.linalg.norm(v) < self.VEL_EPS:
                # spin in place until we move
                left  =  self.SPIN_TORQUE
                right = -self.SPIN_TORQUE
            else:
                turn  = np.clip(self.TURN_GAIN * cross, -1.0, 1.0)
                left  = self.FWD_TORQUE - turn
                right = self.FWD_TORQUE + turn

            # respect actuator limits
            acts[i, 0] = np.clip(left,  self.low[0], self.high[0])
            acts[i, 1] = np.clip(right, self.low[1], self.high[1])

        # convert back to torch; policy outputs don’t need log π / V(s)
        return torch.as_tensor(acts, dtype=torch.float32), None, None, None

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