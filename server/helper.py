import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.mlp import MlpContinuousActorCritic
from typing import List


def encode_wheels(x, y):
    sign1 = "-" if x < 0 else "+"
    x = str(abs(x)).zfill(3)
    sign2 = "-" if y < 0 else "+"
    y = str(abs(y)).zfill(3)
    return f"{x}{sign1}{y}{sign2}"


def clamp(x, min_val=-1, max_val=1):
    return max(min_val, min(max_val, x))


def load_checkpoint(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    actor_hidden_widths: List[int],
    critic_hidden_widths: List[int],
    device: str = "cpu",
):
    agent = MlpContinuousActorCritic(obs_dim, action_dim, actor_hidden_widths, critic_hidden_widths).to(device)
    agent.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    agent.eval()
    return agent
