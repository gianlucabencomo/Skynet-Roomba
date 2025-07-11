import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from roomba.server.state_buffer import State
from roomba.environments.sumo_v1 import Sumo


def encode_wheels(x, y):
    sign1 = "-" if x > 0 else "+"
    x = str(abs(x)).zfill(3)
    sign2 = "-" if y > 0 else "+"
    y = str(abs(y)).zfill(3)
    return f"{x}{sign1}{y}{sign2}"


def clamp(x, min_val=-1, max_val=1):
    return max(min_val, min(max_val, x))


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
):  
    agent = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.eval()
    return agent

def get_framestack_size(agent, env=Sumo()):
    """Determine framestack size by examining agent's expected input dimensions."""
    # Get observation size from environment
    obs = env._get_obs()
    obs_size = obs["maximus"].shape[0]
    
    # Calculate framestack size
    framestack_size = agent.input_dim // obs_size
    
    return framestack_size


def build_obs(max_s: State, max_torque: np.array, com_s: State, com_torque: np.array) -> dict:
    rel_pos = np.array([max_s.x - com_s.x, max_s.y - com_s.y])
    rel_vel = np.array([max_s.vx - com_s.vx, max_s.vy - com_s.vy])
    return {
        "maximus": np.concatenate(([max_s.x, max_s.y, max_s.vx, max_s.vy, *max_torque], rel_pos, rel_vel)).astype(
            np.float32
        ),
        "commodus": np.concatenate(
            ([com_s.x, com_s.y, com_s.vx, com_s.vy, *com_torque], -rel_pos, -rel_vel)
        ).astype(np.float32),
    }