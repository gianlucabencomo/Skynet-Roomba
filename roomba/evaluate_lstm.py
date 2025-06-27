import numpy as np
from typing import Callable, Dict, Any

import torch

def evaluate_self_play(env, maximus, commodus, device, n_seeds=10, render=False):
    all_rewards = {"maximus": [], "commodus": []}
    all_lengths = []

    maximus.eval()
    commodus.eval()

    # Get LSTM shapes dynamically
    def get_init_state(agent):
        n_layers = agent.shared.num_layers
        hidden_dim = agent.shared.hidden_size
        h = torch.zeros(n_layers, 1, hidden_dim, device=device)
        c = torch.zeros(n_layers, 1, hidden_dim, device=device)
        return (h, c)

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed + 100)
        done = {"maximus": False, "commodus": False}
        trunc = {"maximus": False, "commodus": False}
        total_rewards = {"maximus": 0.0, "commodus": 0.0}
        episode_length = 0

        # Reset LSTM states for both agents at the start of each episode
        state_maximus = get_init_state(maximus)
        state_commodus = get_init_state(commodus)

        # Each agent needs its own "done" state for the LSTM
        done_maximus = torch.zeros(1, 1, device=device)
        done_commodus = torch.zeros(1, 1, device=device)

        while not any(done[a] or trunc[a] for a in done):
            obs_tensor_maximus = torch.tensor(
                obs["maximus"].reshape(1, 1, -1), dtype=torch.float32
            ).to(device)
            obs_tensor_commodus = torch.tensor(
                obs["commodus"].reshape(1, 1, -1), dtype=torch.float32
            ).to(device)

            # Forward through the LSTM agent (seq_len=1)
            action_maximus, _, _, _, state_maximus = maximus.get_action_and_value(
                obs_tensor_maximus, state_maximus, done_maximus
            )
            action_commodus, _, _, _, state_commodus = commodus.get_action_and_value(
                obs_tensor_commodus, state_commodus, done_commodus
            )

            actions = {
                "maximus": action_maximus.detach().cpu().numpy().flatten(),
                "commodus": action_commodus.detach().cpu().numpy().flatten(),
            }

            obs, rewards, done, trunc, info = env.step(actions)
            for agent in ["maximus", "commodus"]:
                total_rewards[agent] += rewards.get(agent, 0.0)

            # Update done tensors for LSTM state reset next step
            done_maximus = torch.tensor([[done["maximus"] or trunc["maximus"]]], dtype=torch.float32, device=device)
            done_commodus = torch.tensor([[done["commodus"] or trunc["commodus"]]], dtype=torch.float32, device=device)

            episode_length += 1

            if render:
                env.render()

        all_rewards["maximus"].append(total_rewards["maximus"])
        all_rewards["commodus"].append(total_rewards["commodus"])
        all_lengths.append(episode_length)
        env.close()
    return np.mean(all_rewards["maximus"]), np.mean(all_lengths)