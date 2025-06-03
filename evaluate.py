import numpy as np
from typing import Callable, Dict, Any

import torch

def evaluate_self_play(env, maximus, commodus, device, n_seeds=10, render=False):
    all_rewards = {"maximus": [], "commodus": []}
    all_lengths = []

    maximus.eval()
    commodus.eval()

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed + 100)
        done = {"maximus": False, "commodus": False}
        trunc = {"maximus": False, "commodus": False}
        total_rewards = {"maximus": 0.0, "commodus": 0.0}
        episode_length = 0

        while not any(done[a] or trunc[a] for a in done):
            obs_tensor_maximus = torch.tensor(obs["maximus"].reshape(1, -1), dtype=torch.float32).to(device)
            obs_tensor_commodus = torch.tensor(obs["commodus"].reshape(1, -1), dtype=torch.float32).to(device)

            action_maximus, _, _, _ = maximus.get_action_and_value(obs_tensor_maximus)
            action_commodus, _, _, _ = commodus.get_action_and_value(obs_tensor_commodus)

            actions = {
                "maximus": action_maximus.detach().cpu().numpy().flatten(),
                "commodus": action_commodus.detach().cpu().numpy().flatten(),
            }

            obs, rewards, done, trunc, info = env.step(actions)
            for agent in ["maximus", "commodus"]:
                total_rewards[agent] += rewards.get(agent, 0.0)
            episode_length += 1

            if render:
                env.render()

        all_rewards["maximus"].append(total_rewards["maximus"])
        all_rewards["commodus"].append(total_rewards["commodus"])
        all_lengths.append(episode_length)
        env.close()
    return np.mean(all_rewards['maximus']), np.mean(all_lengths)