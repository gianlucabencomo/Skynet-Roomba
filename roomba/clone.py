import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import typer
from utils import set_random_seeds, get_device


def finetune(
    seed: int = 0,
    data_dir: str = "./data",
    ckpt_path: str = "./checkpoints/s3_fs10_ns512_dr_ts_199753728.pt",
    lr: float = 1e-4,
    epochs: int = 100,
    frame_stack: int = 10,
):
    # -- Setup --
    set_random_seeds(seed)
    device = get_device()

    # -- Load demonstration data --
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".pkl") and os.path.isfile(os.path.join(data_dir, f))
    ])
    total_obs = []
    total_action = []
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f)

        # -- prepare observations and actions --
        raw_obs = np.array([d[0]["commodus"] for d in data])
        raw_action = np.array([d[1]["commodus"] for d in data])

        obs = np.array([raw_obs[i - frame_stack:i].flatten() for i in range(frame_stack, len(raw_obs))])
        action = raw_action[frame_stack:]
        total_obs.append(obs)
        total_action.append(action)

        raw_obs = np.array([d[0]["maximus"] for d in data])
        raw_action = np.array([d[1]["maximus"] for d in data])

        obs = np.array([raw_obs[i - frame_stack:i].flatten() for i in range(frame_stack, len(raw_obs))])
        action = raw_action[frame_stack:]
        total_obs.append(obs)
        total_action.append(action)
    obs = np.concatenate(total_obs, axis=0)      # shape: [N_total, obs_dim]
    action = np.concatenate(total_action, axis=0)  # shape: [N_total, action_dim]

    # -- load agent --
    agent = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = agent.actor_mean

    # -- convert to tensors --
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.float32).to(device)

    # -- optimization setup --
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # -- finetuning loop --
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_actions = model(obs)
        loss = loss_fn(pred_actions, action)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    save_path = ckpt_path.replace(".pt", "_finetuned.pt")
    torch.save(agent, save_path)
    print(f"Saved finetuned agent to: {save_path}")

if __name__ == "__main__":
    typer.run(finetune)