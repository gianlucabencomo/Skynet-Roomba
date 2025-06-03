import os
import typer
import tqdm.auto as tqdm

import copy
import os

import random
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

import time
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from models.mlp import MlpContinuousActorCritic
from models.base import ZeroActionAgent, RandomActionAgent

from utils import *

from environments.sumo import Sumo
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

import supersuit as ss

from evaluate import evaluate_self_play

from collections import deque

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon

    def update(self, x):
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

def clone_policy(src: nn.Module) -> nn.Module:
    tgt = copy.deepcopy(src).cpu().eval()
    for p in tgt.parameters():
        p.requires_grad_(False)
    return tgt

def train(
    seed: int = 0,
    total_timesteps: int = 200_000_000,
    n_envs: int = 1024,
    n_steps: int = 256,
    n_mbs: int = 64, # number of mini-batches
    epochs_per_update: int = 10,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.01,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    frame_stack: int = 1,
    record_videos: bool = False,
    run_name: Optional[str] = None,
    past_agent_buffer_size: int = 50
):
    timestamp = int(time.time())
    if run_name is None:
        run_name = f"roomba__{seed}__{timestamp}"
    else:
        run_name = f"roomba__{run_name}_{seed}__{timestamp}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in locals().items()])),
    )
    # compute ppo params
    batch_size = n_envs * n_steps
    n_iterations = total_timesteps // batch_size + 1
    minibatch_size = batch_size // n_mbs
    past_agents: deque[nn.Module] = deque(maxlen=past_agent_buffer_size)

    # set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # set device
    device = get_device()

    total_envs = n_envs * 2
    env = Sumo()
    env = pettingzoo_env_to_vec_env_v1(env)
    envs = concat_vec_envs_v1(env, num_vec_envs=n_envs, num_cpus=os.cpu_count(), base_class="gymnasium")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    inds = np.arange(total_envs)
    agent_inds, opponent_inds = inds[::2], inds[1::2]

    # add Random and Zero Action Agents to buffer for initial sampling
    past_agents.append(clone_policy(RandomActionAgent(envs)))
    past_agents.append(clone_policy(ZeroActionAgent(envs)))

    # agent + optimizer
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = MlpContinuousActorCritic(obs_dim, action_dim).to(device)

    actor_optimizer = optim.Adam(agent.actor_params(), lr=actor_lr, eps=1e-5)
    critic_optimizer = optim.AdamW(agent.critic_params(), lr=critic_lr, eps=1e-5, weight_decay=1e-4)
    
    # set up buffer
    obs = torch.zeros((n_steps, n_envs) + envs.single_observation_space.shape).to(device) # TODO: fix hard coding
    actions = torch.zeros((n_steps, n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((n_steps, n_envs)).to(device)
    rewards = torch.zeros((n_steps, n_envs)).to(device)
    dones = torch.zeros((n_steps, n_envs)).to(device)
    values = torch.zeros((n_steps, n_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset()

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(total_envs).to(device)

    reward_rms = RunningMeanStd()
    with tqdm.tqdm(total=total_timesteps, desc="Training") as pbar:
        for iteration in range(1, n_iterations + 1):
            agent.train()
            opp_idx = torch.randint(low=0, high=len(past_agents), size=(n_envs,), device=device)
            for step in range(0, n_steps):
                global_step += n_envs
                obs[step] = next_obs[agent_inds].reshape(n_envs, -1)
                dones[step] = next_done[agent_inds]

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs[step])
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                total_action = np.zeros((total_envs, envs.single_action_space.shape[0]), dtype=np.float32)
                total_action[agent_inds] = action.cpu().numpy()
                opp_actions = np.empty((n_envs, envs.single_action_space.shape[0]), dtype=np.float32)
                for idx in opp_idx.unique():
                    idx_int = int(idx.item())
                    mask = (opp_idx == idx)
                    mask_np = mask.cpu().numpy()
                    obs_batch = next_obs[opponent_inds][mask]
                    with torch.no_grad():
                        act_batch, _, _, _ = past_agents[idx_int].get_action_and_value(obs_batch)
                    opp_actions[mask_np] = act_batch.cpu().numpy()
                total_action[opponent_inds] = opp_actions
                next_obs, reward, terminations, truncations, infos = envs.step(total_action)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = (
                    torch.tensor(reward[agent_inds], dtype=torch.float32).to(device).view(-1)
                )
                next_obs, next_done = torch.Tensor(next_obs.reshape(total_envs, -1)).to(device), torch.Tensor(
                    next_done
                ).to(device)

            agent.eval()
            # Normalize and clip rewards across rollout
            with torch.no_grad():
                flat_rewards = rewards.view(-1)
                reward_rms.update(flat_rewards)

                rewards = (rewards - reward_rms.mean) / torch.sqrt(reward_rms.var + 1e-8)
                rewards = torch.clip(rewards, -10.0, 10.0)

            with torch.no_grad():
                next_value = agent.get_value(next_obs[agent_inds].reshape(n_envs, -1)).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(n_steps)):
                    if t == n_steps - 1:
                        nextnonterminal = 1.0 - next_done[agent_inds]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) # TODO: fix hard coding
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(batch_size)
            clipfracs = []
            entropies = []
            for epoch in range(epochs_per_update):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]
                        entropies += [entropy.mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    actor_loss = torch.max(pg_loss1, pg_loss2).mean() - ent_coef * entropy.mean()

                    newvalue = newvalue.view(-1)
                    if clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_coef,
                            clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        critic_loss = 0.5 * v_loss_max.mean()
                    else:
                        critic_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Update actor parameters
                    actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(agent.actor_params(), max_grad_norm)
                    actor_optimizer.step()

                    # Update critic parameters
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(agent.critic_params(), max_grad_norm)
                    critic_optimizer.step()

                if target_kl is not None and approx_kl > target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if iteration % 20 == 0:
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                if not run_name:
                    checkpoint_name = f"agent_{seed}_train_step_{global_step}_timestamp_{timestamp}.pt"
                else:
                    checkpoint_name = f"agent_{run_name}_train_step_{global_step}_timestamp_{timestamp}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(agent.state_dict(), checkpoint_path)
                past_agents.append(clone_policy(agent))

            reward, length = evaluate_self_play(Sumo(train=False), agent, ZeroActionAgent(envs), device)

            writer.add_scalar("results/episode_reward", reward, global_step)
            writer.add_scalar("results/episode_length", length, global_step)
            writer.add_scalar("losses/entropy", np.mean(entropies), global_step)
            writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            # update progress bar
            pbar.update(n_envs * n_steps)


if __name__ == "__main__":
    typer.run(train)