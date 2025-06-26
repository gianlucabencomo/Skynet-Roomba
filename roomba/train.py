import os
import typer
import tqdm.auto as tqdm
import time
from typing import Optional
from collections import deque

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

import supersuit as ss
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from roomba.environments.sumo_v1 import Sumo

from roomba.models.mlp import MlpContinuousActorCritic
from roomba.models.base import ZeroActionAgent, RandomActionAgent
from roomba.utils import get_device, RunningMeanStd, clone_policy, set_random_seeds
from roomba.environments.wrappers import FrameStackWrapper
from roomba.evaluate import evaluate_self_play

from torch.utils.tensorboard import SummaryWriter


def train(
    seed: int = 0,
    env_mode: str = "uwb",
    total_timesteps: int = 100_000_000,
    n_envs: int = 512,  # n parallel environments
    n_steps: int = 512,  # steps per rollout per environment
    n_mbs: int = 64,  # n mini-batches per epoch (determines batch size)
    epochs_per_update: int = 5,  # n epochs per rollout
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-2,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.01,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    anneal_lr: bool = True,
    frame_stack: int = 1,  # n of frames to stack for obs when training MLP
    past_agent_buffer_size: int = 50,  # maximum number of previous agents to play against in asynchronous self-play
    checkpoint_dir: str = "checkpoints",
    save_freq: int = 20,  # after how many updates to save checkpoints / add to buffer
    checkpoint_path: Optional[str] = None,  # for continuing training
    uwb_sensor_noise: float = 0.01,
    action_alpha: float = 0.4,
    obs_alpha: float = 0.6,
    run_name: str = None,
):
    """PPO asynchronous self-play with MLP for Sumo. Heavily referenced https://github.com/vwxyzjn/cleanrl."""
    # -- create unique run name ---
    timestamp = int(time.time())
    if run_name is None: 
        run_name = f"s{seed}_fs{frame_stack}_ns{n_steps}_uwb{uwb_sensor_noise:.2f}".replace('.', '_')

    # -- initialize TensorBoard writer --
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in locals().items()])),
    )
    # -- compute relevant PPO parameters --
    batch_size = n_envs * n_steps
    n_iterations = total_timesteps // batch_size + 1
    minibatch_size = batch_size // n_mbs
    total_envs = n_envs * 2

    # -- initialize self-play buffer --
    past_agents: deque[nn.Module] = deque(maxlen=past_agent_buffer_size)

    # -- set random seeds + get device --
    set_random_seeds(seed)
    device = get_device()

    # -- create environment --
    env = Sumo(mode=env_mode, uwb_sensor_noise=uwb_sensor_noise, action_alpha=action_alpha, obs_alpha=obs_alpha)

    # -- frame stacking --
    if frame_stack > 1:
        env = FrameStackWrapper(env, k=frame_stack)

    # -- convert to vectorized format for parallel rollouts + set metadata --
    env = pettingzoo_env_to_vec_env_v1(env)
    envs = concat_vec_envs_v1(
        env, num_vec_envs=n_envs, num_cpus=os.cpu_count(), base_class="gymnasium"
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True

    # -- even indices = agent, odd = opponent --
    inds = np.arange(total_envs)
    agent_inds, opponent_inds = inds[::2], inds[1::2]

    # -- inititalize self-play buffer with dummy agents --
    past_agents.append(clone_policy(RandomActionAgent(envs)))
    past_agents.append(clone_policy(ZeroActionAgent(envs)))

    # -- initialize agent and optimizers --
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = MlpContinuousActorCritic(obs_dim, action_dim).to(device)
    
    # -- load checkpoint for continuing training if provided -- 
    start_global_step = 0
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        agent.load_state_dict(checkpoint)
        
        # Extract global step from checkpoint filename if following the naming convention
        # e.g., "agent_roomba__test_0__1234567890_train_step_1000000.pt"
        import re
        match = re.search(r'train_step_(\d+)', checkpoint_path)
        if match:
            start_global_step = int(match.group(1))
            print(f"Resuming from global step: {start_global_step}")
    
    actor_optimizer = optim.Adam(agent.actor_params(), lr=actor_lr, eps=1e-5)
    critic_optimizer = optim.Adam(agent.critic_params(), lr=critic_lr, eps=1e-5)

    # -- initialize rollout buffers to store experience data --
    obs = torch.zeros((n_steps, n_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((n_steps, n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((n_steps, n_envs)).to(device)
    rewards = torch.zeros((n_steps, n_envs)).to(device)
    dones = torch.zeros((n_steps, n_envs)).to(device)
    values = torch.zeros((n_steps, n_envs)).to(device)

    # -- training state variables --
    global_step = start_global_step  # Start from checkpoint step instead of 0
    
    # -- obs and reward tracking --
    next_obs, infos = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(total_envs).to(device)
    reward_rms = RunningMeanStd()
    
    with tqdm.tqdm(total=total_timesteps, desc="Training") as pbar:
        # Calculate which iteration to start from
        start_iteration = (start_global_step // batch_size) + 1
        
        for iteration in range(start_iteration, n_iterations + 1):
            if anneal_lr:
                frac = (
                    1.0 - (iteration - 1.0) / n_iterations
                )  # Linear decay from 1.0 to 0.0
                actor_lr_now = frac * actor_lr
                critic_lr_now = frac * critic_lr
                actor_optimizer.param_groups[0]["lr"] = actor_lr_now
                critic_optimizer.param_groups[0]["lr"] = critic_lr_now

            agent.train()
            opp_idx = torch.randint(
                low=0, high=len(past_agents), size=(n_envs,), device=device
            )
            # -- collect rollout --
            for step in range(0, n_steps):
                global_step += n_envs

                obs[step] = next_obs[agent_inds].reshape(n_envs, -1)
                dones[step] = next_done[agent_inds]

                # -- sample actions --
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs[step])
                    values[step] = value.flatten()  # Store value estimates for GAE

                actions[step] = action
                logprobs[step] = logprob

                total_action = np.zeros(
                    (total_envs, envs.single_action_space.shape[0]), dtype=np.float32
                )
                total_action[agent_inds] = action.cpu().numpy()  # Our agent's actions

                opp_actions = np.empty(
                    (n_envs, envs.single_action_space.shape[0]), dtype=np.float32
                )
                for idx in opp_idx.unique():
                    idx_int = int(idx.item())
                    mask = opp_idx == idx
                    mask_np = mask.cpu().numpy()
                    obs_batch = next_obs[opponent_inds][mask]

                    with torch.no_grad():
                        act_batch, _, _, _ = past_agents[idx_int].get_action_and_value(
                            obs_batch
                        )
                    opp_actions[mask_np] = act_batch.cpu().numpy()

                total_action[opponent_inds] = opp_actions
                next_obs, reward, terminations, truncations, infos = envs.step(
                    total_action
                )
                next_done = np.logical_or(terminations, truncations)

                rewards[step] = (
                    torch.tensor(reward[agent_inds], dtype=torch.float32)
                    .to(device)
                    .view(-1)
                )

                next_obs = torch.Tensor(next_obs.reshape(total_envs, -1)).to(device)
                next_done = torch.Tensor(next_done).to(device)

            agent.eval()
            # -- normalize rewards from rollout --
            with torch.no_grad():
                flat_rewards = rewards.view(-1)
                reward_rms.update(flat_rewards)
                rewards = (rewards - reward_rms.mean) / torch.sqrt(
                    reward_rms.var + 1e-8
                )
                rewards = torch.clip(rewards, -10.0, 10.0)

            # -- advantage calculation --
            with torch.no_grad():
                next_value = agent.get_value(
                    next_obs[agent_inds].reshape(n_envs, -1)
                ).reshape(1, -1)
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

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(batch_size)
            clipfracs = []
            entropies = []

            # -- update based on rollout data --
            for epoch in range(epochs_per_update):
                np.random.shuffle(b_inds)  # shuffle data for each epoch

                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # approximate KL divergence between old and new policy
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()

                        # track fraction of updates that were clipped (indicates policy change)
                        clipfracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]
                        entropies += [entropy.mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    actor_loss = (
                        torch.max(pg_loss1, pg_loss2).mean() - ent_coef * entropy.mean()
                    )

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
                        critic_loss = (
                            0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        )

                    # -- update actor network parameters --
                    actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(agent.actor_params(), max_grad_norm)
                    actor_optimizer.step()

                    # -- update critic network parameters --
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(agent.critic_params(), max_grad_norm)
                    critic_optimizer.step()

                if target_kl is not None and approx_kl > target_kl:
                    break

            # -- compute explained variance --
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # -- checkpointing --
            if iteration % save_freq == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_name = run_name + f"_ts_{global_step}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(agent, checkpoint_path)
                # -- add to buffer --
                past_agents.append(clone_policy(agent))

            # -- evaluation --
            eval_env = Sumo(
                mode=env_mode,
                contact_rew_weight=0.0,
                dist_center_weight=0.0,
                symmetry_rew_weight=0.0,
                uwb_sensor_noise=uwb_sensor_noise, action_alpha=action_alpha, obs_alpha=obs_alpha
            )
            if frame_stack > 1:
                eval_env = FrameStackWrapper(eval_env, k=frame_stack)
            # -- test against zero agent --
            reward, length = evaluate_self_play(
                eval_env, agent, ZeroActionAgent(envs), device
            )

            # -- logging --
            writer.add_scalar("results/episode_reward_0", reward, global_step)
            writer.add_scalar("results/episode_length_0", length, global_step)

            # -- test against most two agents ago --
            reward, length = evaluate_self_play(
                eval_env, agent, past_agents[-2], device
            )

            # -- logging --
            writer.add_scalar("results/episode_reward_2", reward, global_step)
            writer.add_scalar("results/episode_length_2", length, global_step)

            writer.add_scalar("losses/entropy", np.mean(entropies), global_step)
            writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            pbar.update(n_envs * n_steps)

    # save the final agent
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        agent,
        os.path.join(checkpoint_dir, run_name + f"_ts_final.pt"),
    )


if __name__ == "__main__":
    typer.run(train)
