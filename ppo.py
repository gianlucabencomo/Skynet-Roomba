"""
Proximal Policy Optimization (PPO) selplay training script for Roomba fighters. Use this script as a starting point for your own training.
"""

# Standard library imports
import os
import typer
import tqdm.auto as tqdm
import copy
import random
import time
from typing import Optional
from collections import deque

# Third-party ML/RL imports
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import supersuit as ss
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

# Local imports
from models.mlp import MlpContinuousActorCritic
from models.base import ZeroActionAgent, RandomActionAgent
from utils import *
from environments.sumo_v2 import Sumo  # Main competition environment
from environments.wrappers import FrameStackWrapper
from evaluate import evaluate_self_play

class RunningMeanStd:
    """
    Computes running mean and standard deviation for reward normalization.
    """
    
    def __init__(self, epsilon=1e-4, shape=()):
        """Initialize running statistics with small epsilon for numerical stability."""
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon  # Small initial count to avoid division by zero

    def update(self, x):
        """Update running mean and variance using Welford's online algorithm."""
        x = x.detach()  # Detach from computation graph to save memory
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()

        # Compute the difference between batch mean and current running mean
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # Update running mean using weighted average
        new_mean = self.mean + delta * batch_count / total_count
        
        # Update running variance using Welford's method
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        # Store updated statistics
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

def clone_policy(src: nn.Module) -> nn.Module:
    """
    Create a frozen copy of a policy network for the past agents buffer. Used to maintain
    a set of past opponents for self-play training.
    """
    tgt = copy.deepcopy(src).cpu().eval()
    for p in tgt.parameters():
        p.requires_grad_(False)  # Freeze all parameters
    return tgt

def train(
    seed: int = 0,                          # Random seed for reproducibility
    total_timesteps: int = 100_000_000,     # Total training timesteps across all environments
    n_envs: int = 1024,                     # Number of parallel vectorized environments
    n_steps: int = 256,                     # Steps per rollout per environment
    n_mbs: int = 64,                        # Number of mini-batches for SGD updates
    epochs_per_update: int = 5,             # Number of epochs to train on each batch of data
    gamma: float = 0.995,                   # Discount factor for future rewards
    gae_lambda: float = 0.95,               # GAE lambda parameter for advantage estimation
    clip_coef: float = 0.2,                 # PPO clipping coefficient
    actor_lr: float = 1e-3,                 # Learning rate for actor network
    critic_lr: float = 1e-2,                # Learning rate for critic network (higher than actor)
    ent_coef: float = 0.0,                  # Entropy coefficient for exploration
    max_grad_norm: float = 0.5,             # Maximum gradient norm for clipping
    target_kl: float = 0.01,                # Target KL divergence for early stopping
    norm_adv: bool = True,                  # Whether to normalize advantages
    clip_vloss: bool = True,                # Whether to clip value function loss
    anneal_lr: bool = True,                 # Whether to anneal learning rate over time
    frame_stack: int = 1,                   # Number of frames to stack for temporal information
    record_videos: bool = False,            # Whether to record training videos
    run_name: Optional[str] = None,         # Optional name for this training run
    past_agent_buffer_size: int = 50        # Size of past agents buffer for self-play
):
    """
    Main PPO training function with self-play for Roomba Sumo agents.
    
    This function implements the complete PPO training loop with several features:
    - Self-play against a buffer of past agent versions
    - Vectorized environments for parallel data collection
    - Reward normalization for training stability
    - GAE for advantage estimation
    - Separate actor and critic optimizers
    - Checkpointing and evaluation
    """
    # Create unique run name with timestamp for logging and checkpoints
    timestamp = int(time.time())
    if run_name is None:
        run_name = f"roomba__{seed}__{timestamp}"
    else:
        run_name = f"roomba__{run_name}_{seed}__{timestamp}"
    
    # Initialize TensorBoard writer for logging training metrics
    writer = SummaryWriter(f"runs/{run_name}")
    # Log all hyperparameters as a markdown table in TensorBoard
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in locals().items()])),
    )
    # Compute derived PPO parameters from hyperparameters
    batch_size = n_envs * n_steps                    # Total samples per iteration
    n_iterations = total_timesteps // batch_size + 1  # Number of training iterations
    minibatch_size = batch_size // n_mbs             # Samples per mini-batch for SGD
    # Initialize buffer for past agents (for self-play)
    past_agents: deque[nn.Module] = deque(maxlen=past_agent_buffer_size)

    # Set random seeds for reproducible training across runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA operations

    # Automatically detect and set the best available device (CUDA/CPU)
    device = get_device()

    # Environment setup: create vectorized parallel environments
    total_envs = n_envs * 2  # Each environment has 2 agents (our agent + opponent)
    env = Sumo()  # Create base sumo environment
    
    # Add frame stacking if requested (for temporal information)
    if frame_stack > 1:
        env = FrameStackWrapper(env, k=frame_stack)
    
    # Convert gymnasium environment to vectorized format for parallel rollouts
    env = pettingzoo_env_to_vec_env_v1(env)
    envs = concat_vec_envs_v1(env, num_vec_envs=n_envs, num_cpus=os.cpu_count(), base_class="gymnasium")
    
    # Set up environment metadata
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    
    # Create indices to separate our training agent from opponents in vectorized env
    inds = np.arange(total_envs)
    agent_inds, opponent_inds = inds[::2], inds[1::2]  # Even indices = agent, odd = opponent

    # Initialize past agents buffer with simple baseline opponents
    # This ensures we have opponents available from the start of training
    past_agents.append(clone_policy(RandomActionAgent(envs)))  # Random actions for exploration
    past_agents.append(clone_policy(ZeroActionAgent(envs)))    # No actions for basic opponent

    # Initialize agent and optimizers
    obs_dim = int(np.prod(envs.single_observation_space.shape))   # Flatten observation dimensions
    action_dim = int(np.prod(envs.single_action_space.shape))     # Flatten action dimensions
    agent = MlpContinuousActorCritic(obs_dim, action_dim).to(device)  # Create agent policy and move to GPU/device

    # Separate optimizers for actor and critic with different learning rates
    actor_optimizer = optim.Adam(agent.actor_params(), lr=actor_lr, eps=1e-5)
    critic_optimizer = optim.Adam(agent.critic_params(), lr=critic_lr, eps=1e-5)
    
    # Initialize rollout buffers to store experience data
    # These buffers store n_steps of experience from n_envs parallel environments
    obs = torch.zeros((n_steps, n_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((n_steps, n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((n_steps, n_envs)).to(device)  # Log probabilities of taken actions
    rewards = torch.zeros((n_steps, n_envs)).to(device)   # Rewards received
    dones = torch.zeros((n_steps, n_envs)).to(device)     # Episode termination flags
    values = torch.zeros((n_steps, n_envs)).to(device)    # Value function estimates

    # Initialize training state variables
    global_step = 0          # Track total steps across all environments
    start_time = time.time() # For computing training speed metrics
    
    # Reset all environments to get initial observations
    next_obs, infos = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)      # Convert to tensor and move to device
    next_done = torch.zeros(total_envs).to(device)    # Track which environments are done

    # Initialize reward normalization tracker for stable training
    reward_rms = RunningMeanStd()
    
    # Main training loop with progress bar
    with tqdm.tqdm(total=total_timesteps, desc="Training") as pbar:
        for iteration in range(1, n_iterations + 1):
            # Learning rate annealing: gradually reduce LR as training progresses
            if anneal_lr:
                frac = 1.0 - (iteration - 1.0) / n_iterations  # Linear decay from 1.0 to 0.0
                actor_lr_now = frac * actor_lr
                critic_lr_now = frac * critic_lr
                actor_optimizer.param_groups[0]["lr"] = actor_lr_now
                critic_optimizer.param_groups[0]["lr"] = critic_lr_now
            
            agent.train()  # Set agent to training mode
            # Randomly select opponents from past agents buffer for each environment
            opp_idx = torch.randint(low=0, high=len(past_agents), size=(n_envs,), device=device)
            # Rollout phase: collect experience from environments
            for step in range(0, n_steps):
                global_step += n_envs  # Increment global step counter
                
                # Store current observations and done flags for our training agent
                obs[step] = next_obs[agent_inds].reshape(n_envs, -1)
                dones[step] = next_done[agent_inds]

                # Get action and value from our training agent (no gradient computation)
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs[step])
                    values[step] = value.flatten()  # Store value estimates for GAE
                
                # Store actions and their log probabilities
                actions[step] = action
                logprobs[step] = logprob

                # Prepare actions for all agents (training agent + opponents)
                total_action = np.zeros((total_envs, envs.single_action_space.shape[0]), dtype=np.float32)
                total_action[agent_inds] = action.cpu().numpy()  # Our agent's actions
                
                # Generate opponent actions from randomly selected past agents
                opp_actions = np.empty((n_envs, envs.single_action_space.shape[0]), dtype=np.float32)
                # Batch process opponents by agent type for efficiency
                for idx in opp_idx.unique():
                    idx_int = int(idx.item())
                    mask = (opp_idx == idx)  # Which environments use this opponent
                    mask_np = mask.cpu().numpy()
                    obs_batch = next_obs[opponent_inds][mask]  # Observations for this opponent type
                    
                    # Get actions from the selected past agent
                    with torch.no_grad():
                        act_batch, _, _, _ = past_agents[idx_int].get_action_and_value(obs_batch)
                    opp_actions[mask_np] = act_batch.cpu().numpy()
                
                total_action[opponent_inds] = opp_actions  # Set opponent actions
                # Execute actions in environment and collect results
                next_obs, reward, terminations, truncations, infos = envs.step(total_action)
                next_done = np.logical_or(terminations, truncations)  # Episode ends on termination OR truncation
                
                # Store rewards for our training agent only
                rewards[step] = (
                    torch.tensor(reward[agent_inds], dtype=torch.float32).to(device).view(-1)
                )
                
                # Prepare observations and done flags for next step
                next_obs = torch.Tensor(next_obs.reshape(total_envs, -1)).to(device)
                next_done = torch.Tensor(next_done).to(device)

            agent.eval()  # Set agent to evaluation mode for value computation
            
            # Reward normalization for training stability
            with torch.no_grad():
                flat_rewards = rewards.view(-1)  # Flatten rewards for statistics
                reward_rms.update(flat_rewards)  # Update running mean and std

                # Normalize rewards to zero mean, unit variance
                rewards = (rewards - reward_rms.mean) / torch.sqrt(reward_rms.var + 1e-8)
                # Clip normalized rewards to prevent extreme values
                rewards = torch.clip(rewards, -10.0, 10.0)

            # Compute advantages using Generalized Advantage Estimation (GAE)
            with torch.no_grad():
                # Get value estimate for the final observation (bootstrap value)
                next_value = agent.get_value(next_obs[agent_inds].reshape(n_envs, -1)).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                
                # Compute advantages backwards through time for GAE
                for t in reversed(range(n_steps)):
                    if t == n_steps - 1:  # Last step uses bootstrap value
                        nextnonterminal = 1.0 - next_done[agent_inds]  # 0 if terminal, 1 if not
                        nextvalues = next_value
                    else:  # Other steps use next step's value
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    
                    # Compute TD error (temporal difference)
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    
                    # Compute GAE advantage with exponential weighting
                    advantages[t] = lastgaelam = (
                        delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    )
                
                # Compute returns as advantages + value estimates
                returns = advantages + values

            # Flatten rollout buffers for mini-batch training
            # Convert from (n_steps, n_envs, ...) to (batch_size, ...)
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)      # Old log probabilities
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)   # Computed advantages
            b_returns = returns.reshape(-1)        # Target values for critic
            b_values = values.reshape(-1)          # Old value estimates

            # Prepare for mini-batch SGD updates
            b_inds = np.arange(batch_size)  # Indices for shuffling data
            clipfracs = []  # Track fraction of clipped policy updates
            entropies = []  # Track policy entropy for exploration
            
            # Multiple epochs of updates on the same batch of data
            for epoch in range(epochs_per_update):
                np.random.shuffle(b_inds)  # Shuffle data for each epoch
                
                # Process data in mini-batches for SGD
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]  # Current mini-batch indices

                    # Forward pass through updated policy
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    
                    # Compute importance sampling ratio for PPO
                    logratio = newlogprob - b_logprobs[mb_inds]  # Log of new_prob / old_prob
                    ratio = logratio.exp()  # new_prob / old_prob

                    # Compute diagnostics (without gradients for efficiency)
                    with torch.no_grad():
                        # Approximate KL divergence between old and new policy
                        old_approx_kl = (-logratio).mean() 
                        approx_kl = ((ratio - 1) - logratio).mean()  
                        
                        # Track fraction of updates that were clipped (indicates policy change)
                        clipfracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]
                        # Track entropy for monitoring exploration
                        entropies += [entropy.mean().item()]

                    # Get advantages for this mini-batch
                    mb_advantages = b_advantages[mb_inds]
                    
                    # Normalize advantages within mini-batch for stable learning
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Compute PPO actor loss with clipping
                    pg_loss1 = -mb_advantages * ratio  # Standard policy gradient loss
                    pg_loss2 = -mb_advantages * torch.clamp(  # Clipped policy gradient loss
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    # Take the more conservative (higher) loss to prevent large updates
                    actor_loss = torch.max(pg_loss1, pg_loss2).mean() - ent_coef * entropy.mean()

                    # Compute critic loss (value function loss)
                    newvalue = newvalue.view(-1)
                    if clip_vloss:
                        # Clipped value loss to prevent large value function updates
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        # Clip value updates similar to policy updates
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_coef,
                            clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        # Take the more conservative (higher) loss
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        critic_loss = 0.5 * v_loss_max.mean()
                    else:
                        # Standard MSE loss for value function
                        critic_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Update actor network parameters
                    actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)  
                    nn.utils.clip_grad_norm_(agent.actor_params(), max_grad_norm)  # Prevent exploding gradients
                    actor_optimizer.step()

                    # Update critic network parameters
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(agent.critic_params(), max_grad_norm)  # Prevent exploding gradients
                    critic_optimizer.step()

                # Early stopping if policy change is too large (prevents policy collapse)
                if target_kl is not None and approx_kl > target_kl:
                    break

            # Compute explained variance for monitoring value function learning
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)  # Variance of true returns
            explained_var = (  # Fraction of return variance explained by value function
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # Periodic checkpointing and past agent buffer update
            if iteration % 20 == 0:
                # Save model checkpoint
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                if not run_name:
                    checkpoint_name = f"agent_{seed}_train_step_{global_step}_timestamp_{timestamp}.pt"
                else:
                    checkpoint_name = f"agent_{run_name}_train_step_{global_step}_timestamp_{timestamp}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(agent.state_dict(), checkpoint_path)
                
                # Add current agent to past agents buffer for future self-play
                past_agents.append(clone_policy(agent))
            # Evaluate current agent performance
            # Use environment without exploration bonuses for fair evaluation
            eval_env = Sumo(contact_rew_weight=0.0, center_cost_weight=0.0)
            if frame_stack > 1:
                eval_env = FrameStackWrapper(eval_env, k=frame_stack)
            # Test agent against simple baseline (zero action) opponent
            reward, length = evaluate_self_play(eval_env, agent, ZeroActionAgent(envs), device)

            # Log all training metrics to TensorBoard
            writer.add_scalar("results/episode_reward", reward, global_step) 
            writer.add_scalar("results/episode_length", length, global_step)          
            writer.add_scalar("losses/entropy", np.mean(entropies), global_step)      
            writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)  
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)    
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)       
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)     
            writer.add_scalar("losses/explained_variance", explained_var, global_step) 

            pbar.update(n_envs * n_steps)


if __name__ == "__main__":
    # Run training function with command-line interface using typer
    # This allows all training hyperparameters to be configured via CLI arguments
    typer.run(train)