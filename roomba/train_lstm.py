import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import os 

from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from roomba.environments.sumo_v1 import Sumo
from roomba.models.mlp import MlpContinuousActorCritic
from roomba.models.lstm import LSTMContinuousActorCritic
from roomba.models.base import LSTMZeroActionAgent
from roomba.utils import get_device, RunningMeanStd, clone_policy, set_random_seeds
from roomba.environments.wrappers import FrameStackWrapper, DomainRandomizationWrapper
from roomba.evaluate_lstm import evaluate_self_play

from torch.utils.tensorboard import SummaryWriter

class PPOTrainer:
    def __init__(
        self,
        env_mode: str = "uwb",
        total_timesteps=1_000_000,
        n_envs=512,
        n_steps=512,
        n_mbs=16,              # number of mini-batches
        epochs_per_update=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        ent_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.05,
        norm_adv=True,
        device=None, 
        extra_update_fn=None,   # an optional callback you might want to run after every update
        uwb_sensor_noise: float = 0.1,
        action_alpha: float = 0.4,
        obs_alpha: float = 0.8,
        domain_randomize: bool = False
    ):
        run_name = f"ns{n_steps}_uwb{uwb_sensor_noise:.2f}".replace('.', '_')
        # -- initialize TensorBoard writer --
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in locals().items()])),
        )
        self.batch_size = n_envs * n_steps
        if self.batch_size > total_timesteps:
            print("[WARNING]: each batch (n_envs * n_steps) has more elements than total_timesteps. Adjust n_steps or n_envs to not train for more timesteps than expected.")
        self.n_iterations = total_timesteps // self.batch_size + 1
        self.minibatch_size = self.batch_size // n_mbs
        self.total_envs = n_envs 
        env = Sumo(mode=env_mode, uwb_sensor_noise=uwb_sensor_noise, action_alpha=action_alpha, obs_alpha=obs_alpha)  # expects a function like make_vec_env(n_envs, **kwargs)
        env = pettingzoo_env_to_vec_env_v1(env)
        self.envs = concat_vec_envs_v1(env, num_vec_envs=n_envs//2, num_cpus=os.cpu_count(), base_class="gymnasium")
        test_obs, _ = self.envs.reset()
        print("Shape of obs after env reset:", np.array(test_obs).shape)
        self.envs.single_observation_space = self.envs.observation_space
        self.envs.single_action_space = self.envs.action_space
        self.envs.is_vector_env = True
        if domain_randomize:
            env = DomainRandomizationWrapper(env)

        obs_dim = int(np.prod(self.envs.observation_space.shape))
        action_dim = int(np.prod(self.envs.action_space.shape))

        # TODO: THINK ABOUT HOW TO ADD THE STARTING AGENT 
        self.agent = LSTMContinuousActorCritic(obs_dim, action_dim).to(device)
        self.total_timesteps = total_timesteps
        self.env_mode = env_mode
        self.uwb_sensor_noise = uwb_sensor_noise
        self.action_alpha = action_alpha
        self.obs_alpha = obs_alpha
        self.domain_randomize = domain_randomize
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_mbs = n_mbs
        self.epochs_per_update = epochs_per_update
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.device = device
        self.extra_update_fn = extra_update_fn

        # Create optimizers for actor and critic; assume the agent provides actor_params() and critic_params()
        self.actor_optimizer = optim.Adam(self.agent.actor_params(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.agent.critic_params(), lr=critic_lr, eps=1e-5)
        self.batch_size = self.n_envs * self.n_steps
        self.minibatch_size = self.batch_size // self.n_mbs

    def collect_rollout(self, seed=0):
        """Collect a rollout by interacting with the environment."""
        # Prepare buffers for observations, actions, etc.
        obs_shape = self.envs.single_observation_space.shape
        act_shape = self.envs.single_action_space.shape
        
        obs = torch.zeros((self.n_steps, self.n_envs) + obs_shape, device=self.device)
        actions = torch.zeros((self.n_steps, self.n_envs) + act_shape, device=self.device)
        logprobs = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        rewards = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        dones = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        values = torch.zeros((self.n_steps, self.n_envs), device=self.device)

        global_step = 0
        next_obs, infos = self.envs.reset(seed=[seed + i for i in range(self.n_envs)])
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.n_envs, device=self.device)

        rollout_info = {"reward": [], "length": [], "step": []}

        for step in range(self.n_steps):
            global_step += self.n_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(self.device)
            next_done = torch.Tensor(next_done_np).to(self.device)

            # Optionally, log episodic returns if provided in infos
            if "episode" in infos:
                for info in infos["episode"]:
                    r = infos["episode"]["r"][infos["episode"]["_r"]].mean()
                    l = infos["episode"]["l"][infos["episode"]["_l"]].mean()
                    if self.writer is not None:
                        self.writer.add_scalar("charts/episodic_return", r, global_step)
                        self.writer.add_scalar("charts/episodic_length", l, global_step)
                    rollout_info["reward"].append(r)
                    rollout_info["length"].append(l)
                    rollout_info["step"].append(global_step)

        # Compute advantages with GAE
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=self.device)
            lastgaelam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        batch = {
            "obs": obs.reshape((-1,) + obs_shape),
            "actions": actions.reshape((-1,) + act_shape),
            "logprobs": logprobs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": values.reshape(-1)
        }
        return batch, rollout_info

    def update_policy(self, batch):
        """Perform the PPO update using mini-batches from the rollout batch."""
        b_obs = batch["obs"]
        b_actions = batch["actions"]
        b_logprobs = batch["logprobs"]
        b_advantages = batch["advantages"]
        b_returns = batch["returns"]
        b_values = batch["values"]

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                B = b_obs[mb_inds].shape[0]
                # Suppose your LSTM has n_layers and hidden_dim
                n_layers = self.agent.shared.num_layers
                hidden_dim = self.agent.shared.hidden_size
                device = b_obs.device

                init_state = (
                    torch.zeros(n_layers, B, hidden_dim, device=device),
                    torch.zeros(n_layers, B, hidden_dim, device=device)
                )
                done_mb = torch.zeros(B, 1, device='cpu')  # Assume seq_len=1 for each mini-batch entry

                _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                    b_obs[mb_inds].unsqueeze(1),  # add time dimension
                    init_state,
                    done_mb,
                    b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                actor_loss = torch.max(pg_loss1, pg_loss2).mean() - self.ent_coef * entropy.mean()

                # Value loss
                critic_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.agent.actor_params(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic_params(), self.max_grad_norm)
                self.critic_optimizer.step()

            # Optionally break early if KL is too high
            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # (Optionally) log the losses and statistics here.
        metrics = {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs)
        }
        return metrics

    def train(self, seed=0):
        """Main training loop."""
        global_step = 0
        n_iterations = self.total_timesteps // self.batch_size + 1

        pbar = tqdm.tqdm(total=self.total_timesteps, desc="Training")
        for iteration in range(1, n_iterations + 1):
            batch, rollout_info = self.collect_rollout(seed=seed)
            metrics = self.update_policy(batch)
            
            global_step += self.batch_size
            pbar.update(self.batch_size)
            
            # Optionally log metrics to a writer
            if self.writer is not None:
                self.writer.add_scalar("losses/actor_loss", metrics["actor_loss"], global_step)
                self.writer.add_scalar("losses/critic_loss", metrics["critic_loss"], global_step)
                self.writer.add_scalar("losses/old_approx_kl", metrics["old_approx_kl"], global_step)
                self.writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
                self.writer.add_scalar("losses/clipfrac", metrics["clipfrac"], global_step)

        # (Optional) Execute any extra update callback passed to the trainer
        if self.extra_update_fn is not None:
            self.extra_update_fn(self.agent, iteration)

        eval_env = Sumo(
            mode=self.env_mode,
            contact_rew_weight=0.0,
            dist_center_weight=0.0,
            symmetry_rew_weight=0.0,
            uwb_sensor_noise=self.uwb_sensor_noise, 
            action_alpha=self.action_alpha, 
            obs_alpha=self.obs_alpha
        )
        action_dim = self.envs.single_action_space.shape[0]

        reward, length = evaluate_self_play(
            eval_env, self.agent, LSTMZeroActionAgent(action_dim), 'cpu'
        )
        self.writer.add_scalar("results/episode_reward_v_static_agent", reward, global_step)
        self.writer.add_scalar("results/episode_length_v_static_agent", length, global_step)

        pbar.close()

        # You can then run evaluation here if needed
        avg_reward = None  # Replace with your evaluation function
        return avg_reward


class PPOTrainerLSTM(PPOTrainer):
    def collect_rollout(self, seed=0):
        # Initialize extra buffer for LSTM hidden and cell states
        lstm_hidden = (
            torch.zeros(self.agent.shared.num_layers, self.n_envs, self.agent.shared.hidden_size, device=self.device),
            torch.zeros(self.agent.shared.num_layers, self.n_envs, self.agent.shared.hidden_size, device=self.device)
        )

        obs_shape = self.envs.single_observation_space.shape
        act_shape = self.envs.single_action_space.shape
        
        obs = torch.zeros((self.n_steps, self.n_envs) + obs_shape, device=self.device)
        actions = torch.zeros((self.n_steps, self.n_envs) + act_shape, device=self.device)
        logprobs = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        rewards = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        dones = torch.zeros((self.n_steps, self.n_envs), device=self.device)
        values = torch.zeros((self.n_steps, self.n_envs), device=self.device)

        global_step = 0
        # TODO: ARE WE SURE THAT WE ARE RESETTING ALL THE ENVS?
        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.n_envs, device=self.device)

        # Save initial LSTM states (the same for all steps)
        initial_state = (lstm_hidden[0].clone(), lstm_hidden[1].clone())

        for step in range(self.n_steps):
            global_step += self.n_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Add a sequence dimension (seq_len=1) for LSTM input
            with torch.no_grad():
                obs_seq = next_obs.unsqueeze(1)
                done_seq = next_done.unsqueeze(1)
                action, logprob, _, value, lstm_hidden = self.agent.get_action_and_value(
                    obs_seq, lstm_hidden, done_seq
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(self.device)
            next_done = torch.Tensor(next_done_np).to(self.device)

        # Compute returns and advantages similarly, making sure to use the final LSTM state if needed.
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs.unsqueeze(1), lstm_hidden, next_done.unsqueeze(1)).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=self.device)
            lastgaelam = 0
            for t in reversed(range(self.n_steps)):
                if t == self.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        batch = {
            "obs": obs.reshape((-1,) + obs_shape),
            "actions": actions.reshape((-1,) + act_shape),
            "logprobs": logprobs.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": values.reshape(-1)
        }

        checkpoint_save_path = 'checkpoints_lstm'
        os.makedirs(checkpoint_save_path, exist_ok=True)

        torch.save(
            self.agent, 
            os.path.join(checkpoint_save_path, 'test.pt')
        )
        return batch, {}  # you can add extra rollout info if needed
    
if __name__ == "__main__":
    trainer = PPOTrainerLSTM()
    trainer.train()