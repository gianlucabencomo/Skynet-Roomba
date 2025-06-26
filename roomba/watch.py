import typer
import torch
from copy import deepcopy

from roomba.environments.sumo_v1 import Sumo
from roomba.environments.wrappers import FrameStackWrapper
from roomba.models.mlp import MlpContinuousActorCritic
from roomba.utils import get_device



def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
):  
    agent = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.eval()
    return agent


def visualize_match(env, agent1, agent2, device, n_episodes=5):
    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")
        observations = env.reset()[0]  # PettingZoo environments return (obs, info)

        observations = {
            agent: torch.tensor(obs, dtype=torch.float32).to(device)
            for agent, obs in observations.items()
        }

        episode_reward1 = 0
        episode_reward2 = 0
        terminated = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}

        while not (any(terminated.values()) or any(truncated.values())):
            # Agent 1 action
            with torch.no_grad():
                action1, _, _, _ = agent1.get_action_and_value(
                    observations["maximus"].reshape(1, -1)
                )

            # Agent 2 action
            with torch.no_grad():
                action2, _, _, _ = agent2.get_action_and_value(
                    observations["commodus"].reshape(1, -1)
                )

            # Convert to numpy and dict format for environment
            actions = {
                "maximus": action1.cpu().numpy()[0],
                "commodus": action2.cpu().numpy()[0],
            }

            # Step the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Convert observations back to tensor
            observations = {
                agent: torch.tensor(obs, dtype=torch.float32).to(device)
                for agent, obs in observations.items()
            }

            # Update metrics
            episode_reward1 += rewards["maximus"]
            episode_reward2 += rewards["commodus"]
            terminated = terminations
            truncated = truncations

            # Render
            env.render()

        print(
            f"Episode rewards - Agent1: {episode_reward1:.2f}, Agent2: {episode_reward2:.2f}"
        )

    env.close()


def main(
    ckpt1: str,
    ckpt2: str = None,
    episodes: int = 5,
    frame_stack: int = 1,
    env_mode: str = "uwb",
):
    device = get_device()
    env = Sumo(mode=env_mode, train=False, render_mode="human")
    if frame_stack > 1:
        env = FrameStackWrapper(env, k=frame_stack)

    obs_dim = env.observation_spaces["maximus"].shape[
        0
    ]  # Use actual shape from environment
    action_dim = env.action_spaces["maximus"].shape[0]

    agent1 = load_checkpoint(ckpt1)
    if ckpt2 is None:
        agent2 = deepcopy(agent1)
    else:
        agent2 = load_checkpoint(ckpt2)

    # Run visualization
    print(f"Starting visualization for {episodes} episodes...")
    visualize_match(env, agent1, agent2, device, n_episodes=episodes)


if __name__ == "__main__":
    typer.run(main)
