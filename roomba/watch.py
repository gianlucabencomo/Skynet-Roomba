import typer
import torch
from copy import deepcopy

from roomba.environments.sumo_v1 import Sumo
from roomba.environments.wrappers import FrameStackWrapper
from roomba.utils import get_device
from roomba.server.helper import get_framestack_size, load_checkpoint



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
                    observations["maximus"].reshape(1, -1),
                    deterministic=True
                )

            # Agent 2 action
            with torch.no_grad():
                action2, _, _, _ = agent2.get_action_and_value(
                    observations["commodus"].reshape(1, -1),
                    deterministic=True
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
    env_mode: str = "uwb",
):
    device = get_device()
    env = Sumo(mode=env_mode, train=False, render_mode="human", uwb_sensor_noise=0.15)
    
    agent1 = load_checkpoint(ckpt1)

    frame_stack1 = get_framestack_size(agent1, env)        

    if ckpt2 is None:
        print('Setting roomba 2 to roomba 1.')
        agent2 = deepcopy(agent1)
        frame_stack2 = frame_stack1
    else:
        agent2 = load_checkpoint(ckpt2)
        frame_stack2 = get_framestack_size(agent2, env)

    env = FrameStackWrapper(env, k=[frame_stack1, frame_stack2])

    # Run visualization
    print(f"Starting visualization for {episodes} episodes...")
    visualize_match(env, agent1, agent2, device, n_episodes=episodes)


if __name__ == "__main__":
    typer.run(main)
