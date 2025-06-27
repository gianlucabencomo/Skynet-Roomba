import typer
import torch
from copy import deepcopy

from roomba.environments.sumo_v1 import Sumo
from roomba.environments.wrappers import FrameStackWrapper
from roomba.utils import get_device
from roomba.server.helper import get_framestack_size, load_checkpoint


def visualize_match(env, agent1, agent2, device, n_episodes=5):
    # --- LSTM: Setup hidden states and done tensors ---
    def get_init_state(agent):
        n_layers = agent.shared.num_layers
        hidden_size = agent.shared.hidden_size
        h = torch.zeros(n_layers, 1, hidden_size, device=device)
        c = torch.zeros(n_layers, 1, hidden_size, device=device)
        return (h, c)
    
    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")
        observations = env.reset()[0]

        observations = {
            agent: torch.tensor(obs, dtype=torch.float32).to(device)
            for agent, obs in observations.items()
        }

        episode_reward1 = 0
        episode_reward2 = 0
        terminated = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}

        # --- LSTM: initialize hidden and done for each agent ---
        state1 = get_init_state(agent1)
        state2 = get_init_state(agent2)
        done1 = torch.zeros(1, 1, device=device)
        done2 = torch.zeros(1, 1, device=device)

        while not (any(terminated.values()) or any(truncated.values())):
            # Add batch and seq dims for LSTM
            obs1 = observations["maximus"].reshape(1, 1, -1)
            obs2 = observations["commodus"].reshape(1, 1, -1)

            # Agent 1 action (LSTM)
            with torch.no_grad():
                action1, _, _, _, state1 = agent1.get_action_and_value(
                    obs1, state1, done1
                )
            # Agent 2 action (LSTM)
            with torch.no_grad():
                action2, _, _, _, state2 = agent2.get_action_and_value(
                    obs2, state2, done2
                )

            actions = {
                "maximus": action1.cpu().numpy().flatten(),
                "commodus": action2.cpu().numpy().flatten(),
            }

            observations, rewards, terminations, truncations, infos = env.step(actions)
            observations = {
                agent: torch.tensor(obs, dtype=torch.float32).to(device)
                for agent, obs in observations.items()
            }

            episode_reward1 += rewards["maximus"]
            episode_reward2 += rewards["commodus"]
            terminated = terminations
            truncated = truncations

            # --- LSTM: update done tensors for each agent ---
            done1 = torch.tensor([[terminated["maximus"] or truncated["maximus"]]], dtype=torch.float32, device=device)
            done2 = torch.tensor([[terminated["commodus"] or truncated["commodus"]]], dtype=torch.float32, device=device)

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
    env = Sumo(mode=env_mode, train=False, render_mode="human")
    
    agent1 = load_checkpoint(ckpt1)

    # frame_stack1 = get_framestack_size(agent1, env)        

    if ckpt2 is None:
        print('Setting roomba 2 to roomba 1.')
        agent2 = deepcopy(agent1)
        # frame_stack2 = frame_stack1
    else:
        agent2 = load_checkpoint(ckpt2)
        # frame_stack2 = get_framestack_size(agent2, env)

# \    env = FrameStackWrapper(env, k=[frame_stack1, frame_stack2])

    # Run visualization
    print(f"Starting visualization for {episodes} episodes...")
    visualize_match(env, agent1, agent2, device, n_episodes=episodes)


if __name__ == "__main__":
    typer.run(main)
