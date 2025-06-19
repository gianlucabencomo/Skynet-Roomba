# AI Roomba Fight Hackathon

![Evil Roomba](docs/assets/room_team_white.png)

Welcome to the **AI Roomba Fight Hackathon**! Train autonomous AI Roombas to fight in wrestling matches using reinforcement learning. The goal is simple: **push your opponent out of the ring while staying in yourself!**

## What is This?

This codebase serves as a starting point to train your fighters built on top of a MuJoCo physics simulation where two Roombas compete in wrestling matches. Your challenge is to train the most effective fighter using any RL techniques you can think of!

## Installation

### Prerequisites

- Python 3.12

### Setup with uv

```bash
uv sync
```

## Train Your First Roomba Fighter

Start training your first Roomba warrior:

```bash
python ppo.py --total-timesteps 1000000 --n-envs 512
```

This will:
- Train a PPO agent for 1M timesteps
- Use 512 parallel environments for faster training
- Save checkpoints every 20 iterations in `checkpoints/`
- Log training metrics to TensorBoard in `runs/`

## Watch Your Fighter in Action

Once you have a trained model, watch it fight:

```bash
python watch.py --ckpt1 checkpoints/your_model.pt --episodes 5
```

## The Sumo Environment

The competition takes place in a physics-based sumo ring environment defined in `environments/sumo_v2.py`. Here's what you need to know:

### Game Rules

- **Objective**: Push your opponent out of the ring while staying in yourself
- **Match Duration**: 60 seconds maximum
- **Victory Conditions**:
  - Push opponent out → +1000 reward, opponent gets -1000
  - Get pushed out → -1000 reward, opponent gets +1000  
  - Time runs out → Both get -1000 (draw)

### Robot Design

The **simulated** Roombas are defined in `environments/assets/roomba_v6.xml` and feature:
- **Actuators**: Two motorized wheels (left/right)
- **Range sensors**: Two rangefinder sensors (`side_left_range` and `side_right_range`) that detect distances to objects

The **real** Roombas have TODO


### Observation Space

TODO

### Action Space

Agents control their Roomba through:
- **Continuous actions**: Torque commands for left and right wheels
- **Action range**: [-1, 1] for each wheel (defined by `ctrlrange` in the XML)
- **Differential drive**: Different wheel speeds create turning, same speeds create straight movement

## PPO Self-Play Training

We've implemented a complete PPO (Proximal Policy Optimization) training system with self-play in `ppo.py` for you to get started. Here's how it works:

### Self-Play Mechanism

The system uses sophisticated self-play where:
- **Past Agent Buffer**: Maintains a buffer of up to 50 previous versions of your agent
- **Opponent Sampling**: Each training episode randomly selects an opponent from this buffer
- **Curriculum Learning**: Starts with simple opponents (random/zero actions) and gradually faces stronger opponents
- **Prevents Overfitting**: Diverse opponents prevent your agent from exploiting a single strategy

### PPO Implementation Features

- **Vectorized Environments**: Trains on parallel environments simultaneously
- **Generalized Advantage Estimation (GAE)**: Advanced advantage computation for stable learning
- **Separate Actor-Critic Optimizers**: Different learning rates for policy and value networks
- **Reward Normalization**: Automatic reward scaling for training stability
- **Gradient Clipping**: Prevents exploding gradients during training

### Key Training Parameters

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `--total-timesteps` | Total training steps | 100M | Start with 1-10M for testing |
| `--n-envs` | Parallel environments | 1024 | Higher = faster (if you have CPU/memory) |
| `--n-steps` | Steps per rollout | 256 | Larger = more stable |
| `--actor-lr` | Actor learning rate | 1e-3 | Lower if training is unstable |
| `--critic-lr` | Critic learning rate | 1e-2 | Usually higher than actor |
| `--frame-stack` | Frame stacking | 1 | 2-4 can help with temporal dynamics |

### Adding New Algorithms

Want to implement a different RL algorithm? The `ppo.py` file is an excellent starting point:
1. **Copy `ppo.py`** to `your_algorithm.py`
2. **Replace the PPO training logic** with your algorithm (SAC, TD3, etc.)
3. **Modify and add model architecture** in `models/` if needed

### Advanced Training Command

For serious training runs, try this configuration:

```bash
python ppo.py \
    --total-timesteps 50000000 \
    --n-envs 2048 \
    --n-steps 512 \
    --actor-lr 5e-4 \
    --critic-lr 5e-3 \
    --frame-stack 3 \
    --run-name "my_champion" \
    --past-agent-buffer-size 100
```

### Monitor Training Progress

View training progress in real-time:

```bash
tensorboard --logdir runs/
```

Open http://localhost:6006 to see:
- Episode rewards and win rates
- Policy and value losses
- Learning curves and statistics

## Tips for Building a Champion Fighter

Here are key areas to focus on to improve your Roomba's fighting ability:

### 1. Reward Engineering (`environments/sumo_v2.py`)

Modify the reward function to encourage specific behaviors:
- **Aggressive contact**: Increase `contact_rew_weight` for more aggressive fighters
- **Center control**: Adjust `center_cost_weight` to encourage/discourage center positioning
- **Custom rewards**: Add your own reward terms for contact forces, distance to opponent, etc.

### 2. The sim2real Gap

TODO

### 3. Model Architecture (`models/mlp.py`)

Experiment with different neural network architectures:
- **Network size**: Try different `HIDDEN_WIDTHS` configurations
- **Layer types**: Add LSTM layers for memory, attention mechanisms, etc.
- **Observation processing**: Modify the observation normalization and processing

### 4. Training Hyperparameters

Key parameters to tune in your training:
- **Learning rates**: Balance actor and critic learning rates
- **Frame stacking**: Use `--frame-stack 2-4` to give your agent memory
- **Environment count**: Increase `--n-envs` for more diverse training data
- **Training duration**: Longer training often leads to better strategies

### 5. Advanced RL Techniques

- **Curriculum learning**: Start with easier opponents and gradually increase difficulty
- **Domain randomization**: Vary environment parameters during training
- **Multi-agent techniques**: Implement population-based training
- **Different algorithms**: Try SAC, TD3, or other state-of-the-art RL algorithms

## Repository Structure

Here's how the codebase is organized:

### Core Training Files
- **`ppo.py`** - Main PPO training script with self-play (your starting point!)
- **`watch.py`** - Visualize and evaluate trained agents
- **`evaluate.py`** - Automated evaluation functions
- **`utils.py`** - Utility functions (device selection, etc.)

### Models (`models/`)
- **`mlp.py`** - MLP-based Actor-Critic networks (modify for new architectures)
- **`base.py`** - Simple baseline agents (ZeroAction, RandomAction)
- **`helper.py`** - Neural network utilities (initialization, normalization)

### Environment (`environments/`)
<!-- - **`sumo_v2.py`** - Main sumo environment (modify rewards here!)
- **`wrappers.py`** - Environment wrappers (frame stacking, preprocessing)
- **`mujoco_env.py`** - MuJoCo physics integration
- **`assets/roomba_v6.xml`** - Robot model definition (modify robot design) -->

### Generated During Training
- **`checkpoints/`** - Saved model weights (`.pt` files)
- **`runs/`** - TensorBoard logs and training metrics

---

Good luck!
