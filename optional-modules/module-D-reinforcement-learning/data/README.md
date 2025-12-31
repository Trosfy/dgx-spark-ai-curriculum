# Module D: Reinforcement Learning - Data

This module primarily uses **simulated environments** rather than static datasets.

## Environments Used

### Gymnasium Environments

All labs use [Gymnasium](https://gymnasium.farama.org/) (successor to OpenAI Gym):

| Environment | Lab | Description |
|-------------|-----|-------------|
| `FrozenLake-v1` | D.1, D.2 | 4x4 grid, navigate to goal without falling in holes |
| `CartPole-v1` | D.3, D.4 | Balance a pole on a cart by moving left/right |
| `LunarLander-v2` | D.4 | Land a spacecraft on a landing pad |

### Installation

```bash
pip install gymnasium
```

### Environment Details

#### FrozenLake-v1

```
SFFF    S = Start
FHFH    F = Frozen (safe)
FFFH    H = Hole (game over)
HFFG    G = Goal
```

- **State space**: 16 discrete states (4x4 grid)
- **Action space**: 4 discrete actions (Left, Down, Right, Up)
- **Reward**: +1 for reaching goal, 0 otherwise
- **Variants**:
  - `is_slippery=False`: Deterministic (easier)
  - `is_slippery=True`: 33% slip chance (harder)

#### CartPole-v1

- **State space**: 4 continuous values
  - Cart position (-4.8 to 4.8)
  - Cart velocity (-Inf to Inf)
  - Pole angle (-0.42 rad to 0.42 rad)
  - Pole angular velocity (-Inf to Inf)
- **Action space**: 2 discrete (Push left, Push right)
- **Reward**: +1 per timestep pole is balanced
- **Episode ends**: Pole > 12 degrees OR cart out of bounds OR 500 steps

#### LunarLander-v2

- **State space**: 8 continuous values
  - x, y position
  - x, y velocity
  - Angle, angular velocity
  - Leg contact (2 booleans)
- **Action space**: 4 discrete (Nothing, Left engine, Main engine, Right engine)
- **Reward**:
  - Landing pad contact: +100 to +140
  - Crash: -100
  - Each leg contact: +10
  - Engine firing: small negative
- **Solved**: Average reward >= 200 over 100 episodes

## Custom Grid Worlds

Lab D.1 includes custom MDP implementations:

### GridWorldMDP

A simple grid world for understanding MDP fundamentals:

```python
from notebooks.mdp_utils import GridWorldMDP

mdp = GridWorldMDP(grid_size=4, slip_prob=0.0)
state = mdp.start_state
next_state, reward, done = mdp.step(state, action=2)  # Move right
```

### ObstacleGridMDP

Extended grid world with obstacles:

```python
from notebooks.mdp_utils import ObstacleGridMDP

mdp = ObstacleGridMDP(grid_size=6, obstacles=[8, 9, 14, 15])
```

## RLHF Data (Lab D.5)

Lab D.5 uses:

1. **Pre-trained model**: `lvwerra/gpt2-imdb` (GPT-2 fine-tuned on movie reviews)
2. **Reward model**: `lvwerra/distilbert-imdb` (Sentiment classifier as proxy reward)

These are downloaded automatically from Hugging Face Hub.

### Preference Data Format

RLHF uses preference data in this format:

```json
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "I think it might be London."
}
```

For real RLHF training, you would need:
- ~10K-100K preference pairs
- Human annotators to rank responses
- Or synthetic preferences from a stronger model

## DGX Spark Considerations

### Memory Usage

| Environment | Typical Memory | Notes |
|-------------|---------------|-------|
| FrozenLake | < 100 MB | Tabular, minimal memory |
| CartPole | < 500 MB | Small networks |
| LunarLander | ~1 GB | Larger replay buffers |
| RLHF (GPT-2) | ~5-10 GB | Language model + reference |

DGX Spark's 128GB unified memory easily handles all these!

### Recommended Settings

```python
# For DQN on DGX Spark
buffer_size = 500000  # Can use larger buffers
batch_size = 256      # Larger batches for GPU efficiency

# For PPO on DGX Spark
rollout_length = 4096  # Longer rollouts
n_envs = 16           # More parallel environments
```

## Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - High-quality RL implementations
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file implementations
- [TRL Library](https://huggingface.co/docs/trl/) - Transformer RL for RLHF

## Generating Custom Data

If you need to create custom RL environments:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.zeros(4)
        return self.state, {}

    def step(self, action):
        # Your environment logic here
        next_state = self.state + 0.1
        reward = 1.0
        terminated = False
        truncated = False
        return next_state, reward, terminated, truncated, {}

# Register and use
gym.register(id='CustomEnv-v0', entry_point='custom_env:CustomEnv')
env = gym.make('CustomEnv-v0')
```
