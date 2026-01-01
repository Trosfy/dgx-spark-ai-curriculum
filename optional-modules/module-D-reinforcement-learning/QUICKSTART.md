# Module D: Reinforcement Learning - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build

A Q-learning agent that learns to navigate a frozen lake - the "hello world" of reinforcement learning.

## ✅ Before You Start

- [ ] Python with NumPy installed
- [ ] `pip install gymnasium` completed

## 🚀 Let's Go!

### Step 1: Create the Environment

```python
import gymnasium as gym
import numpy as np

# Create FrozenLake: Navigate from S to G without falling in H
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

print(env.reset()[0])  # Starting state
print(env.render())    # Visual representation
```

**Expected output:**
```
0
SFFF
FHFH
FFFH
HFFG
```
(S=Start, F=Frozen/safe, H=Hole/bad, G=Goal)

### Step 2: Create Q-Table

```python
# Q-table: stores expected rewards for each state-action pair
# 16 states (4x4 grid), 4 actions (up, right, down, left)
Q = np.zeros((16, 4))
print(f"Q-table shape: {Q.shape}")
```

### Step 3: Train with Q-Learning!

```python
# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.99    # Discount factor
epsilon = 1.0    # Exploration rate
n_episodes = 1000

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Epsilon-greedy: explore or exploit?
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Random
        else:
            action = np.argmax(Q[state])        # Best known

        # Take action, observe result
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s')) - Q(s,a)]
        best_next = np.max(Q[next_state]) if not done else 0
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state

    epsilon *= 0.995  # Decay exploration

print("Training complete!")
print(f"Final Q-values for start state: {Q[0]}")
```

### Step 4: Watch the Agent!

```python
# Test the learned policy
state, _ = env.reset()
done = False
steps = 0

print("Agent's path:")
print(env.render())

while not done:
    action = np.argmax(Q[state])  # Use learned policy
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    steps += 1
    print(f"Step {steps}:")
    print(env.render())

print(f"\n{'SUCCESS!' if reward > 0 else 'Failed :('} in {steps} steps")
```

**Expected output:**
```
Agent's path:
SFFF
FHFH
FFFH
HFFG

Step 1:
SFFF
FHFH
FFFH
HFFG
...
SUCCESS! in 6 steps
```

## 🎉 You Did It!

You just trained an RL agent from scratch! It learned:
- **States**: The 16 positions on the grid
- **Actions**: Up, right, down, left
- **Rewards**: +1 for reaching goal, 0 otherwise
- **Policy**: Which action to take in each state

## ▶️ Next Steps

1. **Deep RL**: Use neural networks for Q-values (Notebook 02: DQN)
2. **Policy gradients**: Directly learn the policy (Notebook 03)
3. **PPO**: The algorithm behind ChatGPT's RLHF (Notebook 04)
4. **RLHF connection**: See how this applies to LLMs (Notebook 05)

---

## 💡 The Key Insight

> **RL is learning from trial and error.**
>
> The agent:
> 1. Tries actions randomly at first
> 2. Gets rewards (or punishments)
> 3. Updates its knowledge (Q-table)
> 4. Gradually learns the best strategy
>
> This is exactly how RLHF trains ChatGPT - but with human preferences as rewards!
