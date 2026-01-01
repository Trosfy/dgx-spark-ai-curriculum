# Module D: Reinforcement Learning - Quick Reference

## 🔑 Core Formulas

### Bellman Equation

```
Q(s, a) = R(s, a) + γ · max_a' Q(s', a')
```

**In words:** Value of state-action = immediate reward + discounted future value

### Q-Learning Update

```
Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') - Q(s, a)]
                         \_______TD Target_______/  \_Current_/
                              \_______TD Error________/
```

### Policy Gradient

```
∇J(θ) = E[∇log π(a|s; θ) · Q(s, a)]
```

**In words:** Increase probability of actions that lead to high rewards

### PPO Clipped Objective

```
L(θ) = min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A)

where r(θ) = π(a|s; θ) / π_old(a|s; θ)
```

---

## 📊 Key Algorithms

### Q-Learning (Tabular)

```python
import numpy as np

def q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.99, epsilon_start=1.0):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = epsilon_start

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # Q-learning update
            best_next = 0 if done else np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            state = next_state

        epsilon *= 0.995  # Decay

    return Q
```

### Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(torch.stack, zip(*[
            (torch.tensor(s), torch.tensor([a]), torch.tensor([r]),
             torch.tensor(s2), torch.tensor([d]))
            for s, a, r, s2, d in batch
        ]))

def train_dqn(q_net, target_net, buffer, optimizer, gamma=0.99):
    if len(buffer.buffer) < 64:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(64)
    actions = actions.long()

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + gamma * next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### PPO (Simplified)

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(hidden, action_dim), nn.Softmax(-1))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

def ppo_update(model, optimizer, states, actions, old_log_probs,
               returns, advantages, clip_eps=0.2, epochs=10):
    for _ in range(epochs):
        probs, values = model(states)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)

        # PPO clipped objective
        ratio = (new_log_probs - old_log_probs).exp()
        clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        # Entropy bonus (encourages exploration)
        entropy = dist.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🎮 Common Environments

### Gymnasium (formerly Gym)

```python
import gymnasium as gym

# Discrete (tabular RL)
env = gym.make("FrozenLake-v1")           # 16 states, 4 actions
env = gym.make("Taxi-v3")                 # 500 states, 6 actions

# Continuous (deep RL)
env = gym.make("CartPole-v1")             # Balance pole on cart
env = gym.make("LunarLander-v2")          # Land spacecraft
env = gym.make("MountainCar-v0")          # Drive car up hill

# Environment API
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

### Environment Anatomy

```python
print(f"State space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"State shape: {env.observation_space.shape}")
print(f"Num actions: {env.action_space.n}")  # For discrete
```

---

## 📐 Key Hyperparameters

### Q-Learning

| Parameter | Symbol | Typical Range | Effect |
|-----------|--------|---------------|--------|
| Learning rate | α | 0.01 - 0.5 | Update speed |
| Discount factor | γ | 0.9 - 0.999 | Future reward weight |
| Exploration rate | ε | 1.0 → 0.01 | Random vs greedy |
| Epsilon decay | - | 0.99 - 0.999 | Exploration schedule |

### DQN

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate | 1e-4 to 1e-3 | Lower than tabular |
| Batch size | 32-128 | From replay buffer |
| Buffer size | 10,000-100,000 | Experience replay |
| Target update | Every 100-1000 steps | Stability |
| Epsilon decay | Over 10,000+ steps | Slower decay |

### PPO

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate | 3e-4 | Adam optimizer |
| Clip epsilon | 0.1-0.3 | Policy constraint |
| GAE lambda | 0.95 | Advantage estimation |
| Entropy coef | 0.01 | Exploration bonus |
| Value coef | 0.5 | Critic loss weight |
| Epochs per update | 3-10 | Reuse experience |
| Horizon | 128-2048 | Steps before update |

---

## 🔗 RLHF Connection

### From RL to RLHF

```python
# Standard RL
reward = environment.step(action)  # From environment

# RLHF for LLMs
reward = reward_model(prompt, response)  # From learned preferences

# The structure is the same!
# - State = prompt
# - Action = generated token (or full response)
# - Policy = the LLM itself
# - Reward = human preference (via reward model)
```

### PPO for LLMs (Conceptual)

```python
def rlhf_step(policy_model, ref_model, reward_model, prompt):
    # 1. Generate response
    response = policy_model.generate(prompt)

    # 2. Get reward
    reward = reward_model(prompt, response)

    # 3. Compute KL penalty (stay close to reference)
    kl_penalty = kl_divergence(policy_model, ref_model, response)

    # 4. Adjusted reward
    total_reward = reward - kl_coef * kl_penalty

    # 5. PPO update
    ppo_update(policy_model, prompt, response, total_reward)
```

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| No exploration | Start with high ε, decay slowly |
| Unstable DQN | Use target network, experience replay |
| PPO not learning | Check advantage normalization |
| Wrong reward scale | Normalize rewards |
| Sparse rewards | Add reward shaping |
| γ too low | Increase for long-horizon tasks |
| γ = 1 | Use < 1 for episodic tasks |

---

## 📈 Expected Performance

### CartPole-v1

| Algorithm | Episodes to Solve | Avg Reward |
|-----------|-------------------|------------|
| Random | - | ~20 |
| DQN | 200-500 | 475+ |
| PPO | 100-300 | 500 |

### FrozenLake-v1 (Deterministic)

| Algorithm | Episodes to Solve | Success Rate |
|-----------|-------------------|--------------|
| Q-Learning | 500-1000 | 100% |
| Random | - | ~1% |

---

## 🔗 Quick Links

- Lab 1: Tabular Q-Learning (FrozenLake)
- Lab 2: Deep Q-Network (CartPole)
- Lab 3: Policy Gradients
- Lab 4: PPO Implementation
- Lab 5: RLHF Connection
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
