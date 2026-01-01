# Optional Module D: Reinforcement Learning

**Category:** Optional - AI Fundamentals
**Duration:** 8-10 hours
**Prerequisites:** Module 1.5 (Neural Networks), Module 2.1 (PyTorch)
**Priority:** P3 (Optional - Foundation for RLHF)

---

## Overview

Reinforcement learning (RL) trains agents to make sequential decisions by learning from rewards. While the curriculum focuses on LLMs, understanding RL is essential for grasping RLHF (Reinforcement Learning from Human Feedback), the technique that makes ChatGPT helpful rather than just coherent.

**Why This Matters:** RLHF transformed language models from next-token predictors into useful assistants. Understanding RL fundamentals helps you fine-tune models for specific behaviors, implement reward modeling, and understand why some training approaches work better than others.

### The Kitchen Table Explanation

Imagine training a dog. You can't explain calculus to a dog, but you can give treats when it does something good (sit!) and withhold treats when it doesn't (ignore commands). Over time, the dog learns what behaviors lead to treats. RL is the mathematical framework for this: an agent (the dog) takes actions in an environment, receives rewards, and learns a policy (what to do) that maximizes total reward. For LLMs, the "action" is generating a response, and the "reward" comes from a human rating or a reward model.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Understand the Markov Decision Process framework
- ✅ Implement Q-learning and Deep Q-Networks
- ✅ Train agents using policy gradient methods (PPO)
- ✅ Connect RL concepts to LLM fine-tuning (RLHF)

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| D.1 | Define and solve Markov Decision Processes | Understand |
| D.2 | Implement Q-learning for tabular environments | Apply |
| D.3 | Build and train Deep Q-Networks (DQN) | Apply |
| D.4 | Implement Proximal Policy Optimization (PPO) | Apply |
| D.5 | Explain how PPO is used in RLHF for LLMs | Understand |

---

## Topics

### D.1 Markov Decision Processes

- **MDP Components**
  - States (S), Actions (A), Transitions (P)
  - Rewards (R), Discount factor (γ)
  - Markov property: future depends only on present

- **Value Functions**
  - State value V(s): expected return from state s
  - Action value Q(s, a): expected return from (s, a)
  - Bellman equations

- **Optimal Policies**
  - Policy π(a|s): probability of action given state
  - Optimal policy π*
  - Value iteration and policy iteration

### D.2 Q-Learning

- **Temporal Difference Learning**
  - Learning from incomplete episodes
  - TD(0) update rule
  - Off-policy vs on-policy learning

- **Q-Learning Algorithm**
  - Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
  - Exploration vs exploitation (ε-greedy)
  - Convergence guarantees

- **Limitations**
  - Curse of dimensionality
  - Continuous state spaces
  - Function approximation needed

### D.3 Deep Q-Networks (DQN)

- **Neural Network Function Approximation**
  - Q-network: Q(s, a; θ) ≈ Q*(s, a)
  - Input: state (or image)
  - Output: Q-values for all actions

- **Key DQN Innovations**
  - Experience replay buffer
  - Target networks for stability
  - Frame stacking for Atari

- **Extensions**
  - Double DQN (reduce overestimation)
  - Dueling DQN (separate value and advantage)
  - Prioritized experience replay

### D.4 Policy Gradient Methods

- **Policy Gradient Theorem**
  - Directly optimize policy parameters
  - ∇J(θ) = E[∇log π(a|s;θ) · Q(s,a)]
  - REINFORCE algorithm

- **Actor-Critic**
  - Actor: learns policy π(a|s;θ)
  - Critic: learns value function V(s;φ)
  - Advantage: A(s,a) = Q(s,a) - V(s)

- **Proximal Policy Optimization (PPO)**
  - Clipped objective for stable updates
  - Trust region approximation
  - Current standard for LLM fine-tuning

### D.5 RL for Language Models

- **RLHF Pipeline**
  - Supervised fine-tuning (SFT)
  - Reward model training
  - PPO optimization against reward model
  - KL divergence constraint

- **Why PPO for LLMs**
  - Text generation as sequential decision
  - Reward at end of sequence
  - Policy is the LM, action is next token

- **Modern Alternatives**
  - DPO: Direct Preference Optimization
  - IPO, KTO: Simpler preference methods
  - Why these avoid explicit RL

---

## Labs

### Lab D.1: Tabular Q-Learning
**Time:** 1.5 hours

Implement Q-learning from scratch on classic environments.

**Instructions:**
1. Implement MDP components (states, actions, transitions)
2. Solve FrozenLake-v1 with Q-learning
3. Implement ε-greedy exploration with decay
4. Visualize Q-table evolution during training
5. Compare different learning rates and discount factors
6. Achieve >70% success rate on FrozenLake

**Deliverable:** Notebook with Q-learning implementation and analysis

---

### Lab D.2: Deep Q-Network on CartPole
**Time:** 2 hours

Build DQN to solve CartPole (balance a pole on a cart).

**Instructions:**
1. Implement Q-network with PyTorch
2. Build experience replay buffer
3. Implement target network with periodic updates
4. Train on CartPole-v1
5. Visualize learning curves
6. Ablate: compare with/without replay and target network
7. Achieve average reward >450 over 100 episodes

**Deliverable:** Working DQN with training visualizations

---

### Lab D.3: Policy Gradient and Actor-Critic
**Time:** 2 hours

Implement REINFORCE and advantage actor-critic.

**Instructions:**
1. Implement REINFORCE algorithm
2. Add baseline (actor-critic)
3. Compare variance of gradient estimates
4. Train on CartPole and LunarLander
5. Implement Generalized Advantage Estimation (GAE)
6. Visualize policy entropy during training

**Deliverable:** Notebook comparing REINFORCE vs actor-critic

---

### Lab D.4: PPO Implementation
**Time:** 2.5 hours

Implement PPO from scratch, the algorithm behind RLHF.

**Instructions:**
1. Implement PPO clipped objective
2. Add value function loss
3. Add entropy bonus for exploration
4. Implement minibatch updates from rollouts
5. Train on CartPole and LunarLander
6. Compare to vanilla policy gradient
7. Experiment with clipping parameter

**Deliverable:** Working PPO implementation

---

### Lab D.5: RLHF Conceptual Lab
**Time:** 2 hours

Connect RL concepts to LLM fine-tuning.

**Instructions:**
1. Review TRL (Transformer Reinforcement Learning) library
2. Trace PPOTrainer code to understand RLHF pipeline
3. Implement simplified reward model (using sentiment classifier)
4. Fine-tune small LM with your reward model
5. Observe KL divergence during training
6. Compare PPO-tuned vs DPO-tuned models

**Deliverable:** Notebook demonstrating RLHF concepts

---

## Guidance

### Markov Decision Process Basics

```python
import numpy as np

class SimpleMDP:
    """
    Simple grid world MDP.
    Agent moves in 4x4 grid, goal is bottom-right corner.
    """

    def __init__(self):
        self.grid_size = 4
        self.n_states = self.grid_size ** 2
        self.n_actions = 4  # up, right, down, left
        self.goal_state = self.n_states - 1
        self.gamma = 0.99

    def step(self, state, action):
        """Take action, return (next_state, reward, done)."""
        row, col = state // self.grid_size, state % self.grid_size

        # Movement deltas: up, right, down, left
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = deltas[action]

        # Apply movement (stay in bounds)
        new_row = max(0, min(self.grid_size - 1, row + dr))
        new_col = max(0, min(self.grid_size - 1, col + dc))
        next_state = new_row * self.grid_size + new_col

        # Reward: +1 at goal, -0.01 otherwise (encourages efficiency)
        done = (next_state == self.goal_state)
        reward = 1.0 if done else -0.01

        return next_state, reward, done

    def value_iteration(self, threshold=1e-6):
        """Solve MDP with value iteration."""
        V = np.zeros(self.n_states)

        while True:
            V_new = np.zeros(self.n_states)
            for s in range(self.n_states):
                if s == self.goal_state:
                    V_new[s] = 0  # Terminal state
                    continue

                values = []
                for a in range(self.n_actions):
                    next_s, r, _ = self.step(s, a)
                    values.append(r + self.gamma * V[next_s])
                V_new[s] = max(values)

            if np.max(np.abs(V_new - V)) < threshold:
                break
            V = V_new

        return V

mdp = SimpleMDP()
V = mdp.value_iteration()
print("Optimal Value Function:")
print(V.reshape(4, 4))
```

### Q-Learning Implementation

```python
import numpy as np
import gymnasium as gym

def q_learning(env, n_episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
    """
    Tabular Q-learning for discrete state/action spaces.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Q-learning update
            best_next = np.max(Q[next_state]) if not done else 0
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            if done:
                break

        episode_rewards.append(total_reward)
        epsilon *= epsilon_decay  # Decay exploration

    return Q, episode_rewards

# Train on FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=False)
Q, rewards = q_learning(env)

# Evaluate learned policy
def evaluate_policy(env, Q, n_episodes=100):
    successes = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        while True:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                successes += reward  # reward=1 only on success
                break
    return successes / n_episodes

print(f"Success rate: {evaluate_policy(env, Q):.2%}")
```

### Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """Experience replay buffer for stable training."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN agent with target network and experience replay."""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma

        # Q-network and target network
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Update Q-network
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy Q-network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

### Proximal Policy Optimization (PPO)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

class PPO:
    """
    Proximal Policy Optimization.

    This is the algorithm used for RLHF in LLMs like ChatGPT.
    """

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, epochs=10, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value, gae_lambda=0.95):
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        """PPO update with clipped objective."""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            probs, values = self.policy(states)
            values = values.squeeze()
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = (new_log_probs - old_log_probs).exp()
            clipped_ratio = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
```

### Connecting to RLHF

```python
# Conceptual code showing how PPO is used in RLHF

def rlhf_training_step(
    policy_model,      # The LLM being fine-tuned
    reference_model,   # Frozen copy of initial LLM
    reward_model,      # Trained to predict human preferences
    prompt_batch,      # Input prompts
    ppo_trainer,       # PPO algorithm
    kl_coef=0.1        # KL penalty coefficient
):
    """
    Single RLHF training step.

    This is a simplified version of what happens in libraries like TRL.
    """
    # 1. Generate responses from current policy
    with torch.no_grad():
        responses = policy_model.generate(prompt_batch)

    # 2. Get reward from reward model
    rewards = reward_model(prompt_batch, responses)

    # 3. Compute KL divergence penalty (stay close to reference)
    policy_logprobs = policy_model.log_prob(responses)
    reference_logprobs = reference_model.log_prob(responses)
    kl_penalty = kl_coef * (policy_logprobs - reference_logprobs)

    # 4. Adjusted reward = reward - KL penalty
    adjusted_rewards = rewards - kl_penalty

    # 5. PPO update
    ppo_trainer.update(
        states=prompt_batch,
        actions=responses,
        rewards=adjusted_rewards
    )

# The key insight: in RLHF, each token generation is an "action",
# but reward is only given at the end of the sequence.
# This makes credit assignment challenging - PPO handles this
# via advantage estimation.
```

### DGX Spark Advantages for RL

> **DGX Spark Tip:** RL training often involves:
> - Running many parallel environments (use all CPU cores)
> - Storing large replay buffers (128GB helps!)
> - Training large policy networks (Blackwell GPU)
> - For RLHF: running inference on full LLMs during training

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Train Q-learning on FrozenLake in 5 minutes |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Bellman, Q-learning, DQN, PPO formulas |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Debug convergence issues and training instability |
| [ELI5.md](./ELI5.md) | Intuitive explanations with dog training and restaurant analogies |

---

## Milestone Checklist

- [ ] Q-learning solving FrozenLake
- [ ] DQN achieving 450+ on CartPole
- [ ] Actor-critic implemented and compared to REINFORCE
- [ ] PPO working on LunarLander
- [ ] Can explain RLHF pipeline conceptually
- [ ] Traced TRL code to understand real RLHF

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Q-learning not converging | Check exploration rate, try different learning rates |
| DQN training unstable | Verify target network updates, check replay buffer size |
| PPO rewards not improving | Tune clip epsilon, check advantage normalization |
| High variance in policy gradient | Add baseline (actor-critic), increase batch size |

---

## Why This Module is Optional

Modern LLM fine-tuning has largely moved away from explicit RL:

1. **DPO replaces PPO** - Direct Preference Optimization avoids reward models entirely
2. **Simpler alternatives work** - KTO, IPO, ORPO achieve similar results
3. **Specialized knowledge** - Only needed if doing RLHF research

However, understanding RL helps you:
- Read RLHF papers (still common in research)
- Debug preference learning when it fails
- Apply RL to other problems (robotics, games)

---

## Next Steps

After completing this module:
1. Implement DPO and compare to your PPO RLHF
2. Read InstructGPT paper with new understanding
3. Apply to Module 3.1 preference learning section

---

## Resources

- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI's RL course (excellent!)
- [TRL Library](https://huggingface.co/docs/trl/) - Transformers RL for LLMs
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single-file RL implementations
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - RLHF for LLMs
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Original PPO algorithm
- [DPO Paper](https://arxiv.org/abs/2305.18290) - RL-free alternative

