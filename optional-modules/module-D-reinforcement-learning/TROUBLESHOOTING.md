# Module D: Reinforcement Learning - Troubleshooting & FAQ

## 🔍 Quick Diagnostic

**Before diving into specific errors:**
1. Check environment setup: `env.reset()` works?
2. Check reward structure: `print(reward)` after each step
3. Check dimensions: State and action space shapes match network?

---

## 🚨 Error Categories

### Q-Learning Issues

#### Issue: Agent Never Learns (Success Rate Stays ~0%)

**Symptoms:**
- Q-values don't change meaningfully
- Agent takes random-looking actions even after training

**Causes and Solutions:**

```python
# 1. Not enough exploration
# Bad: Epsilon too low or decays too fast
epsilon = 0.01  # Agent never explores!

# Good: Start high, decay slowly
epsilon = 1.0
for episode in range(n_episodes):
    ...
    epsilon = max(0.01, epsilon * 0.995)  # Gradual decay

# 2. Learning rate too low or too high
# Bad:
alpha = 0.001  # Too slow
alpha = 0.9    # Too aggressive

# Good:
alpha = 0.1    # Sweet spot for tabular

# 3. Not updating correctly
# Check: Are you updating Q BEFORE moving to next state?
Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
state = next_state  # Move AFTER update

# 4. Environment is slippery (stochastic)
# FrozenLake default is_slippery=True - actions don't always work!
env = gym.make("FrozenLake-v1", is_slippery=False)  # Start with deterministic
```

---

#### Issue: Q-Values Exploding or Going to Infinity

**Symptoms:**
```
Q[0]: [1e15, 1e15, 1e15, 1e15]
```

**Cause:** Learning rate too high or gamma too high.

**Solution:**
```python
# Reduce learning rate
alpha = 0.05  # Was 0.5

# Ensure gamma < 1
gamma = 0.99  # Not 1.0!

# Clip Q-values if needed
Q = np.clip(Q, -100, 100)
```

---

### DQN Issues

#### Issue: DQN Training Unstable (Loss Oscillates Wildly)

**Symptoms:**
- Loss goes up and down dramatically
- Performance doesn't improve

**Solutions:**
```python
# 1. Add target network
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

# Update target periodically (not every step!)
if step % 100 == 0:
    target_net.load_state_dict(q_net.state_dict())

# Use target for computing next Q-values
with torch.no_grad():
    next_q = target_net(next_states).max(1)[0]

# 2. Use experience replay (don't train on sequential data!)
buffer = ReplayBuffer(capacity=10000)
# Sample random batches, not consecutive transitions

# 3. Lower learning rate
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)  # Not 1e-2

# 4. Gradient clipping
torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
```

---

#### Issue: DQN Agent Doesn't Explore

**Symptoms:**
- Agent gets stuck in same behavior
- Never discovers reward

**Solution:**
```python
# Slower epsilon decay
epsilon = max(0.01, 1.0 - episode / 5000)  # Over 5000 episodes

# Or use boltzmann exploration (softmax over Q-values)
def boltzmann_action(q_values, temperature=1.0):
    probs = torch.softmax(q_values / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()
```

---

### PPO Issues

#### Issue: PPO Rewards Not Improving

**Symptoms:**
- Episode rewards stay flat
- Policy doesn't change

**Causes and Solutions:**
```python
# 1. Advantage not normalized
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# 2. Clip epsilon too small or too large
clip_eps = 0.2  # Standard value, 0.1-0.3 range

# 3. Not enough epochs per update
epochs = 10  # Not 1!

# 4. Learning rate too high
lr = 3e-4  # Standard for PPO

# 5. Entropy coefficient too low (no exploration)
entropy_loss = -0.01 * entropy.mean()  # Encourage exploration
```

---

#### Issue: PPO Policy Collapses

**Symptoms:**
- Policy becomes deterministic too quickly
- All actions go to same value

**Solution:**
```python
# Increase entropy bonus
entropy_coef = 0.02  # Was 0.01

# Check for NaN in log probs
log_probs = dist.log_prob(actions)
if torch.isnan(log_probs).any():
    print("NaN in log probs! Check action probabilities")

# Ensure probabilities don't go to 0
probs = probs.clamp(min=1e-8)
```

---

### Environment Issues

#### Error: `gymnasium.error.NameNotFound`

**Symptoms:**
```
gymnasium.error.NameNotFound: Environment 'CartPole-v0' doesn't exist
```

**Solution:**
```python
# Check available environments
import gymnasium
print(gymnasium.envs.registry.keys())

# Use correct version (v0 vs v1)
env = gym.make("CartPole-v1")  # Not v0

# Some environments need extra dependencies
pip install gymnasium[atari]
pip install gymnasium[mujoco]
```

---

#### Issue: Environment Runs Too Slow

**Symptoms:**
- Training takes forever
- Each step is slow

**Solution:**
```python
# Don't render during training!
env = gym.make("CartPole-v1")  # No render_mode

# Only render for visualization
if visualize:
    env = gym.make("CartPole-v1", render_mode="human")

# Vectorize environments for faster training
from gymnasium.vector import SyncVectorEnv
envs = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
```

---

## ❓ Frequently Asked Questions

### Conceptual Questions

#### Q: What's the difference between on-policy and off-policy?

**A:**

| On-Policy (e.g., PPO) | Off-Policy (e.g., DQN) |
|----------------------|------------------------|
| Learn from current policy | Learn from any policy |
| Can't reuse old experience | Use experience replay |
| More stable | More sample efficient |
| Need fresh data each update | Can train on old data |

---

#### Q: When should I use DQN vs PPO?

**A:**

| Use DQN when... | Use PPO when... |
|-----------------|-----------------|
| Discrete actions | Discrete or continuous actions |
| Can store lots of experience | Memory limited |
| Sample efficiency matters | Stability matters |
| Simple environments | Complex environments |

**Default recommendation:** Start with PPO - it's more robust.

---

#### Q: What is the discount factor γ really doing?

**A:** γ determines how much future rewards matter vs immediate rewards.

```
γ = 0: Only care about immediate reward
γ = 1: All future rewards equally important (dangerous!)
γ = 0.99: Care about ~100 steps ahead
γ = 0.9: Care about ~10 steps ahead

Formula: Effective horizon ≈ 1/(1-γ)
```

**Intuition:** High γ = "patient" agent, low γ = "greedy" agent.

---

#### Q: Why does RLHF use PPO instead of DQN?

**A:**

1. **PPO handles text generation** - Each token is an action, needs policy gradients
2. **No replay** - LLM outputs are expensive to store
3. **Stability** - PPO's clipping prevents destructive updates
4. **Continuous-ish** - Token probabilities are continuous even if tokens are discrete

---

#### Q: What's the reward model in RLHF?

**A:** The reward model is a neural network trained to predict human preferences:

```python
# Trained on pairs of responses with human rankings
# Input: (prompt, response) → Output: scalar score

reward_model = RewardModel()
# Trained on: "Response A is better than Response B"
# Learns: P(A > B) = sigmoid(reward(A) - reward(B))

# At RLHF time:
reward = reward_model(prompt, llm_response)
# PPO uses this reward to update the LLM
```

---

### Practical Questions

#### Q: How do I know if my agent is learning?

**A:** Track these metrics:

```python
# 1. Episode reward (should increase)
episode_rewards = []
# Plot: plt.plot(episode_rewards)

# 2. Episode length (task-dependent)
# CartPole: should increase (staying alive longer)
# FrozenLake: should decrease (finding shorter path)

# 3. Q-value magnitude (should stabilize)
avg_q = Q.mean()  # Should converge to reasonable value

# 4. Loss (for DQN/PPO - should decrease then stabilize)
# Don't expect it to go to 0!
```

---

#### Q: How much training is enough?

**A:** Environment-dependent benchmarks:

| Environment | Target | Typical Training |
|-------------|--------|-----------------|
| FrozenLake (deterministic) | 100% success | 1,000 episodes |
| CartPole-v1 | 475+ avg reward | 500-1,000 episodes |
| LunarLander-v2 | 200+ avg reward | 1,000-2,000 episodes |

**Pro tip:** Plot a moving average (100 episodes) of rewards.

---

#### Q: My results are different every run. Is that normal?

**A:** Yes! RL is high-variance. Always:

```python
# 1. Set random seeds
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)

# 2. Run multiple seeds and report mean ± std
results = [train_agent(seed=s) for s in range(5)]
print(f"Reward: {np.mean(results):.1f} ± {np.std(results):.1f}")

# 3. Use longer evaluation (100 episodes, not 10)
```

---

### Beyond the Basics

#### Q: How do I go from PPO here to RLHF for LLMs?

**A:** The jump requires:

1. **TRL library** - HuggingFace's implementation
2. **Reward model** - Train on human preferences
3. **Reference model** - Frozen copy of pre-RLHF LLM
4. **KL penalty** - Prevent policy from diverging too much

```python
# Conceptual flow (see TRL for full implementation)
from trl import PPOTrainer, PPOConfig

config = PPOConfig(learning_rate=1e-5, kl_penalty="kl")
trainer = PPOTrainer(model, ref_model, tokenizer, config)

for batch in dataloader:
    responses = model.generate(batch['prompts'])
    rewards = reward_model(batch['prompts'], responses)
    trainer.step(batch['prompts'], responses, rewards)
```

---

#### Q: What about DPO (Direct Preference Optimization)?

**A:** DPO is an alternative to PPO for RLHF:

| PPO | DPO |
|-----|-----|
| Needs reward model | No reward model |
| More complex | Simpler |
| Can be unstable | More stable |
| Industry standard | Growing popularity |

DPO directly optimizes the LLM on preference pairs without intermediate reward model.

---

## 🔄 Reset Procedures

### Reset Environment

```python
state, info = env.reset()  # New episode

# Reset with specific seed
state, info = env.reset(seed=42)
```

### Reset Q-Table

```python
Q = np.zeros((env.observation_space.n, env.action_space.n))
```

### Reset Neural Network

```python
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
```

---

## 📞 Still Stuck?

1. **Try simpler environment** - FrozenLake before CartPole
2. **Check Spinning Up** - OpenAI's RL tutorial
3. **Use CleanRL** - Single-file implementations
4. **Visualize** - Actually watch your agent to debug
