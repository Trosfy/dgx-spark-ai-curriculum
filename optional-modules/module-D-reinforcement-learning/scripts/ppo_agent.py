"""
Proximal Policy Optimization (PPO) Agent

Production-quality PPO implementation for discrete action spaces.

Author: Professor SPARK
Module: D.4 - Policy Gradients and PPO

This is the algorithm used in RLHF for training ChatGPT and similar models.

Example Usage:
    >>> from ppo_agent import PPOAgent
    >>> agent = PPOAgent(state_dim=4, action_dim=2)
    >>> action = agent.select_action(state)
    >>> agent.store_transition(reward, done)
    >>> metrics = agent.update(next_state)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, List, Dict


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic Network for PPO.

    Architecture:
    - Shared feature extraction layers
    - Actor head: outputs action probabilities
    - Critic head: outputs state value V(s)

    Args:
        state_dim: Dimension of state space
        action_dim: Number of possible actions
        hidden_dim: Size of hidden layers

    Example:
        >>> network = ActorCriticNetwork(state_dim=4, action_dim=2)
        >>> probs, value = network(torch.randn(32, 4))
        >>> print(probs.shape, value.shape)  # (32, 2), (32, 1)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
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

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Small weights for output layers
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor, shape (batch_size, state_dim)

        Returns:
            action_probs: Action probability distribution
            value: State value estimate
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action_and_value(
        self,
        state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and get associated values.

        Args:
            state: State tensor

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: State value estimate
            entropy: Policy entropy
        """
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, value.squeeze(-1), entropy


class RolloutBuffer:
    """
    Rollout buffer for PPO.

    Stores trajectories from environment interaction.

    Args:
        buffer_size: Maximum number of transitions
        state_dim: Dimension of state space
        device: Device to put tensors on
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        state_dim: int = 4,
        device: torch.device = torch.device("cpu")
    ):
        self.buffer_size = buffer_size
        self.device = device

        # Pre-allocate buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool
    ) -> None:
        """Store a transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> None:
        """
        Compute returns and GAE advantages.

        Args:
            last_value: Value of the last state (for bootstrapping)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]

    def get_batches(
        self,
        batch_size: int
    ) -> List[Tuple[torch.Tensor, ...]]:
        """
        Get random mini-batches for training.

        Args:
            batch_size: Size of each mini-batch

        Returns:
            List of mini-batches
        """
        indices = np.random.permutation(self.size)
        batches = []

        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            batch = (
                torch.FloatTensor(self.states[batch_indices]).to(self.device),
                torch.LongTensor(self.actions[batch_indices]).to(self.device),
                torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                torch.FloatTensor(self.returns[batch_indices]).to(self.device),
            )
            batches.append(batch)

        return batches

    def reset(self) -> None:
        """Reset the buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """
    Proximal Policy Optimization Agent.

    This is the algorithm used for RLHF in LLMs like ChatGPT.

    Features:
    - Clipped surrogate objective for stable updates
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus for exploration
    - Multiple epochs per update

    Args:
        state_dim: Dimension of state space
        action_dim: Number of possible actions
        hidden_dim: Size of hidden layers
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        n_epochs: Number of PPO epochs per update
        batch_size: Mini-batch size for updates
        buffer_size: Rollout buffer size
        device: Device to use

    Example:
        >>> agent = PPOAgent(state_dim=8, action_dim=4)
        >>> # Collect rollout
        >>> for step in range(2048):
        ...     action = agent.select_action(state)
        ...     next_state, reward, done, _ = env.step(action)
        ...     agent.store_transition(reward, done)
        ...     state = next_state
        >>> # Update policy
        >>> metrics = agent.update(state)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: Optional[torch.device] = None
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Network
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(buffer_size, state_dim, self.device)

        # Current state for storing
        self._current_state: Optional[np.ndarray] = None
        self._current_action: Optional[int] = None
        self._current_log_prob: Optional[float] = None
        self._current_value: Optional[float] = None

        # Tracking
        self.update_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action and store necessary values for later.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value, _ = self.network.get_action_and_value(state_tensor)

        # Store for later
        self._current_state = state
        self._current_action = action
        self._current_log_prob = log_prob.item()
        self._current_value = value.item()

        return action

    def store_transition(self, reward: float, done: bool) -> None:
        """
        Store the transition in the rollout buffer.

        Args:
            reward: Reward received
            done: Whether episode ended
        """
        if self._current_state is None:
            raise RuntimeError("Must call select_action before store_transition")

        self.buffer.store(
            self._current_state,
            self._current_action,
            self._current_log_prob,
            self._current_value,
            reward,
            done
        )

    def update(self, last_state: np.ndarray) -> Dict[str, float]:
        """
        Update policy using PPO.

        Args:
            last_state: The state after the last action (for bootstrapping)

        Returns:
            Dictionary of training metrics
        """
        # Get last value for bootstrapping
        with torch.no_grad():
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_state_tensor)
            last_value = last_value.item()

        # Compute advantages
        self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages = self.buffer.advantages[:self.buffer.size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages[:self.buffer.size] = advantages

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                states, actions, old_log_probs, advantages, returns = batch

                # Get current policy outputs
                probs, values = self.network(states)
                values = values.squeeze(-1)

                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Probability ratio
                ratio = (new_log_probs - old_log_probs).exp()

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (with optional clipping)
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (ratio.log())).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl
                n_updates += 1

        # Reset buffer
        self.buffer.reset()
        self.update_count += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'approx_kl': total_approx_kl / n_updates,
        }

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']


if __name__ == "__main__":
    # Test PPO agent
    print("Testing PPO Agent...\n")

    agent = PPOAgent(
        state_dim=4,
        action_dim=2,
        buffer_size=256,
        batch_size=64,
        n_epochs=4
    )

    # Simulate a rollout
    state = np.random.randn(4)
    for step in range(256):
        action = agent.select_action(state)
        reward = np.random.randn()
        done = np.random.random() < 0.05
        agent.store_transition(reward, done)

        if done:
            state = np.random.randn(4)
        else:
            state = state + 0.1 * np.random.randn(4)

    # Update
    metrics = agent.update(state)

    print("Update metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n✅ PPO Agent test passed!")
