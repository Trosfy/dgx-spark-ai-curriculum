"""
Deep Q-Network (DQN) Agent

Production-quality DQN implementation with experience replay and target network.

Author: Professor SPARK
Module: D.3 - Deep Q-Networks

Example Usage:
    >>> from dqn_agent import DQNAgent
    >>> agent = DQNAgent(state_dim=4, action_dim=2)
    >>> action = agent.select_action(state)
    >>> agent.store_experience(state, action, reward, next_state, done)
    >>> loss = agent.train_step()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, Optional, List, Dict


class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.

    Architecture: Input → FC(hidden) → ReLU → FC(hidden) → ReLU → Q-values

    Args:
        state_dim: Dimension of state space
        action_dim: Number of possible actions
        hidden_dim: Size of hidden layers

    Example:
        >>> q_net = QNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        >>> q_values = q_net(torch.randn(32, 4))  # Batch of 32 states
        >>> print(q_values.shape)  # torch.Size([32, 2])
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor, shape (batch_size, state_dim) or (state_dim,)

        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.

    Separates Q-value into Value and Advantage streams:
    Q(s, a) = V(s) + A(s, a) - mean(A(s, .))

    This helps the network learn which states are valuable regardless of action.

    Args:
        state_dim: Dimension of state space
        action_dim: Number of possible actions
        hidden_dim: Size of hidden layers
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture."""
        features = self.feature(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Stores transitions and samples random batches for training.

    Args:
        capacity: Maximum number of transitions to store
        state_dim: Dimension of state space (for pre-allocation)

    Example:
        >>> buffer = ReplayBuffer(capacity=100000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=64)
    """

    def __init__(self, capacity: int = 100000, state_dim: Optional[int] = None):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            device: Device to put tensors on

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent.

    Features:
    - Neural network Q-function approximation
    - Experience replay for stable training
    - Target network for stable targets
    - Epsilon-greedy exploration with decay
    - Optional Double DQN for reduced overestimation

    Args:
        state_dim: Dimension of state space
        action_dim: Number of possible actions
        hidden_dim: Size of hidden layers
        lr: Learning rate
        gamma: Discount factor
        buffer_size: Replay buffer capacity
        batch_size: Training batch size
        target_update_freq: Steps between target network updates
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay_steps: Steps to decay epsilon
        double_dqn: Whether to use Double DQN
        dueling: Whether to use Dueling architecture
        device: Device to use (cuda/cpu)

    Example:
        >>> agent = DQNAgent(state_dim=4, action_dim=2)
        >>> for episode in range(1000):
        ...     state = env.reset()
        ...     while not done:
        ...         action = agent.select_action(state)
        ...         next_state, reward, done, _ = env.step(action)
        ...         agent.store_experience(state, action, reward, next_state, done)
        ...         agent.train_step()
        ...         state = next_state
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        double_dqn: bool = False,
        dueling: bool = False,
        device: Optional[torch.device] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Networks
        NetworkClass = DuelingQNetwork if dueling else QNetwork
        self.q_network = NetworkClass(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = NetworkClass(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Tracking
        self.train_steps = 0
        self.losses: List[float] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use exploration; if False, always exploit

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value, or None if buffer too small
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size, self.device
        )

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use Q-network to select, target to evaluate
                best_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Track
        self.train_steps += 1
        self.losses.append(loss.item())

        # Update target network
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        return loss.item()

    def update_target_network(self) -> None:
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']


if __name__ == "__main__":
    # Test DQN agent
    print("Testing DQN Agent...\n")

    # Create agent
    agent = DQNAgent(
        state_dim=4,
        action_dim=2,
        double_dqn=True,
        dueling=True
    )

    # Simulate some experiences
    for _ in range(200):
        state = np.random.randn(4)
        action = agent.select_action(state)
        reward = random.random()
        next_state = np.random.randn(4)
        done = random.random() < 0.1

        agent.store_experience(state, action, reward, next_state, done)

    # Train
    for step in range(100):
        loss = agent.train_step()
        if loss is not None and (step + 1) % 20 == 0:
            print(f"Step {step + 1}: Loss = {loss:.4f}, Epsilon = {agent.epsilon:.3f}")

    print("\n✅ DQN Agent test passed!")
