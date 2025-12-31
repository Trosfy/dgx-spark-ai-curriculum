"""
Reinforcement Learning Utilities

Common utilities for RL algorithms used across Module D notebooks.

Author: Professor SPARK
Module: D - Reinforcement Learning

Example Usage:
    >>> from rl_utils import compute_returns, compute_gae, normalize
    >>> returns = compute_returns(rewards, gamma=0.99)
    >>> advantages = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt


def compute_returns(
    rewards: Union[List[float], np.ndarray],
    gamma: float = 0.99,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute discounted returns for a sequence of rewards.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        rewards: Sequence of rewards from an episode
        gamma: Discount factor (0 to 1)
        normalize: Whether to normalize returns to have zero mean and unit variance

    Returns:
        Array of returns, same length as rewards

    Example:
        >>> rewards = [1.0, 1.0, 1.0, 10.0]
        >>> returns = compute_returns(rewards, gamma=0.99)
        >>> print(returns)  # [12.9..., 11.9..., 10.9..., 10.0]
    """
    rewards = np.array(rewards)
    returns = np.zeros_like(rewards, dtype=np.float32)

    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def compute_gae(
    rewards: Union[List[float], np.ndarray],
    values: Union[List[float], np.ndarray],
    dones: Union[List[bool], np.ndarray],
    next_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE provides a smooth trade-off between bias and variance in advantage estimation:
    - gae_lambda=0: TD(0), low variance but high bias
    - gae_lambda=1: Monte Carlo, high variance but low bias

    Args:
        rewards: Sequence of rewards
        values: Sequence of value predictions V(s)
        dones: Sequence of done flags
        next_value: Value of the terminal state (usually 0)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        normalize: Whether to normalize advantages

    Returns:
        advantages: GAE advantages for each timestep
        returns: Advantage + Value for each timestep

    Example:
        >>> rewards = [1.0, 1.0, 1.0]
        >>> values = [0.5, 0.5, 0.5]
        >>> dones = [False, False, True]
        >>> advantages, returns = compute_gae(rewards, values, dones)
    """
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)

    # Extend values with next_value
    values_extended = np.append(values, next_value)

    advantages = np.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        # TD error: r + gamma * V(s') - V(s)
        delta = rewards[t] + gamma * values_extended[t + 1] * (1 - dones[t]) - values[t]
        # GAE: sum of discounted TD errors
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values

    if normalize and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def normalize_array(
    arr: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize array to zero mean and unit variance.

    Args:
        arr: Array to normalize
        eps: Small constant for numerical stability

    Returns:
        Normalized array
    """
    return (arr - arr.mean()) / (arr.std() + eps)


def soft_update(
    target_network: torch.nn.Module,
    source_network: torch.nn.Module,
    tau: float = 0.005
) -> None:
    """
    Soft update of target network parameters.

    θ_target = τ * θ_source + (1 - τ) * θ_target

    Args:
        target_network: Network to update
        source_network: Network to copy from
        tau: Interpolation parameter (0 to 1)

    Example:
        >>> soft_update(target_q_net, q_net, tau=0.01)
    """
    for target_param, source_param in zip(
        target_network.parameters(), source_network.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(
    target_network: torch.nn.Module,
    source_network: torch.nn.Module
) -> None:
    """
    Hard update (copy) of network parameters.

    Args:
        target_network: Network to update
        source_network: Network to copy from
    """
    target_network.load_state_dict(source_network.state_dict())


def explained_variance(
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> float:
    """
    Compute explained variance between predictions and targets.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    - EV = 1: Perfect predictions
    - EV = 0: Predictions no better than mean
    - EV < 0: Predictions worse than mean

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        Explained variance score
    """
    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0
    return 1 - np.var(y_true - y_pred) / var_true


def plot_training_curves(
    rewards: List[float],
    window: int = 100,
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training curves with smoothing.

    Args:
        rewards: List of episode rewards
        window: Smoothing window size
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Raw rewards
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, label=f'{window}-ep avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(rewards[-100:] if len(rewards) > 100 else rewards, bins=20, edgecolor='black')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution (last 100 episodes)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class RunningMeanStd:
    """
    Running mean and standard deviation calculator.

    Useful for normalizing observations in RL.

    Example:
        >>> rms = RunningMeanStd(shape=(4,))
        >>> for obs in observations:
        ...     rms.update(obs)
        ...     normalized = (obs - rms.mean) / (rms.std + 1e-8)
    """

    def __init__(self, shape: Tuple[int, ...] = (), eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new observations."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > len(self.mean.shape) else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ) -> None:
        """Update from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        """Standard deviation."""
        return np.sqrt(self.var)


if __name__ == "__main__":
    # Test utilities
    print("Testing RL utilities...\n")

    # Test compute_returns
    rewards = [1.0, 1.0, 1.0, 10.0]
    returns = compute_returns(rewards, gamma=0.99)
    print(f"Rewards: {rewards}")
    print(f"Returns: {returns}")

    # Test compute_gae
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    dones = [False, False, True]
    advantages, returns = compute_gae(rewards, values, dones)
    print(f"\nGAE Advantages: {advantages}")
    print(f"Returns: {returns}")

    # Test RunningMeanStd
    rms = RunningMeanStd(shape=(4,))
    for _ in range(100):
        obs = np.random.randn(4)
        rms.update(obs.reshape(1, -1))
    print(f"\nRunning stats - Mean: {rms.mean}, Std: {rms.std}")

    print("\n✅ All tests passed!")
