"""
Reinforcement Learning Visualization Utilities

Plotting and visualization tools for RL experiments.

Author: Professor SPARK
Module: D - Reinforcement Learning

Example Usage:
    >>> from visualization import plot_learning_curve, plot_q_table
    >>> plot_learning_curve(rewards, title="DQN Training")
    >>> plot_q_table(Q, env_shape=(4, 4))
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import matplotlib.patches as mpatches


def plot_learning_curve(
    rewards: List[float],
    window: int = 100,
    title: str = "Learning Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    goal_line: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training rewards with smoothing.

    Args:
        rewards: List of episode rewards
        window: Smoothing window size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        goal_line: Optional horizontal line showing goal reward
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)

    # Raw rewards
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw')

    # Smoothed rewards
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed,
                color='blue', linewidth=2, label=f'{window}-episode average')

    # Goal line
    if goal_line is not None:
        plt.axhline(y=goal_line, color='green', linestyle='--',
                   linewidth=2, label=f'Goal ({goal_line})')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_comparison(
    results: Dict[str, List[float]],
    window: int = 50,
    title: str = "Algorithm Comparison",
    figsize: Tuple[int, int] = (12, 5),
    goal_line: Optional[float] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple algorithms on the same plot.

    Args:
        results: Dictionary mapping algorithm names to reward lists
        window: Smoothing window
        title: Plot title
        figsize: Figure size
        goal_line: Optional goal line
        save_path: Optional path to save
    """
    plt.figure(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, rewards), color in zip(results.items(), colors):
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), smoothed,
                    label=name, color=color, linewidth=2)

    if goal_line is not None:
        plt.axhline(y=goal_line, color='green', linestyle='--',
                   label=f'Goal ({goal_line})')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_q_table(
    Q: np.ndarray,
    grid_shape: Tuple[int, int] = (4, 4),
    action_names: Optional[List[str]] = None,
    title: str = "Q-Table",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a Q-table for a grid world.

    Args:
        Q: Q-table of shape (n_states, n_actions)
        grid_shape: Shape of the grid world
        action_names: Names of actions
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    """
    if action_names is None:
        action_names = ['Left', 'Down', 'Right', 'Up']

    n_states, n_actions = Q.shape
    n_rows, n_cols = grid_shape

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Q-value heatmap
    ax = axes[0]
    im = ax.imshow(Q, cmap='RdYlGn', aspect='auto')
    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(action_names)
    ax.set_title(f'{title} - Q-Values')
    plt.colorbar(im, ax=ax)

    # Grid view with best actions
    ax = axes[1]
    V = Q.max(axis=1).reshape(grid_shape)
    policy = Q.argmax(axis=1).reshape(grid_shape)

    im = ax.imshow(V, cmap='RdYlGn')

    # Arrow symbols for actions
    arrows = ['←', '↓', '→', '↑']

    for i in range(n_rows):
        for j in range(n_cols):
            state_idx = i * n_cols + j
            if Q[state_idx].max() != 0:  # Non-terminal
                ax.text(j, i - 0.15, arrows[policy[i, j]],
                       ha='center', va='center', fontsize=18, fontweight='bold')
            ax.text(j, i + 0.25, f'{V[i, j]:.2f}',
                   ha='center', va='center', fontsize=9)

    ax.set_title(f'{title} - Optimal Policy')
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='State Value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_loss_curves(
    losses: List[float],
    window: int = 100,
    title: str = "Training Loss",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training loss over time.

    Args:
        losses: List of loss values
        window: Smoothing window
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    """
    plt.figure(figsize=figsize)

    plt.plot(losses, alpha=0.3, color='red', label='Raw')

    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), smoothed,
                color='red', linewidth=2, label=f'{window}-step average')

    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_epsilon_decay(
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    n_episodes: int = 1000,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize epsilon decay schedule.

    Args:
        epsilon_start: Initial epsilon
        epsilon_end: Final epsilon
        epsilon_decay: Decay rate per episode
        n_episodes: Number of episodes to plot
        figsize: Figure size
        save_path: Optional path to save
    """
    epsilons = []
    epsilon = epsilon_start

    for _ in range(n_episodes):
        epsilons.append(epsilon)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    plt.figure(figsize=figsize)
    plt.plot(epsilons, linewidth=2)
    plt.axhline(y=epsilon_end, color='r', linestyle='--',
               label=f'Min ε = {epsilon_end}')

    plt.xlabel('Episode')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.title('ε-Greedy Exploration Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate key points
    for threshold in [0.5, 0.1, 0.05]:
        try:
            episode = next(i for i, e in enumerate(epsilons) if e < threshold)
            plt.axvline(x=episode, color='gray', linestyle=':', alpha=0.5)
            plt.annotate(f'ε < {threshold}\n(ep {episode})',
                        xy=(episode, threshold),
                        xytext=(episode + n_episodes*0.05, threshold + 0.1),
                        fontsize=8)
        except StopIteration:
            pass

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_policy_entropy(
    entropies: List[float],
    window: int = 50,
    title: str = "Policy Entropy During Training",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot policy entropy over training.

    Higher entropy = more exploration
    Lower entropy = more deterministic policy

    Args:
        entropies: List of entropy values
        window: Smoothing window
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    """
    plt.figure(figsize=figsize)

    plt.plot(entropies, alpha=0.3, color='purple', label='Raw')

    if len(entropies) >= window:
        smoothed = np.convolve(entropies, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(entropies)), smoothed,
                color='purple', linewidth=2, label=f'{window}-step average')

    plt.xlabel('Update Step')
    plt.ylabel('Entropy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add interpretation
    plt.annotate('High entropy\n(exploring)',
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=9, color='green')
    plt.annotate('Low entropy\n(exploiting)',
                xy=(0.02, 0.05), xycoords='axes fraction',
                fontsize=9, color='blue')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_advantage_distribution(
    advantages: np.ndarray,
    title: str = "Advantage Distribution",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of advantage values.

    Args:
        advantages: Array of advantage values
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save
    """
    plt.figure(figsize=figsize)

    plt.hist(advantages, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Zero (baseline)')
    plt.axvline(x=advantages.mean(), color='g', linestyle='-',
               label=f'Mean ({advantages.mean():.2f})')

    plt.xlabel('Advantage')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def create_training_dashboard(
    rewards: List[float],
    losses: List[float],
    entropies: Optional[List[float]] = None,
    window: int = 50,
    title: str = "Training Dashboard",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive training dashboard.

    Args:
        rewards: Episode rewards
        losses: Training losses
        entropies: Optional policy entropies
        window: Smoothing window
        title: Dashboard title
        figsize: Figure size
        save_path: Optional path to save
    """
    n_plots = 3 if entropies else 2
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Rewards
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3)
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[0, 1]
    ax.plot(losses, alpha=0.3, color='red')
    if len(losses) >= window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses)), smoothed, color='red', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # Reward histogram
    ax = axes[1, 0]
    ax.hist(rewards[-100:] if len(rewards) > 100 else rewards,
           bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution (last 100)')
    ax.grid(True, alpha=0.3)

    # Entropies or statistics
    ax = axes[1, 1]
    if entropies:
        ax.plot(entropies, alpha=0.3, color='purple')
        if len(entropies) >= window:
            smoothed = np.convolve(entropies, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(entropies)), smoothed,
                   color='purple', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
    else:
        # Show statistics
        stats_text = f"""Training Statistics

Episodes: {len(rewards)}
Mean Reward: {np.mean(rewards):.2f}
Max Reward: {np.max(rewards):.2f}
Final 100 Avg: {np.mean(rewards[-100:]):.2f}
Total Steps: {len(losses)}
Final Loss: {losses[-1]:.4f} (if applicable)"""
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               fontfamily='monospace')
        ax.axis('off')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Demo visualizations
    print("Demo: Visualization Utilities\n")

    # Generate fake data
    np.random.seed(42)
    n_episodes = 500

    # Simulate learning curve
    rewards = np.cumsum(np.random.randn(n_episodes) * 10) / np.arange(1, n_episodes + 1)
    rewards = 100 + rewards * 100 + np.linspace(0, 300, n_episodes)
    rewards = rewards + np.random.randn(n_episodes) * 30

    losses = 1.0 / (np.arange(1, 5001) * 0.01 + 1) + np.random.randn(5000) * 0.1

    print("Plotting learning curve...")
    plot_learning_curve(rewards.tolist(), goal_line=400, title="Demo Learning Curve")

    print("Plotting epsilon decay...")
    plot_epsilon_decay()

    print("✅ Visualization demo complete!")
