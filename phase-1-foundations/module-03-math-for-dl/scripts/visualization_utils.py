"""
Visualization utilities for deep learning mathematics.

Provides functions for visualizing:
- Loss landscapes
- Optimization trajectories
- Training curves
- SVD analysis
- Probability distributions

Example usage:
    >>> from visualization_utils import plot_loss_landscape, plot_training_curve
    >>>
    >>> # Plot loss landscape
    >>> plot_loss_landscape(func, x_range=(-3, 3), y_range=(-3, 3))
    >>>
    >>> # Plot training curve
    >>> plot_training_curve(losses, title="Training Loss")
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


def plot_loss_landscape(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-3, 3),
    y_range: Tuple[float, float] = (-3, 3),
    resolution: int = 100,
    title: str = "Loss Landscape",
    figsize: Tuple[int, int] = (14, 5),
    show_3d: bool = True,
    trajectory: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create 2D contour and 3D surface plots of a loss function.

    Args:
        func: Loss function f(x, y) -> z
        x_range: Range for x-axis (min, max)
        y_range: Range for y-axis (min, max)
        resolution: Number of points per axis
        title: Plot title
        figsize: Figure size
        show_3d: Whether to show 3D surface plot
        trajectory: Optional optimization trajectory to overlay
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object

    Example:
        >>> rosenbrock = lambda x, y: (1-x)**2 + 100*(y-x**2)**2
        >>> plot_loss_landscape(rosenbrock, x_range=(-2, 2), y_range=(-1, 3))
    """
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    if show_3d:
        fig = plt.figure(figsize=figsize)

        # 2D Contour
        ax1 = fig.add_subplot(121)
        levels = np.logspace(np.log10(Z.min() + 1), np.log10(Z.max() + 1), 30)
        contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis')
        ax1.set_xlabel('Parameter 1')
        ax1.set_ylabel('Parameter 2')
        ax1.set_title(f'{title} (Contour)')
        plt.colorbar(contour, ax=ax1, label='Loss')

        # Add trajectory if provided
        if trajectory is not None:
            ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2,
                    alpha=0.7, label='Optimization path')
            ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='green',
                       s=100, marker='o', label='Start', zorder=5)
            ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red',
                       s=100, marker='*', label='End', zorder=5)
            ax1.legend()

        # 3D Surface
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, np.log10(Z + 1), cmap='viridis',
                               alpha=0.8, linewidth=0)
        ax2.set_xlabel('Parameter 1')
        ax2.set_ylabel('Parameter 2')
        ax2.set_zlabel('log₁₀(Loss + 1)')
        ax2.set_title(f'{title} (3D)')
    else:
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        levels = np.logspace(np.log10(Z.min() + 1), np.log10(Z.max() + 1), 30)
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis')
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title(title)
        plt.colorbar(contour, ax=ax, label='Loss')

        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2)
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curve(
    losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training (and optionally validation) loss curve.

    Args:
        losses: Training loss values
        val_losses: Optional validation loss values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(losses, 'b-', linewidth=2, label='Training')
    if val_losses:
        ax.plot(val_losses, 'r--', linewidth=2, label='Validation')
        ax.legend()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_optimizer_comparison(
    histories: Dict[str, Dict[str, List]],
    x_range: Tuple[float, float] = (-2, 2),
    y_range: Tuple[float, float] = (-1, 3),
    loss_func: Optional[Callable] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare optimization trajectories of different optimizers.

    Args:
        histories: Dict of {optimizer_name: {'params': [...], 'loss': [...]}}
        x_range: Range for x-axis
        y_range: Range for y-axis
        loss_func: Optional loss function for background contour
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    # Left: Trajectories
    if loss_func:
        x = np.linspace(*x_range, 100)
        y = np.linspace(*y_range, 100)
        X, Y = np.meshgrid(x, y)
        Z = loss_func(X, Y)
        levels = np.logspace(-1, 3, 30)
        axes[0].contour(X, Y, Z, levels=levels, cmap='Greys', alpha=0.5)

    for (name, hist), color in zip(histories.items(), colors):
        path = np.array(hist['params'])
        step = max(1, len(path) // 200)
        axes[0].plot(path[::step, 0], path[::step, 1], '-', color=color,
                    linewidth=1.5, alpha=0.7, label=name)
        axes[0].scatter(path[0, 0], path[0, 1], color=color, s=100,
                       marker='o', edgecolors='black', zorder=5)
        axes[0].scatter(path[-1, 0], path[-1, 1], color=color, s=100,
                       marker='*', edgecolors='black', zorder=5)

    axes[0].set_xlabel('Parameter 1', fontsize=12)
    axes[0].set_ylabel('Parameter 2', fontsize=12)
    axes[0].set_title('Optimization Trajectories', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].set_xlim(x_range)
    axes[0].set_ylim(y_range)

    # Right: Loss curves
    for (name, hist), color in zip(histories.items(), colors):
        axes[1].semilogy(hist['loss'], color=color, linewidth=2,
                        label=name, alpha=0.8)

    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1].set_title('Convergence Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_svd_analysis(
    W: np.ndarray,
    title: str = "SVD Analysis",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize SVD decomposition of a matrix.

    Args:
        W: Input matrix to analyze
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Singular values
    axes[0].bar(range(min(len(S), 50)), S[:50], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Singular Value Index', fontsize=12)
    axes[0].set_ylabel('Singular Value', fontsize=12)
    axes[0].set_title(f'{title} - Singular Values', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Cumulative energy
    energy = (S ** 2) / (S ** 2).sum() * 100
    cumulative = np.cumsum(energy)

    axes[1].plot(cumulative, 'b-', linewidth=2)
    axes[1].axhline(y=95, color='r', linestyle='--', label='95% energy')
    axes[1].axhline(y=99, color='g', linestyle='--', label='99% energy')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Energy (%)', fontsize=12)
    axes[1].set_title('Cumulative Information', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, min(100, len(S)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_distribution(
    x: np.ndarray,
    pdf: np.ndarray,
    title: str = "Probability Distribution",
    xlabel: str = "x",
    ylabel: str = "p(x)",
    fill: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a probability distribution.

    Args:
        x: x values
        pdf: probability density values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        fill: Whether to fill under the curve
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if fill:
        ax.fill_between(x, pdf, alpha=0.3)
    ax.plot(x, pdf, linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_gradient_check(
    analytical: np.ndarray,
    numerical: np.ndarray,
    title: str = "Gradient Verification",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize gradient checking results.

    Args:
        analytical: Analytically computed gradients
        numerical: Numerically computed gradients
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot
    axes[0].scatter(analytical.flatten(), numerical.flatten(), alpha=0.5)
    min_val = min(analytical.min(), numerical.min())
    max_val = max(analytical.max(), numerical.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    axes[0].set_xlabel('Analytical Gradient', fontsize=12)
    axes[0].set_ylabel('Numerical Gradient', fontsize=12)
    axes[0].set_title('Gradient Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Difference histogram
    diff = np.abs(analytical - numerical).flatten()
    axes[1].hist(diff, bins=50, alpha=0.7, color='steelblue')
    axes[1].axvline(x=diff.mean(), color='red', linestyle='--',
                   label=f'Mean: {diff.mean():.2e}')
    axes[1].set_xlabel('Absolute Difference', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Gradient Difference Distribution', fontsize=14)
    axes[1].legend()
    axes[1].set_yscale('log')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_animation_frames(
    trajectory: np.ndarray,
    loss_func: Callable,
    x_range: Tuple[float, float] = (-2, 2),
    y_range: Tuple[float, float] = (-1, 3),
    output_dir: str = "frames"
) -> List[str]:
    """
    Create frames for an optimization animation.

    Args:
        trajectory: Array of (x, y) positions
        loss_func: Loss function for background
        x_range: X-axis range
        y_range: Y-axis range
        output_dir: Directory to save frames

    Returns:
        List of frame file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_func(X, Y)

    frame_paths = []

    for i, pos in enumerate(trajectory):
        fig, ax = plt.subplots(figsize=(8, 6))

        levels = np.logspace(-1, 3, 30)
        ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)

        # Plot trajectory up to current point
        ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], 'r-', linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100)
        ax.scatter(pos[0], pos[1], color='red', s=100, marker='*')

        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_title(f'Optimization Step {i}')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        path = os.path.join(output_dir, f"frame_{i:04d}.png")
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()

        frame_paths.append(path)

    return frame_paths


if __name__ == "__main__":
    print("Visualization Utilities Tests")
    print("=" * 50)

    # Test loss landscape
    rosenbrock = lambda x, y: (1-x)**2 + 100*(y-x**2)**2
    fig = plot_loss_landscape(rosenbrock, title="Rosenbrock Function")
    plt.close()
    print("✓ Loss landscape plot")

    # Test training curve
    losses = [1.0 / (i + 1) for i in range(100)]
    fig = plot_training_curve(losses, title="Test Training Curve")
    plt.close()
    print("✓ Training curve plot")

    # Test SVD analysis
    W = np.random.randn(100, 100)
    fig = plot_svd_analysis(W, title="Random Matrix SVD")
    plt.close()
    print("✓ SVD analysis plot")

    print("\n✅ All visualization tests passed!")
