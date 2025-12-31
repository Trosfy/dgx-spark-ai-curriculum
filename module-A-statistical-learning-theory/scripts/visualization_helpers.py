"""
Visualization Helpers for Module A: Statistical Learning Theory

This module provides reusable visualization utilities for creating
publication-quality plots for learning theory concepts.

Author: Professor SPARK
Module: A - Statistical Learning Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from typing import List, Tuple, Optional, Dict, Any
from itertools import product


def setup_plot_style():
    """
    Set up consistent plotting style for all visualizations.

    Example:
        >>> setup_plot_style()
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.show()
    """
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def plot_shattering_visualization(
    points: np.ndarray,
    title: str = "Shattering Visualization",
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Visualize all possible labelings of points and whether they're separable.

    Args:
        points: Array of shape (n, 2) with 2D point coordinates
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Example:
        >>> points = np.array([[0, 0], [1, 0], [0.5, 1]])
        >>> fig = plot_shattering_visualization(points)
        >>> plt.show()
    """
    from sklearn.svm import SVC

    n = len(points)
    labelings = list(product([0, 1], repeat=n))
    n_labelings = len(labelings)

    cols = min(4, n_labelings)
    rows = (n_labelings + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_labelings == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (ax, labeling) in enumerate(zip(axes, labelings)):
        labels = np.array(labeling)
        colors = ['blue' if l == 0 else 'red' for l in labels]

        # Plot points
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=200,
                  edgecolors='black', linewidths=2, zorder=5)

        # Check separability and plot line if possible
        separable = _check_linear_separability(points, labels)

        if separable and len(np.unique(labels)) > 1:
            _add_separator_line(ax, points, labels)
            ax.set_title(f"{labeling}\nSeparable", fontsize=10, color='green')
        else:
            status = "Trivial" if len(np.unique(labels)) == 1 else "NOT Separable"
            title_color = 'gray' if len(np.unique(labels)) == 1 else 'red'
            ax.set_title(f"{labeling}\n{status}", fontsize=10, color=title_color)

        # Set axis limits
        margin = 0.5
        ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
        ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # Hide unused subplots
    for ax in axes[n_labelings:]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def _check_linear_separability(points: np.ndarray, labels: np.ndarray) -> bool:
    """Helper to check if points are linearly separable."""
    from sklearn.svm import SVC

    if len(np.unique(labels)) == 1:
        return True

    clf = SVC(kernel='linear', C=1e10, max_iter=10000)
    try:
        clf.fit(points, labels)
        return np.all(clf.predict(points) == labels)
    except:
        return False


def _add_separator_line(ax: plt.Axes, points: np.ndarray, labels: np.ndarray):
    """Helper to add separator line to axis."""
    from sklearn.svm import SVC

    clf = SVC(kernel='linear', C=1e10)
    clf.fit(points, labels)

    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1

    xx = np.linspace(x_min, x_max, 100)
    w = clf.coef_[0]
    b = clf.intercept_[0]

    if abs(w[1]) > 1e-10:
        yy = -(w[0] * xx + b) / w[1]
        mask = (yy >= y_min) & (yy <= y_max)
        ax.plot(xx[mask], yy[mask], 'g-', linewidth=2)
    else:
        x_line = -b / w[0]
        ax.axvline(x=x_line, color='g', linewidth=2)


def plot_bias_variance_tradeoff(
    degrees: List[int],
    biases: List[float],
    variances: List[float],
    noise: float,
    title: str = "Bias-Variance Tradeoff",
    figsize: Tuple[int, int] = (12, 7)
) -> plt.Figure:
    """
    Create the classic bias-variance tradeoff plot.

    Args:
        degrees: List of model complexity values
        biases: List of bias² values
        variances: List of variance values
        noise: Irreducible noise level
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Example:
        >>> degrees = [1, 3, 5, 10, 15]
        >>> biases = [0.15, 0.05, 0.02, 0.01, 0.01]
        >>> variances = [0.01, 0.02, 0.03, 0.08, 0.15]
        >>> fig = plot_bias_variance_tradeoff(degrees, biases, variances, 0.09)
        >>> plt.show()
    """
    totals = [b + v + noise for b, v in zip(biases, variances)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(degrees, biases, 'b-o', linewidth=2, markersize=8, label='Bias²')
    ax.plot(degrees, variances, 'r-o', linewidth=2, markersize=8, label='Variance')
    ax.plot(degrees, totals, 'g-o', linewidth=3, markersize=8, label='Total Error')
    ax.axhline(y=noise, color='gray', linestyle='--', linewidth=2,
              label=f'Irreducible Noise (σ²={noise:.2f})')

    # Mark optimal complexity
    optimal_idx = np.argmin(totals)
    optimal_degree = degrees[optimal_idx]
    optimal_error = totals[optimal_idx]

    ax.scatter([optimal_degree], [optimal_error], s=300, color='gold',
              edgecolors='black', linewidths=2, zorder=5, marker='*',
              label=f'Optimal (degree {optimal_degree})')

    # Add annotations
    ax.annotate('Underfitting\n(High Bias)',
               xy=(degrees[1], biases[1] + 0.02), fontsize=11,
               ha='center', color='blue', fontweight='bold')
    ax.annotate('Overfitting\n(High Variance)',
               xy=(degrees[-2], variances[-2] + 0.02), fontsize=11,
               ha='center', color='red', fontweight='bold')

    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(degrees)
    ax.set_ylim(0, max(totals) * 1.1)

    plt.tight_layout()
    return fig


def plot_dartboard(
    scenarios: List[Tuple[str, float, float]],
    n_throws: int = 30,
    figsize: Tuple[int, int] = (12, 12),
    seed: int = 42
) -> plt.Figure:
    """
    Create dartboard visualization of bias vs variance.

    Args:
        scenarios: List of (title, bias, variance) tuples
        n_throws: Number of dart throws per scenario
        figsize: Figure size
        seed: Random seed for reproducibility

    Returns:
        matplotlib Figure object

    Example:
        >>> scenarios = [
        ...     ('High Bias, Low Variance', 2.0, 0.2),
        ...     ('Low Bias, High Variance', 0.3, 1.5),
        ...     ('Low Bias, Low Variance (Goal!)', 0.2, 0.2),
        ... ]
        >>> fig = plot_dartboard(scenarios)
        >>> plt.show()
    """
    np.random.seed(seed)

    n_scenarios = len(scenarios)
    cols = min(2, n_scenarios)
    rows = (n_scenarios + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_scenarios == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, (title, bias, variance) in zip(axes, scenarios):
        # Draw dartboard rings
        for r, color in zip([3, 2, 1, 0.3], ['#eeeeee', '#cccccc', '#aaaaaa', '#ff4444']):
            circle = Circle((0, 0), r, color=color, zorder=0)
            ax.add_patch(circle)

        # Generate dart throws
        throws_x = np.random.normal(bias, np.sqrt(variance), n_throws)
        throws_y = np.random.normal(0, np.sqrt(variance), n_throws)

        # Plot throws
        ax.scatter(throws_x, throws_y, c='blue', s=100, edgecolors='black',
                  linewidths=1, zorder=5, alpha=0.7)

        # Plot center of throws
        ax.scatter([np.mean(throws_x)], [np.mean(throws_y)], c='yellow', s=200,
                  edgecolors='black', linewidths=2, zorder=6, marker='*')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x error')
        ax.set_ylabel('y error')
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')

    # Hide unused subplots
    for ax in axes[n_scenarios:]:
        ax.set_visible(False)

    plt.suptitle('The Dartboard Analogy for Bias-Variance\n'
                '(Blue dots = predictions, Star = average, Red center = target)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_learning_curve(
    train_sizes: List[int],
    train_errors: List[float],
    test_errors: List[float],
    test_error_std: Optional[List[float]] = None,
    pac_bound: Optional[int] = None,
    target_epsilon: float = 0.05,
    title: str = "Learning Curve",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot learning curve showing train/test error vs training set size.

    Args:
        train_sizes: List of training set sizes
        train_errors: List of training errors
        test_errors: List of test errors
        test_error_std: Optional list of test error standard deviations
        pac_bound: Optional PAC sample complexity bound to mark
        target_epsilon: Target error rate to mark
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Example:
        >>> sizes = [100, 500, 1000, 5000]
        >>> train_err = [0.0, 0.01, 0.02, 0.02]
        >>> test_err = [0.15, 0.08, 0.05, 0.04]
        >>> fig = plot_learning_curve(sizes, train_err, test_err)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot test error
    if test_error_std:
        ax.errorbar(train_sizes, test_errors, yerr=test_error_std, fmt='bo-',
                   linewidth=2, markersize=8, capsize=5, label='Test Error')
    else:
        ax.plot(train_sizes, test_errors, 'bo-', linewidth=2, markersize=8,
               label='Test Error')

    # Plot train error
    ax.plot(train_sizes, train_errors, 'g^-', linewidth=2, markersize=8,
           label='Train Error')

    # Mark target epsilon
    ax.axhline(y=target_epsilon, color='red', linestyle='--', linewidth=2,
              label=f'Target ε = {target_epsilon}')

    # Mark PAC bound if provided
    if pac_bound:
        ax.axvline(x=pac_bound, color='purple', linestyle=':', linewidth=2,
                  label=f'PAC bound = {pac_bound:,}')

    # Mark where target is empirically achieved
    for n, err in zip(train_sizes, test_errors):
        if err <= target_epsilon:
            ax.axvline(x=n, color='green', linestyle='-.', linewidth=1, alpha=0.5)
            ax.annotate(f'Achieved at {n:,}', xy=(n, target_epsilon + 0.01),
                       fontsize=10, color='green')
            break

    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(test_errors) * 1.1)

    plt.tight_layout()
    return fig


def plot_double_descent(
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create illustrative double descent curve plot.

    Shows the phenomenon where test error decreases, then increases,
    then decreases again with model complexity.

    Args:
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_double_descent()
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    complexity = np.linspace(0.1, 3, 1000)

    # Classical U-curve
    def classical_curve(x):
        return 0.5 * (x - 1)**2 + 0.1

    # Double descent
    def double_descent(x):
        if x < 0.9:
            return 0.5 * (x - 1)**2 + 0.1
        elif x < 1.1:
            return 0.5 + 0.3 * np.sin((x - 0.9) * np.pi / 0.2)
        else:
            return 0.3 * np.exp(-(x - 1.1)) + 0.05

    y_classical = [classical_curve(x) for x in complexity]
    y_double = [double_descent(x) for x in complexity]

    ax.plot(complexity, y_classical, 'b--', linewidth=2,
           label='Classical theory (bias-variance)')
    ax.plot(complexity, y_double, 'r-', linewidth=3,
           label='Modern deep learning')

    # Mark interpolation threshold
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1)
    ax.annotate('Interpolation\nThreshold', xy=(1.0, 0.6), fontsize=10, ha='center')

    # Mark regions
    ax.fill_between([0.1, 0.7], 0, 1, alpha=0.1, color='blue', label='Underfitting')
    ax.fill_between([0.7, 1.3], 0, 1, alpha=0.1, color='orange', label='Classical overfitting')
    ax.fill_between([1.3, 3], 0, 1, alpha=0.1, color='green', label='Over-parameterized (good!)')

    ax.set_xlabel('Model Complexity (Parameters / Data)', fontsize=12)
    ax.set_ylabel('Test Error', fontsize=12)
    ax.set_title('Double Descent: Why Over-Parameterization Works',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sample_complexity_surface(
    vc_dims: np.ndarray,
    epsilons: np.ndarray,
    delta: float = 0.05,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create 3D surface plot of sample complexity vs VC dimension and epsilon.

    Args:
        vc_dims: Array of VC dimension values
        epsilons: Array of epsilon values
        delta: Fixed delta value
        figsize: Figure size

    Returns:
        matplotlib Figure object

    Example:
        >>> vc_dims = np.arange(10, 110, 10)
        >>> epsilons = np.linspace(0.01, 0.2, 20)
        >>> fig = plot_sample_complexity_surface(vc_dims, epsilons)
        >>> plt.show()
    """
    from mpl_toolkits.mplot3d import Axes3D

    VD, EPS = np.meshgrid(vc_dims, epsilons)

    # Compute sample complexity
    C = 8.0
    M = (C / EPS) * (VD * np.log(16 / EPS) + np.log(2 / delta))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(VD, EPS, M / 1000, cmap='viridis', alpha=0.8)

    ax.set_xlabel('VC Dimension', fontsize=11)
    ax.set_ylabel('Epsilon (ε)', fontsize=11)
    ax.set_zlabel('Sample Complexity (thousands)', fontsize=11)
    ax.set_title(f'PAC Sample Complexity Surface (δ = {delta})',
                fontsize=14, fontweight='bold')

    fig.colorbar(surf, shrink=0.5, aspect=10, label='Samples (K)')

    plt.tight_layout()
    return fig


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Visualization Helpers...")
    setup_plot_style()

    # Test dartboard
    scenarios = [
        ('High Bias, Low Variance', 2.0, 0.2),
        ('Low Bias, High Variance', 0.3, 1.5),
        ('High Bias, High Variance', 2.0, 1.5),
        ('Low Bias, Low Variance (Goal!)', 0.2, 0.2),
    ]
    fig = plot_dartboard(scenarios)
    plt.savefig('/tmp/dartboard_test.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Dartboard plot saved to /tmp/dartboard_test.png")

    # Test double descent
    fig = plot_double_descent()
    plt.savefig('/tmp/double_descent_test.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Double descent plot saved to /tmp/double_descent_test.png")

    print("All visualization tests passed!")
