"""
Visualization Utilities for Classical ML Module

This module provides plotting functions for model comparison,
feature importance, and performance analysis.

Example:
    >>> from visualization_utils import plot_feature_importance, plot_model_comparison
    >>> plot_feature_importance(model, feature_names)
    >>> plot_model_comparison(results_df, metric='accuracy')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any, Union
import warnings

# Set style with fallback for older matplotlib versions
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # Use default style if neither is available


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_k: int = 20,
    figsize: tuple = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        top_k: Number of top features to show
        figsize: Figure size
        title: Plot title (auto-generated if None)
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> model = RandomForestClassifier().fit(data.data, data.target)
        >>> fig = plot_feature_importance(model, list(data.feature_names))
    """
    # Extract importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model has no feature_importances_ or coef_ attribute")

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    # Take top k
    if len(importance_df) > top_k:
        importance_df = importance_df.tail(top_k)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))

    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Feature Importance')
    ax.set_title(title or 'Feature Importance')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    results: Union[pd.DataFrame, List[Dict]],
    metric: str = 'accuracy',
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of model performance.

    Args:
        results: DataFrame or list of dicts with model results
        metric: Metric column name to plot
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> results = pd.DataFrame({
        ...     'Model': ['XGBoost', 'RF', 'LR'],
        ...     'accuracy': [0.95, 0.93, 0.91],
        ...     'train_time': [1.2, 2.5, 0.1]
        ... })
        >>> fig = plot_model_comparison(results, metric='accuracy')
    """
    if isinstance(results, list):
        results = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=figsize)

    # Get model names and metric values
    models = results['Model'].tolist()
    values = results[metric].tolist()

    # Color based on rank
    sorted_indices = np.argsort(values)[::-1]
    colors = ['gold' if i == sorted_indices[0] else 'silver' if i == sorted_indices[1]
              else 'steelblue' for i in range(len(models))]

    bars = ax.bar(models, values, color=colors)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Model Comparison: {metric}')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cv_results(
    results: List[Dict],
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cross-validation results as box plots.

    Args:
        results: List of dicts with 'name' and 'cv_scores' keys
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> results = [
        ...     {'name': 'XGBoost', 'cv_scores': np.array([0.94, 0.95, 0.96])},
        ...     {'name': 'RF', 'cv_scores': np.array([0.92, 0.93, 0.94])}
        ... ]
        >>> fig = plot_cv_results(results)
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = [r['name'] for r in results]
    cv_data = [r['cv_scores'] for r in results]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    bp = ax.boxplot(cv_data, labels=names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('CV Score')
    ax.set_title(title or 'Cross-Validation Score Distribution')
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curves showing train vs validation performance.

    Args:
        train_sizes: Array of training set sizes
        train_scores: Array of training scores (n_sizes, n_folds)
        val_scores: Array of validation scores (n_sizes, n_folds)
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> from sklearn.model_selection import learning_curve
        >>> sizes, train_scores, val_scores = learning_curve(
        ...     model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
        ... )
        >>> fig = plot_learning_curves(sizes, train_scores, val_scores)
    """
    fig, ax = plt.subplots(figsize=figsize)

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # Plot training scores
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='blue')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')

    # Plot validation scores
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='green')
    ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Score')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.set_title(title or 'Learning Curves')
    ax.legend(loc='lower right')
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    title: Optional[str] = None,
    normalize: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        figsize: Figure size
        title: Plot title
        normalize: Normalize by row (show percentages)
        save_path: Path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> fig = plot_confusion_matrix(y_true, y_pred, class_names=['Neg', 'Pos'])
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title or 'Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predicted vs actual values for regression.

    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        >>> fig = plot_prediction_scatter(y_true, y_pred)
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.5, s=30)

    # Perfect prediction line
    lims = [
        np.min([y_true.min(), y_pred.min()]),
        np.max([y_true.max(), y_pred.max()])
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title or 'Predicted vs Actual')
    ax.legend()

    # Add metrics annotation
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot residual analysis for regression.

    Args:
        y_true: True values
        y_pred: Predicted values
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')

    # Residual distribution
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_experiment_summary_plot(
    experiment_results: List[Dict],
    task: str = 'classification',
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary plot for an experiment.

    Args:
        experiment_results: List of ModelResult.to_dict() outputs
        task: 'classification' or 'regression'
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    names = [r['name'] for r in experiment_results]
    n_models = len(names)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_models))

    # 1. Primary metric
    ax1 = axes[0, 0]
    if task == 'classification':
        metric_values = [r['metrics']['accuracy'] for r in experiment_results]
        metric_name = 'Accuracy'
    else:
        metric_values = [r['metrics']['rmse'] for r in experiment_results]
        metric_name = 'RMSE'

    bars = ax1.bar(names, metric_values, color=colors)
    ax1.set_ylabel(metric_name)
    ax1.set_title(f'Model Performance: {metric_name}')
    ax1.tick_params(axis='x', rotation=15)

    for bar, val in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom')

    # 2. CV scores (use cv_mean and cv_std)
    ax2 = axes[0, 1]
    cv_means = [r['cv_mean'] for r in experiment_results]
    cv_stds = [r['cv_std'] for r in experiment_results]

    ax2.bar(names, cv_means, yerr=cv_stds, color=colors, capsize=5)
    ax2.set_ylabel('CV Score')
    ax2.set_title('Cross-Validation Scores (mean +/- std)')
    ax2.tick_params(axis='x', rotation=15)

    # 3. Training time
    ax3 = axes[1, 0]
    train_times = [r['train_time'] for r in experiment_results]
    bars = ax3.bar(names, train_times, color=colors)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time')
    ax3.tick_params(axis='x', rotation=15)

    # 4. Inference time
    ax4 = axes[1, 1]
    infer_times = [r['inference_time'] for r in experiment_results]
    bars = ax4.bar(names, infer_times, color=colors)
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Inference Time')
    ax4.tick_params(axis='x', rotation=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
