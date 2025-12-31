"""
ML Visualization Utilities
==========================

Production-ready visualization functions for ML model analysis.
Designed for publication-quality figures with consistent styling.

Features:
- Training curve plots with automatic overfitting detection
- Confusion matrix heatmaps with normalization options
- Feature importance visualizations
- Distribution analysis plots
- Multi-panel dashboard layouts

Example Usage:
    >>> from utils.visualization import MLVisualizer
    >>>
    >>> viz = MLVisualizer(style='publication')
    >>>
    >>> # Plot training history
    >>> viz.plot_training_curves(history, save_path='training.png')
    >>>
    >>> # Create confusion matrix
    >>> viz.plot_confusion_matrix(y_true, y_pred, class_names=['A', 'B', 'C'])
    >>>
    >>> # Create full dashboard
    >>> viz.create_model_dashboard(model_results, save_path='dashboard.png')
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import warnings

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Visualization functions will not work.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Color palettes
PALETTES = {
    'default': {
        'primary': '#3498db',
        'secondary': '#e74c3c',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'info': '#9b59b6',
        'gray': '#95a5a6'
    },
    'colorblind': {
        'primary': '#0077BB',
        'secondary': '#CC3311',
        'success': '#009988',
        'warning': '#EE7733',
        'info': '#AA3377',
        'gray': '#BBBBBB'
    },
    'nvidia': {
        'primary': '#76B900',  # NVIDIA Green
        'secondary': '#1A1A1A',
        'success': '#76B900',
        'warning': '#FFB800',
        'info': '#0068B5',
        'gray': '#999999'
    }
}


class MLVisualizer:
    """
    Machine Learning Visualization Toolkit.

    A comprehensive visualization class for ML model analysis with
    consistent styling and publication-ready output.

    Attributes:
        style: Visual style preset ('default', 'publication', 'presentation')
        palette: Color palette name ('default', 'colorblind', 'nvidia')
        figsize: Default figure size (width, height)
        dpi: Figure resolution for saving

    Example:
        >>> viz = MLVisualizer(style='publication', palette='colorblind')
        >>> viz.plot_training_curves(history)
        >>> plt.show()
    """

    def __init__(
        self,
        style: str = 'default',
        palette: str = 'default',
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 150
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for MLVisualizer")

        self.style = style
        self.palette = PALETTES.get(palette, PALETTES['default'])
        self.figsize = figsize
        self.dpi = dpi
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib style based on preset."""
        if self.style == 'publication':
            plt.rcParams.update({
                'font.size': 10,
                'font.family': 'serif',
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.dpi': 150,
                'savefig.dpi': 300,
                'axes.grid': True,
                'grid.alpha': 0.3
            })
        elif self.style == 'presentation':
            plt.rcParams.update({
                'font.size': 14,
                'font.family': 'sans-serif',
                'axes.titlesize': 18,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.dpi': 100,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'lines.linewidth': 2.5
            })
        else:  # default
            plt.rcParams.update({
                'font.size': 10,
                'axes.grid': True,
                'grid.alpha': 0.3
            })

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        metrics: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = 'Training History',
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Any]:
        """
        Plot training and validation curves from training history.

        Automatically detects overfitting and marks the best epoch.

        Args:
            history: Dict with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'
            metrics: List of metrics to plot. If None, plots all available.
            figsize: Figure size override
            title: Main figure title
            save_path: If provided, save figure to this path

        Returns:
            Tuple of (Figure, Axes array)
        """
        figsize = figsize or self.figsize

        # Detect available metric pairs
        if metrics is None:
            base_metrics = [k for k in history.keys() if not k.startswith('val_')]
            metrics = [m for m in base_metrics if f'val_{m}' in history]

        n_metrics = len(metrics)
        if n_metrics == 0:
            raise ValueError("No valid metric pairs found in history")

        fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0] * n_metrics / 2, figsize[1]))
        if n_metrics == 1:
            axes = [axes]

        epochs = np.arange(1, len(history[metrics[0]]) + 1)

        for ax, metric in zip(axes, metrics):
            train_values = np.array(history[metric])
            val_key = f'val_{metric}'
            val_values = np.array(history[val_key]) if val_key in history else None

            ax.plot(epochs, train_values,
                   label='Train', color=self.palette['primary'], linewidth=2)

            if val_values is not None:
                ax.plot(epochs, val_values,
                       label='Validation', color=self.palette['secondary'], linewidth=2)

                # Find best epoch
                if 'loss' in metric.lower():
                    best_epoch = np.argmin(val_values) + 1
                    best_value = val_values[best_epoch - 1]
                else:
                    best_epoch = np.argmax(val_values) + 1
                    best_value = val_values[best_epoch - 1]

                ax.axvline(x=best_epoch, color=self.palette['gray'],
                          linestyle='--', alpha=0.7, label=f'Best ({best_epoch})')
                ax.scatter([best_epoch], [best_value], color=self.palette['secondary'],
                          s=100, zorder=5, edgecolors='white', linewidth=2)

                # Fill overfitting region
                if 'loss' in metric.lower() and len(val_values) > best_epoch:
                    if val_values[-1] > best_value * 1.05:
                        ax.fill_between(
                            epochs[best_epoch-1:],
                            train_values[best_epoch-1:],
                            val_values[best_epoch-1:],
                            alpha=0.2, color=self.palette['secondary'],
                            label='Overfitting'
                        )

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend(loc='best')
            ax.set_xlim(1, len(epochs))

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig, axes

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        cmap: str = 'Blues',
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Any]:
        """
        Create a confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for each class
            normalize: If True, show percentages instead of counts
            figsize: Figure size override
            cmap: Colormap name
            title: Plot title
            save_path: If provided, save figure to this path

        Returns:
            Tuple of (Figure, Axes)
        """
        figsize = figsize or (8, 6)

        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            i = np.where(classes == t)[0][0]
            j = np.where(classes == p)[0][0]
            cm[i, j] += 1

        if class_names is None:
            class_names = [str(c) for c in classes]

        fig, ax = plt.subplots(figsize=figsize)

        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = np.nan_to_num(cm_display)
            fmt = '.1%'
        else:
            cm_display = cm
            fmt = 'd'

        if HAS_SEABORN:
            sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        else:
            im = ax.imshow(cm_display, cmap=cmap)
            plt.colorbar(im, ax=ax, label='Count' if not normalize else 'Proportion')
            for i in range(n_classes):
                for j in range(n_classes):
                    value = cm_display[i, j]
                    text = f'{value:{fmt}}'
                    color = 'white' if value > cm_display.max() / 2 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color)
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig, ax

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_k: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = 'Feature Importance',
        horizontal: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Any]:
        """
        Create a feature importance bar chart.

        Args:
            feature_names: List of feature names
            importances: Array of importance values
            top_k: If provided, only show top k features
            figsize: Figure size override
            title: Plot title
            horizontal: If True, create horizontal bar chart
            save_path: If provided, save figure to this path

        Returns:
            Tuple of (Figure, Axes)
        """
        sorted_idx = np.argsort(importances)
        if top_k:
            sorted_idx = sorted_idx[-top_k:]

        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = importances[sorted_idx]

        figsize = figsize or (10, max(4, len(sorted_idx) * 0.3))
        fig, ax = plt.subplots(figsize=figsize)

        if horizontal:
            ax.barh(range(len(sorted_idx)), sorted_values, color=self.palette['primary'])
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Importance')
        else:
            ax.bar(range(len(sorted_idx)), sorted_values, color=self.palette['primary'])
            ax.set_xticks(range(len(sorted_idx)))
            ax.set_xticklabels(sorted_names, rotation=45, ha='right')
            ax.set_ylabel('Importance')

        ax.set_title(title, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig, ax

    def plot_distribution(
        self,
        data: Union[np.ndarray, Any],
        name: str = 'Feature',
        bins: int = 30,
        kde: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Any]:
        """
        Plot distribution of a single feature.

        Args:
            data: Array or Series of values
            name: Feature name for labeling
            bins: Number of histogram bins
            kde: If True and seaborn available, overlay KDE
            figsize: Figure size override
            title: Plot title
            save_path: If provided, save figure to this path

        Returns:
            Tuple of (Figure, Axes)
        """
        figsize = figsize or (8, 5)
        title = title or f'{name} Distribution'

        fig, ax = plt.subplots(figsize=figsize)

        if HAS_SEABORN and kde:
            sns.histplot(data, bins=bins, kde=True, ax=ax, color=self.palette['primary'])
        else:
            ax.hist(data, bins=bins, color=self.palette['primary'], alpha=0.7, edgecolor='white')

        mean_val = np.nanmean(data)
        median_val = np.nanmedian(data)

        ax.axvline(mean_val, color=self.palette['secondary'], linestyle='--',
                   label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color=self.palette['success'], linestyle=':',
                   label=f'Median: {median_val:.2f}')

        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontweight='bold')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig, ax

    def create_model_dashboard(
        self,
        training_history: Optional[Dict[str, List[float]]] = None,
        confusion_data: Optional[Tuple[np.ndarray, np.ndarray, List[str]]] = None,
        feature_importance: Optional[Tuple[List[str], np.ndarray]] = None,
        predictions: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (14, 10),
        title: str = 'Model Analysis Dashboard',
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a comprehensive 2x2 model analysis dashboard.

        Args:
            training_history: Dict with loss/metric history
            confusion_data: Tuple of (y_true, y_pred, class_names)
            feature_importance: Tuple of (feature_names, importance_values)
            predictions: Array of model predictions for distribution plot
            figsize: Figure size
            title: Dashboard title
            save_path: If provided, save figure to this path

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

        # Panel 1: Training curves
        ax1 = fig.add_subplot(gs[0, 0])
        if training_history:
            epochs = np.arange(1, len(list(training_history.values())[0]) + 1)
            loss_key = next((k for k in training_history if 'loss' in k.lower() and not k.startswith('val_')), None)
            if loss_key:
                ax1.plot(epochs, training_history[loss_key],
                        label='Train', color=self.palette['primary'], linewidth=2)
                val_key = f'val_{loss_key}'
                if val_key in training_history:
                    ax1.plot(epochs, training_history[val_key],
                            label='Validation', color=self.palette['secondary'], linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training & Validation Loss', fontweight='bold')
                ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No training history', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12, color=self.palette['gray'])
            ax1.set_title('Training Curves', fontweight='bold')

        # Panel 2: Confusion matrix
        ax2 = fig.add_subplot(gs[0, 1])
        if confusion_data:
            y_true, y_pred, class_names = confusion_data
            classes = np.unique(np.concatenate([y_true, y_pred]))
            n_classes = len(classes)
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(y_true, y_pred):
                i = np.where(classes == t)[0][0]
                j = np.where(classes == p)[0][0]
                cm[i, j] += 1
            if HAS_SEABORN:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names, ax=ax2)
            else:
                ax2.imshow(cm, cmap='Blues')
                for i in range(n_classes):
                    for j in range(n_classes):
                        ax2.text(j, i, str(cm[i, j]), ha='center', va='center')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
            ax2.set_title('Confusion Matrix', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No confusion data', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12, color=self.palette['gray'])
            ax2.set_title('Confusion Matrix', fontweight='bold')

        # Panel 3: Feature importance
        ax3 = fig.add_subplot(gs[1, 0])
        if feature_importance:
            names, values = feature_importance
            sorted_idx = np.argsort(values)[-10:]
            ax3.barh(range(len(sorted_idx)), values[sorted_idx], color=self.palette['primary'])
            ax3.set_yticks(range(len(sorted_idx)))
            ax3.set_yticklabels([names[i] for i in sorted_idx])
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importance', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No feature importance data', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12, color=self.palette['gray'])
            ax3.set_title('Feature Importance', fontweight='bold')

        # Panel 4: Prediction distribution
        ax4 = fig.add_subplot(gs[1, 1])
        if predictions is not None:
            if HAS_SEABORN:
                sns.histplot(predictions, bins=30, kde=True, ax=ax4, color=self.palette['primary'])
            else:
                ax4.hist(predictions, bins=30, color=self.palette['primary'], alpha=0.7, edgecolor='white')
            ax4.axvline(np.mean(predictions), color=self.palette['secondary'],
                       linestyle='--', label=f'Mean: {np.mean(predictions):.2f}')
            ax4.set_xlabel('Prediction Value')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Prediction Distribution', fontweight='bold')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No prediction data', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12, color=self.palette['gray'])
            ax4.set_title('Prediction Distribution', fontweight='bold')

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig


# Standalone convenience functions
def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[Figure, Any]:
    """Convenience function for plotting training curves."""
    viz = MLVisualizer()
    return viz.plot_training_curves(history, save_path=save_path, **kwargs)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[Figure, Any]:
    """Convenience function for plotting confusion matrix."""
    viz = MLVisualizer()
    return viz.plot_confusion_matrix(y_true, y_pred, class_names, save_path=save_path, **kwargs)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[Figure, Any]:
    """Convenience function for plotting feature importance."""
    viz = MLVisualizer()
    return viz.plot_feature_importance(feature_names, importances, save_path=save_path, **kwargs)


def create_model_dashboard(
    training_history: Optional[Dict[str, List[float]]] = None,
    confusion_data: Optional[Tuple[np.ndarray, np.ndarray, List[str]]] = None,
    feature_importance: Optional[Tuple[List[str], np.ndarray]] = None,
    predictions: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    **kwargs
) -> Figure:
    """Convenience function for creating model dashboard."""
    viz = MLVisualizer()
    return viz.create_model_dashboard(
        training_history, confusion_data, feature_importance, predictions,
        save_path=save_path, **kwargs
    )


def plot_learning_rate_finder(
    lrs: np.ndarray,
    losses: np.ndarray,
    suggested_lr: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Any]:
    """
    Plot results from learning rate finder.

    Args:
        lrs: Array of learning rates tested
        losses: Corresponding loss values
        suggested_lr: Suggested learning rate to highlight
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        Tuple of (Figure, Axes)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(lrs, losses, linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Rate Finder', fontweight='bold')

    if suggested_lr:
        ax.axvline(suggested_lr, color='red', linestyle='--',
                   label=f'Suggested LR: {suggested_lr:.2e}')
        ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_correlation_matrix(
    data: Any,
    figsize: Tuple[float, float] = (12, 10),
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None
) -> Tuple[Figure, Any]:
    """
    Create a correlation matrix heatmap.

    Args:
        data: DataFrame with numeric columns or correlation matrix
        figsize: Figure size
        cmap: Colormap
        save_path: If provided, save figure to this path

    Returns:
        Tuple of (Figure, Axes)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required")

    if HAS_PANDAS and hasattr(data, 'select_dtypes'):
        corr = data.select_dtypes(include=[np.number]).corr()
    else:
        corr = data

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                   center=0, vmin=-1, vmax=1, ax=ax)
    else:
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')

    ax.set_title('Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


if __name__ == "__main__":
    print("Visualization Utils Demo")
    print("=" * 50)

    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping demo")
    else:
        viz = MLVisualizer(style='default')

        # Sample training history
        np.random.seed(42)
        epochs = 50
        x = np.arange(epochs)
        history = {
            'loss': 2.0 * np.exp(-0.05 * x) + 0.2 + np.random.normal(0, 0.02, epochs),
            'val_loss': 2.0 * np.exp(-0.04 * x) + 0.3 + np.random.normal(0, 0.03, epochs),
            'accuracy': 1 - 0.4 * np.exp(-0.05 * x) + np.random.normal(0, 0.01, epochs),
            'val_accuracy': 1 - 0.45 * np.exp(-0.04 * x) + np.random.normal(0, 0.015, epochs)
        }

        print("\n1. Creating training curves plot...")
        fig, _ = viz.plot_training_curves(history)
        plt.close()
        print("   Done!")

        print("\n2. Creating confusion matrix...")
        y_true = np.random.choice([0, 1, 2], 100)
        y_pred = y_true.copy()
        y_pred[np.random.choice(100, 20)] = np.random.choice([0, 1, 2], 20)
        fig, _ = viz.plot_confusion_matrix(y_true, y_pred, ['A', 'B', 'C'])
        plt.close()
        print("   Done!")

        print("\nAll visualizations created successfully!")
