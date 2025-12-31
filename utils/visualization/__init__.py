"""
Visualization Utilities for DGX Spark AI Curriculum
====================================================

Production-ready visualization functions for ML model analysis.

This package provides:
- MLVisualizer: Comprehensive ML visualization toolkit
- Training curve plots with automatic overfitting detection
- Confusion matrix heatmaps
- Feature importance visualizations
- Multi-panel dashboard layouts

Usage:
    from utils.visualization import MLVisualizer, plot_training_curves

    # Quick visualization
    viz = MLVisualizer(style='publication')
    viz.plot_training_curves(history, save_path='training.png')

    # Or use standalone functions
    from utils.visualization import plot_confusion_matrix
    plot_confusion_matrix(y_true, y_pred, class_names=['A', 'B', 'C'])
"""

from .ml_plots import (
    MLVisualizer,
    PALETTES,
    plot_learning_rate_finder,
    plot_correlation_matrix,
    plot_training_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    create_model_dashboard,
)

__all__ = [
    "MLVisualizer",
    "PALETTES",
    "plot_learning_rate_finder",
    "plot_correlation_matrix",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "create_model_dashboard",
]
