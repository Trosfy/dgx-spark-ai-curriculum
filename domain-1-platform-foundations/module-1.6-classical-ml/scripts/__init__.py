"""
Module 1.6: Classical ML Foundations - Utility Scripts

This module provides reusable utilities for classical ML experiments:
- baseline_utils: BaselineExperiment class for model comparison
- data_utils: Data loading and preprocessing functions
- visualization_utils: Plotting and visualization functions

Example:
    >>> from scripts.baseline_utils import BaselineExperiment
    >>> from scripts.data_utils import load_dataset, preprocess_features
    >>> from scripts.visualization_utils import plot_feature_importance
"""

from .baseline_utils import BaselineExperiment, ModelResult, quick_baseline
from .data_utils import (
    load_dataset,
    preprocess_features,
    generate_synthetic_data,
    get_dataset_info,
    create_train_test_split
)
from .visualization_utils import (
    plot_feature_importance,
    plot_model_comparison,
    plot_cv_results,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_prediction_scatter
)

__all__ = [
    # Baseline utilities
    'BaselineExperiment',
    'ModelResult',
    'quick_baseline',
    # Data utilities
    'load_dataset',
    'preprocess_features',
    'generate_synthetic_data',
    'get_dataset_info',
    'create_train_test_split',
    # Visualization utilities
    'plot_feature_importance',
    'plot_model_comparison',
    'plot_cv_results',
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_prediction_scatter'
]

__version__ = '1.0.0'
