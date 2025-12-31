"""
Module 1.2: Python for AI/ML - Scripts Package
===============================================

This package contains production-ready utility scripts for the Python for AI module.

Available modules:
- numpy_utils: NumPy utilities for ML operations (distances, similarity, activations)
- preprocessing_pipeline: Data preprocessing utilities with fit/transform pattern
- visualization_utils: Visualization helper functions for ML analysis
- profiling_utils: Code profiling and benchmarking tools

Example:
    >>> from scripts import Preprocessor, MLVisualizer, Timer
    >>> from scripts import pairwise_distances, softmax, cosine_similarity
"""

from .numpy_utils import *
from .preprocessing_pipeline import *
from .visualization_utils import *
from .profiling_utils import *
