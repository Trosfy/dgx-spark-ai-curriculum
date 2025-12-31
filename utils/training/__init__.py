"""
Training Utilities for DGX Spark AI Curriculum
===============================================

Utilities for training neural networks across different frameworks.

Submodules:
- numpy_training: NumPy-based training utilities for learning from scratch
- hf_training: Hugging Face Transformers training utilities

Usage:
    # For NumPy-based training (Module 1.5)
    from utils.training import (
        create_batches,
        train_test_split,
        EarlyStopping,
        TrainingHistory,
        accuracy,
        confusion_matrix
    )

    # For Hugging Face training (Module 2.5+)
    from utils.training import (
        create_training_args,
        compute_metrics_factory,
        MemoryCallback,
        train_and_evaluate,
        cleanup_memory
    )
"""

# NumPy-based training utilities
from .numpy_training import (
    create_batches,
    train_test_split,
    accuracy,
    confusion_matrix,
    precision_recall_f1,
    Dropout,
    l2_regularization_loss,
    l2_regularization_gradient,
    TrainingHistory,
    EarlyStopping,
    train_epoch,
    evaluate,
    train,
    plot_training_history,
    plot_confusion_matrix as plot_cm,
)

# Hugging Face training utilities
from .hf_training import (
    get_device_info,
    get_optimal_batch_size,
    create_training_args,
    compute_metrics_factory,
    MemoryCallback,
    TimingCallback,
    TrainingResult,
    train_and_evaluate,
    cleanup_memory,
)

__all__ = [
    # NumPy-based
    "create_batches",
    "train_test_split",
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "Dropout",
    "l2_regularization_loss",
    "l2_regularization_gradient",
    "TrainingHistory",
    "EarlyStopping",
    "train_epoch",
    "evaluate",
    "train",
    "plot_training_history",
    "plot_cm",
    # Hugging Face
    "get_device_info",
    "get_optimal_batch_size",
    "create_training_args",
    "compute_metrics_factory",
    "MemoryCallback",
    "TimingCallback",
    "TrainingResult",
    "train_and_evaluate",
    "cleanup_memory",
]
