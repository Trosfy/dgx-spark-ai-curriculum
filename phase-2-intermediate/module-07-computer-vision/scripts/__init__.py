"""
Computer Vision Module Scripts

This package contains reusable implementations for the Computer Vision module.

Quick imports:
    from scripts import ResNet18, Trainer, plot_training_history
    from scripts.cnn_architectures import get_model
    from scripts.metrics import compute_iou
"""

# CNN Architectures
from .cnn_architectures import (
    LeNet5,
    AlexNet,
    VGG11,
    ResNet18,
    UNet,
    get_model,
    count_parameters,
)

# Training Utilities
from .training_utils import (
    Trainer,
    get_optimizer,
    get_scheduler,
    EarlyStopping,
    benchmark_inference,
)

# Visualization Utilities
from .visualization_utils import (
    CIFAR10_CLASSES,
    VOC_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CIFAR10_MEAN,
    CIFAR10_STD,
    denormalize,
    plot_training_history,
    visualize_predictions,
    plot_confusion_matrix,
    visualize_segmentation,
)

# Metrics
from .metrics import (
    ClassificationMetrics,
    SegmentationMetrics,
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
)

__all__ = [
    # Architectures
    'LeNet5', 'AlexNet', 'VGG11', 'ResNet18', 'UNet',
    'get_model', 'count_parameters',
    # Training
    'Trainer', 'get_optimizer', 'get_scheduler', 'EarlyStopping', 'benchmark_inference',
    # Visualization
    'CIFAR10_CLASSES', 'VOC_CLASSES', 'IMAGENET_MEAN', 'IMAGENET_STD',
    'CIFAR10_MEAN', 'CIFAR10_STD',
    'denormalize', 'plot_training_history', 'visualize_predictions',
    'plot_confusion_matrix', 'visualize_segmentation',
    # Metrics
    'ClassificationMetrics', 'SegmentationMetrics',
    'compute_iou', 'compute_dice', 'compute_pixel_accuracy',
]
