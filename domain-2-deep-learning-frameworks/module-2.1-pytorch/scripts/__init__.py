"""
PyTorch Deep Learning Scripts - Module 6

Reusable components for PyTorch training on DGX Spark.

Components:
    - custom_dataset: Dataset classes and data loading utilities
    - resnet_blocks: ResNet building blocks and model constructors
    - custom_activations: Novel activation functions with autograd
    - amp_trainer: Mixed precision training utilities
    - checkpoint_manager: Production-ready checkpointing
    - profiler_utils: Performance profiling tools

Example:
    >>> from scripts.custom_dataset import ImageFolderDataset, create_dataloaders
    >>> from scripts.resnet_blocks import resnet18
    >>> from scripts.amp_trainer import AMPTrainer

Author: DGX Spark AI Curriculum
"""

from .custom_dataset import (
    ImageFolderDataset,
    MixupDataset,
    CutmixDataset,
    create_transforms,
    create_dataloaders,
    compute_mean_std,
)
from .resnet_blocks import (
    SEBlock,
    BasicBlock,
    Bottleneck,
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from .custom_activations import (
    swish,
    mish,
    hard_swish,
    star_relu,
    Swish,
    Mish,
    HardSwish,
    StarReLU,
    verify_gradients,
)
from .amp_trainer import (
    TrainingMetrics,
    AMPTrainer,
    benchmark_precision,
    print_benchmark_results,
)
from .checkpoint_manager import (
    save_checkpoint,
    load_checkpoint,
    CheckpointManager,
    ProductionCheckpointManager,
)
from .profiler_utils import (
    ProfilingResult,
    Timer,
    MemoryTracker,
    profile_training_step,
    benchmark_dataloader,
    profile_with_pytorch_profiler,
    find_bottlenecks,
    generate_profile_report,
)

__all__ = [
    # Datasets
    'ImageFolderDataset',
    'MixupDataset',
    'CutmixDataset',
    'create_transforms',
    'create_dataloaders',
    'compute_mean_std',
    # ResNet
    'SEBlock',
    'BasicBlock',
    'Bottleneck',
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    # Activations
    'swish',
    'mish',
    'hard_swish',
    'star_relu',
    'Swish',
    'Mish',
    'HardSwish',
    'StarReLU',
    'verify_gradients',
    # Training
    'TrainingMetrics',
    'AMPTrainer',
    'benchmark_precision',
    'print_benchmark_results',
    # Checkpointing
    'save_checkpoint',
    'load_checkpoint',
    'CheckpointManager',
    'ProductionCheckpointManager',
    # Profiling
    'ProfilingResult',
    'Timer',
    'MemoryTracker',
    'profile_training_step',
    'benchmark_dataloader',
    'profile_with_pytorch_profiler',
    'find_bottlenecks',
    'generate_profile_report',
]
