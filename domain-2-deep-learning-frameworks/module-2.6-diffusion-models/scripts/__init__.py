"""
Module 2.6: Diffusion Models - Utility Scripts

This package provides reusable utilities for working with diffusion models:
- diffusion_utils: Core diffusion operations and schedulers
- image_utils: Image preprocessing and visualization
- training_utils: LoRA and fine-tuning helpers

Example Usage:
    from scripts.diffusion_utils import NoiseScheduler, add_noise, denoise_step
    from scripts.image_utils import show_image_grid, prepare_control_image
    from scripts.training_utils import prepare_dataset, LoRATrainer
"""

from .diffusion_utils import (
    NoiseScheduler,
    add_noise,
    denoise_step,
    get_timestep_embedding,
)

from .image_utils import (
    show_image_grid,
    prepare_control_image,
    get_canny_edges,
    load_and_preprocess,
    tensor_to_pil,
    pil_to_tensor,
)

from .training_utils import (
    prepare_dataset,
    DiffusionDataset,
    compute_snr_weights,
)

__version__ = "1.0.0"
__author__ = "DGX Spark AI Curriculum"
