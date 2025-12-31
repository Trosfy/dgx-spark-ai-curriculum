"""
Module 2.5: Hugging Face Ecosystem - Scripts Package

This package contains utility scripts for the Hugging Face Ecosystem module.

Available modules:
- hub_utils: Utilities for interacting with the HuggingFace Hub
- pipeline_utils: Wrappers for common pipeline tasks
- dataset_utils: Dataset loading and preprocessing helpers
- training_utils: Training helper functions for HuggingFace models
- peft_utils: Parameter-efficient fine-tuning utilities

Example usage:
    from scripts.hub_utils import search_models, document_model
    from scripts.pipeline_utils import PipelineDemo, measure_pipeline_latency
    from scripts.dataset_utils import load_and_analyze_dataset, prepare_dataset_for_training
    from scripts.training_utils import create_training_args, compute_metrics_factory
    from scripts.peft_utils import create_lora_config, compare_memory_usage
"""

from .hub_utils import *
from .pipeline_utils import *
from .dataset_utils import *
from .training_utils import *
from .peft_utils import *
