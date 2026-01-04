#!/usr/bin/env python3
"""
Option E: Training Configuration

Centralized configuration for QLoRA fine-tuning on DGX Spark.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str = "unsloth/gemma-3-270m-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: str = "bfloat16"


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16  # Rank
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.0  # Dropout (0 for small datasets)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4  # Effective batch = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42


@dataclass
class PathConfig:
    """Project paths."""
    project_dir: Path = Path("./troscha-matcha")
    data_dir: Path = field(default=None)
    model_dir: Path = field(default=None)
    output_dir: Path = field(default=None)

    def __post_init__(self):
        self.data_dir = self.project_dir / "data"
        self.model_dir = self.project_dir / "models"
        self.output_dir = self.model_dir / "troscha-lora"

        # Create directories
        for path in [self.data_dir, self.model_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class FullConfig:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def get_config() -> FullConfig:
    """Get the default configuration."""
    return FullConfig()


def print_config(config: FullConfig) -> None:
    """Print configuration summary."""
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print("\nModel:")
    print(f"  Base: {config.model.name}")
    print(f"  Max Seq Length: {config.model.max_seq_length}")
    print(f"  Load in 4-bit: {config.model.load_in_4bit}")

    print("\nLoRA:")
    print(f"  Rank (r): {config.lora.r}")
    print(f"  Alpha: {config.lora.lora_alpha}")
    print(f"  Target Modules: {len(config.lora.target_modules)}")

    print("\nTraining:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective Batch: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print(f"  Learning Rate: {config.training.learning_rate}")

    print("\nPaths:")
    print(f"  Project: {config.paths.project_dir}")
    print(f"  Output: {config.paths.output_dir}")


if __name__ == "__main__":
    config = get_config()
    print_config(config)
