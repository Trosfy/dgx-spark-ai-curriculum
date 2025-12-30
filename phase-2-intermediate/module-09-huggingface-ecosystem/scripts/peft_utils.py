"""
PEFT (Parameter-Efficient Fine-Tuning) Utilities

This module provides helper functions for working with LoRA and other PEFT methods
on the DGX Spark platform.

Example usage:
    from scripts.peft_utils import (
        create_lora_config,
        apply_lora,
        get_trainable_params,
        compare_memory_usage
    )

    # Create LoRA configuration
    config = create_lora_config(rank=16, target_modules=["query", "value"])

    # Apply LoRA to model
    peft_model = apply_lora(base_model, config)

    # Compare memory usage
    comparison = compare_memory_usage(model_name, task="classification")
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import time
import gc
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel
)


@dataclass
class LoRAParams:
    """LoRA configuration parameters."""
    rank: int
    alpha: int
    dropout: float
    target_modules: List[str]
    trainable_params: int
    total_params: int
    trainable_percent: float


@dataclass
class MemoryComparison:
    """Memory comparison between full and LoRA fine-tuning."""
    method: str
    peak_memory_gb: float
    trainable_params: int
    total_params: int
    load_time_seconds: float


def find_target_modules(model: nn.Module, patterns: List[str] = None) -> List[str]:
    """
    Find linear layer names in a model that can be targeted by LoRA.

    Args:
        model: The model to inspect
        patterns: Optional patterns to filter (e.g., ["query", "value"])

    Returns:
        List of module names suitable for LoRA.

    Example:
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> modules = find_target_modules(model)
        >>> print(modules)  # ['query', 'key', 'value', 'dense', ...]
    """
    target_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get the last part of the name
            parts = name.split(".")
            if parts:
                module_name = parts[-1]

                if patterns:
                    # Only include if matches a pattern
                    for pattern in patterns:
                        if pattern in module_name.lower():
                            target_modules.add(module_name)
                            break
                else:
                    target_modules.add(module_name)

    return list(target_modules)


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    task_type: str = "SEQ_CLS",
    bias: str = "none",
    modules_to_save: Optional[List[str]] = None
) -> LoraConfig:
    """
    Create a LoRA configuration with sensible defaults.

    Args:
        rank: Rank of the low-rank matrices (4-64 typical)
        alpha: Scaling factor (usually 2*rank)
        dropout: Dropout probability for LoRA layers
        target_modules: Modules to apply LoRA to (e.g., ["query", "value"])
        task_type: Task type ("SEQ_CLS", "CAUSAL_LM", "TOKEN_CLS", etc.)
        bias: Bias training strategy ("none", "all", "lora_only")
        modules_to_save: Modules to train normally (e.g., ["classifier"])

    Returns:
        Configured LoraConfig instance.

    Example:
        >>> config = create_lora_config(rank=16, target_modules=["query", "value"])
        >>> model = get_peft_model(base_model, config)
    """
    # Map string task type to TaskType enum
    task_type_map = {
        "SEQ_CLS": TaskType.SEQ_CLS,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "QUESTION_ANS": TaskType.QUESTION_ANS,
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION
    }

    task = task_type_map.get(task_type, TaskType.SEQ_CLS)

    # Default target modules for different model types
    if target_modules is None:
        target_modules = ["query", "value"]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=task,
        bias=bias,
        modules_to_save=modules_to_save
    )


def apply_lora(
    model: nn.Module,
    config: LoraConfig
) -> PeftModel:
    """
    Apply LoRA to a model.

    Args:
        model: Base model to add LoRA to
        config: LoRA configuration

    Returns:
        Model with LoRA adapters applied.

    Example:
        >>> config = create_lora_config(rank=8)
        >>> peft_model = apply_lora(base_model, config)
        >>> peft_model.print_trainable_parameters()
    """
    return get_peft_model(model, config)


def get_trainable_params(model: nn.Module) -> LoRAParams:
    """
    Get trainable parameter statistics for a model.

    Args:
        model: Model to analyze (can be PEFT or regular)

    Returns:
        LoRAParams with parameter counts and percentages.

    Example:
        >>> params = get_trainable_params(peft_model)
        >>> print(f"Trainable: {params.trainable_percent:.2f}%")
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Try to get LoRA-specific info
    rank = 0
    alpha = 0
    dropout = 0.0
    target_modules = []

    if hasattr(model, 'peft_config'):
        config = model.peft_config.get('default')
        if config:
            rank = config.r
            alpha = config.lora_alpha
            dropout = config.lora_dropout
            target_modules = list(config.target_modules) if config.target_modules else []

    return LoRAParams(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        trainable_params=trainable,
        total_params=total,
        trainable_percent=100 * trainable / total if total > 0 else 0
    )


def compare_memory_usage(
    model_name: str,
    task: str = "classification",
    lora_configs: Optional[List[Dict[str, Any]]] = None,
    dtype: torch.dtype = torch.bfloat16
) -> List[MemoryComparison]:
    """
    Compare memory usage between full fine-tuning and LoRA configurations.

    Args:
        model_name: Hugging Face model identifier
        task: Task type ("classification", "generation")
        lora_configs: List of LoRA config dictionaries
        dtype: Data type for model weights

    Returns:
        List of MemoryComparison results.

    Example:
        >>> results = compare_memory_usage("bert-base-uncased")
        >>> for r in results:
        ...     print(f"{r.method}: {r.peak_memory_gb:.2f} GB")
    """
    if lora_configs is None:
        lora_configs = [
            {"rank": 8, "alpha": 16},
            {"rank": 16, "alpha": 32},
            {"rank": 32, "alpha": 64}
        ]

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Helper to load model
    def load_model():
        if task == "classification":
            return AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, torch_dtype=dtype
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            )

    # Test full fine-tuning
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    model = load_model().to(device)
    load_time = time.time() - start

    results.append(MemoryComparison(
        method="Full Fine-tuning",
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
        trainable_params=sum(p.numel() for p in model.parameters()),
        total_params=sum(p.numel() for p in model.parameters()),
        load_time_seconds=load_time
    ))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Test LoRA configurations
    for config_dict in lora_configs:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        start = time.time()
        base_model = load_model()

        config = create_lora_config(
            rank=config_dict.get("rank", 8),
            alpha=config_dict.get("alpha", 16),
            target_modules=config_dict.get("target_modules", ["query", "value"]),
            task_type="SEQ_CLS" if task == "classification" else "CAUSAL_LM",
            modules_to_save=["classifier"] if task == "classification" else None
        )

        peft_model = get_peft_model(base_model, config).to(device)
        load_time = time.time() - start

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())

        results.append(MemoryComparison(
            method=f"LoRA (r={config_dict.get('rank', 8)})",
            peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
            trainable_params=trainable,
            total_params=total,
            load_time_seconds=load_time
        ))

        del peft_model, base_model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_comparison(results: List[MemoryComparison]) -> None:
    """
    Print a formatted comparison table.

    Args:
        results: List of MemoryComparison results.
    """
    print("\n" + "=" * 80)
    print("MEMORY COMPARISON")
    print("=" * 80)

    baseline_params = results[0].trainable_params if results else 1

    print(f"\n{'Method':<25} {'Memory (GB)':<15} {'Trainable':<15} {'Reduction':<12}")
    print("-" * 70)

    for r in results:
        reduction = baseline_params / r.trainable_params if r.trainable_params > 0 else 1
        print(f"{r.method:<25} {r.peak_memory_gb:<15.2f} {r.trainable_params:<15,} {reduction:.0f}x")


def save_lora_adapter(model: PeftModel, path: str, tokenizer=None) -> None:
    """
    Save a LoRA adapter to disk.

    Args:
        model: PEFT model with LoRA adapters
        path: Directory path to save to
        tokenizer: Optional tokenizer to save alongside
    """
    model.save_pretrained(path)
    if tokenizer:
        tokenizer.save_pretrained(path)
    print(f"Adapter saved to {path}")


def load_lora_adapter(
    base_model: nn.Module,
    adapter_path: str
) -> PeftModel:
    """
    Load a LoRA adapter onto a base model.

    Args:
        base_model: The base model to load adapters onto
        adapter_path: Path to the saved adapter

    Returns:
        Model with loaded adapters.

    Example:
        >>> base = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> model = load_lora_adapter(base, "./my_adapter")
    """
    return PeftModel.from_pretrained(base_model, adapter_path)


def merge_and_unload(model: PeftModel) -> nn.Module:
    """
    Merge LoRA weights into base model and remove adapter layers.

    This creates a standard model with LoRA changes baked in.
    Useful for deployment when you don't need to switch adapters.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        Merged model without adapter layers.

    Example:
        >>> merged = merge_and_unload(peft_model)
        >>> merged.save_pretrained("./merged_model")
    """
    return model.merge_and_unload()


def calculate_lora_params(
    hidden_size: int,
    num_layers: int,
    rank: int,
    num_target_modules: int = 2
) -> Dict[str, int]:
    """
    Calculate the number of LoRA parameters for a given configuration.

    Args:
        hidden_size: Model hidden dimension (e.g., 768 for BERT-base)
        num_layers: Number of transformer layers
        rank: LoRA rank
        num_target_modules: Number of modules to apply LoRA to per layer

    Returns:
        Dictionary with parameter counts.

    Example:
        >>> params = calculate_lora_params(768, 12, 8, 2)
        >>> print(f"LoRA params: {params['lora_params']:,}")
    """
    # Each LoRA adapter has A (in_features x rank) + B (rank x out_features)
    lora_per_module = hidden_size * rank + rank * hidden_size

    total_lora = lora_per_module * num_target_modules * num_layers

    # Original parameters (rough estimate for attention)
    original_per_layer = 4 * hidden_size * hidden_size  # Q, K, V, O
    total_original = original_per_layer * num_layers

    return {
        "lora_params": total_lora,
        "original_params": total_original,
        "reduction_factor": total_original // total_lora,
        "trainable_percent": 100 * total_lora / total_original
    }


if __name__ == "__main__":
    # Example usage
    print("PEFT Utilities Demo")
    print("=" * 50)

    # Calculate LoRA parameters
    print("\nLoRA Parameter Calculation (BERT-base, rank=8):")
    params = calculate_lora_params(768, 12, 8, 2)
    for k, v in params.items():
        if isinstance(v, int):
            print(f"  {k}: {v:,}")
        else:
            print(f"  {k}: {v:.2f}")

    # Create config
    print("\nCreating LoRA Config:")
    config = create_lora_config(rank=16, target_modules=["query", "value"])
    print(f"  Rank: {config.r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Target modules: {config.target_modules}")

    # Memory comparison (if GPU available)
    if torch.cuda.is_available():
        print("\nComparing memory usage...")
        results = compare_memory_usage("distilbert-base-uncased")
        print_comparison(results)
    else:
        print("\nGPU not available for memory comparison.")
