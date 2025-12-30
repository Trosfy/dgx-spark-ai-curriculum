"""
Training Utilities for Hugging Face Transformers

This module provides helper functions for training models using the Hugging Face
Trainer API, including custom metrics, callbacks, and DGX Spark optimizations.

Example usage:
    from scripts.training_utils import (
        create_training_args,
        compute_metrics_factory,
        MemoryCallback,
        get_optimal_batch_size
    )

    # Create optimized training args
    args = create_training_args(
        output_dir="./results",
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )

    # Create metrics function
    compute_metrics = compute_metrics_factory(["accuracy", "f1", "precision", "recall"])
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import torch
import numpy as np
import evaluate
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from transformers.trainer_utils import EvalPrediction
import gc
import time


def get_device_info() -> Dict[str, Any]:
    """
    Get information about the current compute device.

    Returns:
        Dictionary with device information.
    """
    info = {
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "total_memory_gb": 0,
        "available_memory_gb": 0
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = props.total_memory / 1e9
        info["available_memory_gb"] = (
            props.total_memory - torch.cuda.memory_allocated()
        ) / 1e9

    return info


def get_optimal_batch_size(
    model_size_gb: float,
    sequence_length: int = 256,
    dtype_bytes: int = 2,  # bfloat16
    safety_factor: float = 0.7
) -> int:
    """
    Estimate optimal batch size based on available memory.

    Args:
        model_size_gb: Approximate model size in GB
        sequence_length: Maximum sequence length
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32)
        safety_factor: Fraction of memory to use (0.7 = 70%)

    Returns:
        Recommended batch size.

    Example:
        >>> batch_size = get_optimal_batch_size(0.4)  # ~400MB model
        >>> print(f"Recommended batch size: {batch_size}")
    """
    device_info = get_device_info()

    if not device_info["cuda_available"]:
        return 8  # Conservative default for CPU

    available_gb = device_info["available_memory_gb"] * safety_factor

    # Rough estimate: model + gradients + optimizer states
    # Optimizer (Adam) needs ~4x model size for states
    training_overhead = model_size_gb * 5

    # Memory per sample estimate (embedding + activations)
    hidden_size = 768  # Typical for BERT-base
    memory_per_sample_mb = (
        sequence_length * hidden_size * dtype_bytes * 12  # Layers
    ) / 1e6

    remaining_gb = available_gb - training_overhead
    max_samples = int(remaining_gb * 1000 / memory_per_sample_mb)

    # Return power of 2 batch size
    batch_sizes = [4, 8, 16, 32, 64, 128, 256]
    for bs in reversed(batch_sizes):
        if bs <= max_samples:
            return bs

    return 4  # Minimum


def create_training_args(
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    eval_batch_size: Optional[int] = None,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    load_best_model: bool = True,
    metric_for_best: str = "accuracy",
    use_bf16: bool = True,
    logging_steps: int = 100,
    gradient_accumulation: int = 1,
    dataloader_workers: int = 4,
    seed: int = 42,
    **kwargs
) -> TrainingArguments:
    """
    Create TrainingArguments optimized for DGX Spark.

    Args:
        output_dir: Directory for saving checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size per device
        learning_rate: Initial learning rate
        eval_batch_size: Evaluation batch size (default: 2x training)
        warmup_ratio: Fraction of training for warmup
        weight_decay: L2 regularization weight
        eval_strategy: When to evaluate ("epoch", "steps", "no")
        save_strategy: When to save ("epoch", "steps", "no")
        save_total_limit: Maximum checkpoints to keep
        load_best_model: Load best model at end
        metric_for_best: Metric to use for best model selection
        use_bf16: Use bfloat16 precision (recommended for DGX Spark)
        logging_steps: Steps between logging
        gradient_accumulation: Gradient accumulation steps
        dataloader_workers: Number of data loading workers
        seed: Random seed
        **kwargs: Additional TrainingArguments parameters

    Returns:
        Configured TrainingArguments instance.

    Example:
        >>> args = create_training_args(
        ...     output_dir="./results",
        ...     epochs=3,
        ...     batch_size=16,
        ...     learning_rate=2e-5
        ... )
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size * 2

    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # Duration
        num_train_epochs=epochs,

        # Batch size
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation,

        # Learning rate
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",

        # Evaluation
        eval_strategy=eval_strategy,

        # Checkpointing
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,

        # Precision
        bf16=use_bf16 and torch.cuda.is_available(),

        # Logging
        logging_strategy="steps",
        logging_steps=logging_steps,
        report_to="none",

        # Performance
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=True,

        # Reproducibility
        seed=seed,

        **kwargs
    )


def compute_metrics_factory(
    metrics: List[str] = ["accuracy", "f1", "precision", "recall"],
    average: str = "weighted"
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Create a compute_metrics function for the Trainer.

    Args:
        metrics: List of metric names to compute
        average: Averaging method for multi-class metrics

    Returns:
        A function that computes the specified metrics.

    Example:
        >>> compute_metrics = compute_metrics_factory(["accuracy", "f1"])
        >>> trainer = Trainer(..., compute_metrics=compute_metrics)
    """
    # Load metrics
    loaded_metrics = {}
    for name in metrics:
        try:
            loaded_metrics[name] = evaluate.load(name)
        except Exception as e:
            print(f"Warning: Could not load metric '{name}': {e}")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        results = {}
        for name, metric in loaded_metrics.items():
            try:
                if name in ["f1", "precision", "recall"]:
                    result = metric.compute(
                        predictions=predictions,
                        references=labels,
                        average=average
                    )
                else:
                    result = metric.compute(
                        predictions=predictions,
                        references=labels
                    )

                # Extract the value
                if isinstance(result, dict):
                    results[name] = list(result.values())[0]
                else:
                    results[name] = result
            except Exception as e:
                print(f"Warning: Error computing {name}: {e}")
                results[name] = 0.0

        return results

    return compute_metrics


class MemoryCallback(TrainerCallback):
    """
    Callback to track and log GPU memory usage during training.

    Example:
        >>> trainer = Trainer(..., callbacks=[MemoryCallback()])
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.memory_log = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0:
            if torch.cuda.is_available():
                memory = {
                    "step": state.global_step,
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
                }
                self.memory_log.append(memory)

    def on_train_end(self, args, state, control, **kwargs):
        if self.memory_log:
            max_memory = max(m["max_allocated_gb"] for m in self.memory_log)
            print(f"\n[MemoryCallback] Peak GPU memory: {max_memory:.2f} GB")


class TimingCallback(TrainerCallback):
    """
    Callback to track training timing and throughput.

    Example:
        >>> callback = TimingCallback()
        >>> trainer = Trainer(..., callbacks=[callback])
        >>> trainer.train()
        >>> print(callback.get_summary())
    """

    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.samples_processed = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)

    def on_train_end(self, args, state, control, **kwargs):
        self.total_time = time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_time_seconds": self.total_time,
            "total_time_minutes": self.total_time / 60,
            "epoch_times": self.epoch_times,
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0
        }


@dataclass
class TrainingResult:
    """Results from a training run."""
    model_name: str
    train_loss: float
    eval_loss: float
    eval_metrics: Dict[str, float]
    training_time_seconds: float
    samples_per_second: float
    peak_memory_gb: float
    trainable_params: int
    total_params: int


def train_and_evaluate(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    test_dataset=None,
    output_dir: str = "./results",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    metrics: List[str] = ["accuracy", "f1"],
    early_stopping_patience: Optional[int] = None,
    callbacks: Optional[List[TrainerCallback]] = None
) -> Tuple[Trainer, TrainingResult]:
    """
    Train and evaluate a model with best practices.

    Args:
        model: The model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Optional test dataset
        output_dir: Directory for outputs
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        metrics: Metrics to compute
        early_stopping_patience: Early stopping patience (None to disable)
        callbacks: Additional callbacks

    Returns:
        Tuple of (trainer, TrainingResult)

    Example:
        >>> trainer, result = train_and_evaluate(
        ...     model, tokenizer, train_data, eval_data,
        ...     epochs=3, batch_size=16
        ... )
        >>> print(f"Accuracy: {result.eval_metrics['accuracy']:.4f}")
    """
    # Setup
    training_args = create_training_args(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    compute_metrics = compute_metrics_factory(metrics)

    # Callbacks
    all_callbacks = [MemoryCallback(), TimingCallback()]
    if early_stopping_patience:
        all_callbacks.append(EarlyStoppingCallback(early_stopping_patience))
    if callbacks:
        all_callbacks.extend(callbacks)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=all_callbacks
    )

    # Train
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    train_result = trainer.train()

    # Evaluate
    eval_result = trainer.evaluate()
    if test_dataset:
        test_result = trainer.evaluate(test_dataset)
        eval_metrics = {k.replace("eval_", "test_"): v for k, v in test_result.items()}
    else:
        eval_metrics = {k.replace("eval_", ""): v for k, v in eval_result.items()}

    # Compile results
    result = TrainingResult(
        model_name=model.config.name_or_path if hasattr(model.config, 'name_or_path') else "unknown",
        train_loss=train_result.training_loss,
        eval_loss=eval_result.get("eval_loss", 0),
        eval_metrics=eval_metrics,
        training_time_seconds=train_result.metrics["train_runtime"],
        samples_per_second=train_result.metrics["train_samples_per_second"],
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters())
    )

    return trainer, result


def cleanup_memory():
    """
    Clean up GPU memory.

    Call this between training runs to free memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    # Example usage
    print("Training Utilities Demo")
    print("=" * 50)

    # Device info
    print("\nDevice Info:")
    info = get_device_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Optimal batch size
    print("\nOptimal Batch Size Estimation:")
    bs = get_optimal_batch_size(0.4)  # ~400MB model
    print(f"  Recommended for 400MB model: {bs}")

    # Training args
    print("\nSample TrainingArguments:")
    args = create_training_args("./test", epochs=2, batch_size=16)
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {args.per_device_train_batch_size}")
    print(f"  LR: {args.learning_rate}")
    print(f"  BF16: {args.bf16}")
