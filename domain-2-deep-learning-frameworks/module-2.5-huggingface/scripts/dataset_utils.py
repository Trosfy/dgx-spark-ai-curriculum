"""
Dataset Utilities for Hugging Face Datasets

This module provides helper functions for loading, processing, and analyzing
datasets using the Hugging Face datasets library.

Example usage:
    from scripts.dataset_utils import (
        load_and_analyze_dataset,
        prepare_dataset_for_training,
        create_stratified_splits,
        stream_large_dataset
    )

    # Load and analyze a dataset
    dataset, stats = load_and_analyze_dataset("imdb")

    # Prepare for training with tokenization
    processed = prepare_dataset_for_training(
        dataset, tokenizer, text_column="text", label_column="label"
    )
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass
import torch
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    Features,
    Value,
    ClassLabel
)
from transformers import PreTrainedTokenizer
import numpy as np
from collections import Counter


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    name: str
    num_examples: Dict[str, int]
    num_columns: int
    columns: List[str]
    features: Dict[str, str]
    label_distribution: Optional[Dict[str, Dict[Any, int]]] = None
    text_length_stats: Optional[Dict[str, Dict[str, float]]] = None
    memory_size_mb: float = 0.0


def load_and_analyze_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    **kwargs
) -> Tuple[DatasetDict, DatasetStats]:
    """
    Load a dataset and compute statistics.

    Args:
        dataset_name: Hugging Face dataset name (e.g., "imdb", "glue")
        subset: Optional subset name (e.g., "sst2" for glue)
        text_column: Column containing text (for length analysis)
        label_column: Column containing labels (for distribution analysis)
        **kwargs: Additional arguments for load_dataset

    Returns:
        Tuple of (dataset, statistics)

    Example:
        >>> dataset, stats = load_and_analyze_dataset(
        ...     "imdb",
        ...     text_column="text",
        ...     label_column="label"
        ... )
        >>> print(f"Training examples: {stats.num_examples['train']}")
    """
    # Load dataset
    if subset:
        dataset = load_dataset(dataset_name, subset, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)

    # Ensure we have a DatasetDict
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    # Get basic info
    first_split = list(dataset.keys())[0]
    columns = dataset[first_split].column_names
    features = {
        col: str(dataset[first_split].features[col])
        for col in columns
    }

    # Count examples per split
    num_examples = {split: len(data) for split, data in dataset.items()}

    # Estimate memory size
    memory_size_mb = sum(
        data.data.nbytes / 1e6
        for data in dataset.values()
        if hasattr(data, 'data')
    )

    # Build stats
    stats = DatasetStats(
        name=f"{dataset_name}/{subset}" if subset else dataset_name,
        num_examples=num_examples,
        num_columns=len(columns),
        columns=columns,
        features=features,
        memory_size_mb=memory_size_mb
    )

    # Label distribution
    if label_column and label_column in columns:
        stats.label_distribution = {}
        for split, data in dataset.items():
            stats.label_distribution[split] = dict(
                Counter(data[label_column])
            )

    # Text length stats
    if text_column and text_column in columns:
        stats.text_length_stats = {}
        for split, data in dataset.items():
            lengths = [len(str(t)) for t in data[text_column][:1000]]  # Sample
            stats.text_length_stats[split] = {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": min(lengths),
                "max": max(lengths),
                "median": np.median(lengths)
            }

    return dataset, stats


def print_dataset_stats(stats: DatasetStats) -> None:
    """
    Pretty print dataset statistics.

    Args:
        stats: DatasetStats object
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {stats.name}")
    print(f"{'='*60}")

    print(f"\nSplit sizes:")
    for split, count in stats.num_examples.items():
        print(f"  {split}: {count:,} examples")

    print(f"\nColumns ({stats.num_columns}):")
    for col, dtype in stats.features.items():
        print(f"  - {col}: {dtype}")

    if stats.label_distribution:
        print(f"\nLabel distribution:")
        for split, dist in stats.label_distribution.items():
            print(f"  {split}:")
            for label, count in sorted(dist.items()):
                pct = 100 * count / stats.num_examples[split]
                print(f"    {label}: {count:,} ({pct:.1f}%)")

    if stats.text_length_stats:
        print(f"\nText length (characters):")
        for split, lengths in stats.text_length_stats.items():
            print(f"  {split}: mean={lengths['mean']:.0f}, "
                  f"median={lengths['median']:.0f}, "
                  f"range=[{lengths['min']}-{lengths['max']}]")

    print(f"\nEstimated memory: {stats.memory_size_mb:.2f} MB")


def prepare_dataset_for_training(
    dataset: Union[Dataset, DatasetDict],
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    label_column: Optional[str] = None,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    batched: bool = True,
    batch_size: int = 1000,
    num_proc: int = 4,
    remove_original_columns: bool = True
) -> Union[Dataset, DatasetDict]:
    """
    Prepare a dataset for training by tokenizing text.

    Args:
        dataset: Dataset or DatasetDict to process
        tokenizer: Tokenizer to use
        text_column: Column containing text
        label_column: Optional column with labels (renamed to 'labels')
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        batched: Use batched processing
        batch_size: Batch size for map
        num_proc: Number of processes
        remove_original_columns: Remove non-model columns

    Returns:
        Tokenized dataset ready for training.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> processed = prepare_dataset_for_training(
        ...     dataset["train"],
        ...     tokenizer,
        ...     text_column="text",
        ...     label_column="label"
        ... )
    """
    def tokenize_function(examples):
        result = tokenizer(
            examples[text_column],
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )
        return result

    # Get columns to remove
    if isinstance(dataset, DatasetDict):
        first_split = list(dataset.keys())[0]
        columns = dataset[first_split].column_names
    else:
        columns = dataset.column_names

    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=batched,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing"
    )

    # Rename label column if needed
    if label_column and label_column != "labels" and label_column in columns:
        tokenized = tokenized.rename_column(label_column, "labels")

    # Remove original columns
    if remove_original_columns:
        keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
        remove_cols = [c for c in columns if c not in keep_cols]

        if remove_cols:
            tokenized = tokenized.remove_columns(remove_cols)

    # Set format
    tokenized.set_format("torch")

    return tokenized


def create_stratified_splits(
    dataset: Dataset,
    label_column: str = "label",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Create stratified train/val/test splits.

    Args:
        dataset: Dataset to split
        label_column: Column with labels for stratification
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for test
        seed: Random seed

    Returns:
        DatasetDict with train, validation, test splits.

    Example:
        >>> splits = create_stratified_splits(dataset, label_column="label")
        >>> print(splits)  # DatasetDict with train, validation, test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Sizes must sum to 1.0"

    # First split: train+val vs test
    first_split = dataset.train_test_split(
        test_size=test_size,
        stratify_by_column=label_column,
        seed=seed
    )

    # Second split: train vs val (from the train portion)
    val_fraction = val_size / (train_size + val_size)
    second_split = first_split["train"].train_test_split(
        test_size=val_fraction,
        stratify_by_column=label_column,
        seed=seed
    )

    return DatasetDict({
        "train": second_split["train"],
        "validation": second_split["test"],
        "test": first_split["test"]
    })


def stream_large_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    batch_size: int = 1000,
    shuffle: bool = True,
    seed: int = 42
) -> IterableDataset:
    """
    Load a large dataset in streaming mode.

    Args:
        dataset_name: Hugging Face dataset name
        subset: Optional subset name
        split: Split to stream
        batch_size: Buffer size for shuffling
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        IterableDataset for memory-efficient processing.

    Example:
        >>> stream = stream_large_dataset("c4", split="train")
        >>> for batch in stream.take(10):
        ...     print(batch)
    """
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size, seed=seed)

    return dataset


def create_data_collator(
    tokenizer: PreTrainedTokenizer,
    task: str = "classification",
    mlm_probability: float = 0.15
):
    """
    Create appropriate data collator for a task.

    Args:
        tokenizer: Tokenizer instance
        task: Task type ("classification", "mlm", "clm")
        mlm_probability: Masking probability for MLM

    Returns:
        Appropriate data collator.

    Example:
        >>> collator = create_data_collator(tokenizer, task="classification")
        >>> trainer = Trainer(..., data_collator=collator)
    """
    from transformers import (
        DataCollatorWithPadding,
        DataCollatorForLanguageModeling,
    )

    if task == "classification":
        return DataCollatorWithPadding(tokenizer=tokenizer)
    elif task == "mlm":
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )
    elif task == "clm":
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def filter_by_length(
    dataset: Union[Dataset, DatasetDict],
    text_column: str,
    min_length: int = 10,
    max_length: int = 10000
) -> Union[Dataset, DatasetDict]:
    """
    Filter dataset by text length.

    Args:
        dataset: Dataset to filter
        text_column: Column with text
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        Filtered dataset.

    Example:
        >>> filtered = filter_by_length(dataset, "text", min_length=50)
    """
    def length_filter(example):
        length = len(str(example[text_column]))
        return min_length <= length <= max_length

    return dataset.filter(length_filter)


def balance_dataset(
    dataset: Dataset,
    label_column: str = "label",
    strategy: str = "undersample",
    seed: int = 42
) -> Dataset:
    """
    Balance a dataset by label.

    Args:
        dataset: Dataset to balance
        label_column: Column with labels
        strategy: "undersample" or "oversample"
        seed: Random seed

    Returns:
        Balanced dataset.

    Example:
        >>> balanced = balance_dataset(dataset, label_column="label")
    """
    # Count labels
    label_counts = Counter(dataset[label_column])

    if strategy == "undersample":
        target_count = min(label_counts.values())
    else:  # oversample
        target_count = max(label_counts.values())

    # Split by label
    balanced_data = []
    rng = np.random.RandomState(seed)

    for label in label_counts.keys():
        label_data = dataset.filter(lambda x: x[label_column] == label)
        indices = list(range(len(label_data)))

        if strategy == "undersample":
            selected = rng.choice(indices, size=target_count, replace=False)
        else:
            selected = rng.choice(indices, size=target_count, replace=True)

        balanced_data.append(label_data.select(selected))

    # Concatenate and shuffle
    from datasets import concatenate_datasets
    result = concatenate_datasets(balanced_data)

    return result.shuffle(seed=seed)


def create_few_shot_dataset(
    dataset: Dataset,
    label_column: str = "label",
    shots_per_class: int = 16,
    seed: int = 42
) -> Dataset:
    """
    Create a few-shot dataset with N examples per class.

    Args:
        dataset: Source dataset
        label_column: Column with labels
        shots_per_class: Number of examples per class
        seed: Random seed

    Returns:
        Few-shot dataset.

    Example:
        >>> few_shot = create_few_shot_dataset(dataset, shots_per_class=8)
    """
    from datasets import concatenate_datasets

    label_values = set(dataset[label_column])
    selected_data = []
    rng = np.random.RandomState(seed)

    for label in label_values:
        label_data = dataset.filter(lambda x: x[label_column] == label)

        if len(label_data) >= shots_per_class:
            indices = rng.choice(len(label_data), size=shots_per_class, replace=False)
        else:
            indices = list(range(len(label_data)))

        selected_data.append(label_data.select(indices))

    return concatenate_datasets(selected_data).shuffle(seed=seed)


def get_sample_batch(
    dataset: Dataset,
    batch_size: int = 8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Get a sample batch from a dataset for testing.

    Args:
        dataset: Dataset to sample from
        batch_size: Number of examples
        seed: Random seed

    Returns:
        Dictionary with batched examples.

    Example:
        >>> batch = get_sample_batch(tokenized_dataset, batch_size=4)
        >>> print(batch["input_ids"].shape)
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(batch_size, len(dataset)), replace=False)

    samples = dataset.select(indices)

    if dataset.format["type"] == "torch":
        return {
            key: torch.stack([samples[i][key] for i in range(len(samples))])
            for key in samples.column_names
            if isinstance(samples[0][key], torch.Tensor)
        }

    return {key: samples[key] for key in samples.column_names}


if __name__ == "__main__":
    # Demo
    print("Dataset Utilities Demo")
    print("=" * 50)

    # Load and analyze
    print("\nLoading IMDB dataset...")
    dataset, stats = load_and_analyze_dataset(
        "imdb",
        text_column="text",
        label_column="label"
    )
    print_dataset_stats(stats)

    # Show sample
    print("\nSample:")
    print(f"  Text: {dataset['train'][0]['text'][:100]}...")
    print(f"  Label: {dataset['train'][0]['label']}")
