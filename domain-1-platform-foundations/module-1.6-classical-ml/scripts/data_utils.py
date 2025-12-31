"""
Data Utilities for Classical ML Module

This module provides utilities for loading, preprocessing, and generating
datasets for classical ML experiments.

Example:
    >>> from data_utils import load_dataset, preprocess_features
    >>> X, y, feature_names = load_dataset('california_housing')
    >>> X_processed = preprocess_features(X, scale=True, handle_missing=True)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union, Dict
import warnings

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_wine,
    load_iris,
    load_diabetes,
    make_classification,
    make_regression
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# Available datasets
AVAILABLE_DATASETS = {
    'california_housing': ('regression', fetch_california_housing),
    'breast_cancer': ('classification', load_breast_cancer),
    'wine': ('classification', load_wine),
    'iris': ('classification', load_iris),
    'diabetes': ('regression', load_diabetes)
}


def load_dataset(
    name: str,
    return_df: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, List[str]], pd.DataFrame]:
    """
    Load a standard dataset by name.

    Args:
        name: Dataset name. Options:
            - 'california_housing': California Housing prices (regression)
            - 'breast_cancer': Breast cancer classification
            - 'wine': Wine classification
            - 'iris': Iris flower classification
            - 'diabetes': Diabetes regression
        return_df: If True, return as DataFrame with target column

    Returns:
        If return_df=False: (X, y, feature_names)
        If return_df=True: DataFrame with all features and 'target' column

    Example:
        >>> X, y, names = load_dataset('breast_cancer')
        >>> print(f"Samples: {len(X)}, Features: {len(names)}")

        >>> df = load_dataset('california_housing', return_df=True)
        >>> print(df.head())
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(AVAILABLE_DATASETS.keys())}"
        )

    task, loader = AVAILABLE_DATASETS[name]
    data = loader()

    X = data.data
    y = data.target
    feature_names = list(data.feature_names)

    if return_df:
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df

    return X, y, feature_names


def get_dataset_info(name: str) -> Dict:
    """
    Get information about a dataset.

    Args:
        name: Dataset name

    Returns:
        Dictionary with dataset information

    Example:
        >>> info = get_dataset_info('california_housing')
        >>> print(f"Task: {info['task']}, Samples: {info['n_samples']}")
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    task, loader = AVAILABLE_DATASETS[name]
    data = loader()

    info = {
        'name': name,
        'task': task,
        'n_samples': data.data.shape[0],
        'n_features': data.data.shape[1],
        'feature_names': list(data.feature_names),
        'target_names': getattr(data, 'target_names', None)
    }

    if task == 'classification':
        info['n_classes'] = len(np.unique(data.target))

    return info


def preprocess_features(
    X: np.ndarray,
    scale: bool = True,
    handle_missing: bool = True,
    clip_outliers: bool = False,
    outlier_threshold: float = 5.0
) -> np.ndarray:
    """
    Preprocess features with common transformations.

    Args:
        X: Feature matrix (n_samples, n_features)
        scale: Apply StandardScaler
        handle_missing: Fill NaN with column means
        clip_outliers: Clip values beyond outlier_threshold std devs
        outlier_threshold: Number of standard deviations for outlier clipping

    Returns:
        Preprocessed feature matrix

    Example:
        >>> X_raw = np.array([[1, 2], [3, np.nan], [5, 6], [100, 8]])
        >>> X_clean = preprocess_features(
        ...     X_raw, scale=True, handle_missing=True, clip_outliers=True
        ... )
    """
    X = X.copy().astype(np.float32)

    # Handle missing values
    if handle_missing:
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    # Clip outliers
    if clip_outliers:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        lower = mean - outlier_threshold * std
        upper = mean + outlier_threshold * std
        X = np.clip(X, lower, upper)

    # Scale features
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 20,
    task: str = 'classification',
    n_classes: int = 2,
    n_informative: Optional[int] = None,
    noise: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic data for testing.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        task: 'classification' or 'regression'
        n_classes: Number of classes (for classification)
        n_informative: Number of informative features (default: n_features // 2)
        noise: Noise level (for regression)
        random_state: Random seed

    Returns:
        (X, y, feature_names)

    Example:
        >>> X, y, names = generate_synthetic_data(
        ...     n_samples=100000,
        ...     n_features=50,
        ...     task='classification'
        ... )
        >>> print(f"Generated: {X.shape}")
    """
    if n_informative is None:
        n_informative = max(2, n_features // 2)

    feature_names = [f'feature_{i}' for i in range(n_features)]

    if task == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative - 2,
            n_classes=n_classes,
            random_state=random_state
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise * np.sqrt(n_informative),
            random_state=random_state
        )

    return X.astype(np.float32), y, feature_names


def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify: bool = True,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create train/validation/test splits.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        stratify: Use stratified split for classification
        random_state: Random seed

    Returns:
        Dictionary with 'X_train', 'X_test', 'y_train', 'y_test',
        and optionally 'X_val', 'y_val'

    Example:
        >>> splits = create_train_test_split(X, y, test_size=0.2, val_size=0.1)
        >>> X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    """
    stratify_arg = y if stratify and len(np.unique(y)) < 50 else None

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_arg,
        random_state=random_state
    )

    result = {
        'X_train': X_temp,
        'X_test': X_test,
        'y_train': y_temp,
        'y_test': y_test
    }

    # Second split: train vs val
    if val_size > 0:
        val_proportion = val_size / (1 - test_size)
        stratify_arg = y_temp if stratify and len(np.unique(y)) < 50 else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_proportion,
            stratify=stratify_arg,
            random_state=random_state
        )

        result['X_train'] = X_train
        result['X_val'] = X_val
        result['y_train'] = y_train
        result['y_val'] = y_val

    return result


def encode_categorical(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical columns using LabelEncoder.

    Args:
        df: DataFrame with features
        categorical_columns: List of columns to encode (auto-detect if None)

    Returns:
        (encoded_df, encoders_dict)

    Example:
        >>> df = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': [1, 2, 3]})
        >>> df_encoded, encoders = encode_categorical(df, ['color'])
        >>> print(df_encoded)
    """
    df = df.copy()
    encoders = {}

    # Auto-detect categorical columns
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders


def get_memory_usage(X: np.ndarray) -> str:
    """
    Get human-readable memory usage of array.

    Args:
        X: NumPy array

    Returns:
        String with memory usage (e.g., "125.5 MB")

    Example:
        >>> X = np.random.randn(100000, 100).astype(np.float32)
        >>> print(get_memory_usage(X))  # "38.1 MB"
    """
    bytes_used = X.nbytes
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024**2:
        return f"{bytes_used/1024:.1f} KB"
    elif bytes_used < 1024**3:
        return f"{bytes_used/1024**2:.1f} MB"
    else:
        return f"{bytes_used/1024**3:.1f} GB"
