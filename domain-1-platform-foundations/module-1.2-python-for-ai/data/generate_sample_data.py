#!/usr/bin/env python3
"""
Sample Data Generator for Module 1.2: Python for AI/ML
=====================================================

This script generates synthetic datasets for the module exercises.
All data is reproducible with the same random seed.

Usage:
    python generate_sample_data.py

Output files:
    - sample_customers.csv
    - sample_training_history.json
    - sample_embeddings.npy

Author: Professor SPARK
Date: 2024
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


def generate_customer_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data with realistic characteristics.

    Includes:
    - Missing values at realistic rates
    - Outliers in income
    - Correlated features
    - Class imbalance in target

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with customer features and target
    """
    np.random.seed(seed)

    # Generate base features
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples).astype(float),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples).astype(float),
        'years_employed': np.random.exponential(5, n_samples),
        'num_accounts': np.random.poisson(3, n_samples),
        'education': np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD', None],
            n_samples,
            p=[0.30, 0.35, 0.20, 0.10, 0.05]
        ),
        'employment_type': np.random.choice(
            ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'],
            n_samples,
            p=[0.60, 0.15, 0.15, 0.10]
        ),
        'region': np.random.choice(
            ['North', 'South', 'East', 'West'],
            n_samples
        )
    }

    df = pd.DataFrame(data)

    # Add realistic correlations
    # Higher education tends to have higher income
    edu_bonus = df['education'].map({
        'High School': 0.8,
        'Bachelor': 1.0,
        'Master': 1.2,
        'PhD': 1.4,
        None: 1.0
    })
    df['income'] = df['income'] * edu_bonus

    # Introduce missing values
    n_missing_age = int(n_samples * 0.05)
    n_missing_income = int(n_samples * 0.08)
    n_missing_credit = int(n_samples * 0.03)

    df.loc[np.random.choice(n_samples, n_missing_age, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(n_samples, n_missing_income, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(n_samples, n_missing_credit, replace=False), 'credit_score'] = np.nan

    # Add outliers (millionaires)
    n_outliers = 5
    df.loc[np.random.choice(n_samples, n_outliers, replace=False), 'income'] = \
        np.random.uniform(5e6, 1e7, n_outliers)

    # Generate target (default) with class imbalance
    # Higher probability of default for: lower credit, lower income, unemployed
    default_prob = 0.15  # Base rate
    default_score = (
        (df['credit_score'].fillna(600) < 600).astype(float) * 0.1 +
        (df['income'].fillna(50000) < 30000).astype(float) * 0.1 +
        (df['employment_type'] == 'Unemployed').astype(float) * 0.2
    )
    default_prob_adjusted = np.clip(default_prob + default_score, 0, 0.8)
    df['default'] = (np.random.random(n_samples) < default_prob_adjusted).astype(int)

    # Round numeric values for cleaner output
    df['income'] = df['income'].round(2)
    df['years_employed'] = df['years_employed'].round(2)

    return df


def generate_training_history(n_epochs: int = 100, seed: int = 42) -> dict:
    """
    Generate synthetic training history with realistic patterns.

    Patterns included:
    - Decreasing loss over time
    - Overfitting after certain epoch
    - Noise in metrics

    Args:
        n_epochs: Number of epochs to simulate
        seed: Random seed for reproducibility

    Returns:
        Dictionary with training metrics
    """
    np.random.seed(seed)

    x = np.arange(1, n_epochs + 1)

    # Training loss: exponential decay + noise
    train_loss = 2.5 * np.exp(-0.05 * x) + 0.15 + np.random.normal(0, 0.02, n_epochs)
    train_loss = np.maximum(train_loss, 0.1)  # Floor at 0.1

    # Validation loss: similar but starts diverging (overfitting)
    val_loss = 2.5 * np.exp(-0.04 * x) + 0.25 + np.random.normal(0, 0.03, n_epochs)
    overfitting_epoch = 50
    val_loss[overfitting_epoch:] += 0.003 * (x[overfitting_epoch:] - overfitting_epoch)
    val_loss = np.maximum(val_loss, 0.15)

    # Training accuracy: inverse of loss pattern
    train_acc = 1 - 0.4 * np.exp(-0.05 * x) + np.random.normal(0, 0.01, n_epochs)
    train_acc = np.clip(train_acc, 0.5, 0.99)

    # Validation accuracy: similar with overfitting
    val_acc = 1 - 0.45 * np.exp(-0.04 * x) + np.random.normal(0, 0.015, n_epochs)
    val_acc[overfitting_epoch:] -= 0.002 * (x[overfitting_epoch:] - overfitting_epoch)
    val_acc = np.clip(val_acc, 0.5, 0.98)

    # Additional metrics
    learning_rate = 0.001 * (0.95 ** (x // 10))  # LR decay

    return {
        'epochs': x.tolist(),
        'loss': train_loss.round(4).tolist(),
        'val_loss': val_loss.round(4).tolist(),
        'accuracy': train_acc.round(4).tolist(),
        'val_accuracy': val_acc.round(4).tolist(),
        'learning_rate': learning_rate.round(6).tolist(),
        'metadata': {
            'n_epochs': n_epochs,
            'overfitting_epoch': overfitting_epoch,
            'best_val_loss_epoch': int(np.argmin(val_loss) + 1),
            'best_val_acc_epoch': int(np.argmax(val_acc) + 1)
        }
    }


def generate_embeddings(n_samples: int = 1000, n_dims: int = 128, seed: int = 42) -> np.ndarray:
    """
    Generate random embeddings for distance/similarity exercises.

    The embeddings are normalized to have similar magnitudes,
    mimicking real embedding spaces.

    Args:
        n_samples: Number of embedding vectors
        n_dims: Dimensionality of each vector
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, n_dims)
    """
    np.random.seed(seed)

    # Generate from standard normal
    embeddings = np.random.randn(n_samples, n_dims).astype(np.float32)

    # Normalize to unit length (like many real embedding spaces)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return embeddings


def generate_confusion_matrix_data(n_samples: int = 1000, n_classes: int = 5, seed: int = 42) -> dict:
    """
    Generate predictions and labels for confusion matrix visualization.

    Creates realistic prediction patterns where adjacent classes
    are more likely to be confused.

    Args:
        n_samples: Total number of predictions
        n_classes: Number of classes
        seed: Random seed for reproducibility

    Returns:
        Dictionary with true labels and predictions
    """
    np.random.seed(seed)

    class_names = [f'Class_{i}' for i in range(n_classes)]

    # True labels (roughly balanced)
    y_true = np.random.randint(0, n_classes, n_samples)

    # Predictions: usually correct, but sometimes wrong
    # Adjacent classes more likely to be confused
    y_pred = y_true.copy()

    n_errors = int(n_samples * 0.25)  # 25% error rate
    error_indices = np.random.choice(n_samples, n_errors, replace=False)

    for idx in error_indices:
        true_class = y_true[idx]
        # Prefer adjacent classes for errors
        error_weights = [0.4 ** abs(c - true_class) for c in range(n_classes)]
        error_weights[true_class] = 0  # Can't be correct
        error_weights = np.array(error_weights) / sum(error_weights)
        y_pred[idx] = np.random.choice(n_classes, p=error_weights)

    return {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'class_names': class_names
    }


def main():
    """Generate all sample datasets."""
    # Get script directory
    script_dir = Path(__file__).parent

    print("Generating sample data for Module 2...")
    print("=" * 50)

    # 1. Customer data
    print("\n1. Generating customer data...")
    df_customers = generate_customer_data(n_samples=1000)
    customer_path = script_dir / 'sample_customers.csv'
    df_customers.to_csv(customer_path, index=False)
    print(f"   Saved: {customer_path}")
    print(f"   Shape: {df_customers.shape}")
    print(f"   Missing values:\n{df_customers.isnull().sum().to_string()}")

    # 2. Training history
    print("\n2. Generating training history...")
    history = generate_training_history(n_epochs=100)
    history_path = script_dir / 'sample_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   Saved: {history_path}")
    print(f"   Epochs: {history['metadata']['n_epochs']}")
    print(f"   Best val_loss epoch: {history['metadata']['best_val_loss_epoch']}")

    # 3. Embeddings
    print("\n3. Generating embeddings...")
    embeddings = generate_embeddings(n_samples=1000, n_dims=128)
    embeddings_path = script_dir / 'sample_embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"   Saved: {embeddings_path}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Memory: {embeddings.nbytes / 1024:.1f} KB")

    # 4. Confusion matrix data
    print("\n4. Generating confusion matrix data...")
    cm_data = generate_confusion_matrix_data(n_samples=1000, n_classes=5)
    cm_path = script_dir / 'sample_confusion_data.json'
    with open(cm_path, 'w') as f:
        json.dump(cm_data, f)
    print(f"   Saved: {cm_path}")
    print(f"   Classes: {cm_data['class_names']}")

    print("\n" + "=" * 50)
    print("All sample data generated successfully!")
    print("\nYou can now run the Module 2 notebooks.")


if __name__ == '__main__':
    main()
