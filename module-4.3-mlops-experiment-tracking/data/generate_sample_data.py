"""
Sample Data Generator for Module 4.3 Exercises.

Generates synthetic datasets for:
- Sentiment classification
- Drift detection
- Model monitoring

Usage:
    python generate_sample_data.py --output sample_sentiment.csv --n_samples 1000
    python generate_sample_data.py --output drifted_data.csv --drift_intensity 0.5
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def generate_sentiment_data(
    n_samples: int = 1000,
    seed: int = 42,
    drift_intensity: float = 0.0,
    missing_ratio: float = 0.0
) -> pd.DataFrame:
    """
    Generate synthetic sentiment classification data.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        drift_intensity: How much to shift distributions (0-1)
        missing_ratio: Ratio of missing values to introduce

    Returns:
        DataFrame with features and labels
    """
    np.random.seed(seed)

    # Base feature distributions
    data = {
        'text_length': np.random.normal(150 + 50 * drift_intensity, 50, n_samples).clip(20, 500),
        'word_count': np.random.normal(30, 10, n_samples).clip(5, 100),
        'avg_word_length': np.random.normal(5.5, 1.0, n_samples).clip(3, 10),
        'sentiment_keywords': np.random.poisson(3 + int(2 * drift_intensity), n_samples),
        'exclamation_count': np.random.poisson(1 + int(3 * drift_intensity), n_samples),
        'question_marks': np.random.poisson(0.5, n_samples),
        'uppercase_ratio': np.random.beta(2 + 5 * drift_intensity, 20, n_samples),
    }

    df = pd.DataFrame(data)

    # Generate target based on features
    score = (
        0.3 * (df['sentiment_keywords'] > 2) +
        0.2 * (df['exclamation_count'] > 0) +
        0.2 * (df['text_length'] > 100) +
        0.3 * np.random.random(n_samples)
    )
    df['target'] = (score > 0.5).astype(int)

    # Simulate predictions (with some noise)
    noise = np.random.random(n_samples) < (0.1 + drift_intensity * 0.2)
    df['prediction'] = df['target'].copy()
    df.loc[noise, 'prediction'] = 1 - df.loc[noise, 'prediction']

    # Add missing values if requested
    if missing_ratio > 0:
        for col in ['sentiment_keywords', 'text_length']:
            mask = np.random.random(n_samples) < missing_ratio
            df.loc[mask, col] = np.nan

    return df


def generate_feature_data(
    n_samples: int = 2000,
    n_features: int = 10,
    seed: int = 42,
    include_timestamp: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic feature data for drift detection exercises.

    Args:
        n_samples: Number of samples
        n_features: Number of numerical features
        seed: Random seed
        include_timestamp: Add timestamp column

    Returns:
        DataFrame with features
    """
    np.random.seed(seed)

    data = {}

    # Numerical features with different distributions
    for i in range(n_features):
        if i % 3 == 0:
            data[f'feature_{i+1}'] = np.random.normal(0, 1, n_samples)
        elif i % 3 == 1:
            data[f'feature_{i+1}'] = np.random.exponential(1, n_samples)
        else:
            data[f'feature_{i+1}'] = np.random.uniform(-1, 1, n_samples)

    # Categorical feature
    categories = ['A', 'B', 'C', 'D']
    data['category'] = np.random.choice(categories, n_samples, p=[0.4, 0.3, 0.2, 0.1])

    # Train/val/test split
    split_probs = [0.8, 0.1, 0.1]
    data['split'] = np.random.choice(['train', 'val', 'test'], n_samples, p=split_probs)

    df = pd.DataFrame(data)

    # Add timestamp
    if include_timestamp:
        base_time = datetime.now() - timedelta(days=30)
        timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]
        df['timestamp'] = timestamps

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample data for MLOps module")
    parser.add_argument("--output", type=str, default="sample_data.csv",
                       help="Output file path")
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--drift_intensity", type=float, default=0.0,
                       help="Drift intensity (0-1)")
    parser.add_argument("--missing_ratio", type=float, default=0.0,
                       help="Ratio of missing values")
    parser.add_argument("--type", type=str, default="sentiment",
                       choices=["sentiment", "features"],
                       help="Type of data to generate")

    args = parser.parse_args()

    if args.type == "sentiment":
        df = generate_sentiment_data(
            n_samples=args.n_samples,
            seed=args.seed,
            drift_intensity=args.drift_intensity,
            missing_ratio=args.missing_ratio
        )
    else:
        df = generate_feature_data(
            n_samples=args.n_samples,
            seed=args.seed
        )

    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} samples -> {args.output}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
