# Data Files for Module 1.6: Classical ML Foundations

This directory contains data files and instructions for datasets used in the Classical ML labs.

## Built-in Datasets (Recommended)

These datasets are included in scikit-learn and load automatically:

| Dataset | Size | Task | Lab Usage |
|---------|------|------|-----------|
| California Housing | 20,640 × 8 | Regression | Labs 1.6.1, 1.6.2 |
| Breast Cancer | 569 × 30 | Classification | Lab 1.6.4 |
| Wine | 178 × 13 | Classification | Lab 1.6.4 |
| Iris | 150 × 4 | Classification | Testing |
| Diabetes | 442 × 10 | Regression | Testing |

### Loading Built-in Datasets

```python
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_wine,
    load_iris,
    load_diabetes
)

# California Housing (Regression)
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Breast Cancer (Binary Classification)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Using our utility module
from scripts.data_utils import load_dataset
X, y, names = load_dataset('california_housing')
```

## External Datasets for Benchmarking

For large-scale benchmarks, consider these external datasets:

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| Adult Income | 48,842 × 14 | Classification | UCI ML Repository |
| Credit Card Fraud | 284,807 × 30 | Imbalanced Classification | Kaggle |
| Higgs Boson | 11M × 28 | Classification | UCI ML Repository |
| Cover Type | 581,012 × 54 | Multi-class Classification | sklearn |

### Loading External Datasets

```python
# Adult Income Dataset
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
df = pd.read_csv(url, names=columns, skipinitialspace=True)

# Cover Type (Built-in, large)
from sklearn.datasets import fetch_covtype
covtype = fetch_covtype()
X, y = covtype.data, covtype.target
print(f"Shape: {X.shape}")  # (581012, 54)
```

## Generating Synthetic Data

For controlled experiments, generate synthetic data:

```python
from sklearn.datasets import make_classification, make_regression

# Classification
X, y = make_classification(
    n_samples=100_000,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    n_classes=2,
    random_state=42
)

# Regression
X, y = make_regression(
    n_samples=100_000,
    n_features=50,
    n_informative=30,
    noise=10,
    random_state=42
)

# Using our utility module
from scripts.data_utils import generate_synthetic_data
X, y, names = generate_synthetic_data(
    n_samples=100_000,
    n_features=50,
    task='classification'
)
```

## RAPIDS/cuDF Data Loading

For GPU acceleration with large datasets:

```python
import cudf

# From pandas DataFrame
pdf = pd.read_csv('large_data.csv')
gdf = cudf.DataFrame.from_pandas(pdf)

# Direct loading (faster for large files)
gdf = cudf.read_csv('large_data.csv')

# From NumPy arrays
gdf = cudf.DataFrame(X)
```

## Memory Considerations on DGX Spark

With the DGX Spark's 128GB unified memory:

| Dataset Size | Estimated Memory | Fits in GPU? |
|--------------|------------------|--------------|
| 1M × 50 (float32) | ~200 MB | Yes |
| 10M × 50 (float32) | ~2 GB | Yes |
| 100M × 50 (float32) | ~20 GB | Yes |
| Higgs (11M × 28) | ~3 GB | Yes |

Tips:
- Always use `float32` instead of `float64` to halve memory usage
- DGX Spark's 128GB unified memory eliminates CPU↔GPU transfer overhead
- No explicit data transfers needed on DGX Spark

## Data Preprocessing Pipeline

```python
from scripts.data_utils import preprocess_features, create_train_test_split

# Preprocess
X_clean = preprocess_features(
    X,
    scale=True,           # StandardScaler
    handle_missing=True,  # Fill NaN with mean
    clip_outliers=True    # Clip extreme values
)

# Split
splits = create_train_test_split(
    X, y,
    test_size=0.2,
    val_size=0.1,  # Optional validation set
    stratify=True  # For classification
)
X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
```

## File Structure

```
data/
├── README.md           # This file
└── (datasets loaded dynamically from sklearn or generated)
```

Note: Large datasets are not stored in this repository. They are either:
1. Loaded from sklearn (built-in)
2. Downloaded from external sources
3. Generated synthetically

This keeps the repository lightweight while supporting large-scale experiments.
