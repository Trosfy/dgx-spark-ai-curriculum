# Data Directory - Module 2: Python for AI/ML

This directory contains sample datasets and data generation utilities for the Module 2 exercises.

---

## ⚠️ IMPORTANT: Generate Data Before Running Notebooks

**The notebooks require sample data files to be generated first.**

Run this command before starting the exercises:

```bash
cd phase-1-foundations/module-02-python-for-ai/data
python generate_sample_data.py
```

This creates the required files:
- `sample_customers.csv`
- `sample_training_history.json`
- `sample_embeddings.npy`
- `sample_confusion_data.json`

**Note:** The notebooks include automatic data generation checks, but running the generator manually ensures all files are ready.

---

## Contents

### Generated Files (created during exercises)

- `preprocessor.pkl` - Saved preprocessor object from Task 2.2
- `model_dashboard.png` - Saved visualization from Task 2.3

### Utility Scripts

- `generate_sample_data.py` - Script to generate synthetic datasets for exercises

## Generating Sample Data

To generate fresh sample data for the exercises:

```bash
cd phase-1-foundations/module-02-python-for-ai/data
python generate_sample_data.py
```

This will create:
- `sample_customers.csv` - Synthetic customer data with missing values and outliers
- `sample_training_history.json` - Simulated model training history
- `sample_embeddings.npy` - Random embeddings for distance/similarity exercises

## Data Descriptions

### sample_customers.csv

Synthetic customer data for preprocessing exercises:
- **age**: Customer age (18-80, with ~5% missing)
- **income**: Annual income (log-normal distribution, with ~8% missing)
- **credit_score**: Credit score (300-850, with ~3% missing)
- **years_employed**: Years at current job (exponential distribution)
- **education**: Education level (High School, Bachelor, Master, PhD, ~5% missing)
- **employment_type**: Employment status (Full-time, Part-time, Self-employed, Unemployed)
- **default**: Target variable (0=no default, 1=default, ~15% positive rate)

### sample_training_history.json

Simulated training metrics for visualization exercises:
- **loss**: Training loss per epoch
- **val_loss**: Validation loss per epoch (with overfitting pattern)
- **accuracy**: Training accuracy per epoch
- **val_accuracy**: Validation accuracy per epoch

### sample_embeddings.npy

Random embeddings for numerical exercises:
- Shape: (1000, 128) - 1000 vectors of 128 dimensions
- dtype: float32
- Useful for distance matrix and similarity computations

## Notes

- All synthetic data is reproducible with seed=42
- Data sizes are chosen to be manageable on any system
- For DGX Spark exercises, feel free to scale up the data sizes

## Memory Estimates

| Dataset | Size | Memory |
|---------|------|--------|
| sample_customers.csv | 1000 rows | ~100 KB |
| sample_embeddings.npy | 1000 × 128 | ~500 KB |
| sample_training_history.json | 100 epochs | ~10 KB |

For larger-scale experiments on DGX Spark (128GB memory), you can safely scale these by 100-1000x.
