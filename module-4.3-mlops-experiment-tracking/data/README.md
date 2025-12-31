# Data Files for Module 4.3: MLOps & Experiment Tracking

This directory contains sample data files and data generation utilities for the module exercises.

## Contents

### Sample Datasets

| File | Description | Size | Format |
|------|-------------|------|--------|
| `sample_sentiment.csv` | Synthetic sentiment classification data | ~1000 rows | CSV |
| `sample_features.csv` | Feature dataset for drift detection exercises | ~2000 rows | CSV |

### Data Generation

Most notebooks generate synthetic data inline. The `generate_sample_data.py` script can be used to regenerate sample datasets:

```bash
python generate_sample_data.py --output sample_sentiment.csv --n_samples 1000
```

## Data Schema

### sample_sentiment.csv

| Column | Type | Description |
|--------|------|-------------|
| text_length | float | Character count of text |
| word_count | float | Word count |
| avg_word_length | float | Average word length |
| sentiment_keywords | int | Count of sentiment words |
| exclamation_count | int | Exclamation marks |
| question_marks | int | Question marks |
| uppercase_ratio | float | Ratio of uppercase chars |
| target | int | Ground truth label (0/1) |
| prediction | int | Model prediction (0/1) |

### sample_features.csv

| Column | Type | Description |
|--------|------|-------------|
| feature_1 to feature_10 | float | Numerical features |
| category | str | Categorical feature |
| timestamp | datetime | Sample timestamp |
| split | str | train/val/test split |

## Usage Notes

1. **Seed for Reproducibility**: All data generation uses seed=42 by default
2. **Drift Simulation**: Use `--drift_intensity` parameter to generate drifted data
3. **Missing Values**: Use `--missing_ratio` to add missing values for quality testing

## License

Sample data is synthetic and can be freely used for educational purposes.
