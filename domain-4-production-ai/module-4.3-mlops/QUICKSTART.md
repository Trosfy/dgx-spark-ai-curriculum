# Module 4.3: MLOps & Experiment Tracking - Quickstart

## Time: ~5 minutes

## What You'll Build

Set up MLflow experiment tracking and log your first model training run.

## Before You Start

- [ ] DGX Spark container running
- [ ] Basic Python knowledge

## Let's Go!

### Step 1: Install MLflow

```bash
pip install mlflow -q
```

### Step 2: Start MLflow Tracking Server

```bash
# In background (or separate terminal)
mlflow server --host 0.0.0.0 --port 5000 &
```

### Step 3: Create a Simple Training Script

```python
import mlflow
import random

# Connect to tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("quickstart-demo")

# Start a run
with mlflow.start_run(run_name="my-first-run"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)

    # Simulate training and log metrics
    for epoch in range(10):
        loss = 1.0 / (epoch + 1) + random.random() * 0.1
        accuracy = 0.5 + (epoch * 0.05) + random.random() * 0.02
        mlflow.log_metrics({
            "loss": loss,
            "accuracy": accuracy
        }, step=epoch)

    # Log final metrics
    mlflow.log_metric("final_accuracy", accuracy)

    print("Run logged successfully!")
```

### Step 4: View in MLflow UI

Open in browser: `http://localhost:5000`

You'll see:
- Your experiment "quickstart-demo"
- The run with parameters and metrics
- Charts showing loss decreasing

### Step 5: Compare Multiple Runs

```python
# Run a few more experiments with different params
for lr in [0.01, 0.001, 0.0001]:
    with mlflow.start_run(run_name=f"lr-{lr}"):
        mlflow.log_param("learning_rate", lr)
        # ... training code ...
        mlflow.log_metric("final_accuracy", 0.8 + random.random() * 0.1)
```

**Expected output in MLflow UI**:
- Multiple runs to compare
- Sortable by any metric
- Charts comparing runs

## You Did It!

You just set up experiment tracking that logs parameters, metrics, and allows comparison across runs! In the full module, you'll learn:

- **W&B Integration**: Alternative tracking with more features
- **LLM Benchmarks**: MMLU, HellaSwag, TruthfulQA with lm-eval
- **Drift Detection**: Monitor model performance over time
- **Model Registry**: Version and stage your models
- **Reproducibility**: Ensure experiments can be replicated

## Next Steps

1. **Add real training**: Replace simulation with actual model training
2. **Log artifacts**: Save model weights, plots, configs
3. **Full tutorial**: Start with [LAB_PREP.md](./LAB_PREP.md)
