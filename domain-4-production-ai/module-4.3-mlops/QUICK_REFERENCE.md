# Module 4.3: MLOps & Experiment Tracking - Quick Reference

## Essential Commands

### NGC Container Setup

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 5000:5000 \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Install Dependencies

```bash
pip install mlflow>=2.10.0 wandb>=0.16.0
pip install lm-eval>=0.4.0
pip install evidently>=0.4.0
```

---

## MLflow

### Start Tracking Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Basic Logging

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment")

with mlflow.start_run(run_name="run-001"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    # Log metrics
    mlflow.log_metric("loss", 0.5)
    mlflow.log_metric("accuracy", 0.85)

    # Log artifacts
    mlflow.log_artifact("config.yaml")

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Track Training Loop

```python
with mlflow.start_run():
    mlflow.log_params({"lr": lr, "epochs": epochs, "batch_size": bs})

    for epoch in range(epochs):
        train_loss = train_one_epoch()
        val_loss, val_acc = evaluate()

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }, step=epoch)

    mlflow.log_metric("final_accuracy", val_acc)
    mlflow.pytorch.log_model(model, "model")
```

### Model Registry

```python
# Register model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="MyModel"
)

# Transition stage
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pytorch.load_model("models:/MyModel/Production")
```

---

## Weights & Biases

### Setup

```python
import wandb

wandb.login()  # Interactive login
# Or: wandb.login(key="your-api-key")

wandb.init(
    project="my-project",
    config={
        "learning_rate": 0.001,
        "epochs": 10
    }
)
```

### Logging

```python
# Log metrics
wandb.log({"loss": 0.5, "accuracy": 0.85})

# Log with step
for step in range(100):
    wandb.log({"loss": loss, "step": step})

# Log table
table = wandb.Table(columns=["input", "output"])
table.add_data("hello", "world")
wandb.log({"examples": table})

# Finish
wandb.finish()
```

### Hyperparameter Sweeps

```yaml
# sweep.yaml
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
```

```python
sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train, count=10)
```

---

## LM Evaluation Harness

### Basic Benchmark

```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16 \
    --tasks mmlu,hellaswag \
    --batch_size 8 \
    --output_path ./results
```

### Common Tasks

| Task | Command | Measures |
|------|---------|----------|
| MMLU | `mmlu` | Knowledge (57 subjects) |
| HellaSwag | `hellaswag` | Common sense |
| ARC Easy | `arc_easy` | Science reasoning |
| ARC Challenge | `arc_challenge` | Hard science |
| TruthfulQA | `truthfulqa_mc2` | Factuality |
| WinoGrande | `winogrande` | Coreference |
| GSM8K | `gsm8k` | Math reasoning |

### With Quantization

```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,load_in_4bit=True \
    --tasks mmlu \
    --batch_size 4
```

### Save Results to MLflow

```python
import mlflow
import json

# Run benchmark
# lm_eval outputs to ./results/results.json

with open("./results/results.json") as f:
    results = json.load(f)

with mlflow.start_run(run_name="benchmark"):
    for task, scores in results["results"].items():
        for metric, value in scores.items():
            mlflow.log_metric(f"{task}_{metric}", value)
```

---

## Evidently AI (Drift Detection)

### Data Drift Report

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

column_mapping = ColumnMapping(
    target="label",
    prediction="prediction"
)

report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=reference_df,
    current_data=current_df,
    column_mapping=column_mapping
)

report.save_html("drift_report.html")
```

### Model Performance Report

```python
from evidently.metric_preset import ClassificationPreset

report = Report(metrics=[ClassificationPreset()])
report.run(
    reference_data=reference_df,
    current_data=current_df,
    column_mapping=column_mapping
)
```

---

## Reproducibility

### Set All Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Log Environment

```python
import mlflow

# Auto-log environment info
mlflow.log_param("python_version", sys.version)
mlflow.log_param("torch_version", torch.__version__)
mlflow.log_param("cuda_version", torch.version.cuda)

# Log requirements
os.system("pip freeze > requirements.txt")
mlflow.log_artifact("requirements.txt")
```

---

## Naming Conventions

| Context | Pattern | Example |
|---------|---------|---------|
| MLflow Experiment | `{Task}-{ModelFamily}` | `Sentiment-Llama` |
| MLflow Run | `{model}-{variant}` | `llama-8b-lora-r16` |
| Benchmark Output | `{model}_{benchmark}` | `llama8b_mmlu` |
| Model Registry | `PascalCase` | `SentimentClassifier` |
| W&B Project | `lowercase-hyphens` | `llm-finetuning` |

---

## Common Patterns

### Pattern: Compare Before/After Fine-tuning

```python
# Benchmark base model
base_results = run_benchmark("meta-llama/Llama-3.1-8B")

# Benchmark fine-tuned model
ft_results = run_benchmark("./my-finetuned-model")

# Log comparison
with mlflow.start_run(run_name="comparison"):
    mlflow.log_metrics({
        "base_mmlu": base_results["mmlu"],
        "ft_mmlu": ft_results["mmlu"],
        "improvement": ft_results["mmlu"] - base_results["mmlu"]
    })
```

### Pattern: CI/CD Benchmark Gate

```python
def check_benchmark_regression(new_results, threshold=0.02):
    """Fail if new model is worse than baseline by more than threshold."""
    baseline = load_baseline_results()

    for task, score in new_results.items():
        if score < baseline[task] - threshold:
            raise ValueError(f"Regression on {task}: {score} < {baseline[task]}")

    print("All benchmarks passed!")
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting to close runs | Use `with mlflow.start_run()` context |
| Not logging seeds | Always log random seeds for reproducibility |
| Wrong metric aggregation | Use step parameter for per-epoch metrics |
| Benchmark OOM | Reduce batch_size or use quantization |
| Drift false positives | Tune sensitivity thresholds |

---

## Quick Links

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [W&B Documentation](https://docs.wandb.ai/)
- [lm-eval GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Evidently AI](https://docs.evidentlyai.com/)
