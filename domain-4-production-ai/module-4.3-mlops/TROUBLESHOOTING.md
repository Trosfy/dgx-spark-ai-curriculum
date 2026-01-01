# Module 4.3: MLOps & Experiment Tracking - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, check these:**

1. Is MLflow server running? `curl http://localhost:5000`
2. Check port mapping: `docker ps` shows `-p 5000:5000`?
3. GPU available? `nvidia-smi`
4. HuggingFace token set? `echo $HF_TOKEN`

---

## MLflow Errors

### Error: `Connection refused` to MLflow

**Symptoms**:
```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=5000)
```

**Solutions**:
```bash
# Solution 1: Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 &

# Solution 2: Check if already running
ps aux | grep mlflow

# Solution 3: Use different port if 5000 is taken
mlflow server --host 0.0.0.0 --port 5001 &
mlflow.set_tracking_uri("http://localhost:5001")
```

---

### Error: `Run already active`

**Symptoms**:
```
mlflow.exceptions.MlflowException: Run with UUID xxx is already active
```

**Solutions**:
```python
# Solution 1: End the active run
mlflow.end_run()

# Solution 2: Always use context manager
with mlflow.start_run():  # Auto-closes
    mlflow.log_param("test", "value")

# Solution 3: Check and end any active run
if mlflow.active_run():
    mlflow.end_run()
```

---

### Error: `Artifact URI not valid`

**Symptoms**:
```
mlflow.exceptions.MlflowException: Artifact URI is not a valid path
```

**Solutions**:
```bash
# Set artifact location explicitly when starting server
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root /workspace/mlflow/artifacts
```

---

## W&B Errors

### Error: `wandb.errors.UsageError: api_key not configured`

**Solutions**:
```python
# Option 1: Interactive login
import wandb
wandb.login()

# Option 2: Set environment variable
export WANDB_API_KEY="your-key"

# Option 3: Pass key directly
wandb.login(key="your-key")
```

---

### Error: `wandb: Network error`

**Solutions**:
```python
# Option 1: Offline mode
wandb.init(mode="offline")

# Option 2: Check network
import requests
requests.get("https://api.wandb.ai")

# Later, sync offline runs
# wandb sync ./wandb/offline-run-xxx
```

---

## lm-eval Errors

### Error: `CUDA out of memory` during benchmarking

**Symptoms**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions**:
```bash
# Solution 1: Reduce batch size
lm_eval --model hf \
    --model_args pretrained=MODEL \
    --tasks mmlu \
    --batch_size 1  # Start with 1

# Solution 2: Use quantization
lm_eval --model hf \
    --model_args pretrained=MODEL,load_in_4bit=True \
    --tasks mmlu

# Solution 3: Clear memory first
python -c "import torch; torch.cuda.empty_cache()"
```

---

### Error: `Task 'xxx' not found`

**Solutions**:
```bash
# List available tasks
lm_eval --tasks list

# Common correct task names:
# - mmlu (not MMLU)
# - hellaswag (not HellaSwag)
# - truthfulqa_mc2 (not truthfulqa)
# - arc_easy, arc_challenge (not ARC)
```

---

### Error: `Cannot access gated model`

**Symptoms**:
```
huggingface_hub.utils._errors.GatedRepoError
```

**Solutions**:
```bash
# Option 1: Set token
export HF_TOKEN="your-token"

# Option 2: Login
huggingface-cli login

# Option 3: Pass in model_args
lm_eval --model hf \
    --model_args pretrained=MODEL,use_auth_token=true
```

---

### Error: Benchmark hangs or is very slow

**Causes**: Wrong batch size or model too large

**Solutions**:
```bash
# Use limit for testing
lm_eval --model hf \
    --model_args pretrained=MODEL \
    --tasks hellaswag \
    --limit 100  # Only 100 examples
    --batch_size 1

# Check GPU utilization during run
watch nvidia-smi
```

---

## Evidently Errors

### Error: `KeyError` in Evidently report

**Cause**: Column mapping doesn't match data

**Solutions**:
```python
from evidently import ColumnMapping

# Check your columns
print(df.columns.tolist())

# Match column mapping to actual names
column_mapping = ColumnMapping(
    target="actual_label",      # Must exist in df
    prediction="predicted_label" # Must exist in df
)
```

---

### Error: Drift report shows all features as drifted

**Cause**: Too sensitive threshold or small sample

**Solutions**:
```python
from evidently.metric_preset import DataDriftPreset

# Increase threshold (less sensitive)
report = Report(metrics=[
    DataDriftPreset(stattest_threshold=0.1)  # Default is 0.05
])

# Use more data
# Evidently needs sufficient samples for statistical tests
assert len(reference_data) >= 100
assert len(current_data) >= 100
```

---

## Model Registry Errors

### Error: `RESOURCE_ALREADY_EXISTS`

**Cause**: Model name already registered

**Solutions**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Check existing versions
versions = client.search_model_versions(f"name='{model_name}'")
print([v.version for v in versions])

# Create new version instead of new model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name  # Same name = new version
)
```

---

### Error: `Cannot transition to Production`

**Cause**: Permissions or model not found

**Solutions**:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# List all versions
for v in client.search_model_versions(f"name='MyModel'"):
    print(f"Version {v.version}: {v.current_stage}")

# Transition correct version
client.transition_model_version_stage(
    name="MyModel",
    version=1,  # Check this is correct
    stage="Production"
)
```

---

## Reproducibility Errors

### Error: Results differ between runs

**Cause**: Random seeds not set

**Solutions**:
```python
import torch
import numpy as np
import random

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For complete reproducibility (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call at start of training
set_all_seeds(42)

# Log seed
mlflow.log_param("seed", 42)
```

---

### Error: Can't reproduce older experiment

**Solutions**:
```python
# Check what was logged
run = mlflow.get_run(run_id)
print("Parameters:", run.data.params)
print("Metrics:", run.data.metrics)

# Check artifacts
artifacts = mlflow.artifacts.list_artifacts(run_id)
print("Artifacts:", [a.path for a in artifacts])

# Download requirements if logged
mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="requirements.txt",
    dst_path="./old_requirements/"
)
```

---

## Reset Procedures

### Reset MLflow

```bash
# Stop server
pkill -f "mlflow server"

# Clear data (if needed)
rm -rf ./mlruns
rm -rf ./mlflow

# Restart
mlflow server --host 0.0.0.0 --port 5000 &
```

### Reset W&B

```bash
# Clear local cache
rm -rf ./wandb

# Re-login
wandb login --relogin
```

---

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: MLflow vs W&B - which should I use?**

A: Both are excellent. Choose based on your needs:

| Factor | MLflow | W&B |
|--------|--------|-----|
| Cost | Free (self-hosted) | Free tier, paid for teams |
| Hosting | Self-hosted or managed | Cloud-hosted |
| Setup | More setup required | Quick to start |
| UI | Functional | More polished |
| Collaboration | Basic | Excellent |
| Offline | Full support | Sync later |

**Recommendation**: Learn MLflow (it's the open standard), use W&B for team projects.

---

**Q: How do I access MLflow UI from outside the container?**

A: Ensure port mapping and host binding:

```bash
# Start container with port mapping
docker run ... -p 5000:5000 ...

# Start MLflow bound to all interfaces
mlflow server --host 0.0.0.0 --port 5000

# Access from host browser
# http://localhost:5000
```

---

**Q: What benchmarks should I run for my LLM?**

A: Depends on your use case:

| Use Case | Essential Benchmarks |
|----------|---------------------|
| General | MMLU, HellaSwag |
| Code | HumanEval, MBPP |
| Reasoning | GSM8K, ARC |
| Safety | TruthfulQA, BBQ |
| Chat | MT-Bench, AlpacaEval |

**Minimum for any LLM**: MMLU + HellaSwag + TruthfulQA

---

### Concepts

**Q: What's the difference between tracking and logging?**

A:

- **Logging** (print, files): Temporary, unstructured, hard to compare
- **Tracking** (MLflow, W&B): Permanent, structured, queryable, comparable

Tracking stores metadata in a database with UI, search, comparison features.

---

**Q: Why do benchmark scores vary between runs?**

A: Several factors:

1. **Random seeds**: Not set → different results
2. **Batch size**: Affects some metrics (especially generation)
3. **Precision**: FP32 vs BF16 vs INT4
4. **Library versions**: Updates can change behavior
5. **Sampling**: Some benchmarks sample subsets

**Solution**: Always log seeds, versions, and use same settings for comparison.

---

**Q: What is model drift and why should I care?**

A: Drift is when model performance degrades over time because:

1. **Data drift**: Real-world data distribution changes
2. **Concept drift**: The relationship between inputs and outputs changes
3. **Upstream drift**: Dependencies (APIs, models) change

**Example**: A sentiment model trained on 2020 tweets fails on 2024 slang.

**Why care**: Silent failures. Users get worse results, you don't know until complaints.

---

**Q: What should I log in experiments?**

A: Log everything you'd need to reproduce:

**Always log**:
- Hyperparameters (lr, batch_size, epochs)
- Random seeds
- Model architecture/name
- Dataset version
- Final metrics

**Should log**:
- Per-step metrics (loss curves)
- Hardware info
- Library versions
- Training time

**Consider logging**:
- Model checkpoints
- Sample predictions
- Config files

---

### Additional Troubleshooting FAQs

**Q: My benchmarks are taking forever**

A: Optimize benchmark runs:

```bash
# Use limit for quick tests
lm_eval --tasks mmlu --limit 100

# Reduce batch size if OOM (but slower)
lm_eval --batch_size 1

# Use quantization for large models
lm_eval --model_args ...,load_in_4bit=True

# Run subset of MMLU
lm_eval --tasks mmlu_abstract_algebra,mmlu_astronomy
```

---

**Q: MLflow server keeps dying**

A: Common causes and fixes:

```bash
# 1. Run in proper background
nohup mlflow server --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &

# 2. Check logs
tail -f mlflow.log

# 3. Use screen/tmux for persistence
screen -S mlflow
mlflow server --host 0.0.0.0 --port 5000
# Ctrl+A, D to detach
```

---

**Q: My runs aren't showing up in MLflow UI**

A: Check:

1. **Tracking URI matches**: Same URI in code and server
2. **Experiment exists**: Check experiment name
3. **Run was closed**: Incomplete runs may not show
4. **Refresh UI**: Hard refresh the browser

```python
# Debug
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Active run: {mlflow.active_run()}")
print(f"Experiment: {mlflow.get_experiment_by_name('my-exp')}")
```

---

### Beyond the Basics

**Q: How do I set up CI/CD with benchmarks?**

A: Basic pattern:

```yaml
# .github/workflows/benchmark.yml
name: Benchmark
on: [push]
jobs:
  benchmark:
    runs-on: self-hosted  # With GPU
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          lm_eval --model hf \
            --model_args pretrained=./model \
            --tasks mmlu,hellaswag \
            --output_path results.json
      - name: Check regression
        run: python scripts/check_regression.py results.json
```

---

**Q: Can I compare models trained with different frameworks?**

A: Yes, benchmarks are framework-agnostic:

```bash
# PyTorch model
lm_eval --model hf --model_args pretrained=./pytorch-model

# Same benchmark, different model
lm_eval --model hf --model_args pretrained=./other-model

# Compare in MLflow
# Both logged to same experiment, compare in UI
```

---

**Q: How do I handle experiments that fail?**

A: Log failures too:

```python
with mlflow.start_run(run_name="experiment"):
    try:
        # Training code
        mlflow.log_param("status", "success")
    except Exception as e:
        mlflow.log_param("status", "failed")
        mlflow.log_param("error", str(e))
        mlflow.set_tag("failed", "true")
        raise
```

Then filter: `mlflow.search_runs(filter_string="tags.failed = 'true'")`

---

**Q: What's the best way to version datasets?**

A: Options:

| Method | Pros | Cons |
|--------|------|------|
| DVC | Git-like for data | Learning curve |
| HF Datasets | Easy, cached | Limited to HF format |
| Manual versioning | Simple | Error-prone |
| Hash-based | Automatic | No browsing |

**Minimum**: Log dataset hash and source in every experiment.

```python
import hashlib

def hash_dataset(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]

mlflow.log_param("dataset_hash", hash_dataset("train.jsonl"))
```

---

**Q: How do I share experiments with my team?**

A: Options:

1. **MLflow**: Share tracking server URL
2. **W&B**: Invite to team workspace
3. **Export**: `mlflow.search_runs().to_csv("results.csv")`
4. **Reports**: W&B Reports for polished sharing

---

## Still Stuck?

1. **Check MLflow docs**: [mlflow.org/docs](https://mlflow.org/docs/latest/)
2. **Check lm-eval issues**: [github.com/EleutherAI/lm-evaluation-harness/issues](https://github.com/EleutherAI/lm-evaluation-harness/issues)
3. **Check Evidently docs**: [docs.evidentlyai.com](https://docs.evidentlyai.com/)
4. **Include in error reports**: Full traceback, versions, command used
