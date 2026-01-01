# Module 4.3: MLOps & Experiment Tracking - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 4.3.1 MLflow Setup | 10 min | 2h | 2.2h |
| 4.3.2 W&B Integration | 10 min | 2h | 2.2h |
| 4.3.3 Benchmark Suite | 15 min | 2h | 2.25h |
| 4.3.4 Custom Evaluation | 10 min | 2h | 2.2h |
| 4.3.5 Drift Detection | 10 min | 2h | 2.2h |
| 4.3.6 Model Registry | 5 min | 2h | 2h |
| 4.3.7 Reproducibility | 5 min | 2h | 2h |

---

## Required Downloads

### Models for Benchmarking

```bash
# Primary model for benchmarking (~16GB)
huggingface-cli download Qwen/Qwen3-8B-Instruct

# Smaller model for quick tests (~4GB)
huggingface-cli download microsoft/phi-2
```

**Total download size**: ~20GB
**Estimated download time**: 20-30 minutes

---

## Environment Setup

### 1. Start NGC Container with Port Mappings

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 5000:5000 \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Install MLOps Dependencies

```bash
# Experiment tracking
pip install mlflow>=2.10.0
pip install wandb>=0.16.0

# Benchmarking
pip install lm-eval>=0.4.0

# Drift detection
pip install evidently>=0.4.0

# Additional utilities
pip install pandas matplotlib seaborn scikit-learn
```

### 3. Start MLflow Server

```bash
# In background
mlflow server --host 0.0.0.0 --port 5000 &
```

### 4. Verify Setup

```python
import mlflow
import wandb

# Check MLflow
mlflow.set_tracking_uri("http://localhost:5000")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# Check lm-eval
import lm_eval
print(f"lm-eval version: {lm_eval.__version__}")
```

**Expected output**:
```
MLflow tracking URI: http://localhost:5000
lm-eval version: 0.4.x
```

---

## Pre-Lab Checklists

### Lab 4.3.1: MLflow Setup

- [ ] MLflow installed
- [ ] Port 5000 mapped
- [ ] MLflow server running
- [ ] Can access http://localhost:5000

**Quick Test**:
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    mlflow.log_param("test", "value")
print("MLflow working!")
```

---

### Lab 4.3.2: W&B Integration

- [ ] W&B account created (free at wandb.ai)
- [ ] wandb installed
- [ ] API key ready

**Quick Test**:
```python
import wandb
wandb.login()  # Will prompt for key
wandb.init(project="test", mode="disabled")
print("W&B ready!")
```

---

### Lab 4.3.3: Benchmark Suite

- [ ] lm-eval installed
- [ ] HuggingFace model accessible
- [ ] At least 20GB GPU memory free

**Quick Test**:
```bash
# Quick sanity check with tiny benchmark
lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks hellaswag \
    --limit 10 \
    --batch_size 1
```

---

### Lab 4.3.4: Custom Evaluation

- [ ] Ollama running (for LLM-as-judge)
- [ ] Sample evaluation dataset ready
- [ ] Completed Lab 4.3.3

---

### Lab 4.3.5: Drift Detection

- [ ] evidently installed
- [ ] Reference dataset prepared
- [ ] Completed Labs 4.3.1-4

**Quick Test**:
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
print("Evidently ready!")
```

---

### Lab 4.3.6: Model Registry

- [ ] MLflow server running
- [ ] Completed Lab 4.3.1
- [ ] A trained model to register

---

### Lab 4.3.7: Reproducibility Audit

- [ ] All previous labs completed
- [ ] Saved training scripts
- [ ] Documented environment

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Port 5000 not mapped | Can't access MLflow UI | Use `-p 5000:5000` |
| MLflow server not running | Tracking fails | Start with `mlflow server &` |
| Wrong W&B project | Experiments scattered | Set project consistently |
| lm-eval batch_size too high | OOM during benchmark | Start with batch_size=1 |
| Not setting HF token | Can't access gated models | Set HF_TOKEN |

---

## Expected Directory Structure

```
/workspace/
├── mlflow/
│   ├── mlruns/           # MLflow data
│   └── artifacts/        # Logged artifacts
├── experiments/
│   ├── exp-001/
│   │   ├── config.yaml
│   │   └── results/
│   └── exp-002/
├── benchmarks/
│   ├── results/
│   └── reports/
└── drift_monitoring/
    └── reports/
```

---

## Quick Start Commands

```bash
# Copy-paste to set up everything:
cd /workspace
mkdir -p mlflow experiments benchmarks drift_monitoring

# Install all dependencies
pip install mlflow>=2.10.0 wandb>=0.16.0 lm-eval>=0.4.0 evidently>=0.4.0 pandas matplotlib seaborn scikit-learn

# Start MLflow
mlflow server --host 0.0.0.0 --port 5000 &
sleep 3

# Verify
python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); print('MLflow ready!')"
python -c "import lm_eval; print('lm-eval ready!')"
```
