# Module 15: Benchmarking, Evaluation & MLOps

**Phase:** 3 - Advanced  
**Duration:** Weeks 25-26 (10-12 hours)  
**Prerequisites:** Modules 10-14

---

## Overview

How do you know if your model is actually good? This module covers systematic evaluation with standard benchmarks, custom metrics, experiment tracking, and MLOps practices for reproducible AI development.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Evaluate LLMs using standard benchmarks
- ✅ Implement comprehensive evaluation frameworks
- ✅ Set up experiment tracking and model versioning
- ✅ Create reproducible ML pipelines

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 15.1 | Run standard LLM benchmarks (MMLU, HellaSwag, etc.) | Apply |
| 15.2 | Design custom evaluation suites | Create |
| 15.3 | Track experiments with MLflow | Apply |
| 15.4 | Version datasets and models systematically | Apply |

---

## Topics

### 15.1 LLM Benchmarks
- MMLU, HellaSwag, ARC, WinoGrande
- HumanEval (code)
- MT-Bench (chat)
- LM Evaluation Harness

### 15.2 Custom Evaluation
- Task-specific metrics
- LLM-as-judge
- Human evaluation
- A/B testing

### 15.3 Experiment Tracking
- MLflow setup
- Weights & Biases
- Hyperparameter logging
- Artifact management

### 15.4 MLOps Practices
- Model versioning (DVC, HF Hub)
- Dataset versioning
- Reproducibility
- CI/CD for ML

---

## Tasks

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 15.1 | Benchmark Suite | 3h | lm-eval on multiple models |
| 15.2 | Custom Eval Framework | 2h | Task-specific + LLM-as-judge |
| 15.3 | MLflow Setup | 2h | Tracking server with dashboards |
| 15.4 | Model Registry | 2h | Version control workflow |
| 15.5 | Reproducibility Audit | 2h | Verify training reproducibility |

---

## Guidance

### DGX Spark Environment Setup

All commands should be run inside an NGC container. Start your container with:

```bash
# Start NGC container with GPU and shared memory support
docker run --gpus all --ipc=host -it \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -p 5000:5000 \
    nvcr.io/nvidia/pytorch:25.01-py3

# Inside the container, install additional tools:
pip install lm-eval mlflow
```

**Important flags:**
- `--gpus all` - Enable GPU access
- `--ipc=host` - Required for PyTorch DataLoader with multiple workers
- `-p 5000:5000` - Expose MLflow UI port

### LM Evaluation Harness

Inside your NGC container:

```bash
# Install lm-eval (pure Python, works on ARM64)
pip install lm-eval

# Run benchmarks
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16 \
    --tasks mmlu,hellaswag,arc_easy \
    --batch_size 8 \
    --output_path ./results
```

**Note:** Use `dtype=bfloat16` for native Blackwell GPU support.

### MLflow on DGX Spark

```bash
# Install MLflow (pure Python, works on ARM64)
pip install mlflow

# Start tracking server (accessible from host via -p 5000:5000)
mlflow server --host 0.0.0.0 --port 5000

# In training script
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("llm-finetuning")

with mlflow.start_run():
    mlflow.log_params(training_args)
    mlflow.log_metrics({"loss": loss, "accuracy": acc})
    mlflow.pytorch.log_model(model, "model")
```

### LLM-as-Judge

```python
judge_prompt = """
Rate the following response on a scale of 1-10:
Question: {question}
Response: {response}
Criteria: Accuracy, Helpfulness, Clarity
Output JSON: {"score": X, "reasoning": "..."}
"""
```

### Experiment Naming Convention

For consistency across the module, use these naming patterns:

| Context | Pattern | Example |
|---------|---------|---------|
| MLflow Experiment | `{Task}-{ModelFamily}` | `Sentiment-Analysis-Models` |
| MLflow Run | `{model}-{variant}` | `phi2-lora-r16` |
| Benchmark Output | `{model}_{benchmark}` | `phi2_quick_test` |
| Model Registry | `PascalCase` | `SentimentClassifier` |

**Example workflow:**
```python
# Experiment name groups related runs
mlflow.set_experiment("LLM-Finetuning-Demo")

# Run name identifies the specific configuration
with mlflow.start_run(run_name="llama-8b-lora-r32"):
    ...

# Registered model name is the artifact
mlflow.pytorch.log_model(model, registered_model_name="InstructionFollower")
```

---

## Milestone Checklist

- [ ] Benchmark results for multiple models
- [ ] Custom evaluation framework
- [ ] MLflow tracking server running
- [ ] Model registry workflow established
- [ ] Reproducibility verified

---

## Resources

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MLflow](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/)
- [HELM](https://crfm.stanford.edu/helm/)
