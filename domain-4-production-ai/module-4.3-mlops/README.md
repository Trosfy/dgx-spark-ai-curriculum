# Module 4.3: MLOps & Experiment Tracking

**Domain:** 4 - Production AI
**Duration:** Weeks 30-31 (12-15 hours)
**Prerequisites:** Module 4.2 (AI Safety)
**Priority:** P0/P1 Expanded (MLflow, W&B, Drift Detection)

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
| 4.3.1 | Set up and use MLflow/W&B for experiment tracking | Apply |
| 4.3.2 | Run standard LLM benchmarks (MMLU, HellaSwag) | Apply |
| 4.3.3 | Implement drift detection with Evidently AI | Apply |
| 4.3.4 | Version datasets and models systematically | Apply |

---

## Topics

### 4.3.1 Experiment Tracking

- **MLflow Setup**
  - Tracking server configuration
  - Logging parameters, metrics, artifacts
  - Model registry
  - UI dashboards

- **Weights & Biases Integration**
  - W&B setup and configuration
  - Training dashboards
  - Sweep configurations
  - Team collaboration

### 4.3.2 LLM Benchmarks

- **Standard Benchmarks**
  - MMLU, HellaSwag, ARC, WinoGrande
  - HumanEval (code generation)
  - MT-Bench (chat quality)
  - LM Evaluation Harness

- **Custom Evaluation**
  - Task-specific metrics
  - LLM-as-judge evaluation
  - Human evaluation protocols

### 4.3.3 Model Monitoring

- **Drift Detection**
  - Concept drift
  - Data drift
  - Performance degradation

- **Evidently AI Integration**
  - Data quality reports
  - Model monitoring dashboards
  - Alert configuration

### 4.3.4 Versioning and Reproducibility

- **Model Versioning**
  - Hugging Face Hub
  - DVC for large files
  - Git LFS alternatives

- **Dataset Versioning**
  - Dataset snapshots
  - Lineage tracking

- **Environment Reproducibility**
  - Requirements pinning
  - Random seed management
  - Container versioning

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 4.3.1 | MLflow Setup | 2h | Tracking server with fine-tuning logs |
| 4.3.2 | W&B Integration | 2h | Training dashboards and sweeps |
| 4.3.3 | Benchmark Suite | 2h | lm-eval on multiple models |
| 4.3.4 | Custom Evaluation | 2h | Task-specific + LLM-as-judge |
| 4.3.5 | Drift Detection | 2h | Evidently AI monitoring setup |
| 4.3.6 | Model Registry | 2h | Version control workflow |
| 4.3.7 | Reproducibility Audit | 2h | Verify training reproducibility |

---

## Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Set up MLflow and log first run in 5 minutes |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives and module roadmap |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Commands, configs, and benchmark patterns |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and checklist |
| [WORKFLOWS.md](./WORKFLOWS.md) | Step-by-step MLOps workflows |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common errors, fixes, and FAQs |

---

## Guidance

### DGX Spark Environment Setup

All commands should be run inside an NGC container. Start your container with:

```bash
# Start NGC container with GPU and shared memory support
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 5000:5000 \
    nvcr.io/nvidia/pytorch:25.11-py3

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

## Next Steps

After completing this module:
1. Document your experiment tracking setup
2. Keep your benchmark results for the capstone
3. Proceed to [Module 4.4: Containerization & Deployment](../module-4.4-containerization-deployment/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 4.2: AI Safety](../module-4.2-ai-safety/) | **Module 4.3: MLOps** | [Module 4.4: Containerization](../module-4.4-containerization-deployment/) |

---

## Resources

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MLflow](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/)
- [Evidently AI](https://www.evidentlyai.com/)
- [HELM](https://crfm.stanford.edu/helm/)
