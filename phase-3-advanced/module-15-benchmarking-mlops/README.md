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

### LM Evaluation Harness

```bash
pip install lm-eval

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B \
    --tasks mmlu,hellaswag,arc_easy \
    --batch_size 8 \
    --output_path ./results
```

### MLflow on DGX Spark

```bash
# Start tracking server
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
