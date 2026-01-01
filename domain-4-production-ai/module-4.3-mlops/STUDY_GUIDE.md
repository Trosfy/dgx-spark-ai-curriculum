# Module 4.3: MLOps & Experiment Tracking - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Set up experiment tracking** with MLflow and Weights & Biases
2. **Run LLM benchmarks** using lm-evaluation-harness
3. **Implement drift detection** with Evidently AI
4. **Version models and datasets** systematically
5. **Ensure reproducibility** in ML experiments

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-4.3.1-mlflow-setup.ipynb | MLflow Tracking | ~2h | Tracking server with logged runs |
| 2 | lab-4.3.2-wandb-integration.ipynb | Weights & Biases | ~2h | Training dashboards and sweeps |
| 3 | lab-4.3.3-benchmark-suite.ipynb | LLM Benchmarks | ~2h | MMLU, HellaSwag results |
| 4 | lab-4.3.4-custom-evaluation.ipynb | Custom Eval | ~2h | Task-specific + LLM-as-judge |
| 5 | lab-4.3.5-drift-detection.ipynb | Drift Detection | ~2h | Evidently AI monitoring |
| 6 | lab-4.3.6-model-registry.ipynb | Model Registry | ~2h | Version control workflow |
| 7 | lab-4.3.7-reproducibility-audit.ipynb | Reproducibility | ~2h | Verify training reproducibility |

**Total Time**: ~14 hours

---

## Core Concepts

### Experiment Tracking
**What**: Recording all parameters, metrics, and artifacts from ML experiments in a centralized system.
**Why it matters**: Without tracking, you can't compare runs, reproduce results, or understand what worked.
**First appears in**: Lab 4.3.1

### LLM Benchmarks
**What**: Standardized tests (MMLU, HellaSwag, etc.) that measure model capabilities across various tasks.
**Why it matters**: Objective comparison between models; required for claiming performance improvements.
**First appears in**: Lab 4.3.3

### Model Registry
**What**: A versioned repository for trained models with stage transitions (staging → production).
**Why it matters**: Production deployment needs controlled, versioned model artifacts.
**First appears in**: Lab 4.3.6

### Drift Detection
**What**: Monitoring changes in model performance or data distribution over time.
**Why it matters**: Models degrade silently; drift detection catches problems before users do.
**First appears in**: Lab 4.3.5

### Reproducibility
**What**: The ability to recreate exact experimental results with the same inputs.
**Why it matters**: Science requires reproducibility; debugging requires understanding exact conditions.
**First appears in**: Lab 4.3.7

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 4.2              ──►  Module 4.3           ──►   Module 4.4
AI Safety                    MLOps & Eval              Containerization
[Safety benchmarks           [Tracking,                [Deploy models
 feed into MLOps]            versioning,               with container
                             drift detection]          and monitoring]
```

**Builds on**:
- Safety benchmarks from Module 4.2 (now tracked in MLflow)
- Fine-tuning from Module 3.1 (now with experiment tracking)
- Quantization from Module 3.2 (benchmark different configurations)

**Prepares for**:
- Module 4.4 uses model registry for deployment
- Module 4.6 capstone requires documented experiments
- Production systems need monitoring (drift detection)

---

## DGX Spark Advantage

| Task | Memory Benefit |
|------|----------------|
| Benchmark large models | Run 70B models without quantization |
| Multiple parallel runs | Track experiments simultaneously |
| Large evaluation datasets | Keep datasets in memory |
| Full fine-tuning + tracking | No memory constraints on tracking |

---

## Recommended Approach

**Standard Path** (14 hours):
1. Labs 4.3.1-2: Set up tracking infrastructure
2. Labs 4.3.3-4: Run and understand benchmarks
3. Lab 4.3.5: Monitor for drift
4. Labs 4.3.6-7: Versioning and reproducibility

**Quick Path** (if familiar with MLOps, 7-8 hours):
1. Lab 4.3.1: Quick MLflow setup
2. Lab 4.3.3: Focus on lm-eval benchmarks
3. Lab 4.3.6: Model registry essentials

**Deep-Dive Path** (20+ hours):
1. All labs with extended exercises
2. Set up full CI/CD pipeline with benchmarks
3. Compare W&B vs MLflow thoroughly
4. Implement custom drift monitors

---

## Key Benchmark Reference

| Benchmark | Measures | Typical Scores |
|-----------|----------|----------------|
| MMLU | Knowledge (57 subjects) | GPT-4: ~87%, Llama-8B: ~65% |
| HellaSwag | Common sense | GPT-4: ~95%, Llama-8B: ~79% |
| ARC | Science reasoning | Varies by difficulty |
| TruthfulQA | Factuality | Higher = more truthful |
| HumanEval | Code generation | pass@1 percentage |

---

## Before You Start

- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success
- See [WORKFLOWS.md](./WORKFLOWS.md) for MLOps workflows
