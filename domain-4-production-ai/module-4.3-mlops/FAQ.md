# Module 4.3: MLOps & Experiment Tracking - Frequently Asked Questions

## Setup & Environment

### Q: MLflow vs W&B - which should I use?

**A**: Both are excellent. Choose based on your needs:

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

### Q: How do I access MLflow UI from outside the container?

**A**: Ensure port mapping and host binding:

```bash
# Start container with port mapping
docker run ... -p 5000:5000 ...

# Start MLflow bound to all interfaces
mlflow server --host 0.0.0.0 --port 5000

# Access from host browser
# http://localhost:5000
```

---

### Q: What benchmarks should I run for my LLM?

**A**: Depends on your use case:

| Use Case | Essential Benchmarks |
|----------|---------------------|
| General | MMLU, HellaSwag |
| Code | HumanEval, MBPP |
| Reasoning | GSM8K, ARC |
| Safety | TruthfulQA, BBQ |
| Chat | MT-Bench, AlpacaEval |

**Minimum for any LLM**: MMLU + HellaSwag + TruthfulQA

---

## Concepts

### Q: What's the difference between tracking and logging?

**A**:

- **Logging** (print, files): Temporary, unstructured, hard to compare
- **Tracking** (MLflow, W&B): Permanent, structured, queryable, comparable

Tracking stores metadata in a database with UI, search, comparison features.

---

### Q: Why do benchmark scores vary between runs?

**A**: Several factors:

1. **Random seeds**: Not set → different results
2. **Batch size**: Affects some metrics (especially generation)
3. **Precision**: FP32 vs BF16 vs INT4
4. **Library versions**: Updates can change behavior
5. **Sampling**: Some benchmarks sample subsets

**Solution**: Always log seeds, versions, and use same settings for comparison.

---

### Q: What is model drift and why should I care?

**A**: Drift is when model performance degrades over time because:

1. **Data drift**: Real-world data distribution changes
2. **Concept drift**: The relationship between inputs and outputs changes
3. **Upstream drift**: Dependencies (APIs, models) change

**Example**: A sentiment model trained on 2020 tweets fails on 2024 slang.

**Why care**: Silent failures. Users get worse results, you don't know until complaints.

---

### Q: What should I log in experiments?

**A**: Log everything you'd need to reproduce:

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

## Troubleshooting

### Q: My benchmarks are taking forever

**A**: Optimize benchmark runs:

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

### Q: MLflow server keeps dying

**A**: Common causes and fixes:

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

### Q: My runs aren't showing up in MLflow UI

**A**: Check:

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

## Beyond the Basics

### Q: How do I set up CI/CD with benchmarks?

**A**: Basic pattern:

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

### Q: Can I compare models trained with different frameworks?

**A**: Yes, benchmarks are framework-agnostic:

```bash
# PyTorch model
lm_eval --model hf --model_args pretrained=./pytorch-model

# Same benchmark, different model
lm_eval --model hf --model_args pretrained=./other-model

# Compare in MLflow
# Both logged to same experiment, compare in UI
```

---

### Q: How do I handle experiments that fail?

**A**: Log failures too:

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

### Q: What's the best way to version datasets?

**A**: Options:

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

### Q: How do I share experiments with my team?

**A**: Options:

1. **MLflow**: Share tracking server URL
2. **W&B**: Invite to team workspace
3. **Export**: `mlflow.search_runs().to_csv("results.csv")`
4. **Reports**: W&B Reports for polished sharing

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [WORKFLOWS.md](./WORKFLOWS.md) for process guidance
- See module [Resources](./README.md#resources) for official documentation
