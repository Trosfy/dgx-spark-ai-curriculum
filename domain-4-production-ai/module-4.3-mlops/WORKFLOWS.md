# Module 4.3: MLOps & Experiment Tracking - Workflow Cheatsheets

## Workflow 1: Experiment Tracking Setup

### When to Use
Use this workflow when starting a new ML project or adding tracking to an existing one.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Start Tracking Server                               │
├─────────────────────────────────────────────────────────────┤
│ □ Start MLflow server                                       │
│ □ Verify UI accessible                                      │
│                                                             │
│ Code:                                                       │
│ ```bash                                                     │
│ mlflow server --host 0.0.0.0 --port 5000 &                  │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: http://localhost:5000 shows MLflow UI         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Create Experiment                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Set tracking URI                                          │
│ □ Create or set experiment                                  │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ import mlflow                                               │
│ mlflow.set_tracking_uri("http://localhost:5000")            │
│ mlflow.set_experiment("my-project")                         │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Experiment visible in UI                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Add Tracking to Training                            │
├─────────────────────────────────────────────────────────────┤
│ □ Wrap training in run context                              │
│ □ Log parameters at start                                   │
│ □ Log metrics during training                               │
│ □ Log model at end                                          │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ with mlflow.start_run(run_name="experiment-1"):             │
│     mlflow.log_params(config)                               │
│     for epoch in range(epochs):                             │
│         loss = train()                                      │
│         mlflow.log_metric("loss", loss, step=epoch)         │
│     mlflow.pytorch.log_model(model, "model")                │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Run appears with params, metrics, model       │
└─────────────────────────────────────────────────────────────┘

### ⚠️ Common Pitfalls
| At Step | Watch Out For |
|---------|---------------|
| 1 | Port already in use - kill previous server |
| 3 | Forgetting to close runs - always use `with` context |
| 3 | Logging too frequently - slows training |

### ✅ Success Criteria
- [ ] MLflow UI accessible
- [ ] Experiments organized by project
- [ ] Runs show parameters, metrics, model

---

## Workflow 2: Running LLM Benchmarks

### When to Use
Use this workflow when evaluating model quality before/after training or comparing models.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Prepare Model                                       │
├─────────────────────────────────────────────────────────────┤
│ □ Identify model to benchmark                               │
│ □ Ensure model is accessible (HF or local)                  │
│ □ Check memory requirements                                 │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ # Check model size                                          │
│ from transformers import AutoConfig                         │
│ config = AutoConfig.from_pretrained(model_name)             │
│ print(f"Parameters: {config.num_parameters:,}")             │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Model loads successfully                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Select Benchmarks                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Choose relevant tasks                                     │
│ □ Consider time constraints                                 │
│                                                             │
│ Quick (~10 min): hellaswag, arc_easy                        │
│ Standard (~1 hr): mmlu, hellaswag, truthfulqa_mc2           │
│ Full (~4 hr): All major benchmarks                          │
│                                                             │
│ ✓ Checkpoint: Task list defined                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Run Benchmarks                                      │
├─────────────────────────────────────────────────────────────┤
│ □ Start with small batch size                               │
│ □ Use bfloat16 for efficiency                               │
│ □ Save results                                              │
│                                                             │
│ Code:                                                       │
│ ```bash                                                     │
│ lm_eval --model hf \                                        │
│     --model_args pretrained=MODEL,dtype=bfloat16 \          │
│     --tasks mmlu,hellaswag \                                │
│     --batch_size 4 \                                        │
│     --output_path ./results                                 │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Results JSON created                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Log to Tracking                                     │
├─────────────────────────────────────────────────────────────┤
│ □ Parse results JSON                                        │
│ □ Log to MLflow/W&B                                         │
│ □ Add context (model, date, notes)                          │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ with mlflow.start_run(run_name="benchmark-llama8b"):        │
│     mlflow.log_param("model", model_name)                   │
│     for task, scores in results["results"].items():         │
│         for metric, val in scores.items():                  │
│             mlflow.log_metric(f"{task}_{metric}", val)      │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Benchmarks tracked, comparable                │
└─────────────────────────────────────────────────────────────┘

### ⚠️ Common Pitfalls
| At Step | Watch Out For |
|---------|---------------|
| 1 | Gated models need HF token |
| 3 | OOM - reduce batch_size |
| 3 | Wrong task names - check with `--tasks list` |

---

## Workflow 3: Model Registry & Deployment

### When to Use
Use this workflow when promoting models through stages (dev → staging → production).

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Register Model                                      │
├─────────────────────────────────────────────────────────────┤
│ □ Complete training run                                     │
│ □ Register from run artifacts                               │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ # After training                                            │
│ run_id = mlflow.active_run().info.run_id                    │
│ mlflow.register_model(                                      │
│     model_uri=f"runs:/{run_id}/model",                      │
│     name="MyModel"                                          │
│ )                                                           │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Model visible in Registry                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Validate in Staging                                 │
├─────────────────────────────────────────────────────────────┤
│ □ Transition to Staging                                     │
│ □ Run validation tests                                      │
│ □ Check benchmarks meet threshold                           │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ client = MlflowClient()                                     │
│ client.transition_model_version_stage(                      │
│     name="MyModel", version=1, stage="Staging"              │
│ )                                                           │
│                                                             │
│ # Load and validate                                         │
│ model = mlflow.pytorch.load_model("models:/MyModel/Staging")│
│ assert validate(model), "Validation failed"                 │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Model passes validation                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Promote to Production                               │
├─────────────────────────────────────────────────────────────┤
│ □ Verify staging tests passed                               │
│ □ Transition to Production                                  │
│ □ Archive previous production version                       │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ client.transition_model_version_stage(                      │
│     name="MyModel", version=1, stage="Production"           │
│ )                                                           │
│                                                             │
│ # Archive old version                                       │
│ client.transition_model_version_stage(                      │
│     name="MyModel", version=0, stage="Archived"             │
│ )                                                           │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: New production model live                     │
└─────────────────────────────────────────────────────────────┘

---

## Workflow 4: Drift Detection Setup

### When to Use
Use this workflow when deploying models to production that need ongoing monitoring.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Capture Reference Data                              │
├─────────────────────────────────────────────────────────────┤
│ □ Save validation data as reference                         │
│ □ Include features and predictions                          │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ reference_data = pd.DataFrame({                             │
│     "features": features,                                   │
│     "prediction": predictions,                              │
│     "target": targets                                       │
│ })                                                          │
│ reference_data.to_parquet("reference_data.parquet")         │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Reference data saved                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Set Up Monitoring                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Configure Evidently report                                │
│ □ Define thresholds                                         │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ from evidently.report import Report                         │
│ from evidently.metric_preset import DataDriftPreset         │
│                                                             │
│ report = Report(metrics=[                                   │
│     DataDriftPreset(stattest_threshold=0.05)                │
│ ])                                                          │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Report configured                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Schedule Regular Checks                             │
├─────────────────────────────────────────────────────────────┤
│ □ Create monitoring script                                  │
│ □ Schedule (cron/Airflow)                                   │
│ □ Set up alerts                                             │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ def check_drift():                                          │
│     current = load_recent_predictions()                     │
│     report.run(reference_data, current)                     │
│     if report.as_dict()["drift_detected"]:                  │
│         send_alert("Data drift detected!")                  │
│     report.save_html(f"drift_{date.today()}.html")          │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Automated monitoring running                  │
└─────────────────────────────────────────────────────────────┘

---

## Decision Flowchart: Which Tracking Tool?

```
                    Start
                      │
                      ▼
              ┌───────────────┐
              │ Team size?    │
              └───────┬───────┘
                      │
            ┌─────────┴─────────┐
            │ Solo/Small        │ Large Team
            ▼                   ▼
    ┌───────────────┐   ┌───────────────┐
    │ Self-hosted   │   │ Collaboration │
    │ important?    │   │ features?     │
    └───────┬───────┘   └───────┬───────┘
            │                   │
      ┌─────┴─────┐       ┌─────┴─────┐
      │ Yes       │ No    │ Yes       │ No
      ▼           ▼       ▼           ▼
   [MLflow]   [W&B]    [W&B]      [MLflow]

Both work! W&B has better UI, MLflow is fully open-source.
```
