"""
MLflow Utilities for Experiment Tracking.

This module provides utilities for setting up and using MLflow
for experiment tracking, model logging, and run management.

Example usage:
    from scripts.mlflow_utils import setup_mlflow, log_training_run

    # Setup MLflow
    setup_mlflow(experiment_name="my-experiment")

    # Log a training run
    log_training_run(
        params={"learning_rate": 1e-4},
        metrics={"accuracy": 0.95},
        model=my_model
    )
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Optional MLflow imports
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

# Optional PyTorch integration
try:
    import mlflow.pytorch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "default",
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Set up MLflow tracking for experiments.

    Args:
        tracking_uri: MLflow tracking URI. If None, uses local file storage.
        experiment_name: Name of the experiment to create or use.
        artifact_location: Where to store artifacts. If None, uses default.
        tags: Optional tags to add to the experiment.

    Returns:
        The experiment ID.

    Example:
        >>> experiment_id = setup_mlflow(
        ...     experiment_name="llm-finetuning",
        ...     tags={"project": "curriculum", "module": "15"}
        ... )
    """
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow is not installed. Run: pip install mlflow")

    # Set tracking URI
    if tracking_uri is None:
        # Default to local file storage
        mlflow_dir = Path.cwd() / "mlflow"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file://{mlflow_dir.absolute()}"

    mlflow.set_tracking_uri(tracking_uri)

    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
            tags=tags or {},
        )
        print(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")

    mlflow.set_experiment(experiment_name)

    return experiment_id


def create_experiment(
    name: str,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create a new MLflow experiment.

    Args:
        name: Name of the experiment.
        description: Optional description.
        tags: Optional tags for the experiment.

    Returns:
        The experiment ID.

    Example:
        >>> exp_id = create_experiment(
        ...     name="sentiment-analysis",
        ...     description="Fine-tuning BERT for sentiment",
        ...     tags={"model_type": "transformer"}
        ... )
    """
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is not None:
        print(f"Experiment '{name}' already exists with ID: {experiment.experiment_id}")
        return experiment.experiment_id

    # Create with tags
    experiment_tags = tags or {}
    if description:
        experiment_tags["mlflow.note.content"] = description

    experiment_id = mlflow.create_experiment(name, tags=experiment_tags)
    print(f"Created experiment '{name}' with ID: {experiment_id}")

    return experiment_id


def log_training_run(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    run_name: Optional[str] = None,
    nested: bool = False,
) -> str:
    """
    Log a complete training run to MLflow.

    Args:
        params: Training parameters (learning_rate, batch_size, etc.).
        metrics: Final metrics (accuracy, loss, etc.).
        model: Optional PyTorch model to log.
        model_name: Optional name to register the model.
        artifacts: Optional dict of {name: file_path} to log as artifacts.
        tags: Optional tags for the run.
        run_name: Optional name for the run.
        nested: Whether this is a nested run.

    Returns:
        The run ID.

    Example:
        >>> run_id = log_training_run(
        ...     params={"learning_rate": 1e-4, "epochs": 3},
        ...     metrics={"accuracy": 0.92, "loss": 0.35},
        ...     model=trained_model,
        ...     model_name="SentimentClassifier"
        ... )
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # Log artifacts
        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_path=artifact_name)

        # Log model
        if model is not None and PYTORCH_AVAILABLE:
            try:
                import torch

                if isinstance(model, torch.nn.Module):
                    mlflow.pytorch.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=model_name,
                    )
            except Exception as e:
                print(f"Could not log model: {e}")

        return run.info.run_id


def log_metrics_over_time(
    metrics: Dict[str, List[float]],
    step_name: str = "epoch",
) -> None:
    """
    Log metrics that change over time (e.g., per epoch).

    Args:
        metrics: Dict mapping metric names to lists of values.
        step_name: Name of the step (for logging purposes).

    Example:
        >>> log_metrics_over_time({
        ...     "train_loss": [0.5, 0.3, 0.2],
        ...     "val_loss": [0.6, 0.4, 0.3]
        ... })
    """
    if not metrics:
        return

    # Get the length from the first metric
    first_metric = next(iter(metrics.values()))
    num_steps = len(first_metric)

    for step in range(num_steps):
        step_metrics = {}
        for metric_name, values in metrics.items():
            if step < len(values):
                step_metrics[metric_name] = values[step]

        mlflow.log_metrics(step_metrics, step=step)


def get_best_run(
    experiment_name: str,
    metric: str = "accuracy",
    ascending: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric to optimize.
        ascending: If True, lower is better. If False, higher is better.

    Returns:
        Dictionary with run info, or None if no runs found.

    Example:
        >>> best = get_best_run("my-experiment", metric="accuracy")
        >>> print(f"Best accuracy: {best['metrics']['accuracy']}")
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None

    # Search for runs
    order = "ASC" if ascending else "DESC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if runs.empty:
        print(f"No runs found in experiment '{experiment_name}'")
        return None

    best_run = runs.iloc[0]

    return {
        "run_id": best_run["run_id"],
        "params": {
            k.replace("params.", ""): v
            for k, v in best_run.items()
            if k.startswith("params.")
        },
        "metrics": {
            k.replace("metrics.", ""): v
            for k, v in best_run.items()
            if k.startswith("metrics.")
        },
        "tags": {
            k.replace("tags.", ""): v
            for k, v in best_run.items()
            if k.startswith("tags.") and not k.startswith("tags.mlflow.")
        },
    }


def compare_runs(
    experiment_name: str,
    metric: str = "accuracy",
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Compare top N runs from an experiment.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric to rank by.
        top_n: Number of top runs to return.

    Returns:
        List of run dictionaries sorted by metric.

    Example:
        >>> runs = compare_runs("my-experiment", metric="accuracy", top_n=3)
        >>> for run in runs:
        ...     print(f"Run {run['run_id']}: {run['metrics']['accuracy']}")
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
    )

    results = []
    for _, row in runs.iterrows():
        results.append(
            {
                "run_id": row["run_id"],
                "params": {
                    k.replace("params.", ""): v
                    for k, v in row.items()
                    if k.startswith("params.")
                },
                "metrics": {
                    k.replace("metrics.", ""): v
                    for k, v in row.items()
                    if k.startswith("metrics.")
                },
            }
        )

    return results


def load_model_from_run(run_id: str, artifact_path: str = "model") -> Any:
    """
    Load a PyTorch model from an MLflow run.

    Args:
        run_id: The run ID containing the model.
        artifact_path: Path to the model artifact.

    Returns:
        The loaded PyTorch model.

    Example:
        >>> model = load_model_from_run("abc123", artifact_path="model")
    """
    if not PYTORCH_AVAILABLE:
        raise RuntimeError("mlflow.pytorch not available. Install PyTorch and mlflow together.")

    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.pytorch.load_model(model_uri)


def get_run_artifacts(run_id: str) -> List[str]:
    """
    Get list of artifacts for a run.

    Args:
        run_id: The run ID.

    Returns:
        List of artifact paths.
    """
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    return [a.path for a in artifacts]


class ExperimentTracker:
    """
    Context manager for tracking experiments.

    Example:
        >>> with ExperimentTracker("my-experiment") as tracker:
        ...     tracker.log_param("learning_rate", 1e-4)
        ...     # ... training ...
        ...     tracker.log_metric("accuracy", 0.95)
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags
        self.run = None

    def __enter__(self):
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)

        if self.tags:
            for key, value in self.tags.items():
                mlflow.set_tag(key, value)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        return False

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self.run.info.run_id if self.run else None


if __name__ == "__main__":
    # Example usage
    print("MLflow Utilities Example")
    print("=" * 40)

    # Setup
    setup_mlflow(experiment_name="demo-experiment")

    # Log a run
    run_id = log_training_run(
        params={"learning_rate": 1e-4, "batch_size": 32},
        metrics={"accuracy": 0.92, "loss": 0.35},
        run_name="demo-run",
    )

    print(f"Logged run: {run_id}")

    # Get best run
    best = get_best_run("demo-experiment", metric="accuracy")
    if best:
        print(f"Best run: {best['run_id']}")
