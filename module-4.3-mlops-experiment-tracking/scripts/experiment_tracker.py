"""
Unified Experiment Tracker for MLflow and W&B.

This module provides a unified interface for tracking experiments
across different backends (MLflow, Weights & Biases).

Example:
    from experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(backend="mlflow")
    tracker.start_experiment("my-experiment")

    with tracker.start_run("run-1"):
        tracker.log_params({"lr": 0.01, "epochs": 10})
        for epoch in range(10):
            tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=epoch)
        tracker.log_model(model, "model")
"""

from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class RunInfo:
    """Information about an experiment run."""
    run_id: str
    experiment_name: str
    run_name: str
    start_time: datetime
    status: str = "running"
    end_time: Optional[datetime] = None


class TrackerBackend(ABC):
    """Abstract base class for experiment tracking backends."""

    @abstractmethod
    def start_run(self, run_name: str) -> RunInfo:
        """Start a new run."""
        pass

    @abstractmethod
    def end_run(self) -> None:
        """End the current run."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file."""
        pass


class MLflowBackend(TrackerBackend):
    """MLflow tracking backend."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow backend.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        import mlflow
        self.mlflow = mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._active_run = None

    def set_experiment(self, name: str) -> None:
        """Set the current experiment."""
        self.mlflow.set_experiment(name)

    def start_run(self, run_name: str) -> RunInfo:
        """Start a new MLflow run."""
        run = self.mlflow.start_run(run_name=run_name)
        self._active_run = run

        return RunInfo(
            run_id=run.info.run_id,
            experiment_name=self.mlflow.get_experiment(run.info.experiment_id).name,
            run_name=run_name,
            start_time=datetime.now()
        )

    def end_run(self) -> None:
        """End the current MLflow run."""
        self.mlflow.end_run()
        self._active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        self.mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to MLflow."""
        self.mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model: Any, artifact_path: str, registered_model_name: Optional[str] = None) -> None:
        """Log a PyTorch model to MLflow."""
        self.mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )


class WandbBackend(TrackerBackend):
    """Weights & Biases tracking backend."""

    def __init__(self, project: str, entity: Optional[str] = None):
        """
        Initialize W&B backend.

        Args:
            project: W&B project name
            entity: W&B entity (username or team)
        """
        import wandb
        self.wandb = wandb
        self.project = project
        self.entity = entity
        self._run = None

    def start_run(self, run_name: str, config: Optional[Dict] = None) -> RunInfo:
        """Start a new W&B run."""
        self._run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config,
            reinit=True
        )

        return RunInfo(
            run_id=self._run.id,
            experiment_name=self.project,
            run_name=run_name,
            start_time=datetime.now()
        )

    def end_run(self) -> None:
        """End the current W&B run."""
        self.wandb.finish()
        self._run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to W&B config."""
        self.wandb.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if step is not None:
            metrics["step"] = step
        self.wandb.log(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to W&B."""
        artifact = self.wandb.Artifact(
            name=artifact_path or Path(local_path).stem,
            type="file"
        )
        artifact.add_file(local_path)
        self.wandb.log_artifact(artifact)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a model to W&B."""
        import torch

        model_path = f"{artifact_path}.pt"
        torch.save(model.state_dict(), model_path)
        self.wandb.save(model_path)
        os.remove(model_path)


class ExperimentTracker:
    """
    Unified experiment tracker supporting multiple backends.

    Example:
        # MLflow backend
        tracker = ExperimentTracker(backend="mlflow")
        tracker.start_experiment("my-experiment")

        with tracker.start_run("run-1"):
            tracker.log_params({"lr": 0.01})
            tracker.log_metrics({"loss": 0.5})

        # W&B backend
        tracker = ExperimentTracker(backend="wandb", project="my-project")

        with tracker.start_run("run-1", config={"lr": 0.01}):
            tracker.log_metrics({"loss": 0.5})
    """

    def __init__(
        self,
        backend: str = "mlflow",
        tracking_uri: Optional[str] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None
    ):
        """
        Initialize the experiment tracker.

        Args:
            backend: Backend to use ("mlflow" or "wandb")
            tracking_uri: MLflow tracking URI (for mlflow backend)
            project: W&B project name (for wandb backend)
            entity: W&B entity (for wandb backend)
        """
        self.backend_name = backend

        if backend == "mlflow":
            self._backend = MLflowBackend(tracking_uri)
        elif backend == "wandb":
            if not project:
                raise ValueError("Project name required for W&B backend")
            self._backend = WandbBackend(project, entity)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._current_run: Optional[RunInfo] = None

    def start_experiment(self, name: str) -> None:
        """Set the current experiment name (MLflow only)."""
        if hasattr(self._backend, 'set_experiment'):
            self._backend.set_experiment(name)

    def start_run(self, run_name: str, config: Optional[Dict] = None) -> "ExperimentTracker":
        """
        Start a new run.

        Can be used as a context manager:
            with tracker.start_run("my-run"):
                tracker.log_metrics(...)
        """
        if isinstance(self._backend, WandbBackend):
            self._current_run = self._backend.start_run(run_name, config)
        else:
            self._current_run = self._backend.start_run(run_name)
            if config:
                self.log_params(config)
        return self

    def end_run(self) -> None:
        """End the current run."""
        self._backend.end_run()
        if self._current_run:
            self._current_run.status = "completed"
            self._current_run.end_time = datetime.now()
        self._current_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        self._backend.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        self._backend.log_metrics(metrics, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file."""
        self._backend.log_artifact(local_path, artifact_path)

    def log_model(self, model: Any, artifact_path: str = "model", registered_name: Optional[str] = None) -> None:
        """Log a model."""
        if isinstance(self._backend, MLflowBackend):
            self._backend.log_model(model, artifact_path, registered_name)
        else:
            self._backend.log_model(model, artifact_path)

    @property
    def current_run(self) -> Optional[RunInfo]:
        """Get the current run info."""
        return self._current_run


def main():
    """Example usage of ExperimentTracker."""
    import torch
    import torch.nn as nn

    # Create a simple model
    model = nn.Linear(10, 1)

    # Example with MLflow
    tracker = ExperimentTracker(backend="mlflow")
    tracker.start_experiment("Demo-Experiment")

    with tracker.start_run("demo-run"):
        tracker.log_params({
            "model": "Linear",
            "input_dim": 10,
            "lr": 0.01
        })

        for step in range(5):
            loss = 1.0 / (step + 1)
            tracker.log_metrics({"loss": loss}, step=step)

        # Note: model logging requires mlflow.pytorch
        # tracker.log_model(model, "model")

    print("Tracking complete!")


if __name__ == "__main__":
    main()
