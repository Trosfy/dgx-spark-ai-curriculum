#!/usr/bin/env python3
"""
Model Registry

Track, version, and manage trained models.
This is a starting point - extend this for your capstone!
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import shutil


@dataclass
class ModelVersion:
    """A versioned model entry."""
    name: str
    version: str
    path: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    base_model: str = ""
    training_method: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.version}"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "created_at": self.created_at,
            "base_model": self.base_model,
            "training_method": self.training_method,
            "metrics": self.metrics,
            "tags": self.tags,
            "metadata": self.metadata
        }


class ModelRegistry:
    """
    Registry for managing trained models.

    Features:
    - Version tracking
    - Metrics storage
    - Model comparison
    - Promotion workflow
    """

    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_path / "index.json"
        self.models: Dict[str, List[ModelVersion]] = {}

        self._load_index()

    def _load_index(self):
        """Load registry index from disk."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                data = json.load(f)
                for name, versions in data.items():
                    self.models[name] = [
                        ModelVersion(**v) for v in versions
                    ]
            print(f"Loaded {len(self.models)} models from registry")
        else:
            print("Initialized empty model registry")

    def _save_index(self):
        """Save registry index to disk."""
        data = {
            name: [v.to_dict() for v in versions]
            for name, versions in self.models.items()
        }
        with open(self.index_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        path: str,
        base_model: str = "",
        training_method: str = "SFT",
        metrics: Dict[str, float] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model name
            path: Path to model files
            base_model: Base model used for fine-tuning
            training_method: SFT, DPO, etc.
            metrics: Training/eval metrics
            tags: Searchable tags
            metadata: Additional metadata

        Returns:
            Created ModelVersion
        """
        # Generate version
        if name not in self.models:
            self.models[name] = []
            version = "v1"
        else:
            latest = len(self.models[name])
            version = f"v{latest + 1}"

        # Create version entry
        model_version = ModelVersion(
            name=name,
            version=version,
            path=path,
            base_model=base_model,
            training_method=training_method,
            metrics=metrics or {},
            tags=tags or [],
            metadata=metadata or {}
        )

        self.models[name].append(model_version)
        self._save_index()

        print(f"Registered: {model_version.full_name}")
        return model_version

    def get(self, name: str, version: str = "latest") -> Optional[ModelVersion]:
        """Get a specific model version."""
        if name not in self.models:
            return None

        versions = self.models[name]
        if version == "latest":
            return versions[-1] if versions else None

        for v in versions:
            if v.version == version:
                return v
        return None

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return self.models.get(name, [])

    def compare(self, model1: str, model2: str) -> Dict:
        """Compare metrics between two models."""
        v1 = self.get(model1.split(":")[0], model1.split(":")[-1] if ":" in model1 else "latest")
        v2 = self.get(model2.split(":")[0], model2.split(":")[-1] if ":" in model2 else "latest")

        if not v1 or not v2:
            return {"error": "Model not found"}

        comparison = {
            "model1": v1.full_name,
            "model2": v2.full_name,
            "metrics_comparison": {}
        }

        # Compare common metrics
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            m1 = v1.metrics.get(metric)
            m2 = v2.metrics.get(metric)
            comparison["metrics_comparison"][metric] = {
                v1.full_name: m1,
                v2.full_name: m2,
                "diff": (m2 - m1) if m1 and m2 else None
            }

        return comparison

    def promote(self, name: str, version: str, stage: str = "production"):
        """Promote a model version to a stage."""
        model = self.get(name, version)
        if model:
            model.tags.append(f"stage:{stage}")
            self._save_index()
            print(f"Promoted {model.full_name} to {stage}")

    def search(self, tag: str = None, min_metric: Dict[str, float] = None) -> List[ModelVersion]:
        """Search models by criteria."""
        results = []

        for versions in self.models.values():
            for v in versions:
                # Tag filter
                if tag and tag not in v.tags:
                    continue

                # Metric filter
                if min_metric:
                    meets_criteria = True
                    for metric, threshold in min_metric.items():
                        if v.metrics.get(metric, 0) < threshold:
                            meets_criteria = False
                            break
                    if not meets_criteria:
                        continue

                results.append(v)

        return results

    def export_report(self) -> str:
        """Generate a registry report."""
        lines = ["# Model Registry Report", ""]

        for name, versions in self.models.items():
            lines.append(f"## {name}")
            lines.append(f"Total versions: {len(versions)}")
            lines.append("")

            for v in versions:
                lines.append(f"### {v.full_name}")
                lines.append(f"- Created: {v.created_at}")
                lines.append(f"- Base: {v.base_model}")
                lines.append(f"- Method: {v.training_method}")
                if v.metrics:
                    lines.append("- Metrics:")
                    for m, val in v.metrics.items():
                        lines.append(f"  - {m}: {val:.4f}")
                if v.tags:
                    lines.append(f"- Tags: {', '.join(v.tags)}")
                lines.append("")

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("Model Registry Demo")
    print("=" * 50)

    # Create registry
    registry = ModelRegistry("./demo_registry")

    # Register some models
    print("\n1. REGISTERING MODELS")
    print("-" * 40)

    registry.register(
        name="domain-assistant",
        path="./outputs/checkpoint-500",
        base_model="Llama-3.3-8B",
        training_method="SFT",
        metrics={"loss": 1.2, "accuracy": 0.85},
        tags=["sft", "domain-specific"]
    )

    registry.register(
        name="domain-assistant",
        path="./outputs/checkpoint-1000",
        base_model="Llama-3.3-8B",
        training_method="DPO",
        metrics={"loss": 0.9, "accuracy": 0.92},
        tags=["dpo", "domain-specific", "improved"]
    )

    registry.register(
        name="code-helper",
        path="./outputs/code-v1",
        base_model="Llama-3.3-8B",
        training_method="SFT",
        metrics={"loss": 1.1, "pass_rate": 0.78},
        tags=["coding", "sft"]
    )

    # List models
    print("\n2. LISTING MODELS")
    print("-" * 40)
    print(f"Models: {registry.list_models()}")

    for name in registry.list_models():
        versions = registry.list_versions(name)
        print(f"  {name}: {[v.version for v in versions]}")

    # Compare models
    print("\n3. COMPARING VERSIONS")
    print("-" * 40)
    comparison = registry.compare("domain-assistant:v1", "domain-assistant:v2")
    print(json.dumps(comparison, indent=2))

    # Search
    print("\n4. SEARCHING")
    print("-" * 40)
    results = registry.search(tag="dpo")
    print(f"Models with 'dpo' tag: {[r.full_name for r in results]}")

    # Export report
    print("\n5. REPORT")
    print("-" * 40)
    print(registry.export_report()[:500] + "...")
