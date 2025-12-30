"""
Versioning Utilities for Models and Datasets.

This module provides utilities for versioning models and datasets,
including hash-based tracking and MLflow Model Registry integration.

Example usage:
    from scripts.versioning_utils import compute_data_hash, create_data_manifest

    # Compute hash of a data file
    data_hash = compute_data_hash("data/train.json")

    # Create a manifest for a data directory
    manifest = create_data_manifest("data/")
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Optional PyTorch integration for model loading
try:
    import mlflow.pytorch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


def compute_data_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute hash of a data file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm to use ("sha256", "md5", etc.).

    Returns:
        Hex digest of the file hash (first 12 characters for brevity).

    Example:
        >>> hash_value = compute_data_hash("data/train.json")
        >>> print(f"Data hash: {hash_value}")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)

    return hasher.hexdigest()[:12]


def compute_directory_hash(
    directory: Union[str, Path],
    patterns: Optional[List[str]] = None,
    algorithm: str = "sha256",
) -> str:
    """
    Compute combined hash of all files in a directory.

    Args:
        directory: Path to the directory.
        patterns: Optional list of glob patterns to include (e.g., ["*.json", "*.csv"]).
        algorithm: Hash algorithm to use.

    Returns:
        Combined hash of all matching files.

    Example:
        >>> dir_hash = compute_directory_hash("data/", patterns=["*.json"])
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    hasher = hashlib.new(algorithm)

    # Get all files
    if patterns:
        files = []
        for pattern in patterns:
            files.extend(directory.rglob(pattern))
    else:
        files = [f for f in directory.rglob("*") if f.is_file()]

    # Sort for consistent ordering
    files = sorted(files)

    for file_path in files:
        # Include relative path in hash for structure awareness
        rel_path = file_path.relative_to(directory)
        hasher.update(str(rel_path).encode())

        # Include file content
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

    return hasher.hexdigest()[:12]


@dataclass
class DataManifest:
    """Manifest describing a dataset version."""

    version: str
    created_at: str
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "files": self.files,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataManifest":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            files=data.get("files", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DataManifest":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Union[str, Path]) -> None:
        """Save manifest to file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataManifest":
        """Load manifest from file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())


def create_data_manifest(
    data_dir: Union[str, Path],
    version: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataManifest:
    """
    Create a manifest of all data files with hashes.

    Args:
        data_dir: Path to the data directory.
        version: Optional version string. If None, uses hash-based version.
        patterns: Optional list of glob patterns to include.
        metadata: Optional additional metadata.

    Returns:
        DataManifest object with file information.

    Example:
        >>> manifest = create_data_manifest(
        ...     "data/",
        ...     version="1.0.0",
        ...     patterns=["*.json", "*.csv"]
        ... )
        >>> manifest.save("data_manifest.json")
    """
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    files = {}

    # Get files
    if patterns:
        file_list = []
        for pattern in patterns:
            file_list.extend(data_dir.rglob(pattern))
    else:
        file_list = [f for f in data_dir.rglob("*") if f.is_file()]

    # Skip hidden files and common non-data files
    skip_patterns = {".gitkeep", ".gitignore", ".DS_Store", "__pycache__"}

    for file_path in sorted(file_list):
        if file_path.name in skip_patterns:
            continue

        rel_path = str(file_path.relative_to(data_dir))
        files[rel_path] = {
            "hash": compute_data_hash(file_path),
            "size": file_path.stat().st_size,
            "modified": datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat(),
        }

    # Generate version from overall hash if not provided
    if version is None:
        if files:
            combined_hash = hashlib.sha256(
                "".join(f["hash"] for f in files.values()).encode()
            ).hexdigest()[:8]
            version = f"auto-{combined_hash}"
        else:
            version = "empty"

    return DataManifest(
        version=version,
        created_at=datetime.now().isoformat(),
        files=files,
        metadata=metadata or {},
    )


def register_model_version(
    model_name: str,
    run_id: str,
    artifact_path: str = "model",
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Register a model version in MLflow Model Registry.

    Args:
        model_name: Name for the registered model.
        run_id: Run ID containing the model artifact.
        artifact_path: Path to model artifact within the run.
        description: Optional description for this version.
        tags: Optional tags for this version.

    Returns:
        Version number as string, or None if registration failed.

    Example:
        >>> version = register_model_version(
        ...     model_name="SentimentClassifier",
        ...     run_id="abc123",
        ...     description="Improved model with attention"
        ... )
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Cannot register model.")
        return None

    client = MlflowClient()

    # Check if model exists, create if not
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)
        print(f"Created registered model: {model_name}")

    # Register the version
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=description,
    )

    version = result.version
    print(f"Registered {model_name} version {version}")

    # Add tags
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(model_name, version, key, value)

    return version


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str,
    archive_existing: bool = False,
) -> bool:
    """
    Transition a model version to a new stage.

    Args:
        model_name: Name of the registered model.
        version: Version number.
        stage: Target stage ("Staging", "Production", "Archived").
        archive_existing: If True, archive existing models in the target stage.

    Returns:
        True if successful, False otherwise.

    Example:
        >>> transition_model_stage(
        ...     model_name="SentimentClassifier",
        ...     version="2",
        ...     stage="Production",
        ...     archive_existing=True
        ... )
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Cannot transition model.")
        return False

    valid_stages = {"None", "Staging", "Production", "Archived"}
    if stage not in valid_stages:
        print(f"Invalid stage '{stage}'. Must be one of: {valid_stages}")
        return False

    client = MlflowClient()

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        print(f"Transitioned {model_name} v{version} to {stage}")
        return True
    except Exception as e:
        print(f"Failed to transition model: {e}")
        return False


def get_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """
    Get all versions of a registered model.

    Args:
        model_name: Name of the registered model.

    Returns:
        List of version dictionaries with info.
    """
    if not MLFLOW_AVAILABLE:
        return []

    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "created_at": v.creation_timestamp,
                "description": v.description,
                "tags": dict(v.tags) if v.tags else {},
            }
            for v in versions
        ]
    except Exception:
        return []


def load_production_model(model_name: str) -> Any:
    """
    Load the production version of a model.

    Args:
        model_name: Name of the registered model.

    Returns:
        The loaded model.

    Example:
        >>> model = load_production_model("SentimentClassifier")
    """
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow not available")

    if not PYTORCH_AVAILABLE:
        raise RuntimeError("mlflow.pytorch not available. Install PyTorch and mlflow together.")

    model_uri = f"models:/{model_name}/Production"
    return mlflow.pytorch.load_model(model_uri)


def compare_data_versions(
    manifest1: DataManifest,
    manifest2: DataManifest,
) -> Dict[str, Any]:
    """
    Compare two data manifests to find differences.

    Args:
        manifest1: First manifest (older).
        manifest2: Second manifest (newer).

    Returns:
        Dictionary with added, removed, and modified files.

    Example:
        >>> old_manifest = DataManifest.load("manifest_v1.json")
        >>> new_manifest = DataManifest.load("manifest_v2.json")
        >>> diff = compare_data_versions(old_manifest, new_manifest)
    """
    files1 = set(manifest1.files.keys())
    files2 = set(manifest2.files.keys())

    added = files2 - files1
    removed = files1 - files2
    common = files1 & files2

    modified = set()
    for f in common:
        if manifest1.files[f]["hash"] != manifest2.files[f]["hash"]:
            modified.add(f)

    return {
        "added": list(added),
        "removed": list(removed),
        "modified": list(modified),
        "unchanged": list(common - modified),
        "summary": {
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "unchanged_count": len(common - modified),
        },
    }


if __name__ == "__main__":
    # Example usage
    print("Versioning Utilities Example")
    print("=" * 40)

    # Create a test file
    test_dir = Path("/tmp/versioning_test")
    test_dir.mkdir(exist_ok=True)

    test_file = test_dir / "test_data.json"
    test_file.write_text('{"test": "data"}')

    # Compute hash
    file_hash = compute_data_hash(test_file)
    print(f"File hash: {file_hash}")

    # Create manifest
    manifest = create_data_manifest(test_dir, version="1.0.0")
    print(f"Manifest: {manifest.to_json()}")

    # Cleanup
    test_file.unlink()
    test_dir.rmdir()
