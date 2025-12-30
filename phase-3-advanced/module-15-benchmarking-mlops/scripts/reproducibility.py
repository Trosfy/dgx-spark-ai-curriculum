"""
Reproducibility Utilities for ML Training.

This module provides utilities for ensuring reproducible machine learning
experiments, including seed setting, environment capture, and verification.

Example usage:
    from scripts.reproducibility import set_seed, capture_environment

    # Set all random seeds
    set_seed(42)

    # Capture environment for logging
    env = capture_environment()
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# NumPy import with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# PyTorch import with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (CPU and CUDA, if available)
    - PYTHONHASHSEED environment variable

    Args:
        seed: The random seed to use.
        deterministic: If True, enable deterministic algorithms (slower but reproducible).

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
        >>> import random
        >>> random.random()  # Same value every time with seed 42
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    if NUMPY_AVAILABLE:
        np.random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # Deterministic behavior settings
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # For PyTorch 1.8+
            try:
                torch.use_deterministic_algorithms(True)
            except (AttributeError, RuntimeError):
                pass  # Not available or not supported

    print(f"All random seeds set to {seed} (deterministic={deterministic})")


def get_seed_state() -> Dict[str, Any]:
    """
    Get current state of all random generators.

    Returns:
        Dictionary with states of random, numpy, and torch generators.
    """
    state = {
        "python_random": random.getstate(),
    }

    if NUMPY_AVAILABLE:
        state["numpy_random"] = np.random.get_state()

    if TORCH_AVAILABLE:
        state["torch_cpu"] = torch.get_rng_state().tolist()
        if torch.cuda.is_available():
            state["torch_cuda"] = [
                s.tolist() for s in torch.cuda.get_rng_state_all()
            ]

    return state


def set_seed_state(state: Dict[str, Any]) -> None:
    """
    Restore random generator states.

    Args:
        state: State dictionary from get_seed_state().
    """
    if "python_random" in state:
        random.setstate(state["python_random"])

    if "numpy_random" in state and NUMPY_AVAILABLE:
        np.random.set_state(state["numpy_random"])

    if TORCH_AVAILABLE:
        if "torch_cpu" in state:
            torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))

        if "torch_cuda" in state and torch.cuda.is_available():
            states = [
                torch.tensor(s, dtype=torch.uint8) for s in state["torch_cuda"]
            ]
            torch.cuda.set_rng_state_all(states)


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the execution environment."""

    timestamp: str
    python: Dict[str, str]
    platform_info: Dict[str, str]
    packages: Dict[str, str]
    cuda: Dict[str, Any]
    environment_variables: Dict[str, str]
    git: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "python": self.python,
            "platform": self.platform_info,
            "packages": self.packages,
            "cuda": self.cuda,
            "environment_variables": self.environment_variables,
            "git": self.git,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Save to file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "EnvironmentSnapshot":
        """Load from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            timestamp=data["timestamp"],
            python=data["python"],
            platform_info=data["platform"],
            packages=data["packages"],
            cuda=data["cuda"],
            environment_variables=data.get("environment_variables", {}),
            git=data.get("git", {}),
        )


def capture_environment(
    include_env_vars: bool = False,
    env_var_prefixes: Optional[List[str]] = None,
) -> EnvironmentSnapshot:
    """
    Capture the full environment for reproducibility.

    Args:
        include_env_vars: Whether to include environment variables.
        env_var_prefixes: If include_env_vars, only include vars with these prefixes.

    Returns:
        EnvironmentSnapshot with full environment information.

    Example:
        >>> env = capture_environment()
        >>> env.save("environment.json")
    """
    # Python info
    python_info = {
        "version": sys.version,
        "executable": sys.executable,
        "path": str(Path(sys.executable).parent),
    }

    # Platform info
    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
    }

    # Package versions
    packages = {}
    package_list = ["torch", "numpy", "transformers", "mlflow", "datasets", "peft"]

    for pkg in package_list:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # CUDA info
    cuda_info: Dict[str, Any] = {"available": False}

    if TORCH_AVAILABLE and torch.cuda.is_available():
        cuda_info = {
            "available": True,
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": torch.cuda.get_device_capability(0),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }

    # Environment variables
    env_vars = {}
    if include_env_vars:
        prefixes = env_var_prefixes or ["CUDA", "TORCH", "HF_", "TRANSFORMERS"]
        for key, value in os.environ.items():
            if any(key.startswith(p) for p in prefixes):
                env_vars[key] = value

    # Git info
    git_info = _get_git_info()

    return EnvironmentSnapshot(
        timestamp=datetime.now().isoformat(),
        python=python_info,
        platform_info=platform_info,
        packages=packages,
        cuda=cuda_info,
        environment_variables=env_vars,
        git=git_info,
    )


def _get_git_info() -> Dict[str, str]:
    """Get current git repository information."""
    git_info = {}

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()[:12]

        # Get branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["dirty"] = bool(result.stdout.strip())

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return git_info


def verify_reproducibility(
    fn: Callable[[], Any],
    seed: int = 42,
    n_runs: int = 3,
    compare_fn: Optional[Callable[[Any, Any], bool]] = None,
) -> Tuple[bool, List[Any]]:
    """
    Verify that a function produces reproducible results.

    Args:
        fn: Function to test (should return comparable results).
        seed: Seed to use for each run.
        n_runs: Number of runs to compare.
        compare_fn: Optional custom comparison function. If None, uses equality.

    Returns:
        Tuple of (is_reproducible, list_of_results).

    Example:
        >>> def my_training():
        ...     set_seed(42)
        ...     return np.random.rand(5)
        >>> is_repro, results = verify_reproducibility(my_training)
        >>> print(f"Reproducible: {is_repro}")
    """
    results = []

    for i in range(n_runs):
        set_seed(seed)
        result = fn()
        results.append(result)

    # Compare all results to the first
    if compare_fn is None:
        # Default comparison
        def default_compare(a: Any, b: Any) -> bool:
            if NUMPY_AVAILABLE:
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    return np.allclose(a, b)
            if TORCH_AVAILABLE and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return torch.allclose(a, b)
            return a == b

        compare_fn = default_compare

    is_reproducible = all(compare_fn(results[0], r) for r in results[1:])

    return is_reproducible, results


@dataclass
class AuditResult:
    """Result of a reproducibility audit."""

    passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)

    @property
    def is_reproducible(self) -> bool:
        """Check if all critical checks passed."""
        return len(self.failed) == 0

    def print_report(self) -> None:
        """Print formatted audit report."""
        print("\nReproducibility Audit Results")
        print("=" * 50)

        print(f"\n Passed ({len(self.passed)})")
        for item in self.passed:
            print(f"   {item}")

        print(f"\n Warnings ({len(self.warnings)})")
        for item in self.warnings:
            print(f"   {item}")

        print(f"\n Failed ({len(self.failed)})")
        for item in self.failed:
            print(f"   {item}")

        status = "PASS" if self.is_reproducible else "FAIL"
        print(f"\nOverall Status: {status}")


def audit_reproducibility() -> AuditResult:
    """
    Audit the current environment for reproducibility.

    Returns:
        AuditResult with detailed findings.

    Example:
        >>> result = audit_reproducibility()
        >>> result.print_report()
    """
    result = AuditResult()

    # Check PYTHONHASHSEED
    if "PYTHONHASHSEED" in os.environ:
        result.passed.append(f"PYTHONHASHSEED set to {os.environ['PYTHONHASHSEED']}")
    else:
        result.warnings.append("PYTHONHASHSEED not set")

    # Check PyTorch settings
    if TORCH_AVAILABLE:
        # CUDNN deterministic
        if torch.backends.cudnn.deterministic:
            result.passed.append("CUDNN deterministic mode enabled")
        else:
            result.warnings.append("CUDNN deterministic mode not enabled")

        # CUDNN benchmark
        if not torch.backends.cudnn.benchmark:
            result.passed.append("CUDNN benchmark mode disabled")
        else:
            result.warnings.append("CUDNN benchmark mode enabled (may cause non-determinism)")

        # Check for deterministic algorithms
        try:
            if torch.are_deterministic_algorithms_enabled():
                result.passed.append("Deterministic algorithms enabled")
            else:
                result.warnings.append("Deterministic algorithms not enabled")
        except AttributeError:
            result.warnings.append("Cannot check deterministic algorithms (older PyTorch)")

        # CUDA availability
        if torch.cuda.is_available():
            result.passed.append(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            result.warnings.append("CUDA not available")

    else:
        result.warnings.append("PyTorch not available")

    # Check for git cleanliness
    git_info = _get_git_info()
    if git_info.get("commit"):
        result.passed.append(f"Git commit: {git_info['commit']}")
        if git_info.get("dirty"):
            result.warnings.append("Git repository has uncommitted changes")
        else:
            result.passed.append("Git repository is clean")
    else:
        result.warnings.append("Not in a git repository")

    return result


def generate_requirements(output_path: Optional[str] = None) -> str:
    """
    Generate a requirements.txt for the current environment.

    Args:
        output_path: Optional path to save the requirements file.

    Returns:
        Requirements string.
    """
    packages = [
        "torch",
        "numpy",
        "transformers",
        "datasets",
        "peft",
        "mlflow",
        "accelerate",
        "bitsandbytes",
    ]

    requirements = []

    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", None)
            if version:
                requirements.append(f"{pkg}=={version}")
        except ImportError:
            pass

    req_string = "\n".join(requirements)

    if output_path:
        with open(output_path, "w") as f:
            f.write(req_string)
        print(f"Requirements saved to {output_path}")

    return req_string


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize DataLoader workers with reproducible seeds.

    Use this as the worker_init_fn argument to DataLoader.

    Args:
        worker_id: Worker ID (provided by DataLoader).

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init_fn
        ... )
    """
    if TORCH_AVAILABLE:
        seed = torch.initial_seed() % 2**32
    else:
        seed = 42 + worker_id

    if NUMPY_AVAILABLE:
        np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # Example usage
    print("Reproducibility Utilities Example")
    print("=" * 40)

    # Set seed
    set_seed(42)

    # Test reproducibility
    def test_random():
        if NUMPY_AVAILABLE:
            return np.random.rand(5)
        return [random.random() for _ in range(5)]

    is_repro, results = verify_reproducibility(test_random, seed=42, n_runs=3)
    print(f"\nReproducibility test: {'PASS' if is_repro else 'FAIL'}")

    # Capture environment
    env = capture_environment()
    print(f"\nEnvironment captured at: {env.timestamp}")
    print(f"Python: {env.python['version'].split()[0]}")
    print(f"Platform: {env.platform_info['system']} {env.platform_info['machine']}")

    # Audit
    audit = audit_reproducibility()
    audit.print_report()
