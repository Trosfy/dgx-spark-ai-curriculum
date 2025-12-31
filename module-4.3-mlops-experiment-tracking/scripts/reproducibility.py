"""
Reproducibility Utilities for ML Experiments.

This module provides utilities for ensuring reproducibility in ML experiments,
including seed management, environment capture, and reproducibility auditing.

Example:
    from reproducibility import set_seed, capture_environment, ReproducibilityAuditor

    # Set all random seeds
    set_seed(42, deterministic=True)

    # Capture environment info
    env_info = capture_environment()

    # Audit reproducibility
    auditor = ReproducibilityAuditor()
    checks = auditor.audit(config, env_info, train_fn)
"""

import os
import sys
import json
import random
import platform
import subprocess
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Environment variable PYTHONHASHSEED

    Args:
        seed: The seed value to use
        deterministic: If True, use deterministic algorithms (slower but reproducible)

    Example:
        set_seed(42)
        # Now all random operations are reproducible
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # For PyTorch 1.8+
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError:
                    # Some operations don't have deterministic implementations
                    pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    except ImportError:
        pass

    print(f"Random seeds set to {seed} (deterministic={deterministic})")


def get_dataloader_worker_init_fn(seed: int = 42):
    """
    Get a worker_init_fn for reproducible DataLoader.

    Args:
        seed: Base seed value

    Returns:
        Function to pass to DataLoader's worker_init_fn

    Example:
        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=get_dataloader_worker_init_fn(42)
        )
    """
    def seed_worker(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def capture_environment() -> Dict[str, Any]:
    """
    Capture comprehensive environment information for reproducibility.

    Returns:
        Dict containing Python version, platform info, package versions, and GPU info.

    Example:
        env_info = capture_environment()
        print(f"Python: {env_info['python']['version']}")
        print(f"PyTorch: {env_info['packages'].get('torch', 'N/A')}")
    """
    env = {
        "timestamp": datetime.now().isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "packages": {},
        "gpu": {},
        "environment_variables": {}
    }

    # Key packages to track
    key_packages = [
        'torch', 'numpy', 'transformers', 'datasets',
        'mlflow', 'wandb', 'peft', 'accelerate', 'bitsandbytes',
        'scipy', 'sklearn', 'pandas', 'evidently'
    ]

    for pkg in key_packages:
        try:
            module = __import__(pkg)
            env["packages"][pkg] = getattr(module, '__version__', 'unknown')
        except ImportError:
            env["packages"][pkg] = 'not installed'

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            }
        else:
            env["gpu"] = {"available": False}
    except ImportError:
        env["gpu"] = {"available": False, "error": "PyTorch not installed"}

    # Relevant environment variables
    relevant_env_vars = [
        'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED',
        'TRANSFORMERS_CACHE', 'HF_HOME'
    ]
    for var in relevant_env_vars:
        if var in os.environ:
            env["environment_variables"][var] = os.environ[var]

    return env


def generate_requirements(
    output_path: str = "requirements.txt",
    include_header: bool = True
) -> bool:
    """
    Generate a requirements.txt with pinned package versions.

    Args:
        output_path: Path to save the requirements file
        include_header: Include comment header with metadata

    Returns:
        True if successful, False otherwise

    Example:
        generate_requirements("requirements_frozen.txt")
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True
        )

        with open(output_path, "w") as f:
            if include_header:
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Python: {sys.version.split()[0]}\n")
                f.write(f"# Platform: {platform.system()} {platform.machine()}\n\n")
            f.write(result.stdout)

        print(f"Requirements saved to {output_path}")
        return True

    except Exception as e:
        print(f"Error generating requirements: {e}")
        return False


@dataclass
class ReproducibilityCheck:
    """Result of a reproducibility check."""
    name: str
    passed: bool
    message: str
    details: Dict = field(default_factory=dict)


class ReproducibilityAuditor:
    """
    Audit experiments for reproducibility.

    Example:
        auditor = ReproducibilityAuditor()
        checks = auditor.audit(config, env_info, train_fn)
        auditor.print_report()
    """

    def __init__(self, tolerance: float = 1e-5):
        """
        Initialize the auditor.

        Args:
            tolerance: Maximum allowed difference for numerical reproducibility
        """
        self.tolerance = tolerance
        self.checks: List[ReproducibilityCheck] = []

    def check_seeds(self, config: Dict) -> ReproducibilityCheck:
        """Check if seeds are properly configured."""
        seed_keys = ['seed', 'random_seed', 'numpy_seed', 'torch_seed']
        found_seeds = [k for k in seed_keys if k in config]

        if not found_seeds:
            return ReproducibilityCheck(
                name="seed_configuration",
                passed=False,
                message="No seed found in config",
                details={"searched": seed_keys}
            )

        return ReproducibilityCheck(
            name="seed_configuration",
            passed=True,
            message=f"Seed found: {found_seeds[0]} = {config[found_seeds[0]]}",
            details={"seed_key": found_seeds[0], "seed_value": config[found_seeds[0]]}
        )

    def check_deterministic_mode(self) -> ReproducibilityCheck:
        """Check if PyTorch deterministic mode is enabled."""
        try:
            import torch

            is_deterministic = torch.backends.cudnn.deterministic
            benchmark_disabled = not torch.backends.cudnn.benchmark

            passed = is_deterministic and benchmark_disabled

            return ReproducibilityCheck(
                name="deterministic_mode",
                passed=passed,
                message="Deterministic mode " + ("enabled" if passed else "disabled"),
                details={
                    "cudnn_deterministic": is_deterministic,
                    "cudnn_benchmark": not benchmark_disabled
                }
            )
        except ImportError:
            return ReproducibilityCheck(
                name="deterministic_mode",
                passed=False,
                message="PyTorch not available",
                details={}
            )

    def check_environment_captured(self, env_info: Dict) -> ReproducibilityCheck:
        """Check if environment info is properly captured."""
        required = ['python', 'packages', 'platform']
        missing = [k for k in required if k not in env_info]

        if missing:
            return ReproducibilityCheck(
                name="environment_capture",
                passed=False,
                message=f"Missing environment info: {missing}",
                details={"missing": missing}
            )

        return ReproducibilityCheck(
            name="environment_capture",
            passed=True,
            message="Environment properly captured",
            details={
                "python": env_info['python'].get('version', '')[:20],
                "packages_count": len(env_info.get('packages', {}))
            }
        )

    def verify_reproducibility(
        self,
        train_fn: Callable,
        config: Dict,
        n_runs: int = 2
    ) -> ReproducibilityCheck:
        """
        Verify training is reproducible by running multiple times.

        Args:
            train_fn: Training function that takes config and returns metrics dict
            config: Training configuration
            n_runs: Number of runs for verification

        Returns:
            ReproducibilityCheck with results
        """
        results = []

        for i in range(n_runs):
            # Reset seeds before each run
            seed = config.get('seed', 42)
            set_seed(seed)

            # Run training
            result = train_fn(config)
            results.append(result)

        # Compare results
        first_result = results[0]
        differences = []

        for i, result in enumerate(results[1:], 1):
            if isinstance(first_result, dict) and isinstance(result, dict):
                for key in first_result:
                    if key in result:
                        diff = abs(first_result[key] - result[key])
                        if diff > self.tolerance:
                            differences.append({
                                "run": i,
                                "metric": key,
                                "diff": diff
                            })
            elif isinstance(first_result, (int, float)):
                diff = abs(first_result - result)
                if diff > self.tolerance:
                    differences.append({"run": i, "diff": diff})

        passed = len(differences) == 0

        return ReproducibilityCheck(
            name="reproducibility_verification",
            passed=passed,
            message=f"Runs {'identical' if passed else 'differ'} within tolerance {self.tolerance}",
            details={
                "n_runs": n_runs,
                "differences": differences,
                "tolerance": self.tolerance
            }
        )

    def audit(
        self,
        config: Dict,
        env_info: Optional[Dict] = None,
        train_fn: Optional[Callable] = None
    ) -> List[ReproducibilityCheck]:
        """
        Run full reproducibility audit.

        Args:
            config: Training configuration
            env_info: Environment information (captured if not provided)
            train_fn: Training function for verification (optional)

        Returns:
            List of ReproducibilityCheck results
        """
        self.checks = []

        # Check seeds
        self.checks.append(self.check_seeds(config))

        # Check deterministic mode
        self.checks.append(self.check_deterministic_mode())

        # Check environment capture
        if env_info is None:
            env_info = capture_environment()
        self.checks.append(self.check_environment_captured(env_info))

        # Verify reproducibility
        if train_fn:
            self.checks.append(self.verify_reproducibility(train_fn, config))

        return self.checks

    def print_report(self) -> None:
        """Print a formatted audit report."""
        print("\n" + "=" * 60)
        print("REPRODUCIBILITY AUDIT REPORT")
        print("=" * 60)

        passed_count = sum(1 for c in self.checks if c.passed)
        total_count = len(self.checks)

        print(f"\nOverall: {passed_count}/{total_count} checks passed")
        print("-" * 40)

        for check in self.checks:
            icon = "PASS" if check.passed else "FAIL"
            print(f"\n[{icon}] {check.name}")
            print(f"      {check.message}")
            if check.details and not check.passed:
                print(f"      Details: {check.details}")

        print("\n" + "=" * 60)

    def to_dict(self) -> Dict:
        """Convert audit results to dictionary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "passed": all(c.passed for c in self.checks),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details
                }
                for c in self.checks
            ]
        }


def generate_reproducibility_report(
    config: Dict,
    env_info: Dict,
    output_path: str = "REPRODUCIBILITY.md",
    git_commit: Optional[str] = None
) -> str:
    """
    Generate a markdown reproducibility report.

    Args:
        config: Training configuration
        env_info: Environment information
        output_path: Path to save the report
        git_commit: Optional git commit hash

    Returns:
        Report content as string
    """
    report = f"""# Reproducibility Report

Generated: {datetime.now().isoformat()}

## Configuration

```json
{json.dumps(config, indent=2, default=str)}
```

## Environment

| Component | Version |
|-----------|---------|
| Python | {env_info['python']['version'].split()[0]} |
| Platform | {env_info['platform']['system']} {env_info['platform']['machine']} |
| PyTorch | {env_info['packages'].get('torch', 'N/A')} |
| NumPy | {env_info['packages'].get('numpy', 'N/A')} |

## GPU Information

| Property | Value |
|----------|-------|
| Available | {env_info['gpu'].get('available', False)} |
| Device | {env_info['gpu'].get('device_name', 'N/A')} |
| CUDA | {env_info['gpu'].get('cuda_version', 'N/A')} |
| Memory | {env_info['gpu'].get('memory_total_gb', 0):.1f} GB |

## Reproducibility Settings

- Random Seed: `{config.get('seed', 'NOT SET')}`
- Deterministic Mode: `True`

## How to Reproduce

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set the seed: `set_seed({config.get('seed', 42)})`
4. Run training with the same config
"""

    if git_commit:
        report += f"\n## Git Information\n\nCommit: `{git_commit}`\n"

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    return report


def main():
    """Example usage of reproducibility utilities."""
    # Set seeds
    set_seed(42, deterministic=True)

    # Capture environment
    env_info = capture_environment()
    print("Environment captured:")
    print(f"  Python: {env_info['python']['version'].split()[0]}")
    print(f"  Platform: {env_info['platform']['system']}")

    # Example config
    config = {"seed": 42, "lr": 0.01, "epochs": 10}

    # Run audit
    auditor = ReproducibilityAuditor()
    auditor.audit(config, env_info)
    auditor.print_report()


if __name__ == "__main__":
    main()
