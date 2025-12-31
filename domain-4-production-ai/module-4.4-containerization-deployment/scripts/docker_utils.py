"""
Docker Utilities for ML Containerization

This module provides utilities for building optimized Docker images for ML applications,
managing Docker Compose stacks, and analyzing Dockerfile best practices.

Designed for DGX Spark with 128GB unified memory.

Example usage:
    from docker_utils import DockerImageBuilder, DockerComposeManager

    # Build an optimized Docker image
    builder = DockerImageBuilder("my-inference-server")
    builder.add_base("nvcr.io/nvidia/pytorch:25.11-py3")
    builder.add_python_deps(["transformers", "accelerate", "vllm"])
    builder.add_entrypoint("python serve.py")
    dockerfile = builder.generate()

    # Manage Docker Compose stack
    compose = DockerComposeManager("./docker-compose.yml")
    compose.start()
    compose.check_health()
"""

import os
import re
import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path


@dataclass
class DockerLayer:
    """Represents a single Docker layer."""

    instruction: str
    size_mb: float
    created: str
    comment: str = ""

    def __repr__(self) -> str:
        return f"DockerLayer({self.instruction[:50]}..., size={self.size_mb:.1f}MB)"


@dataclass
class DockerImageInfo:
    """Information about a Docker image."""

    name: str
    tag: str
    size_mb: float
    layers: List[DockerLayer] = field(default_factory=list)
    base_image: str = ""
    created: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.name}:{self.tag}"

    def __repr__(self) -> str:
        return f"DockerImage({self.full_name}, size={self.size_mb:.1f}MB, layers={len(self.layers)})"


class DockerImageBuilder:
    """
    Builder for creating optimized ML Docker images.

    Follows best practices for:
    - Multi-stage builds to reduce image size
    - Layer caching optimization
    - GPU support configuration
    - Health check endpoints

    Example:
        >>> builder = DockerImageBuilder("my-model-server")
        >>> builder.add_base("nvcr.io/nvidia/pytorch:25.11-py3")
        >>> builder.add_python_deps(["torch", "transformers"])
        >>> builder.add_model_weights("/models/llama-8b")
        >>> builder.add_entrypoint("python serve.py")
        >>> dockerfile = builder.generate()
        >>> print(dockerfile)
    """

    # NGC base images optimized for DGX Spark (ARM64 + CUDA)
    NGC_IMAGES = {
        "pytorch": "nvcr.io/nvidia/pytorch:25.11-py3",
        "pytorch-24.12": "nvcr.io/nvidia/pytorch:24.12-py3",
        "tensorrt": "nvcr.io/nvidia/tensorrt:25.01-py3",
        "triton": "nvcr.io/nvidia/tritonserver:25.01-py3",
        "nemo": "nvcr.io/nvidia/nemo:25.01",
        "cuda-base": "nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu22.04",
    }

    def __init__(self, name: str, use_multistage: bool = True):
        """
        Initialize the Docker image builder.

        Args:
            name: Name for the Docker image
            use_multistage: Whether to use multi-stage build (recommended)
        """
        self.name = name
        self.use_multistage = use_multistage
        self.base_image = self.NGC_IMAGES["pytorch"]
        self.python_deps: List[str] = []
        self.system_deps: List[str] = []
        self.env_vars: Dict[str, str] = {}
        self.copy_files: List[Tuple[str, str]] = []
        self.model_paths: List[str] = []
        self.entrypoint: Optional[str] = None
        self.cmd: Optional[str] = None
        self.healthcheck: Optional[Dict[str, Any]] = None
        self.exposed_ports: List[int] = []
        self.workdir = "/app"
        self.user: Optional[str] = None

    def add_base(self, image: str) -> "DockerImageBuilder":
        """Set the base image."""
        self.base_image = image
        return self

    def add_python_deps(
        self,
        deps: List[str],
        requirements_file: Optional[str] = None
    ) -> "DockerImageBuilder":
        """
        Add Python dependencies.

        Args:
            deps: List of pip package names
            requirements_file: Optional path to requirements.txt
        """
        self.python_deps.extend(deps)
        if requirements_file:
            self.copy_files.append((requirements_file, "/tmp/requirements.txt"))
        return self

    def add_system_deps(self, deps: List[str]) -> "DockerImageBuilder":
        """Add system dependencies (apt packages)."""
        self.system_deps.extend(deps)
        return self

    def add_env(self, key: str, value: str) -> "DockerImageBuilder":
        """Add environment variable."""
        self.env_vars[key] = value
        return self

    def add_copy(self, src: str, dst: str) -> "DockerImageBuilder":
        """Add file/directory to copy into image."""
        self.copy_files.append((src, dst))
        return self

    def add_model_weights(self, path: str) -> "DockerImageBuilder":
        """Add model weights path (will be mounted, not copied)."""
        self.model_paths.append(path)
        return self

    def add_entrypoint(self, cmd: str) -> "DockerImageBuilder":
        """Set the container entrypoint."""
        self.entrypoint = cmd
        return self

    def add_cmd(self, cmd: str) -> "DockerImageBuilder":
        """Set the default command."""
        self.cmd = cmd
        return self

    def add_healthcheck(
        self,
        endpoint: str = "/health",
        port: int = 8000,
        interval: int = 30,
        timeout: int = 10,
        retries: int = 3
    ) -> "DockerImageBuilder":
        """Add health check configuration."""
        self.healthcheck = {
            "endpoint": endpoint,
            "port": port,
            "interval": interval,
            "timeout": timeout,
            "retries": retries
        }
        return self

    def expose(self, port: int) -> "DockerImageBuilder":
        """Expose a port."""
        self.exposed_ports.append(port)
        return self

    def set_workdir(self, path: str) -> "DockerImageBuilder":
        """Set working directory."""
        self.workdir = path
        return self

    def set_user(self, user: str) -> "DockerImageBuilder":
        """Set the user to run as (for security)."""
        self.user = user
        return self

    def generate(self) -> str:
        """Generate the Dockerfile content."""
        lines = []

        if self.use_multistage:
            # Stage 1: Builder
            lines.append(f"# Multi-stage build for optimized ML inference image")
            lines.append(f"# Generated for: {self.name}")
            lines.append("")
            lines.append("# ============================================")
            lines.append("# Stage 1: Builder - Install dependencies")
            lines.append("# ============================================")
            lines.append(f"FROM {self.base_image} AS builder")
            lines.append("")

            # System deps in builder
            if self.system_deps:
                lines.append("# Install system dependencies")
                lines.append("RUN apt-get update && apt-get install -y --no-install-recommends \\")
                for i, dep in enumerate(self.system_deps):
                    if i < len(self.system_deps) - 1:
                        lines.append(f"    {dep} \\")
                    else:
                        lines.append(f"    {dep} \\")
                lines.append("    && rm -rf /var/lib/apt/lists/*")
                lines.append("")

            # Python deps in builder (install to --user for copying)
            if self.python_deps:
                lines.append("# Install Python dependencies to user directory")
                deps_str = " ".join(f'"{dep}"' for dep in self.python_deps)
                lines.append(f"RUN pip install --user --no-cache-dir {deps_str}")
                lines.append("")

            # Stage 2: Production
            lines.append("# ============================================")
            lines.append("# Stage 2: Production - Minimal runtime image")
            lines.append("# ============================================")
            lines.append(f"FROM {self.base_image}")
            lines.append("")

            # Copy installed packages from builder
            if self.python_deps:
                lines.append("# Copy Python packages from builder")
                lines.append("COPY --from=builder /root/.local /root/.local")
                lines.append("ENV PATH=/root/.local/bin:$PATH")
                lines.append("")
        else:
            # Single stage build
            lines.append(f"# Docker image for: {self.name}")
            lines.append(f"FROM {self.base_image}")
            lines.append("")

            # System deps
            if self.system_deps:
                lines.append("# Install system dependencies")
                lines.append("RUN apt-get update && apt-get install -y --no-install-recommends \\")
                for dep in self.system_deps:
                    lines.append(f"    {dep} \\")
                lines.append("    && rm -rf /var/lib/apt/lists/*")
                lines.append("")

            # Python deps
            if self.python_deps:
                lines.append("# Install Python dependencies")
                deps_str = " ".join(f'"{dep}"' for dep in self.python_deps)
                lines.append(f"RUN pip install --no-cache-dir {deps_str}")
                lines.append("")

        # Environment variables
        if self.env_vars:
            lines.append("# Environment configuration")
            for key, value in self.env_vars.items():
                lines.append(f"ENV {key}={value}")
            lines.append("")

        # Working directory
        lines.append(f"# Set working directory")
        lines.append(f"WORKDIR {self.workdir}")
        lines.append("")

        # Copy application files
        if self.copy_files:
            lines.append("# Copy application files")
            for src, dst in self.copy_files:
                if "requirements" not in src:  # Skip requirements.txt
                    lines.append(f"COPY {src} {dst}")
            lines.append("")

        # Expose ports
        if self.exposed_ports:
            lines.append("# Expose ports")
            for port in self.exposed_ports:
                lines.append(f"EXPOSE {port}")
            lines.append("")

        # Health check
        if self.healthcheck:
            lines.append("# Health check")
            hc = self.healthcheck
            lines.append(
                f'HEALTHCHECK --interval={hc["interval"]}s --timeout={hc["timeout"]}s '
                f'--retries={hc["retries"]} \\'
            )
            lines.append(
                f'    CMD curl -f http://localhost:{hc["port"]}{hc["endpoint"]} || exit 1'
            )
            lines.append("")

        # User (for security)
        if self.user:
            lines.append("# Run as non-root user")
            lines.append(f"USER {self.user}")
            lines.append("")

        # Entrypoint and CMD
        if self.entrypoint:
            lines.append("# Start the application")
            # Parse command into list format
            parts = self.entrypoint.split()
            cmd_str = ", ".join(f'"{p}"' for p in parts)
            lines.append(f"ENTRYPOINT [{cmd_str}]")

        if self.cmd:
            parts = self.cmd.split()
            cmd_str = ", ".join(f'"{p}"' for p in parts)
            lines.append(f"CMD [{cmd_str}]")

        return "\n".join(lines)

    def save(self, path: str = "Dockerfile") -> str:
        """Save the Dockerfile to disk."""
        content = self.generate()
        with open(path, "w") as f:
            f.write(content)
        return path

    def build(
        self,
        tag: Optional[str] = None,
        context: str = ".",
        no_cache: bool = False
    ) -> bool:
        """
        Build the Docker image.

        Args:
            tag: Image tag (default: latest)
            context: Build context directory
            no_cache: Whether to disable layer caching

        Returns:
            True if build succeeded
        """
        tag = tag or "latest"
        image_name = f"{self.name}:{tag}"

        # Save Dockerfile
        dockerfile_path = os.path.join(context, "Dockerfile")
        self.save(dockerfile_path)

        # Build command
        cmd = ["docker", "build", "-t", image_name, "-f", dockerfile_path]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(context)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully built {image_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build failed: {e.stderr}")
            return False


class DockerComposeManager:
    """
    Manager for Docker Compose ML stacks.

    Handles multi-service architectures typically used in production:
    - Inference server (GPU)
    - Vector database (ChromaDB, Qdrant)
    - Monitoring (Prometheus, Grafana)
    - Load balancer (Traefik, Nginx)

    Example:
        >>> compose = DockerComposeManager()
        >>> compose.add_inference_service("my-model", gpu=True)
        >>> compose.add_vector_db("chromadb")
        >>> compose.add_monitoring("prometheus")
        >>> compose.save("docker-compose.yml")
        >>> compose.start()
    """

    def __init__(self, compose_file: str = "docker-compose.yml"):
        """
        Initialize Docker Compose manager.

        Args:
            compose_file: Path to docker-compose.yml
        """
        self.compose_file = compose_file
        self.services: Dict[str, Dict[str, Any]] = {}
        self.volumes: Dict[str, Dict[str, Any]] = {}
        self.networks: Dict[str, Dict[str, Any]] = {"default": {"driver": "bridge"}}

    def add_service(
        self,
        name: str,
        image: str,
        ports: Optional[List[str]] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[List[str]] = None,
        depends_on: Optional[List[str]] = None,
        gpu: bool = False,
        gpu_count: int = 1,
        healthcheck: Optional[Dict[str, Any]] = None,
        command: Optional[str] = None,
        build: Optional[str] = None,
    ) -> "DockerComposeManager":
        """
        Add a generic service.

        Args:
            name: Service name
            image: Docker image
            ports: Port mappings (host:container)
            environment: Environment variables
            volumes: Volume mounts
            depends_on: Service dependencies
            gpu: Whether to enable GPU access
            gpu_count: Number of GPUs to allocate
            healthcheck: Health check configuration
            command: Override command
            build: Build context path (instead of image)
        """
        service = {}

        if build:
            service["build"] = build
        else:
            service["image"] = image

        if ports:
            service["ports"] = ports
        if environment:
            service["environment"] = environment
        if volumes:
            service["volumes"] = volumes
        if depends_on:
            service["depends_on"] = depends_on
        if command:
            service["command"] = command

        if gpu:
            service["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [{
                            "driver": "nvidia",
                            "count": gpu_count,
                            "capabilities": ["gpu"]
                        }]
                    }
                }
            }

        if healthcheck:
            service["healthcheck"] = healthcheck

        self.services[name] = service
        return self

    def add_inference_service(
        self,
        name: str = "inference",
        image: str = "my-inference:latest",
        port: int = 8000,
        model_path: str = "/models",
        gpu: bool = True,
    ) -> "DockerComposeManager":
        """Add a GPU-enabled inference service."""
        return self.add_service(
            name=name,
            image=image,
            ports=[f"{port}:{port}"],
            environment={"MODEL_PATH": model_path},
            volumes=[f"./models:{model_path}"],
            gpu=gpu,
            healthcheck={
                "test": ["CMD", "curl", "-f", f"http://localhost:{port}/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
            }
        )

    def add_vector_db(
        self,
        db_type: str = "chromadb",
        port: int = 8001,
    ) -> "DockerComposeManager":
        """
        Add a vector database service.

        Args:
            db_type: One of "chromadb", "qdrant", "milvus"
            port: External port mapping
        """
        configs = {
            "chromadb": {
                "image": "chromadb/chroma:latest",
                "ports": [f"{port}:8000"],
                "volumes": ["chroma_data:/chroma/chroma"],
            },
            "qdrant": {
                "image": "qdrant/qdrant:latest",
                "ports": [f"{port}:6333"],
                "volumes": ["qdrant_data:/qdrant/storage"],
            },
            "milvus": {
                "image": "milvusdb/milvus:latest",
                "ports": [f"{port}:19530"],
                "volumes": ["milvus_data:/var/lib/milvus"],
            },
        }

        if db_type not in configs:
            raise ValueError(f"Unknown vector DB: {db_type}")

        config = configs[db_type]
        self.add_service(name="vectordb", **config)

        # Add volume
        volume_name = config["volumes"][0].split(":")[0]
        self.volumes[volume_name] = {}

        return self

    def add_monitoring(
        self,
        stack: str = "prometheus",
        prometheus_port: int = 9090,
        grafana_port: int = 3000,
    ) -> "DockerComposeManager":
        """
        Add monitoring stack.

        Args:
            stack: "prometheus" or "full" (prometheus + grafana)
            prometheus_port: Prometheus UI port
            grafana_port: Grafana UI port
        """
        # Prometheus
        self.add_service(
            name="prometheus",
            image="prom/prometheus:latest",
            ports=[f"{prometheus_port}:9090"],
            volumes=["./prometheus.yml:/etc/prometheus/prometheus.yml"],
        )

        if stack == "full":
            # Grafana
            self.add_service(
                name="grafana",
                image="grafana/grafana:latest",
                ports=[f"{grafana_port}:3000"],
                environment={
                    "GF_SECURITY_ADMIN_PASSWORD": "admin",
                    "GF_USERS_ALLOW_SIGN_UP": "false",
                },
                depends_on=["prometheus"],
            )

        return self

    def generate(self) -> str:
        """Generate docker-compose.yml content."""
        compose = {
            "version": "3.8",
            "services": self.services,
        }

        if self.volumes:
            compose["volumes"] = self.volumes

        if len(self.networks) > 1:
            compose["networks"] = self.networks

        # Manual YAML generation (to avoid pyyaml dependency)
        lines = ["version: '3.8'", "", "services:"]

        for name, service in self.services.items():
            lines.append(f"  {name}:")
            lines.extend(self._dict_to_yaml(service, indent=4))
            lines.append("")

        if self.volumes:
            lines.append("volumes:")
            for vol_name, vol_config in self.volumes.items():
                if vol_config:
                    lines.append(f"  {vol_name}:")
                    lines.extend(self._dict_to_yaml(vol_config, indent=4))
                else:
                    lines.append(f"  {vol_name}:")

        return "\n".join(lines)

    def _dict_to_yaml(self, d: Dict, indent: int = 0) -> List[str]:
        """Convert dictionary to YAML lines."""
        lines = []
        prefix = " " * indent

        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(self._dict_to_yaml(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        # First key on same line with dash
                        first_key = list(item.keys())[0]
                        lines.append(f"{prefix}  - {first_key}: {item[first_key]}")
                        for k, v in list(item.items())[1:]:
                            lines.append(f"{prefix}    {k}: {v}")
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return lines

    def save(self, path: Optional[str] = None) -> str:
        """Save docker-compose.yml to disk."""
        path = path or self.compose_file
        content = self.generate()
        with open(path, "w") as f:
            f.write(content)
        return path

    def start(self, detach: bool = True) -> bool:
        """Start the Docker Compose stack."""
        cmd = ["docker", "compose", "-f", self.compose_file, "up"]
        if detach:
            cmd.append("-d")

        try:
            subprocess.run(cmd, check=True)
            print("Stack started successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to start stack: {e}")
            return False

    def stop(self) -> bool:
        """Stop the Docker Compose stack."""
        cmd = ["docker", "compose", "-f", self.compose_file, "down"]
        try:
            subprocess.run(cmd, check=True)
            print("Stack stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop stack: {e}")
            return False

    def check_health(self) -> Dict[str, str]:
        """Check health of all services."""
        cmd = ["docker", "compose", "-f", self.compose_file, "ps", "--format", "json"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            services = json.loads(result.stdout) if result.stdout.strip() else []

            health = {}
            for svc in services:
                if isinstance(svc, dict):
                    name = svc.get("Name", svc.get("Service", "unknown"))
                    state = svc.get("State", svc.get("Status", "unknown"))
                    health[name] = state
            return health
        except Exception as e:
            print(f"Failed to check health: {e}")
            return {}


def calculate_image_layers(image_name: str) -> List[DockerLayer]:
    """
    Analyze Docker image layers.

    Args:
        image_name: Full image name with tag

    Returns:
        List of DockerLayer objects
    """
    cmd = ["docker", "history", "--no-trunc", "--format",
           "{{.Size}}|{{.CreatedAt}}|{{.CreatedBy}}", image_name]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        layers = []

        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.split("|", 2)
                size_str = parts[0].strip()

                # Parse size
                size_mb = 0.0
                if "GB" in size_str:
                    size_mb = float(size_str.replace("GB", "")) * 1024
                elif "MB" in size_str:
                    size_mb = float(size_str.replace("MB", ""))
                elif "kB" in size_str:
                    size_mb = float(size_str.replace("kB", "")) / 1024
                elif "B" in size_str and "GB" not in size_str and "MB" not in size_str:
                    size_mb = float(size_str.replace("B", "")) / (1024 * 1024)

                layers.append(DockerLayer(
                    instruction=parts[2] if len(parts) > 2 else "",
                    size_mb=size_mb,
                    created=parts[1] if len(parts) > 1 else "",
                ))

        return layers
    except Exception as e:
        print(f"Failed to analyze image: {e}")
        return []


def optimize_dockerfile(dockerfile_path: str) -> List[str]:
    """
    Analyze Dockerfile and suggest optimizations.

    Args:
        dockerfile_path: Path to Dockerfile

    Returns:
        List of optimization suggestions
    """
    suggestions = []

    with open(dockerfile_path, "r") as f:
        content = f.read()
        lines = content.split("\n")

    # Check for multi-stage build
    if content.count("FROM ") < 2:
        suggestions.append(
            "Consider using multi-stage build to reduce final image size. "
            "Install dependencies in a builder stage, then copy only what's needed."
        )

    # Check for apt-get cleanup
    if "apt-get install" in content and "rm -rf /var/lib/apt/lists" not in content:
        suggestions.append(
            "Add 'rm -rf /var/lib/apt/lists/*' after apt-get install to reduce layer size."
        )

    # Check for pip cache
    if "pip install" in content and "--no-cache-dir" not in content:
        suggestions.append(
            "Add '--no-cache-dir' to pip install commands to avoid caching packages."
        )

    # Check for COPY ordering (dependencies before code)
    copy_indices = [i for i, line in enumerate(lines) if line.strip().startswith("COPY")]
    if copy_indices:
        # Suggest copying requirements.txt first
        for i, idx in enumerate(copy_indices):
            if "requirements" in lines[idx].lower():
                if i != 0:
                    suggestions.append(
                        "Copy requirements.txt before other files to leverage layer caching. "
                        "This way, dependencies are only reinstalled when requirements change."
                    )
                break

    # Check for health check
    if "HEALTHCHECK" not in content:
        suggestions.append(
            "Add a HEALTHCHECK instruction for container orchestration. "
            "This helps Kubernetes and Docker Swarm manage container lifecycle."
        )

    # Check for non-root user
    if "USER" not in content or "USER root" in content:
        suggestions.append(
            "Consider running as a non-root user for security. "
            "Add 'RUN useradd -m appuser' and 'USER appuser'."
        )

    # Check for .dockerignore
    dockerfile_dir = os.path.dirname(dockerfile_path) or "."
    if not os.path.exists(os.path.join(dockerfile_dir, ".dockerignore")):
        suggestions.append(
            "Create a .dockerignore file to exclude unnecessary files from build context. "
            "Include: __pycache__, *.pyc, .git, .env, models/, data/"
        )

    # Check for layer combining
    run_lines = [line for line in lines if line.strip().startswith("RUN")]
    if len(run_lines) > 5:
        suggestions.append(
            f"Found {len(run_lines)} RUN instructions. Consider combining related "
            "commands with '&&' to reduce layers."
        )

    return suggestions


def get_image_size(image_name: str) -> float:
    """Get Docker image size in MB."""
    cmd = ["docker", "image", "inspect", "--format", "{{.Size}}", image_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        size_bytes = int(result.stdout.strip())
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def create_dockerignore(path: str = ".dockerignore") -> str:
    """Create an optimized .dockerignore file for ML projects."""
    content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.eggs/
*.egg-info/
*.egg

# Virtual environments
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# ML artifacts (mount these instead)
models/
*.pt
*.pth
*.bin
*.safetensors
*.gguf
checkpoints/

# Data (mount these instead)
data/
*.csv
*.json
*.parquet

# Logs
logs/
*.log
wandb/
mlruns/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Environment
.env
.env.*
secrets/

# Build artifacts
dist/
build/
*.tar.gz

# Notebooks (usually not needed in production)
*.ipynb
.ipynb_checkpoints/

# Documentation
docs/
*.md
!README.md
"""

    with open(path, "w") as f:
        f.write(content)

    return path


if __name__ == "__main__":
    # Example usage
    print("=== Docker Image Builder Example ===")

    builder = DockerImageBuilder("llm-inference")
    builder.add_base("nvcr.io/nvidia/pytorch:25.11-py3")
    builder.add_python_deps([
        "transformers>=4.37.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
    ])
    builder.add_env("MODEL_PATH", "/models")
    builder.add_env("CUDA_VISIBLE_DEVICES", "0")
    builder.add_copy("app/", "/app/")
    builder.expose(8000)
    builder.add_healthcheck("/health", 8000)
    builder.add_entrypoint("python -m uvicorn main:app --host 0.0.0.0 --port 8000")

    print(builder.generate())
    print("\n" + "="*50)

    print("\n=== Docker Compose Example ===")

    compose = DockerComposeManager()
    compose.add_inference_service("llm-server", gpu=True)
    compose.add_vector_db("chromadb")
    compose.add_monitoring("prometheus")

    print(compose.generate())
