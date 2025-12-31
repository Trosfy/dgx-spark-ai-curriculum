"""
Module 4.4: Containerization & Cloud Deployment Utilities

This module provides utilities for:
- Docker image building and management
- Cloud deployment (AWS SageMaker, GCP Vertex AI)
- Kubernetes deployment helpers
- Demo building with Gradio and Streamlit

Example usage:
    from scripts.docker_utils import DockerImageBuilder
    from scripts.cloud_utils import SageMakerDeployer, VertexAIDeployer
    from scripts.k8s_utils import K8sDeploymentManager
"""

from .docker_utils import (
    DockerImageBuilder,
    DockerComposeManager,
    calculate_image_layers,
    optimize_dockerfile,
)

from .cloud_utils import (
    SageMakerDeployer,
    VertexAIDeployer,
    estimate_cloud_costs,
    compare_platforms,
)

from .k8s_utils import (
    K8sDeploymentManager,
    generate_deployment_manifest,
    generate_service_manifest,
    generate_hpa_manifest,
)

from .demo_utils import (
    create_gradio_chat_interface,
    create_streamlit_dashboard,
    StreamingLLMClient,
)

__all__ = [
    # Docker
    "DockerImageBuilder",
    "DockerComposeManager",
    "calculate_image_layers",
    "optimize_dockerfile",
    # Cloud
    "SageMakerDeployer",
    "VertexAIDeployer",
    "estimate_cloud_costs",
    "compare_platforms",
    # Kubernetes
    "K8sDeploymentManager",
    "generate_deployment_manifest",
    "generate_service_manifest",
    "generate_hpa_manifest",
    # Demo
    "create_gradio_chat_interface",
    "create_streamlit_dashboard",
    "StreamingLLMClient",
]

__version__ = "1.0.0"
