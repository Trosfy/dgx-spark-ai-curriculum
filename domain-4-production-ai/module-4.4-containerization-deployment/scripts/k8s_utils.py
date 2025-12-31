"""
Kubernetes Utilities for ML Deployments

This module provides utilities for:
- Generating Kubernetes manifests for ML deployments
- Managing GPU-enabled deployments
- Configuring auto-scaling (HPA)
- Health monitoring

Example usage:
    from k8s_utils import K8sDeploymentManager, generate_deployment_manifest

    # Generate manifests
    deployment = generate_deployment_manifest(
        name="llm-inference",
        image="my-inference:latest",
        replicas=2,
        gpu_count=1
    )

    # Apply manifests
    manager = K8sDeploymentManager()
    manager.apply_manifest(deployment)
"""

import os
import json
import subprocess
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union
from pathlib import Path


@dataclass
class K8sResource:
    """Represents a Kubernetes resource."""

    kind: str
    name: str
    namespace: str = "default"
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.manifest, default_flow_style=False)

    def save(self, path: str) -> str:
        """Save manifest to file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())
        return path


@dataclass
class DeploymentStatus:
    """Status of a Kubernetes deployment."""

    name: str
    namespace: str
    replicas: int
    ready_replicas: int
    available_replicas: int
    conditions: List[Dict[str, str]]

    @property
    def is_ready(self) -> bool:
        return self.ready_replicas == self.replicas

    def __repr__(self) -> str:
        status = "Ready" if self.is_ready else "Pending"
        return f"Deployment({self.name}, {self.ready_replicas}/{self.replicas} replicas, {status})"


class K8sDeploymentManager:
    """
    Manager for Kubernetes ML deployments.

    Provides high-level interface for:
    - Applying manifests
    - Managing deployments
    - Monitoring status
    - Port forwarding for local testing

    Example:
        >>> manager = K8sDeploymentManager(namespace="ml-serving")
        >>> manager.apply_manifest(deployment_yaml)
        >>> status = manager.get_deployment_status("llm-inference")
        >>> print(status)
    """

    def __init__(
        self,
        namespace: str = "default",
        kubeconfig: Optional[str] = None,
        context: Optional[str] = None,
    ):
        """
        Initialize K8s deployment manager.

        Args:
            namespace: Default namespace for operations
            kubeconfig: Path to kubeconfig file
            context: Kubernetes context to use
        """
        self.namespace = namespace
        self.kubeconfig = kubeconfig or os.environ.get("KUBECONFIG")
        self.context = context
        self._check_kubectl()

    def _check_kubectl(self) -> bool:
        """Check if kubectl is available."""
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _kubectl_cmd(self, *args) -> List[str]:
        """Build kubectl command with common options."""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            cmd.extend(["--context", self.context])
        cmd.extend(["-n", self.namespace])
        cmd.extend(args)
        return cmd

    def apply_manifest(
        self,
        manifest: Union[str, Dict, K8sResource],
        dry_run: bool = False,
    ) -> bool:
        """
        Apply a Kubernetes manifest.

        Args:
            manifest: YAML string, dict, or K8sResource
            dry_run: If True, only validate without applying

        Returns:
            True if successful
        """
        # Convert to YAML string
        if isinstance(manifest, K8sResource):
            yaml_str = manifest.to_yaml()
        elif isinstance(manifest, dict):
            yaml_str = yaml.dump(manifest, default_flow_style=False)
        else:
            yaml_str = manifest

        cmd = self._kubectl_cmd("apply", "-f", "-")
        if dry_run:
            cmd.append("--dry-run=client")

        try:
            result = subprocess.run(
                cmd,
                input=yaml_str,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"Applied successfully: {result.stdout.strip()}")
                return True
            else:
                print(f"Apply failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error applying manifest: {e}")
            return False

    def apply_file(self, path: str, dry_run: bool = False) -> bool:
        """Apply manifest from file."""
        cmd = self._kubectl_cmd("apply", "-f", path)
        if dry_run:
            cmd.append("--dry-run=client")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Applied: {result.stdout.strip()}")
                return True
            else:
                print(f"Failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def delete_resource(
        self,
        kind: str,
        name: str,
        wait: bool = True,
    ) -> bool:
        """
        Delete a Kubernetes resource.

        Args:
            kind: Resource kind (deployment, service, etc.)
            name: Resource name
            wait: Wait for deletion to complete
        """
        cmd = self._kubectl_cmd("delete", kind, name)
        if not wait:
            cmd.append("--wait=false")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Deleted: {kind}/{name}")
                return True
            else:
                print(f"Delete failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def get_deployment_status(self, name: str) -> Optional[DeploymentStatus]:
        """Get status of a deployment."""
        cmd = self._kubectl_cmd(
            "get", "deployment", name,
            "-o", "json"
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            status = data.get("status", {})

            return DeploymentStatus(
                name=name,
                namespace=self.namespace,
                replicas=status.get("replicas", 0),
                ready_replicas=status.get("readyReplicas", 0),
                available_replicas=status.get("availableReplicas", 0),
                conditions=status.get("conditions", []),
            )
        except Exception as e:
            print(f"Error getting status: {e}")
            return None

    def get_pods(self, label_selector: Optional[str] = None) -> List[Dict]:
        """Get pods, optionally filtered by label."""
        cmd = self._kubectl_cmd("get", "pods", "-o", "json")
        if label_selector:
            cmd.extend(["-l", label_selector])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return []

            data = json.loads(result.stdout)
            return data.get("items", [])
        except Exception:
            return []

    def get_logs(
        self,
        pod_name: str,
        container: Optional[str] = None,
        tail: int = 100,
        follow: bool = False,
    ) -> str:
        """Get logs from a pod."""
        cmd = self._kubectl_cmd("logs", pod_name, f"--tail={tail}")
        if container:
            cmd.extend(["-c", container])
        if follow:
            cmd.append("-f")

        try:
            if follow:
                # Stream logs
                subprocess.run(cmd)
                return ""
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.stdout
        except Exception as e:
            return f"Error getting logs: {e}"

    def port_forward(
        self,
        resource: str,  # e.g., "deployment/inference-server"
        local_port: int,
        remote_port: int,
    ) -> subprocess.Popen:
        """
        Start port forwarding to a resource.

        Returns:
            Popen object (call .terminate() to stop)
        """
        cmd = self._kubectl_cmd(
            "port-forward",
            resource,
            f"{local_port}:{remote_port}"
        )

        return subprocess.Popen(cmd)

    def scale_deployment(self, name: str, replicas: int) -> bool:
        """Scale a deployment to specified replicas."""
        cmd = self._kubectl_cmd(
            "scale", "deployment", name,
            f"--replicas={replicas}"
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Scaled {name} to {replicas} replicas")
                return True
            else:
                print(f"Scale failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def wait_for_ready(
        self,
        name: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> bool:
        """Wait for deployment to be ready."""
        cmd = self._kubectl_cmd(
            "rollout", "status",
            f"deployment/{name}",
            f"--timeout={timeout}s"
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False


def generate_deployment_manifest(
    name: str,
    image: str,
    replicas: int = 1,
    port: int = 8000,
    gpu_count: int = 0,
    gpu_type: str = "nvidia.com/gpu",
    memory_request: str = "8Gi",
    memory_limit: str = "16Gi",
    cpu_request: str = "2",
    cpu_limit: str = "4",
    env_vars: Optional[Dict[str, str]] = None,
    health_path: str = "/health",
    command: Optional[List[str]] = None,
    args: Optional[List[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None,
    node_selector: Optional[Dict[str, str]] = None,
    tolerations: Optional[List[Dict]] = None,
    volumes: Optional[List[Dict]] = None,
    volume_mounts: Optional[List[Dict]] = None,
) -> K8sResource:
    """
    Generate a Kubernetes Deployment manifest for ML inference.

    Args:
        name: Deployment name
        image: Container image
        replicas: Number of replicas
        port: Container port
        gpu_count: Number of GPUs per pod
        gpu_type: GPU resource type
        memory_request: Memory request
        memory_limit: Memory limit
        cpu_request: CPU request
        cpu_limit: CPU limit
        env_vars: Environment variables
        health_path: Health check endpoint path
        command: Container command
        args: Container arguments
        labels: Additional labels
        annotations: Pod annotations
        node_selector: Node selector
        tolerations: Pod tolerations
        volumes: Volumes to mount
        volume_mounts: Volume mount configurations

    Returns:
        K8sResource object
    """
    # Build labels
    all_labels = {"app": name}
    if labels:
        all_labels.update(labels)

    # Build container spec
    container = {
        "name": name,
        "image": image,
        "ports": [{"containerPort": port}],
        "resources": {
            "requests": {
                "memory": memory_request,
                "cpu": cpu_request,
            },
            "limits": {
                "memory": memory_limit,
                "cpu": cpu_limit,
            },
        },
        "livenessProbe": {
            "httpGet": {
                "path": health_path,
                "port": port,
            },
            "initialDelaySeconds": 60,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3,
        },
        "readinessProbe": {
            "httpGet": {
                "path": health_path,
                "port": port,
            },
            "initialDelaySeconds": 30,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3,
        },
    }

    # Add GPU resources
    if gpu_count > 0:
        container["resources"]["limits"][gpu_type] = gpu_count
        container["resources"]["requests"][gpu_type] = gpu_count

    # Add environment variables
    if env_vars:
        container["env"] = [
            {"name": k, "value": str(v)}
            for k, v in env_vars.items()
        ]

    # Add command and args
    if command:
        container["command"] = command
    if args:
        container["args"] = args

    # Add volume mounts
    if volume_mounts:
        container["volumeMounts"] = volume_mounts

    # Build pod spec
    pod_spec = {
        "containers": [container],
    }

    # Add volumes
    if volumes:
        pod_spec["volumes"] = volumes

    # Add node selector
    if node_selector:
        pod_spec["nodeSelector"] = node_selector

    # Add tolerations for GPU nodes
    pod_tolerations = []
    if gpu_count > 0:
        pod_tolerations.append({
            "key": "nvidia.com/gpu",
            "operator": "Exists",
            "effect": "NoSchedule",
        })
    if tolerations:
        pod_tolerations.extend(tolerations)
    if pod_tolerations:
        pod_spec["tolerations"] = pod_tolerations

    # Build template metadata
    template_metadata = {"labels": all_labels}
    if annotations:
        template_metadata["annotations"] = annotations

    # Build deployment manifest
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "labels": all_labels,
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {"app": name},
            },
            "template": {
                "metadata": template_metadata,
                "spec": pod_spec,
            },
        },
    }

    return K8sResource(
        kind="Deployment",
        name=name,
        manifest=manifest,
    )


def generate_service_manifest(
    name: str,
    port: int = 80,
    target_port: int = 8000,
    service_type: str = "ClusterIP",  # ClusterIP, NodePort, LoadBalancer
    labels: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None,
    node_port: Optional[int] = None,
) -> K8sResource:
    """
    Generate a Kubernetes Service manifest.

    Args:
        name: Service name (should match deployment)
        port: Service port
        target_port: Container port
        service_type: Service type
        labels: Additional labels
        annotations: Service annotations
        node_port: NodePort (for NodePort type)

    Returns:
        K8sResource object
    """
    all_labels = {"app": name}
    if labels:
        all_labels.update(labels)

    port_spec = {
        "port": port,
        "targetPort": target_port,
        "protocol": "TCP",
    }

    if service_type == "NodePort" and node_port:
        port_spec["nodePort"] = node_port

    manifest = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{name}-service",
            "labels": all_labels,
        },
        "spec": {
            "type": service_type,
            "ports": [port_spec],
            "selector": {"app": name},
        },
    }

    if annotations:
        manifest["metadata"]["annotations"] = annotations

    return K8sResource(
        kind="Service",
        name=f"{name}-service",
        manifest=manifest,
    )


def generate_hpa_manifest(
    deployment_name: str,
    min_replicas: int = 1,
    max_replicas: int = 5,
    cpu_target: int = 70,  # Percentage
    memory_target: Optional[int] = None,  # Percentage
    custom_metrics: Optional[List[Dict]] = None,
) -> K8sResource:
    """
    Generate a Horizontal Pod Autoscaler manifest.

    Args:
        deployment_name: Name of the deployment to scale
        min_replicas: Minimum replicas
        max_replicas: Maximum replicas
        cpu_target: Target CPU utilization percentage
        memory_target: Target memory utilization percentage
        custom_metrics: Custom metric specifications

    Returns:
        K8sResource object
    """
    metrics = []

    # CPU metric
    metrics.append({
        "type": "Resource",
        "resource": {
            "name": "cpu",
            "target": {
                "type": "Utilization",
                "averageUtilization": cpu_target,
            },
        },
    })

    # Memory metric
    if memory_target:
        metrics.append({
            "type": "Resource",
            "resource": {
                "name": "memory",
                "target": {
                    "type": "Utilization",
                    "averageUtilization": memory_target,
                },
            },
        })

    # Custom metrics
    if custom_metrics:
        metrics.extend(custom_metrics)

    manifest = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": f"{deployment_name}-hpa",
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name,
            },
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": metrics,
        },
    }

    return K8sResource(
        kind="HorizontalPodAutoscaler",
        name=f"{deployment_name}-hpa",
        manifest=manifest,
    )


def generate_configmap_manifest(
    name: str,
    data: Dict[str, str],
    labels: Optional[Dict[str, str]] = None,
) -> K8sResource:
    """
    Generate a ConfigMap manifest.

    Args:
        name: ConfigMap name
        data: Key-value data
        labels: Additional labels

    Returns:
        K8sResource object
    """
    manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": name,
        },
        "data": data,
    }

    if labels:
        manifest["metadata"]["labels"] = labels

    return K8sResource(
        kind="ConfigMap",
        name=name,
        manifest=manifest,
    )


def generate_secret_manifest(
    name: str,
    data: Dict[str, str],
    secret_type: str = "Opaque",
    labels: Optional[Dict[str, str]] = None,
) -> K8sResource:
    """
    Generate a Secret manifest.

    Note: Values should be base64 encoded in production.

    Args:
        name: Secret name
        data: Key-value data (will be encoded)
        secret_type: Secret type
        labels: Additional labels

    Returns:
        K8sResource object
    """
    import base64

    # Base64 encode values
    encoded_data = {
        k: base64.b64encode(v.encode()).decode()
        for k, v in data.items()
    }

    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
        },
        "type": secret_type,
        "data": encoded_data,
    }

    if labels:
        manifest["metadata"]["labels"] = labels

    return K8sResource(
        kind="Secret",
        name=name,
        manifest=manifest,
    )


def generate_pvc_manifest(
    name: str,
    storage_size: str = "10Gi",
    storage_class: Optional[str] = None,
    access_modes: List[str] = None,
) -> K8sResource:
    """
    Generate a PersistentVolumeClaim manifest.

    Args:
        name: PVC name
        storage_size: Storage size
        storage_class: Storage class name
        access_modes: Access modes

    Returns:
        K8sResource object
    """
    access_modes = access_modes or ["ReadWriteOnce"]

    manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
        },
        "spec": {
            "accessModes": access_modes,
            "resources": {
                "requests": {
                    "storage": storage_size,
                },
            },
        },
    }

    if storage_class:
        manifest["spec"]["storageClassName"] = storage_class

    return K8sResource(
        kind="PersistentVolumeClaim",
        name=name,
        manifest=manifest,
    )


def generate_complete_ml_stack(
    name: str,
    image: str,
    port: int = 8000,
    replicas: int = 2,
    gpu_count: int = 1,
    enable_hpa: bool = True,
    enable_pvc: bool = False,
    storage_size: str = "50Gi",
    env_vars: Optional[Dict[str, str]] = None,
) -> List[K8sResource]:
    """
    Generate a complete ML inference stack.

    Includes:
    - Deployment
    - Service
    - HPA (optional)
    - PVC (optional)

    Args:
        name: Stack name
        image: Container image
        port: Container port
        replicas: Initial replicas
        gpu_count: GPUs per pod
        enable_hpa: Enable auto-scaling
        enable_pvc: Enable persistent storage
        storage_size: PVC size
        env_vars: Environment variables

    Returns:
        List of K8sResource objects
    """
    resources = []

    # Volume configuration
    volumes = None
    volume_mounts = None

    if enable_pvc:
        # Add PVC
        pvc = generate_pvc_manifest(f"{name}-storage", storage_size)
        resources.append(pvc)

        volumes = [{
            "name": "model-storage",
            "persistentVolumeClaim": {
                "claimName": f"{name}-storage",
            },
        }]
        volume_mounts = [{
            "name": "model-storage",
            "mountPath": "/models",
        }]

    # Deployment
    deployment = generate_deployment_manifest(
        name=name,
        image=image,
        replicas=replicas,
        port=port,
        gpu_count=gpu_count,
        memory_request="16Gi",
        memory_limit="32Gi",
        cpu_request="4",
        cpu_limit="8",
        env_vars=env_vars,
        volumes=volumes,
        volume_mounts=volume_mounts,
    )
    resources.append(deployment)

    # Service
    service = generate_service_manifest(
        name=name,
        port=80,
        target_port=port,
        service_type="LoadBalancer",
    )
    resources.append(service)

    # HPA
    if enable_hpa:
        hpa = generate_hpa_manifest(
            deployment_name=name,
            min_replicas=1,
            max_replicas=replicas * 2,
            cpu_target=70,
        )
        resources.append(hpa)

    return resources


def save_manifests(
    resources: List[K8sResource],
    output_dir: str,
    single_file: bool = False,
) -> List[str]:
    """
    Save manifests to files.

    Args:
        resources: List of K8sResource objects
        output_dir: Output directory
        single_file: Combine all manifests into one file

    Returns:
        List of saved file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_files = []

    if single_file:
        # Combine all manifests
        combined = "\n---\n".join(r.to_yaml() for r in resources)
        path = os.path.join(output_dir, "all-manifests.yaml")
        with open(path, "w") as f:
            f.write(combined)
        saved_files.append(path)
    else:
        # Save individual files
        for resource in resources:
            filename = f"{resource.kind.lower()}-{resource.name}.yaml"
            path = os.path.join(output_dir, filename)
            resource.save(path)
            saved_files.append(path)

    return saved_files


if __name__ == "__main__":
    # Example usage
    print("=== Generate ML Inference Stack ===\n")

    # Generate complete stack
    resources = generate_complete_ml_stack(
        name="llm-inference",
        image="my-registry/llm-inference:v1.0",
        port=8000,
        replicas=2,
        gpu_count=1,
        enable_hpa=True,
        enable_pvc=True,
        env_vars={
            "MODEL_PATH": "/models/llama-8b",
            "MAX_BATCH_SIZE": "32",
        },
    )

    print("Generated resources:")
    for resource in resources:
        print(f"  - {resource.kind}: {resource.name}")

    print("\n=== Sample Deployment Manifest ===\n")
    deployment = generate_deployment_manifest(
        name="inference-server",
        image="my-inference:latest",
        replicas=2,
        gpu_count=1,
        env_vars={"MODEL_PATH": "/models"},
    )
    print(deployment.to_yaml())

    print("\n=== Sample Service Manifest ===\n")
    service = generate_service_manifest(
        name="inference-server",
        service_type="LoadBalancer",
    )
    print(service.to_yaml())

    print("\n=== Sample HPA Manifest ===\n")
    hpa = generate_hpa_manifest(
        deployment_name="inference-server",
        min_replicas=1,
        max_replicas=5,
        cpu_target=70,
    )
    print(hpa.to_yaml())
