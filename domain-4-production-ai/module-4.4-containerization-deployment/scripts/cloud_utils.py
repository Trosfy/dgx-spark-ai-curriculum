"""
Cloud Deployment Utilities for ML Models

This module provides utilities for deploying ML models to:
- AWS SageMaker
- GCP Vertex AI
- Azure ML (placeholder)

Includes cost estimation and platform comparison tools.

Example usage:
    from cloud_utils import SageMakerDeployer, VertexAIDeployer

    # Deploy to AWS SageMaker
    sagemaker = SageMakerDeployer(region="us-west-2")
    endpoint = sagemaker.deploy_huggingface_model(
        model_id="Qwen/Qwen3-8B-Instruct",
        instance_type="ml.g5.xlarge"
    )

    # Deploy to GCP Vertex AI
    vertex = VertexAIDeployer(project="my-project", region="us-central1")
    endpoint = vertex.deploy_model(
        model_path="gs://bucket/model",
        machine_type="n1-standard-4"
    )
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class CloudEndpoint:
    """Represents a deployed model endpoint."""

    name: str
    platform: str  # "sagemaker", "vertex", "azure"
    endpoint_url: str
    instance_type: str
    instance_count: int
    status: str
    created_at: datetime
    cost_per_hour: float
    region: str
    model_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "platform": self.platform,
            "endpoint_url": self.endpoint_url,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "cost_per_hour": self.cost_per_hour,
            "region": self.region,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }


@dataclass
class CloudCostEstimate:
    """Cost estimation for cloud deployment."""

    platform: str
    instance_type: str
    hourly_cost: float
    monthly_cost: float  # Assuming 24/7 operation
    cost_per_1k_requests: float
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"{self.platform} ({self.instance_type}): "
            f"${self.hourly_cost:.2f}/hr, ${self.monthly_cost:.0f}/month, "
            f"${self.cost_per_1k_requests:.4f}/1k requests"
        )


class CloudDeployer(ABC):
    """Abstract base class for cloud deployers."""

    @abstractmethod
    def deploy_model(self, **kwargs) -> CloudEndpoint:
        """Deploy a model to the cloud."""
        pass

    @abstractmethod
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a deployed endpoint."""
        pass

    @abstractmethod
    def list_endpoints(self) -> List[CloudEndpoint]:
        """List all deployed endpoints."""
        pass

    @abstractmethod
    def invoke_endpoint(self, endpoint_name: str, payload: Dict) -> Dict:
        """Invoke a deployed endpoint."""
        pass


class SageMakerDeployer(CloudDeployer):
    """
    AWS SageMaker deployment manager.

    Handles:
    - HuggingFace model deployment
    - Custom container deployment
    - Auto-scaling configuration
    - Endpoint management

    Example:
        >>> deployer = SageMakerDeployer(region="us-west-2")
        >>> endpoint = deployer.deploy_huggingface_model(
        ...     model_id="Qwen/Qwen3-8B-Instruct",
        ...     instance_type="ml.g5.xlarge"
        ... )
        >>> response = deployer.invoke_endpoint(
        ...     endpoint.name,
        ...     {"inputs": "Hello, how are you?"}
        ... )
    """

    # SageMaker instance pricing (approximate, us-west-2)
    INSTANCE_PRICING = {
        # GPU instances
        "ml.g5.xlarge": 1.006,      # 1x A10G
        "ml.g5.2xlarge": 1.515,     # 1x A10G
        "ml.g5.4xlarge": 2.533,     # 1x A10G
        "ml.g5.12xlarge": 7.598,    # 4x A10G
        "ml.g5.48xlarge": 20.262,   # 8x A10G
        "ml.p4d.24xlarge": 32.773,  # 8x A100
        "ml.p4de.24xlarge": 40.966, # 8x A100 80GB
        "ml.p5.48xlarge": 98.322,   # 8x H100
        # CPU instances
        "ml.m5.xlarge": 0.230,
        "ml.m5.2xlarge": 0.461,
        "ml.c5.xlarge": 0.204,
        "ml.c5.2xlarge": 0.408,
    }

    def __init__(
        self,
        region: str = "us-west-2",
        role_arn: Optional[str] = None,
    ):
        """
        Initialize SageMaker deployer.

        Args:
            region: AWS region
            role_arn: SageMaker execution role ARN
        """
        self.region = region
        self.role_arn = role_arn or os.environ.get("SAGEMAKER_ROLE_ARN")
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required AWS dependencies are available."""
        self._boto3_available = False
        self._sagemaker_available = False

        try:
            import boto3
            self._boto3_available = True
        except ImportError:
            pass

        try:
            import sagemaker
            self._sagemaker_available = True
        except ImportError:
            pass

    def _get_client(self):
        """Get SageMaker client."""
        if not self._boto3_available:
            raise ImportError("boto3 is required. Install with: pip install boto3")
        import boto3
        return boto3.client("sagemaker", region_name=self.region)

    def _get_session(self):
        """Get SageMaker session."""
        if not self._sagemaker_available:
            raise ImportError("sagemaker is required. Install with: pip install sagemaker")
        import sagemaker
        return sagemaker.Session()

    def deploy_huggingface_model(
        self,
        model_id: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        endpoint_name: Optional[str] = None,
        transformers_version: str = "4.37",
        pytorch_version: str = "2.1",
        quantization: Optional[str] = None,  # "bitsandbytes8", "bitsandbytes4", "gptq"
        environment: Optional[Dict[str, str]] = None,
    ) -> CloudEndpoint:
        """
        Deploy a HuggingFace model to SageMaker.

        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-8B-Instruct")
            instance_type: SageMaker instance type
            instance_count: Number of instances
            endpoint_name: Custom endpoint name
            transformers_version: Transformers library version
            pytorch_version: PyTorch version
            quantization: Quantization method
            environment: Additional environment variables

        Returns:
            CloudEndpoint object
        """
        if not self._sagemaker_available:
            return self._simulate_deploy(
                model_id, instance_type, instance_count, "sagemaker"
            )

        from sagemaker.huggingface import HuggingFaceModel

        # Generate endpoint name if not provided
        endpoint_name = endpoint_name or f"hf-{model_id.split('/')[-1][:20]}-{int(time.time())}"

        # Build environment
        env = {
            "HF_MODEL_ID": model_id,
            "SM_NUM_GPUS": "1",  # Adjust based on instance
            "MAX_INPUT_LENGTH": "2048",
            "MAX_TOTAL_TOKENS": "4096",
        }

        if quantization:
            if quantization == "bitsandbytes8":
                env["HF_MODEL_QUANTIZE"] = "bitsandbytes"
            elif quantization == "bitsandbytes4":
                env["HF_MODEL_QUANTIZE"] = "bitsandbytes-nf4"
            elif quantization == "gptq":
                env["HF_MODEL_QUANTIZE"] = "gptq"

        if environment:
            env.update(environment)

        # Create model
        model = HuggingFaceModel(
            role=self.role_arn,
            transformers_version=transformers_version,
            pytorch_version=pytorch_version,
            py_version="py310",
            env=env,
        )

        # Deploy
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

        # Get cost info
        hourly_cost = self.INSTANCE_PRICING.get(instance_type, 0) * instance_count

        return CloudEndpoint(
            name=endpoint_name,
            platform="sagemaker",
            endpoint_url=f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
            instance_type=instance_type,
            instance_count=instance_count,
            status="InService",
            created_at=datetime.now(),
            cost_per_hour=hourly_cost,
            region=self.region,
            model_id=model_id,
        )

    def deploy_custom_container(
        self,
        image_uri: str,
        model_data: str,  # S3 URI to model.tar.gz
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        endpoint_name: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> CloudEndpoint:
        """
        Deploy a custom container to SageMaker.

        Args:
            image_uri: ECR image URI
            model_data: S3 URI to model artifacts
            instance_type: Instance type
            instance_count: Number of instances
            endpoint_name: Custom endpoint name
            environment: Environment variables
        """
        if not self._sagemaker_available:
            return self._simulate_deploy(
                image_uri, instance_type, instance_count, "sagemaker"
            )

        from sagemaker.model import Model

        endpoint_name = endpoint_name or f"custom-{int(time.time())}"

        model = Model(
            image_uri=image_uri,
            model_data=model_data,
            role=self.role_arn,
            env=environment or {},
        )

        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

        hourly_cost = self.INSTANCE_PRICING.get(instance_type, 0) * instance_count

        return CloudEndpoint(
            name=endpoint_name,
            platform="sagemaker",
            endpoint_url=f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
            instance_type=instance_type,
            instance_count=instance_count,
            status="InService",
            created_at=datetime.now(),
            cost_per_hour=hourly_cost,
            region=self.region,
        )

    def _simulate_deploy(
        self,
        model_id: str,
        instance_type: str,
        instance_count: int,
        platform: str
    ) -> CloudEndpoint:
        """Simulate deployment for demo/testing purposes."""
        endpoint_name = f"sim-{model_id.split('/')[-1][:15]}-{int(time.time())}"
        hourly_cost = self.INSTANCE_PRICING.get(instance_type, 1.0) * instance_count

        return CloudEndpoint(
            name=endpoint_name,
            platform=platform,
            endpoint_url=f"https://simulated.endpoint/{endpoint_name}",
            instance_type=instance_type,
            instance_count=instance_count,
            status="Simulated",
            created_at=datetime.now(),
            cost_per_hour=hourly_cost,
            region=self.region,
            model_id=model_id,
            metadata={"note": "Simulated endpoint - install sagemaker SDK for real deployment"}
        )

    def configure_autoscaling(
        self,
        endpoint_name: str,
        min_capacity: int = 1,
        max_capacity: int = 5,
        target_invocations: int = 70,  # Per instance per minute
    ) -> Dict[str, Any]:
        """
        Configure auto-scaling for an endpoint.

        Args:
            endpoint_name: Name of the endpoint
            min_capacity: Minimum instances
            max_capacity: Maximum instances
            target_invocations: Target invocations per instance per minute
        """
        if not self._boto3_available:
            return {
                "status": "simulated",
                "min_capacity": min_capacity,
                "max_capacity": max_capacity,
            }

        import boto3

        autoscaling = boto3.client("application-autoscaling", region_name=self.region)

        # Register scalable target
        resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

        autoscaling.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity,
        )

        # Create scaling policy
        autoscaling.put_scaling_policy(
            PolicyName=f"{endpoint_name}-scaling-policy",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": target_invocations,
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
                },
                "ScaleInCooldown": 600,  # 10 minutes
                "ScaleOutCooldown": 300,  # 5 minutes
            },
        )

        return {
            "status": "configured",
            "min_capacity": min_capacity,
            "max_capacity": max_capacity,
            "target_invocations": target_invocations,
        }

    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a SageMaker endpoint."""
        if not self._boto3_available:
            print(f"[Simulated] Deleted endpoint: {endpoint_name}")
            return True

        client = self._get_client()

        try:
            # Delete endpoint
            client.delete_endpoint(EndpointName=endpoint_name)

            # Delete endpoint config
            client.delete_endpoint_config(EndpointConfigName=endpoint_name)

            print(f"Deleted endpoint: {endpoint_name}")
            return True
        except Exception as e:
            print(f"Failed to delete endpoint: {e}")
            return False

    def list_endpoints(self) -> List[CloudEndpoint]:
        """List all SageMaker endpoints."""
        if not self._boto3_available:
            return []

        client = self._get_client()
        endpoints = []

        try:
            response = client.list_endpoints()
            for ep in response["Endpoints"]:
                endpoints.append(CloudEndpoint(
                    name=ep["EndpointName"],
                    platform="sagemaker",
                    endpoint_url=f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{ep['EndpointName']}/invocations",
                    instance_type="",  # Would need describe_endpoint for this
                    instance_count=0,
                    status=ep["EndpointStatus"],
                    created_at=ep["CreationTime"],
                    cost_per_hour=0,
                    region=self.region,
                ))
        except Exception as e:
            print(f"Failed to list endpoints: {e}")

        return endpoints

    def invoke_endpoint(self, endpoint_name: str, payload: Dict) -> Dict:
        """Invoke a SageMaker endpoint."""
        if not self._boto3_available:
            return {"error": "boto3 not available", "simulated_response": "Hello!"}

        import boto3

        runtime = boto3.client("sagemaker-runtime", region_name=self.region)

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        return json.loads(response["Body"].read().decode())

    def deploy_model(self, **kwargs) -> CloudEndpoint:
        """Generic deploy method."""
        if "model_id" in kwargs:
            return self.deploy_huggingface_model(**kwargs)
        elif "image_uri" in kwargs:
            return self.deploy_custom_container(**kwargs)
        else:
            raise ValueError("Must provide either model_id or image_uri")


class VertexAIDeployer(CloudDeployer):
    """
    GCP Vertex AI deployment manager.

    Handles:
    - Model upload and deployment
    - Custom container deployment
    - Auto-scaling configuration
    - Endpoint management

    Example:
        >>> deployer = VertexAIDeployer(project="my-project", region="us-central1")
        >>> endpoint = deployer.deploy_model(
        ...     model_path="gs://bucket/model",
        ...     machine_type="n1-standard-4",
        ...     accelerator_type="NVIDIA_TESLA_T4"
        ... )
    """

    # Vertex AI pricing (approximate, us-central1)
    MACHINE_PRICING = {
        # CPU machines (per hour)
        "n1-standard-2": 0.095,
        "n1-standard-4": 0.190,
        "n1-standard-8": 0.380,
        "n1-standard-16": 0.760,
        # GPU pricing (additional, per hour)
        "NVIDIA_TESLA_T4": 0.35,
        "NVIDIA_TESLA_V100": 2.48,
        "NVIDIA_TESLA_A100": 2.93,
        "NVIDIA_A100_80GB": 3.67,
        "NVIDIA_L4": 0.70,
        "NVIDIA_H100_80GB": 10.00,  # Approximate
    }

    def __init__(
        self,
        project: str,
        region: str = "us-central1",
        staging_bucket: Optional[str] = None,
    ):
        """
        Initialize Vertex AI deployer.

        Args:
            project: GCP project ID
            region: GCP region
            staging_bucket: GCS bucket for staging
        """
        self.project = project
        self.region = region
        self.staging_bucket = staging_bucket
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required GCP dependencies are available."""
        self._aiplatform_available = False

        try:
            from google.cloud import aiplatform
            self._aiplatform_available = True
        except ImportError:
            pass

    def _init_aiplatform(self):
        """Initialize AI Platform SDK."""
        if not self._aiplatform_available:
            raise ImportError(
                "google-cloud-aiplatform is required. "
                "Install with: pip install google-cloud-aiplatform"
            )

        from google.cloud import aiplatform
        aiplatform.init(
            project=self.project,
            location=self.region,
            staging_bucket=self.staging_bucket,
        )
        return aiplatform

    def upload_model(
        self,
        display_name: str,
        artifact_uri: str,
        serving_container_image_uri: str,
        serving_container_predict_route: str = "/predict",
        serving_container_health_route: str = "/health",
        environment_variables: Optional[Dict[str, str]] = None,
    ):
        """
        Upload a model to Vertex AI Model Registry.

        Args:
            display_name: Model display name
            artifact_uri: GCS path to model artifacts
            serving_container_image_uri: Container image for serving
            serving_container_predict_route: Prediction route
            serving_container_health_route: Health check route
            environment_variables: Container environment variables
        """
        if not self._aiplatform_available:
            return {
                "name": display_name,
                "status": "simulated",
                "artifact_uri": artifact_uri,
            }

        aiplatform = self._init_aiplatform()

        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
            serving_container_predict_route=serving_container_predict_route,
            serving_container_health_route=serving_container_health_route,
            serving_container_environment_variables=environment_variables or {},
        )

        return model

    def deploy_model(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        serving_container_image_uri: Optional[str] = None,
        machine_type: str = "n1-standard-4",
        accelerator_type: Optional[str] = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        min_replica_count: int = 1,
        max_replica_count: int = 3,
        endpoint_name: Optional[str] = None,
    ) -> CloudEndpoint:
        """
        Deploy a model to Vertex AI endpoint.

        Args:
            model_path: GCS path to model artifacts (for new model)
            model_name: Existing model resource name
            serving_container_image_uri: Container image for serving
            machine_type: GCE machine type
            accelerator_type: GPU type
            accelerator_count: Number of GPUs
            min_replica_count: Minimum replicas
            max_replica_count: Maximum replicas
            endpoint_name: Custom endpoint name
        """
        if not self._aiplatform_available:
            return self._simulate_deploy(
                model_path or model_name or "unknown",
                machine_type,
                accelerator_type,
                min_replica_count,
            )

        aiplatform = self._init_aiplatform()

        # Get or create model
        if model_name:
            model = aiplatform.Model(model_name)
        elif model_path:
            model = self.upload_model(
                display_name=endpoint_name or f"model-{int(time.time())}",
                artifact_uri=model_path,
                serving_container_image_uri=serving_container_image_uri,
            )
        else:
            raise ValueError("Must provide either model_path or model_name")

        # Deploy to endpoint
        endpoint = model.deploy(
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
        )

        # Calculate cost
        hourly_cost = self.MACHINE_PRICING.get(machine_type, 0)
        if accelerator_type:
            hourly_cost += self.MACHINE_PRICING.get(accelerator_type, 0) * accelerator_count
        hourly_cost *= min_replica_count

        return CloudEndpoint(
            name=endpoint.display_name,
            platform="vertex",
            endpoint_url=endpoint.resource_name,
            instance_type=f"{machine_type}+{accelerator_type}",
            instance_count=min_replica_count,
            status="DEPLOYED",
            created_at=datetime.now(),
            cost_per_hour=hourly_cost,
            region=self.region,
            metadata={"max_replicas": max_replica_count},
        )

    def _simulate_deploy(
        self,
        model_id: str,
        machine_type: str,
        accelerator_type: Optional[str],
        replica_count: int,
    ) -> CloudEndpoint:
        """Simulate deployment for demo/testing purposes."""
        endpoint_name = f"sim-vertex-{int(time.time())}"

        hourly_cost = self.MACHINE_PRICING.get(machine_type, 0.19)
        if accelerator_type:
            hourly_cost += self.MACHINE_PRICING.get(accelerator_type, 0.35)
        hourly_cost *= replica_count

        return CloudEndpoint(
            name=endpoint_name,
            platform="vertex",
            endpoint_url=f"projects/{self.project}/locations/{self.region}/endpoints/{endpoint_name}",
            instance_type=f"{machine_type}+{accelerator_type or 'cpu'}",
            instance_count=replica_count,
            status="Simulated",
            created_at=datetime.now(),
            cost_per_hour=hourly_cost,
            region=self.region,
            model_id=model_id,
            metadata={"note": "Simulated endpoint - install google-cloud-aiplatform for real deployment"},
        )

    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a Vertex AI endpoint."""
        if not self._aiplatform_available:
            print(f"[Simulated] Deleted endpoint: {endpoint_name}")
            return True

        try:
            aiplatform = self._init_aiplatform()
            endpoint = aiplatform.Endpoint(endpoint_name)
            endpoint.delete(force=True)
            print(f"Deleted endpoint: {endpoint_name}")
            return True
        except Exception as e:
            print(f"Failed to delete endpoint: {e}")
            return False

    def list_endpoints(self) -> List[CloudEndpoint]:
        """List all Vertex AI endpoints."""
        if not self._aiplatform_available:
            return []

        try:
            aiplatform = self._init_aiplatform()
            endpoints = aiplatform.Endpoint.list()

            return [
                CloudEndpoint(
                    name=ep.display_name,
                    platform="vertex",
                    endpoint_url=ep.resource_name,
                    instance_type="",
                    instance_count=0,
                    status="DEPLOYED",
                    created_at=ep.create_time,
                    cost_per_hour=0,
                    region=self.region,
                )
                for ep in endpoints
            ]
        except Exception as e:
            print(f"Failed to list endpoints: {e}")
            return []

    def invoke_endpoint(self, endpoint_name: str, payload: Dict) -> Dict:
        """Invoke a Vertex AI endpoint."""
        if not self._aiplatform_available:
            return {"error": "aiplatform not available", "simulated_response": "Hello!"}

        aiplatform = self._init_aiplatform()
        endpoint = aiplatform.Endpoint(endpoint_name)

        instances = payload.get("instances", [payload])
        response = endpoint.predict(instances=instances)

        return {"predictions": response.predictions}


def estimate_cloud_costs(
    model_size_gb: float,
    expected_requests_per_day: int,
    avg_latency_ms: float = 100,
) -> List[CloudCostEstimate]:
    """
    Estimate costs across cloud platforms.

    Args:
        model_size_gb: Model size in GB
        expected_requests_per_day: Expected daily request volume
        avg_latency_ms: Average request latency

    Returns:
        List of cost estimates for different configurations
    """
    estimates = []

    # Calculate required instance types based on model size
    # Rule of thumb: Need ~2x model size for GPU memory

    # SageMaker estimates
    if model_size_gb <= 16:
        sm_instance = "ml.g5.xlarge"
        sm_price = 1.006
    elif model_size_gb <= 24:
        sm_instance = "ml.g5.2xlarge"
        sm_price = 1.515
    elif model_size_gb <= 48:
        sm_instance = "ml.g5.4xlarge"
        sm_price = 2.533
    else:
        sm_instance = "ml.g5.12xlarge"
        sm_price = 7.598

    # Calculate throughput (requests per hour)
    requests_per_second = 1000 / avg_latency_ms
    requests_per_hour = requests_per_second * 3600

    # Cost per 1k requests
    cost_per_1k = (sm_price / requests_per_hour) * 1000

    estimates.append(CloudCostEstimate(
        platform="AWS SageMaker",
        instance_type=sm_instance,
        hourly_cost=sm_price,
        monthly_cost=sm_price * 24 * 30,
        cost_per_1k_requests=cost_per_1k,
        notes="1x A10G GPU, good for inference",
    ))

    # Vertex AI estimates
    if model_size_gb <= 16:
        vertex_machine = "n1-standard-4"
        vertex_gpu = "NVIDIA_TESLA_T4"
        vertex_price = 0.19 + 0.35
    elif model_size_gb <= 40:
        vertex_machine = "n1-standard-8"
        vertex_gpu = "NVIDIA_L4"
        vertex_price = 0.38 + 0.70
    else:
        vertex_machine = "n1-standard-8"
        vertex_gpu = "NVIDIA_TESLA_A100"
        vertex_price = 0.38 + 2.93

    cost_per_1k = (vertex_price / requests_per_hour) * 1000

    estimates.append(CloudCostEstimate(
        platform="GCP Vertex AI",
        instance_type=f"{vertex_machine} + {vertex_gpu}",
        hourly_cost=vertex_price,
        monthly_cost=vertex_price * 24 * 30,
        cost_per_1k_requests=cost_per_1k,
        notes="Flexible GPU options",
    ))

    # Spot instance estimates (60-80% cheaper)
    spot_discount = 0.3
    estimates.append(CloudCostEstimate(
        platform="AWS SageMaker (Spot)",
        instance_type=sm_instance,
        hourly_cost=sm_price * spot_discount,
        monthly_cost=sm_price * spot_discount * 24 * 30,
        cost_per_1k_requests=cost_per_1k * spot_discount,
        notes="70% cheaper but may be interrupted",
    ))

    return estimates


def compare_platforms(
    model_id: str,
    instance_configs: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Compare cloud platforms for ML deployment.

    Args:
        model_id: Model identifier
        instance_configs: Platform-specific instance types

    Returns:
        Comparison dictionary
    """
    configs = instance_configs or {
        "sagemaker": "ml.g5.xlarge",
        "vertex": "n1-standard-4 + NVIDIA_TESLA_T4",
    }

    comparison = {
        "model": model_id,
        "platforms": {
            "sagemaker": {
                "instance": configs["sagemaker"],
                "hourly_cost": SageMakerDeployer.INSTANCE_PRICING.get(
                    configs["sagemaker"], 1.0
                ),
                "setup_complexity": "Medium",
                "auto_scaling": "Yes (Application Auto Scaling)",
                "cold_start": "~2-5 minutes",
                "max_payload_size": "6 MB",
                "auth": "IAM + VPC",
                "pros": [
                    "Tight AWS integration",
                    "Managed infrastructure",
                    "Good monitoring (CloudWatch)",
                ],
                "cons": [
                    "AWS lock-in",
                    "Complex IAM setup",
                    "Higher costs than self-managed",
                ],
            },
            "vertex": {
                "instance": configs["vertex"],
                "hourly_cost": 0.54,  # n1-standard-4 + T4
                "setup_complexity": "Medium",
                "auto_scaling": "Yes (built-in)",
                "cold_start": "~2-5 minutes",
                "max_payload_size": "1.5 MB",
                "auth": "IAM + VPC",
                "pros": [
                    "Good BigQuery integration",
                    "Simpler pricing",
                    "Good ML tooling (AutoML)",
                ],
                "cons": [
                    "GCP lock-in",
                    "Smaller GPU selection",
                    "Less mature than SageMaker",
                ],
            },
        },
        "recommendation": (
            "SageMaker for AWS shops with complex ML pipelines. "
            "Vertex AI for GCP shops or simpler deployments. "
            "Consider self-managed (EKS/GKE + vLLM) for cost optimization."
        ),
    }

    return comparison


def create_deployment_checklist() -> List[Dict[str, Any]]:
    """
    Create a deployment readiness checklist.

    Returns:
        List of checklist items
    """
    return [
        {
            "category": "Model Preparation",
            "items": [
                "Model tested locally with representative data",
                "Model size fits in target GPU memory",
                "Model serialized in compatible format (safetensors/GGUF)",
                "Input/output schema documented",
                "Model versioned and tagged",
            ],
        },
        {
            "category": "Container",
            "items": [
                "Dockerfile tested and builds successfully",
                "Container runs locally with GPU",
                "Health check endpoint implemented (/health)",
                "Prediction endpoint tested (/predict)",
                "Container pushed to registry (ECR/GCR)",
            ],
        },
        {
            "category": "Cloud Infrastructure",
            "items": [
                "IAM roles/service accounts created",
                "VPC and security groups configured",
                "Model artifacts uploaded to storage (S3/GCS)",
                "Endpoint quota checked and requested if needed",
                "Budget alerts configured",
            ],
        },
        {
            "category": "Deployment",
            "items": [
                "Endpoint deployed successfully",
                "Health check passing",
                "Test inference successful",
                "Auto-scaling configured",
                "Monitoring dashboard created",
            ],
        },
        {
            "category": "Production Readiness",
            "items": [
                "Load testing completed",
                "Latency within SLA",
                "Error handling tested",
                "Rollback procedure documented",
                "On-call rotation established",
            ],
        },
    ]


if __name__ == "__main__":
    # Example usage
    print("=== Cloud Cost Estimation ===")
    estimates = estimate_cloud_costs(
        model_size_gb=14.0,  # 7B model at fp16
        expected_requests_per_day=10000,
        avg_latency_ms=150,
    )

    for est in estimates:
        print(est)

    print("\n=== Platform Comparison ===")
    comparison = compare_platforms("Qwen/Qwen3-8B-Instruct")
    print(json.dumps(comparison, indent=2, default=str))

    print("\n=== Deployment Checklist ===")
    checklist = create_deployment_checklist()
    for category in checklist:
        print(f"\n{category['category']}:")
        for item in category["items"]:
            print(f"  [ ] {item}")
