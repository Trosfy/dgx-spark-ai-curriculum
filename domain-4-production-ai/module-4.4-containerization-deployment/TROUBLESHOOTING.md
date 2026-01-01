# Module 4.4: Containerization & Cloud Deployment - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, check these:**

1. Docker running? `docker ps`
2. GPU accessible? `nvidia-smi`
3. Ports available? `netstat -tlnp | grep <port>`
4. Container logs? `docker logs <container_id>`

---

## Docker Errors

### Error: `docker: Error response from daemon: could not select device driver`

**Symptoms**:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

**Cause**: NVIDIA Container Toolkit not installed or configured.

**Solutions**:
```bash
# Solution 1: Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Solution 2: Verify installation
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

### Error: `Cannot connect to the Docker daemon`

**Symptoms**:
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solutions**:
```bash
# Solution 1: Start Docker service
sudo systemctl start docker

# Solution 2: Add user to docker group
sudo usermod -aG docker $USER
# Then log out and back in

# Solution 3: Check socket permissions
sudo chmod 666 /var/run/docker.sock
```

---

### Error: Docker image too large

**Symptoms**: Image size 10+ GB, slow builds

**Solutions**:
```dockerfile
# Solution 1: Use multi-stage builds
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM nvcr.io/nvidia/pytorch:25.11-py3
COPY --from=builder /root/.local /root/.local
# Much smaller final image

# Solution 2: Clear caches in same layer
RUN pip install -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

# Solution 3: Use .dockerignore
# Create .dockerignore file:
# __pycache__
# *.pyc
# .git
# models/
# data/
```

---

### Error: Port already in use

**Symptoms**:
```
Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use
```

**Solutions**:
```bash
# Solution 1: Find and kill process using port
lsof -i :8000
kill <PID>

# Solution 2: Use different port
docker run -p 8001:8000 my-image

# Solution 3: Stop all containers using the port
docker ps | grep 8000
docker stop <container_id>
```

---

### Error: Container exits immediately

**Symptoms**: Container starts and stops, status shows "Exited"

**Solutions**:
```bash
# Solution 1: Check logs
docker logs <container_id>

# Solution 2: Run interactively to debug
docker run -it my-image /bin/bash

# Solution 3: Keep container running for debugging
docker run -d my-image tail -f /dev/null

# Solution 4: Check if CMD/ENTRYPOINT is correct
docker inspect my-image | grep -A5 "Cmd"
```

---

## Docker Compose Errors

### Error: GPU not available in Compose

**Symptoms**: `torch.cuda.is_available()` returns `False` in container

**Solutions**:
```yaml
# Solution: Use deploy.resources for GPU
services:
  inference:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

### Error: Service dependency issues

**Symptoms**: Services can't communicate, connection refused

**Solutions**:
```yaml
# Solution 1: Use depends_on with health checks
services:
  app:
    depends_on:
      db:
        condition: service_healthy

  db:
    healthcheck:
      test: ["CMD", "pg_isready"]
      interval: 5s
      timeout: 5s
      retries: 5

# Solution 2: Use service names as hostnames
# In app code:
# db_host = "db"  # Not "localhost"
# url = "http://inference:8000"  # Not "http://localhost:8000"
```

---

### Error: Volumes not persisting data

**Symptoms**: Data lost after `docker compose down`

**Solutions**:
```yaml
# Solution 1: Use named volumes
volumes:
  model_data:
  chroma_data:

services:
  inference:
    volumes:
      - model_data:/models  # Named volume
      - ./local:/app/data   # Bind mount for dev

# Solution 2: Don't use -v flag with down
docker compose down      # Preserves volumes
docker compose down -v   # DELETES volumes
```

---

## AWS SageMaker Errors

### Error: `AccessDeniedException`

**Symptoms**:
```
botocore.exceptions.ClientError: An error occurred (AccessDeniedException)
```

**Solutions**:
```bash
# Solution 1: Check IAM role permissions
aws iam get-role --role-name SageMakerExecutionRole

# Solution 2: Attach required policies
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Solution 3: Check S3 bucket policy
aws s3api get-bucket-policy --bucket your-bucket
```

---

### Error: `ModelError` - Endpoint invocation failed

**Symptoms**:
```
ModelError: An error occurred (ModelError) when calling the InvokeEndpoint operation
```

**Solutions**:
```python
# Solution 1: Check endpoint logs
import boto3
client = boto3.client('logs')
# Look for /aws/sagemaker/Endpoints/<endpoint-name>

# Solution 2: Test model locally first
# Ensure model works before deploying

# Solution 3: Check model.tar.gz structure
# Required structure:
# model.tar.gz
# ├── code/
# │   ├── inference.py
# │   └── requirements.txt
# └── model/
#     └── pytorch_model.bin
```

---

### Error: SageMaker endpoint timeout

**Symptoms**: Endpoint returns timeout errors during inference

**Solutions**:
```python
# Solution 1: Increase timeout in endpoint config
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    model_server_timeout=300,  # Increase timeout
)

# Solution 2: Use async inference for long requests
from sagemaker.async_inference import AsyncInferenceConfig
async_config = AsyncInferenceConfig(
    output_path="s3://bucket/async-output/"
)
predictor = model.deploy(
    async_inference_config=async_config,
    ...
)
```

---

## GCP Vertex AI Errors

### Error: `PermissionDenied`

**Symptoms**:
```
google.api_core.exceptions.PermissionDenied: 403 Permission denied
```

**Solutions**:
```bash
# Solution 1: Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Solution 2: Grant IAM roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:you@email.com" \
    --role="roles/aiplatform.user"

# Solution 3: Check service account
gcloud auth list
gcloud config get-value account
```

---

### Error: Container image not found

**Symptoms**:
```
Container image not found in Container Registry
```

**Solutions**:
```bash
# Solution 1: Push to correct registry
docker tag my-image gcr.io/PROJECT_ID/my-image:latest
docker push gcr.io/PROJECT_ID/my-image:latest

# Solution 2: Use Artifact Registry (newer)
docker tag my-image us-docker.pkg.dev/PROJECT_ID/repo/my-image:latest
docker push us-docker.pkg.dev/PROJECT_ID/repo/my-image:latest

# Solution 3: Configure Docker auth
gcloud auth configure-docker
# or for Artifact Registry:
gcloud auth configure-docker us-docker.pkg.dev
```

---

## Kubernetes Errors

### Error: Pod stuck in `Pending` state

**Symptoms**: `kubectl get pods` shows Pending status

**Solutions**:
```bash
# Solution 1: Check why it's pending
kubectl describe pod <pod-name>

# Common causes:
# - Insufficient resources: Scale down or request less
# - No GPU available: Check nvidia-device-plugin

# Solution 2: Check GPU availability
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# Solution 3: Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

---

### Error: `ImagePullBackOff`

**Symptoms**: Pod can't pull container image

**Solutions**:
```bash
# Solution 1: Check image name
kubectl describe pod <pod-name> | grep Image

# Solution 2: Create image pull secret
kubectl create secret docker-registry regcred \
    --docker-server=<registry> \
    --docker-username=<user> \
    --docker-password=<password>

# Solution 3: Add secret to deployment
spec:
  imagePullSecrets:
  - name: regcred
  containers:
  - name: app
    image: private-registry/my-image
```

---

### Error: Service not accessible

**Symptoms**: Can't reach service from outside cluster

**Solutions**:
```bash
# Solution 1: Check service type
kubectl get svc
# Use LoadBalancer or NodePort for external access

# Solution 2: Check endpoints
kubectl get endpoints <service-name>
# Should show pod IPs

# Solution 3: Port forward for testing
kubectl port-forward svc/<service-name> 8000:80
```

---

## Gradio Errors

### Error: Gradio app slow on first request

**Symptoms**: 30+ second delay on first inference

**Solutions**:
```python
# Solution 1: Preload model at startup
import gradio as gr
import ollama

# Load model before creating interface
client = ollama.Client()
# Warm up with dummy request
client.generate(model="llama3.1:8b", prompt="Hello")

def chat(message, history):
    # Model already loaded
    ...

demo = gr.ChatInterface(fn=chat)
demo.launch()

# Solution 2: Use gr.Blocks with state
with gr.Blocks() as demo:
    # Load once
    model_state = gr.State(value=load_model())
```

---

### Error: `Error: Connection closed`

**Symptoms**: Gradio demo crashes mid-response

**Solutions**:
```python
# Solution 1: Add error handling
def safe_chat(message, history):
    try:
        return chat(message, history)
    except Exception as e:
        return f"Error: {str(e)}. Please try again."

# Solution 2: Increase queue timeout
demo.launch(max_threads=10)

# Solution 3: Use try/except with streaming
def stream_chat(message, history):
    try:
        for chunk in ollama.chat(..., stream=True):
            yield chunk["message"]["content"]
    except Exception as e:
        yield f"Error: {e}"
```

---

### Error: Gradio `share=True` not working

**Symptoms**: Can't get public share link

**Solutions**:
```python
# Solution 1: Check network/firewall
# Some networks block Gradio's share tunnel

# Solution 2: Use server_name for local network
demo.launch(server_name="0.0.0.0", server_port=7860)
# Access via http://<your-ip>:7860

# Solution 3: Deploy to Hugging Face Spaces instead
```

---

## Streamlit Errors

### Error: Streamlit reruns on every interaction

**Symptoms**: Model reloads, state resets

**Solutions**:
```python
# Solution 1: Use cache for models
@st.cache_resource
def load_model():
    return ollama.Client()

client = load_model()  # Only loads once

# Solution 2: Use session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Solution 3: Use cache_data for data
@st.cache_data(ttl=3600)
def fetch_data():
    return expensive_query()
```

---

### Error: Streamlit memory usage grows

**Symptoms**: App slows down, eventually crashes

**Solutions**:
```python
# Solution 1: Clear cache periodically
st.cache_data.clear()

# Solution 2: Limit session state size
if len(st.session_state.messages) > 100:
    st.session_state.messages = st.session_state.messages[-50:]

# Solution 3: Use ttl on caches
@st.cache_data(ttl=600)  # Expire after 10 minutes
def get_embeddings(text):
    ...
```

---

### Error: File upload fails

**Symptoms**: Large file uploads timeout or fail

**Solutions**:
```python
# Solution 1: Increase max upload size
# .streamlit/config.toml
[server]
maxUploadSize = 200  # MB

# Solution 2: Process in chunks
uploaded_file = st.file_uploader("Upload", type=["pdf"])
if uploaded_file:
    # Process in chunks instead of loading all at once
    for chunk in read_in_chunks(uploaded_file):
        process(chunk)
```

---

## Reset Procedures

### Full Docker Reset

```bash
# Stop all containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm $(docker ps -aq)

# Remove all images (careful!)
docker rmi $(docker images -q)

# Clean up
docker system prune -a --volumes
```

### Kubernetes Reset

```bash
# Delete all resources in namespace
kubectl delete all --all -n default

# Or reset minikube completely
minikube delete
minikube start --gpus all
```

### Gradio/Streamlit Reset

```bash
# Kill processes on ports
fuser -k 7860/tcp  # Gradio
fuser -k 8501/tcp  # Streamlit

# Or find and kill
lsof -i :7860 | awk 'NR>1 {print $2}' | xargs kill
```

---

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: Which cloud platform should I learn first?**

**A**: If you have no preference, **AWS SageMaker** has more examples and tutorials. Both SageMaker and Vertex AI are similar in capability.

| Choose AWS if | Choose GCP if |
|---------------|---------------|
| Company uses AWS | Company uses GCP |
| Need HuggingFace integration | Need TPU access |
| Want more tutorials | Prefer GCP console |

---

**Q: Do I need GPU instances for all deployments?**

**A**: No! Consider your use case:

| Use Case | Instance |
|----------|----------|
| Inference (small model) | CPU may be enough |
| Inference (LLM) | GPU required |
| Inference (quantized) | GPU (smaller) |
| Training | GPU required |

---

**Q: How do I reduce Docker image size?**

**A**: Use these techniques:

1. **Multi-stage builds**: Separate build from runtime
2. **Slim base images**: `python:3.11-slim` vs `python:3.11`
3. **Clean up in same layer**: `apt-get && rm -rf /var/lib/apt/lists/*`
4. **Use .dockerignore**: Exclude test files, docs, .git

```dockerfile
# Multi-stage example
FROM python:3.11 AS builder
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY app.py .
CMD ["python", "app.py"]
```

---

### Concepts

**Q: What's the difference between Docker Compose and Kubernetes?**

**A**:

| Docker Compose | Kubernetes |
|----------------|------------|
| Single machine | Multi-machine cluster |
| Development/testing | Production |
| Simple YAML | More complex YAML |
| No auto-healing | Auto-healing |
| Manual scaling | Auto-scaling |
| Quick to learn | Steeper learning curve |

**Use Docker Compose for development, Kubernetes for production.**

---

**Q: When should I use managed services vs Kubernetes?**

**A**:

| Use Managed (SageMaker/Vertex) | Use Kubernetes |
|--------------------------------|----------------|
| Small team, no DevOps | Have DevOps team |
| Want quick deployment | Need full control |
| Standard ML workflows | Custom requirements |
| Predictable workloads | Complex scaling needs |

---

**Q: How do I handle secrets in containers?**

**A**: Never put secrets in Dockerfiles!

| Method | When to Use |
|--------|-------------|
| Environment variables | Development |
| Docker secrets | Docker Swarm |
| K8s Secrets | Kubernetes |
| AWS Secrets Manager | AWS production |
| GCP Secret Manager | GCP production |
| HashiCorp Vault | Multi-cloud |

```yaml
# Kubernetes secret example
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
type: Opaque
data:
  HF_TOKEN: base64_encoded_token
```

---

### Troubleshooting

**Q: Docker build fails with "no space left on device"**

**A**: Clean up Docker resources:

```bash
# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Remove all unused data
docker system prune -a
```

---

**Q: Container runs but can't access GPU**

**A**: Check these:

1. **NVIDIA Container Toolkit installed**:
   ```bash
   docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

2. **Use `--gpus all` flag**:
   ```bash
   docker run --gpus all myimage
   ```

3. **Check driver compatibility**:
   ```bash
   nvidia-smi  # Check driver version
   ```

---

**Q: SageMaker endpoint times out**

**A**: Common causes and fixes:

1. **Model loading too slow**: Increase `model_server_timeout` in endpoint config
2. **Container unhealthy**: Add health check endpoint
3. **Memory insufficient**: Use larger instance type
4. **Cold start**: Use provisioned concurrency

---

**Q: Kubernetes pods stuck in Pending**

**A**: Check resources:

```bash
kubectl describe pod <pod-name>
# Look for "Events" section

# Common causes:
# - Insufficient CPU/memory: Use smaller requests
# - No GPU available: Check nvidia-device-plugin
# - Image pull failed: Check image name and credentials
```

---

### Beyond the Basics

**Q: How do I set up auto-scaling?**

**A**: Depends on platform:

**SageMaker**:
```python
client.put_scaling_policy(
    PolicyName='scale-on-invocations',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 1000.0,  # Invocations per minute
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

**Kubernetes**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

**Q: How do I do blue-green deployments?**

**A**: Basic pattern:

1. Deploy new version alongside old (blue + green)
2. Test new version
3. Switch traffic to new version
4. Keep old version for rollback
5. Delete old version after confidence

**SageMaker**: Use endpoint update with `RetainAllVariantProperties`
**Kubernetes**: Use separate Deployments with Service selector switch

---

**Q: How do I monitor costs?**

**A**:

| Platform | Tool |
|----------|------|
| AWS | AWS Cost Explorer, Budgets |
| GCP | Cloud Billing reports |
| K8s | Kubecost, OpenCost |

**Tips**:
- Set budget alerts
- Use spot/preemptible for non-critical
- Right-size instances
- Auto-scale down during low traffic

---

## Still Stuck?

1. **Check container logs**: `docker logs <id>` or `kubectl logs <pod>`
2. **Check cloud console**: AWS CloudWatch, GCP Cloud Logging
3. **Search error message**: Stack Overflow, GitHub Issues
4. **Include in reports**: Full error, Dockerfile, docker-compose.yml, manifest
