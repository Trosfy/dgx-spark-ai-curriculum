# Module 4.4: Containerization & Cloud Deployment

**Domain:** 4 - Production AI
**Duration:** Weeks 32-33 (12-15 hours)
**Prerequisites:** Module 4.3 (MLOps)
**Priority:** P0/P1 (Docker Critical, Cloud High)

---

## Overview

Your model works on your DGX Spark. Now how do you get it running in production? This module covers the container-to-cloud journey: Docker for reproducible environments, Kubernetes for orchestration, and cloud platforms (AWS SageMaker, GCP Vertex AI) for scalable deployment.

**Why This Matters:** 58% of organizations use Kubernetes for AI workloads. Containerization isn't optional for production AI - it's the foundation everything else builds on.

### The Kitchen Table Explanation

Think of Docker like a shipping container. Your model, its dependencies, and the exact environment that makes it work - all packed together. Whether you ship it to AWS, GCP, or your colleague's laptop, it works the same way. Kubernetes is the port that manages thousands of these containers, deciding where each one goes and making sure they keep running.

---

## Learning Outcomes

By the end of this module, you will be able to:

- Containerize ML applications with Docker
- Deploy models to cloud platforms (AWS SageMaker, GCP Vertex AI)
- Use Kubernetes basics for ML deployment
- Build demo applications with Gradio and Streamlit

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.4.1 | Create Docker images for ML applications | Apply |
| 4.4.2 | Deploy models to AWS SageMaker or GCP Vertex AI | Apply |
| 4.4.3 | Use basic Kubernetes for ML deployments | Apply |
| 4.4.4 | Build interactive demos with Gradio | Apply |

---

## Topics

### 4.4.1 Docker for ML [P0 Critical]

- **Dockerfile Best Practices**
  - Base image selection (NGC containers)
  - Multi-stage builds
  - Layer optimization
  - Minimizing image size

- **GPU Container Configuration**
  - NVIDIA Container Toolkit
  - CUDA version matching
  - Driver compatibility

- **Docker Compose for ML Stacks**
  - Multi-service architectures
  - Inference server + monitoring + vector DB
  - Environment management

- **NGC Container Customization**
  - Extending NGC base images
  - Adding custom dependencies
  - Building production images

### 4.4.2 Cloud Deployment [P1 High]

- **AWS SageMaker**
  - SageMaker endpoints
  - Model packaging
  - Auto-scaling configuration
  - Cost optimization

- **GCP Vertex AI**
  - Vertex AI deployment
  - Custom containers
  - Prediction endpoints
  - Monitoring integration

- **Cost Optimization**
  - Spot/preemptible instances
  - Right-sizing instances
  - Serverless vs always-on

### 4.4.3 Kubernetes Basics

- **Core Concepts**
  - Pods, Deployments, Services
  - ConfigMaps and Secrets
  - Namespaces

- **GPU Scheduling**
  - NVIDIA device plugin
  - Resource requests/limits
  - Node affinity

- **Scaling**
  - Horizontal Pod Autoscaler
  - Replicas for inference

### 4.4.4 Demo Building

- **Gradio for ML Interfaces**
  - Input/output components
  - Custom themes
  - Hugging Face Spaces deployment

- **Streamlit for Data Apps**
  - Multi-page applications
  - Session state
  - Caching strategies

### 4.4.5 CI/CD for ML

- **GitHub Actions**
  - Model validation in CI
  - Automated testing
  - Container builds

- **Deployment Pipelines**
  - Staging environments
  - Canary deployments
  - A/B testing

---

## Labs

### Lab 4.4.1: Docker ML Image
**Time:** 2 hours

Create an optimized Docker image for your inference server.

**Instructions:**
1. Create Dockerfile extending NGC PyTorch container
2. Add your model dependencies
3. Implement multi-stage build for smaller image
4. Configure GPU support
5. Add health check endpoint
6. Build and test locally
7. Document image size optimization

**Deliverable:** Optimized Dockerfile with inference server

---

### Lab 4.4.2: Docker Compose Stack
**Time:** 2 hours

Create a complete ML stack with Docker Compose.

**Instructions:**
1. Define services: inference server, vector DB (ChromaDB), monitoring (Prometheus)
2. Configure networking between services
3. Set up volume mounts for persistence
4. Add GPU allocation for inference service
5. Create health checks for all services
6. Test the complete stack locally

**Deliverable:** Working Docker Compose stack with 3+ services

---

### Lab 4.4.3: AWS SageMaker Deployment
**Time:** 2 hours

Deploy your model to AWS SageMaker.

**Instructions:**
1. Package model for SageMaker
2. Create SageMaker endpoint configuration
3. Deploy to real-time endpoint
4. Test inference endpoint
5. Configure auto-scaling
6. Benchmark latency and throughput
7. Calculate cost per 1000 requests

**Deliverable:** Working SageMaker endpoint with cost analysis

---

### Lab 4.4.4: GCP Vertex AI Deployment
**Time:** 2 hours

Deploy the same model to Google Cloud.

**Instructions:**
1. Create custom container for Vertex AI
2. Upload model to Model Registry
3. Deploy to prediction endpoint
4. Test inference
5. Compare latency with SageMaker
6. Document differences between platforms

**Deliverable:** Working Vertex AI endpoint with platform comparison

---

### Lab 4.4.5: Kubernetes Deployment
**Time:** 2 hours

Deploy inference server to local Kubernetes.

**Instructions:**
1. Install minikube or kind with GPU support
2. Create Deployment manifest for inference server
3. Create Service for external access
4. Configure GPU resource requests
5. Test scaling with replicas
6. Monitor pod health
7. Implement Horizontal Pod Autoscaler

**Deliverable:** K8s deployment manifest with HPA configuration

---

### Lab 4.4.6: Gradio Demo
**Time:** 2 hours

Build an interactive demo for your fine-tuned model.

**Instructions:**
1. Create Gradio interface for chat
2. Add file upload for RAG
3. Display model reasoning/sources
4. Implement streaming responses
5. Add custom theme
6. Deploy to Hugging Face Spaces
7. Share public link

**Deliverable:** Gradio demo deployed to Hugging Face Spaces

---

### Lab 4.4.7: Streamlit Dashboard
**Time:** 2 hours

Create a comprehensive model dashboard.

**Instructions:**
1. Build multi-page Streamlit app
2. Page 1: Model playground (chat interface)
3. Page 2: Performance metrics
4. Page 3: Model comparison
5. Add session state for conversation history
6. Implement caching for slow operations
7. Deploy to Streamlit Cloud

**Deliverable:** Multi-page Streamlit dashboard

---

## Guidance

### DGX Spark Docker Setup

```bash
# Verify NVIDIA Container Toolkit
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi

# Build from NGC base
docker build -t my-inference:latest .
docker run --gpus all -p 8000:8000 my-inference:latest
```

### Optimized Dockerfile for Inference

```dockerfile
# Multi-stage build for smaller image
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS builder

# Install dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production image
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Copy only installed packages
COPY --from=builder /root/.local /root/.local

# Copy application
WORKDIR /app
COPY app/ /app/

# Set path
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run inference server
EXPOSE 8000
CMD ["python", "serve.py"]
```

### Docker Compose for ML Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  inference:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/models
    environment:
      - MODEL_PATH=/models/llama-8b
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  vectordb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  chroma_data:
```

### AWS SageMaker Deployment

```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Create HuggingFace Model
huggingface_model = HuggingFaceModel(
    model_data="s3://bucket/model.tar.gz",
    role=sagemaker.get_execution_role(),
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py311",  # Use latest available; check SageMaker docs for py313 support
)

# Deploy to endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="my-llm-endpoint",
)

# Test inference
response = predictor.predict({
    "inputs": "What is the capital of France?"
})
```

### GCP Vertex AI Deployment

```python
from google.cloud import aiplatform

# Initialize
aiplatform.init(project="my-project", location="us-central1")

# Upload model
model = aiplatform.Model.upload(
    display_name="my-llm",
    artifact_uri="gs://bucket/model/",
    serving_container_image_uri="us-docker.pkg.dev/my-project/my-repo/inference:latest",
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=3,
)

# Test
response = endpoint.predict(instances=[{"prompt": "Hello!"}])
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: inference
        image: my-inference:latest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "16Gi"
            cpu: "4"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: inference
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Gradio Demo

```python
import gradio as gr
import ollama

def chat(message, history):
    """Chat with the model."""
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    response = ollama.chat(
        model="llama3.1:8b",
        messages=messages,
        stream=True
    )

    partial = ""
    for chunk in response:
        partial += chunk["message"]["content"]
        yield partial

# Create interface
demo = gr.ChatInterface(
    fn=chat,
    title="My Fine-tuned LLM",
    description="Chat with a model fine-tuned on DGX Spark",
    examples=["Hello!", "Explain quantum computing", "Write a poem"],
    theme=gr.themes.Soft(),
)

demo.launch(share=True)
```

### Streamlit Dashboard

```python
import streamlit as st
import ollama

st.set_page_config(page_title="Model Dashboard", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Page", ["Chat", "Metrics", "Compare"])

if page == "Chat":
    st.title("Model Playground")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = ollama.chat(
                model="llama3.1:8b",
                messages=st.session_state.messages
            )
            reply = response["message"]["content"]
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

elif page == "Metrics":
    st.title("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", "120ms", "-10ms")
    col2.metric("Throughput", "50 req/s", "+5")
    col3.metric("GPU Util", "78%", "+3%")

elif page == "Compare":
    st.title("Model Comparison")
    # Add comparison logic
```

---

## Cloud Cost Comparison

| Platform | Instance | GPU | Cost/hr | Notes |
|----------|----------|-----|---------|-------|
| AWS SageMaker | ml.g5.xlarge | A10G | ~$1.00 | Good for inference |
| AWS SageMaker | ml.g5.2xlarge | A10G | ~$1.50 | More memory |
| GCP Vertex AI | n1-standard-4 + T4 | T4 | ~$0.95 | Cheaper GPU |
| GCP Vertex AI | n1-standard-8 + A100 | A100 | ~$3.50 | High performance |

**Tip:** Use spot/preemptible instances for 60-80% cost savings on non-critical workloads.

---

## Milestone Checklist

- [ ] Optimized Docker image created
- [ ] Docker Compose stack running locally
- [ ] AWS SageMaker deployment working
- [ ] GCP Vertex AI deployment working
- [ ] Kubernetes deployment with HPA
- [ ] Gradio demo on Hugging Face Spaces
- [ ] Streamlit dashboard created

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Docker GPU not detected | Install NVIDIA Container Toolkit, add `--gpus all` |
| Image too large | Use multi-stage builds, remove build dependencies |
| SageMaker timeout | Increase model_server_timeout in endpoint config |
| K8s GPU scheduling fails | Verify nvidia-device-plugin daemonset running |
| Gradio slow first load | Preload model in Spaces container |

---

## Next Steps

After completing this module:
1. Document your deployment configurations
2. Keep your Docker and K8s manifests for capstone
3. Proceed to [Module 4.5: Demo Building](../module-4.5-demo-building/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 4.3: MLOps](../module-4.3-mlops/) | **Module 4.4: Containerization** | [Module 4.5: Demo Building](../module-4.5-demo-building/) |

---

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/)
- [GCP Vertex AI](https://cloud.google.com/vertex-ai/docs)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [NGC Container Catalog](https://catalog.ngc.nvidia.com/)
