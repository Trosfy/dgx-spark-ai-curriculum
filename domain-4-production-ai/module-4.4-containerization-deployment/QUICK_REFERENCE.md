# Module 4.4: Containerization & Cloud Deployment - Quick Reference

## Essential Commands

### NGC Container Setup

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8000:8000 \
    -p 8501:8501 \
    -p 7860:7860 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Install Dependencies

```bash
pip install gradio>=4.0.0 streamlit>=1.30.0
pip install sagemaker>=2.200.0
pip install google-cloud-aiplatform>=1.40.0
```

---

## Docker Basics

### Build & Run

```bash
# Build image
docker build -t my-inference:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 my-inference:latest

# Run interactive
docker run --gpus all -it my-inference:latest /bin/bash

# Check running containers
docker ps

# Check logs
docker logs <container_id>
```

### Dockerfile Template (Multi-stage)

```dockerfile
# Build stage
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM nvcr.io/nvidia/pytorch:25.11-py3
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
COPY app/ /app/

HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "serve.py"]
```

### Image Size Optimization

| Technique | Savings |
|-----------|---------|
| Multi-stage builds | 40-60% |
| Remove build deps | 10-20% |
| Use slim base | 20-30% |
| Clear pip cache | 5-10% |

```dockerfile
# Clear caches in same layer
RUN pip install -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/*
```

---

## Docker Compose

### Basic ML Stack

```yaml
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

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  chroma_data:
```

### Compose Commands

```bash
# Start stack
docker compose up -d

# View logs
docker compose logs -f inference

# Stop stack
docker compose down

# Rebuild and restart
docker compose up -d --build
```

---

## AWS SageMaker

### Endpoint Deployment

```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

role = sagemaker.get_execution_role()

model = HuggingFaceModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    transformers_version="4.41",
    pytorch_version="2.3",
    py_version="py311",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="my-endpoint",
)

# Test
response = predictor.predict({
    "inputs": "What is AI?"
})
```

### Instance Types

| Instance | GPU | Memory | Cost/hr |
|----------|-----|--------|---------|
| ml.g5.xlarge | A10G | 24GB | ~$1.00 |
| ml.g5.2xlarge | A10G | 24GB | ~$1.50 |
| ml.g5.4xlarge | A10G | 24GB | ~$2.50 |
| ml.p4d.24xlarge | 8x A100 | 320GB | ~$32.00 |

### Auto-Scaling

```python
from sagemaker.autoscaling import AutoScaler

autoscaler = AutoScaler(predictor)
autoscaler.set_min_capacity(1)
autoscaler.set_max_capacity(5)
autoscaler.add_scaling_policy(
    policy_name="requests-scaling",
    metric_type="SageMakerVariantInvocationsPerInstance",
    target_value=100,
    scale_in_cooldown=300,
    scale_out_cooldown=60,
)
```

---

## GCP Vertex AI

### Endpoint Deployment

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

model = aiplatform.Model.upload(
    display_name="my-llm",
    artifact_uri="gs://bucket/model/",
    serving_container_image_uri="us-docker.pkg.dev/my-project/my-repo/inference:latest",
)

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

### Machine Types

| Machine + GPU | Memory | Cost/hr |
|---------------|--------|---------|
| n1-standard-4 + T4 | 16GB | ~$0.95 |
| n1-standard-8 + T4 | 16GB | ~$1.20 |
| n1-standard-4 + A100 | 40GB | ~$3.50 |
| a2-highgpu-1g | 40GB A100 | ~$3.75 |

---

## Kubernetes

### Deployment Manifest

```yaml
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
```

### Service & HPA

```yaml
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

### kubectl Commands

```bash
# Apply manifests
kubectl apply -f deployment.yaml

# Check pods
kubectl get pods

# Check logs
kubectl logs -f <pod-name>

# Scale manually
kubectl scale deployment inference-server --replicas=3

# Check HPA status
kubectl get hpa
```

---

## Gradio

### Quick Chat Interface

```python
import gradio as gr
import ollama

def chat(message, history):
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

demo = gr.ChatInterface(
    fn=chat,
    title="My LLM",
    examples=["Hello!", "Explain AI"],
    theme=gr.themes.Soft(),
)

demo.launch(share=True)
```

### Blocks API Layout

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# My App")

    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox()

        with gr.TabItem("Settings"):
            with gr.Row():
                model = gr.Dropdown(choices=["8b", "70b"])
                temp = gr.Slider(0, 1, 0.7)

    msg.submit(fn=chat, inputs=[msg, chatbot], outputs=[chatbot])

demo.launch()
```

### Deploy to Spaces

```yaml
# README.md header for Spaces
---
title: My Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
---
```

---

## Streamlit

### Basic App

```python
import streamlit as st
import ollama

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Model Dashboard")

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
```

### Multi-page Structure

```
my_app/
├── Home.py
├── pages/
│   ├── 1_Chat.py
│   ├── 2_Metrics.py
│   └── 3_Settings.py
└── .streamlit/
    └── config.toml
```

### Caching

```python
@st.cache_resource
def load_model():
    """Cache model loading (runs once per session)."""
    return ollama.Client()

@st.cache_data(ttl=3600)
def get_embeddings(text):
    """Cache embeddings for 1 hour."""
    return client.embeddings(model="nomic-embed-text", prompt=text)
```

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `--gpus all` missing | Add `--gpus all` to docker run |
| Port not exposed | Add `-p 8000:8000` and EXPOSE in Dockerfile |
| Docker image too large | Use multi-stage builds |
| SageMaker timeout | Increase `model_server_timeout` |
| K8s GPU not scheduled | Check nvidia-device-plugin daemonset |
| Gradio slow first load | Preload models at startup |
| Streamlit reruns everything | Use `@st.cache_resource` for models |

---

## Port Reference

| Service | Default Port |
|---------|--------------|
| FastAPI/Inference | 8000 |
| Gradio | 7860 |
| Streamlit | 8501 |
| MLflow | 5000 |
| Prometheus | 9090 |
| Grafana | 3000 |
| ChromaDB | 8000 |

---

## Quick Links

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/)
- [GCP Vertex AI](https://cloud.google.com/vertex-ai/docs)
- [Kubernetes](https://kubernetes.io/docs/)
- [Gradio](https://gradio.app/docs/)
- [Streamlit](https://docs.streamlit.io/)
