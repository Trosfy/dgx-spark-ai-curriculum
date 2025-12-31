# Data Files for Module 4.4: Containerization & Deployment

This directory contains templates, configurations, and sample files for containerization labs.

## Dockerfile Templates

### Inference Server Template

```dockerfile
# templates/Dockerfile.inference
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ /app/

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "serve.py"]
```

### Multi-stage Build Template

```dockerfile
# templates/Dockerfile.multistage
# Build stage
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Copy only what we need
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
COPY app/ /app/

EXPOSE 8000
CMD ["python", "serve.py"]
```

## Docker Compose Templates

### Basic ML Stack

```yaml
# templates/docker-compose.basic.yml
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
      - MODEL_PATH=/models/model.pt
```

### Full ML Stack with Monitoring

```yaml
# templates/docker-compose.full.yml
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
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  chroma_data:
  prometheus_data:
  grafana_data:
```

## Kubernetes Manifests

### Deployment Template

```yaml
# templates/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
  labels:
    app: inference
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
        image: ${IMAGE}:${TAG}
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            cpu: "4"
            memory: "8Gi"
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/model.pt"
        volumeMounts:
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### Service Template

```yaml
# templates/k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: inference
```

### HorizontalPodAutoscaler Template

```yaml
# templates/k8s/hpa.yaml
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
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Cloud Deployment Templates

### AWS SageMaker

```python
# templates/sagemaker/deploy.py
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

def deploy_to_sagemaker(
    model_data: str,
    endpoint_name: str,
    instance_type: str = "ml.g5.xlarge"
):
    """Deploy model to SageMaker endpoint."""

    huggingface_model = HuggingFaceModel(
        model_data=model_data,
        role=sagemaker.get_execution_role(),
        transformers_version="4.41",  # Check SageMaker docs for latest
        pytorch_version="2.3",
        py_version="py311",  # Use latest available; check SageMaker docs for py313 support
    )

    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    return predictor
```

### GCP Vertex AI

```python
# templates/vertex/deploy.py
from google.cloud import aiplatform

def deploy_to_vertex(
    project_id: str,
    model_name: str,
    artifact_uri: str,
    container_uri: str,
    machine_type: str = "n1-standard-4",
    gpu_type: str = "NVIDIA_TESLA_T4"
):
    """Deploy model to Vertex AI endpoint."""

    aiplatform.init(project=project_id, location="us-central1")

    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=container_uri,
    )

    endpoint = model.deploy(
        machine_type=machine_type,
        accelerator_type=gpu_type,
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=3,
    )

    return endpoint
```

## Gradio Templates

### Basic Chat Interface

```python
# templates/gradio/chat_basic.py
import gradio as gr

def chat(message, history):
    # Your model inference here
    return f"Response to: {message}"

demo = gr.ChatInterface(
    fn=chat,
    title="My LLM",
    description="Chat with the model",
)

if __name__ == "__main__":
    demo.launch()
```

### Advanced RAG Interface

```python
# templates/gradio/rag_advanced.py
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Chat Application")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload Documents", file_count="multiple")
            index_btn = gr.Button("Index Documents")
            index_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(label="Message", placeholder="Ask about your documents...")
            clear = gr.Button("Clear")

    with gr.Accordion("Sources", open=False):
        sources = gr.Markdown()

    # Event handlers
    def index_docs(files):
        # Index documents
        return f"Indexed {len(files)} documents"

    def respond(message, history):
        # RAG response
        return history + [[message, "Response"]], "Source: doc.pdf"

    index_btn.click(index_docs, [file_upload], [index_status])
    msg.submit(respond, [msg, chatbot], [chatbot, sources])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
```

## Streamlit Templates

### Multi-page App Structure

```
templates/streamlit/
├── app.py
├── pages/
│   ├── 1_Chat.py
│   ├── 2_Metrics.py
│   └── 3_Compare.py
└── utils/
    └── model.py
```

### Main App Template

```python
# templates/streamlit/app.py
import streamlit as st

st.set_page_config(
    page_title="Model Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("Welcome to Model Dashboard")
st.markdown("""
Navigate using the sidebar:
- **Chat**: Interactive model playground
- **Metrics**: Performance monitoring
- **Compare**: Model comparison
""")
```

## Prometheus Configuration

```yaml
# templates/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'inference-server'
    static_configs:
      - targets: ['inference:8000']
    metrics_path: /metrics

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

## Sample requirements.txt

```
# templates/requirements.txt
fastapi>=0.109.0
uvicorn>=0.27.0
transformers>=4.46.0
torch>=2.5.0
accelerate>=0.28.0
prometheus-client>=0.19.0
```

## GitHub Actions Workflow

```yaml
# templates/.github/workflows/deploy.yml
name: Deploy Model

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: user/inference:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to K8s
        run: |
          kubectl set image deployment/inference-server \
            inference=user/inference:${{ github.sha }}
```

## Usage

1. Copy the appropriate template for your use case
2. Customize for your model and requirements
3. Test locally before deploying to cloud

All templates are designed to work with DGX Spark's NGC container base images.
