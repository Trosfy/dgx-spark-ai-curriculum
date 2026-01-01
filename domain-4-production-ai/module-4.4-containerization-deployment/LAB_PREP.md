# Module 4.4: Containerization & Cloud Deployment - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 4.4.1 Docker ML Image | 15 min | 2h | 2.25h |
| 4.4.2 Docker Compose Stack | 10 min | 2h | 2.2h |
| 4.4.3 AWS SageMaker | 20 min | 2h | 2.3h |
| 4.4.4 GCP Vertex AI | 20 min | 2h | 2.3h |
| 4.4.5 Kubernetes | 15 min | 2h | 2.25h |
| 4.4.6 Gradio Demo | 10 min | 2h | 2.2h |
| 4.4.7 Streamlit Dashboard | 10 min | 2h | 2.2h |

---

## Required Tools

### Docker & Container Tools

```bash
# Verify Docker installation
docker --version
# Expected: Docker version 24.x or higher

# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
# Should show your GPU info
```

### Cloud CLIs (Labs 4.4.3-4.4.4)

```bash
# AWS CLI
pip install awscli boto3 sagemaker

# GCP CLI (if using Vertex AI)
pip install google-cloud-aiplatform

# Or install gcloud SDK
# curl https://sdk.cloud.google.com | bash
```

### Kubernetes Tools (Lab 4.4.5)

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mv kubectl /usr/local/bin/

# Install minikube (for local K8s)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube-linux-amd64
mv minikube-linux-amd64 /usr/local/bin/minikube
```

### Demo Libraries (Labs 4.4.6-4.4.7)

```bash
pip install gradio>=4.0.0 streamlit>=1.30.0
```

---

## Environment Setup

### 1. Start NGC Container with All Ports

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --ipc=host \
    -p 8000:8000 \
    -p 8501:8501 \
    -p 7860:7860 \
    -p 9090:9090 \
    nvcr.io/nvidia/pytorch:25.11-py3
```

> **Note:** Mounting `/var/run/docker.sock` allows Docker-in-Docker for Labs 4.4.1-4.4.2

### 2. Install All Dependencies

```bash
# Core deployment tools
pip install gradio>=4.0.0 streamlit>=1.30.0
pip install fastapi uvicorn
pip install ollama chromadb

# Cloud SDKs (optional - for cloud labs)
pip install sagemaker>=2.200.0 boto3
pip install google-cloud-aiplatform>=1.40.0
```

### 3. Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

**Expected output**:
```
CUDA available: True
Device: NVIDIA GH200 480GB
Memory: 128.0 GB
```

---

## Pre-Lab Checklists

### Lab 4.4.1: Docker ML Image

- [ ] Docker installed and running
- [ ] NVIDIA Container Toolkit installed
- [ ] Can run `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi`
- [ ] Have a simple inference script ready

**Quick Test**:
```bash
# Create a simple Dockerfile
cat > /tmp/Dockerfile << 'EOF'
FROM nvcr.io/nvidia/pytorch:25.11-py3
WORKDIR /app
COPY . /app
CMD ["python", "-c", "import torch; print(torch.cuda.is_available())"]
EOF

# Build and test
cd /tmp
docker build -t test-gpu .
docker run --gpus all test-gpu
# Should print: True
```

---

### Lab 4.4.2: Docker Compose Stack

- [ ] Docker Compose installed (`docker compose version`)
- [ ] Completed Lab 4.4.1
- [ ] Ollama running locally

**Quick Test**:
```bash
docker compose version
# Should show: Docker Compose version v2.x
```

---

### Lab 4.4.3: AWS SageMaker Deployment

- [ ] AWS account with SageMaker access
- [ ] AWS CLI configured (`aws configure`)
- [ ] IAM role with SageMaker permissions
- [ ] S3 bucket for model artifacts
- [ ] Model packaged as `model.tar.gz`

**Quick Test**:
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check SageMaker access
aws sagemaker list-endpoints
```

**IAM Permissions Needed**:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (for model artifacts)

---

### Lab 4.4.4: GCP Vertex AI Deployment

- [ ] GCP project with Vertex AI API enabled
- [ ] gcloud CLI authenticated
- [ ] Service account with Vertex AI permissions
- [ ] GCS bucket for model artifacts
- [ ] Container Registry set up

**Quick Test**:
```bash
# Check GCP auth
gcloud auth list

# Check project
gcloud config get-value project

# Verify Vertex AI API
gcloud services list --enabled | grep aiplatform
```

**Required APIs**:
- Vertex AI API
- Container Registry API
- Cloud Storage API

---

### Lab 4.4.5: Kubernetes Deployment

- [ ] kubectl installed
- [ ] minikube or kind installed (local K8s)
- [ ] NVIDIA device plugin (for GPU scheduling)
- [ ] Completed Lab 4.4.1 (have Docker image)

**Quick Test**:
```bash
# Start minikube with GPU
minikube start --driver=docker --gpus all

# Verify
kubectl get nodes
kubectl get pods -n kube-system | grep nvidia
```

---

### Lab 4.4.6: Gradio Demo

- [ ] Gradio installed (`pip install gradio>=4.0.0`)
- [ ] Ollama running with a model
- [ ] Port 7860 available
- [ ] (Optional) Hugging Face account for Spaces deployment

**Quick Test**:
```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(server_port=7860)
# Access at http://localhost:7860
```

---

### Lab 4.4.7: Streamlit Dashboard

- [ ] Streamlit installed (`pip install streamlit>=1.30.0`)
- [ ] Ollama running with a model
- [ ] Port 8501 available
- [ ] (Optional) GitHub account for Streamlit Cloud deployment

**Quick Test**:
```python
# test_streamlit.py
import streamlit as st
st.title("Test App")
st.write("Streamlit is working!")
```

```bash
streamlit run test_streamlit.py --server.port 8501
# Access at http://localhost:8501
```

---

## Cloud Account Setup

### AWS SageMaker (Lab 4.4.3)

1. **Create IAM Role**:
   - Go to IAM → Roles → Create Role
   - Select "SageMaker" as trusted entity
   - Attach `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`
   - Name it `SageMakerExecutionRole`

2. **Create S3 Bucket**:
   ```bash
   aws s3 mb s3://my-sagemaker-models-<unique-suffix>
   ```

3. **Configure AWS CLI**:
   ```bash
   aws configure
   # Enter Access Key ID, Secret Key, Region (e.g., us-west-2)
   ```

### GCP Vertex AI (Lab 4.4.4)

1. **Enable APIs**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

2. **Create Service Account**:
   ```bash
   gcloud iam service-accounts create vertex-ai-sa \
       --display-name="Vertex AI Service Account"

   gcloud projects add-iam-policy-binding PROJECT_ID \
       --member="serviceAccount:vertex-ai-sa@PROJECT_ID.iam.gserviceaccount.com" \
       --role="roles/aiplatform.user"
   ```

3. **Create GCS Bucket**:
   ```bash
   gsutil mb gs://my-vertex-models-<unique-suffix>
   ```

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Docker socket not mounted | Can't build images inside container | Add `-v /var/run/docker.sock:/var/run/docker.sock` |
| Missing `--gpus all` | No GPU in container | Always include for inference |
| Wrong port mapping | Can't access service | Check `-p host:container` matches |
| AWS credentials not set | SageMaker calls fail | Run `aws configure` first |
| GCP project not set | Vertex AI calls fail | Run `gcloud config set project` |
| minikube not GPU-enabled | K8s can't schedule GPU pods | Start with `--gpus all` |

---

## Expected Directory Structure

After preparation, your workspace should look like:

```
/workspace/
├── module-4.4/
│   ├── lab-4.4.1-docker/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       └── serve.py
│   ├── lab-4.4.2-compose/
│   │   ├── docker-compose.yml
│   │   ├── prometheus.yml
│   │   └── inference/
│   ├── lab-4.4.3-sagemaker/
│   │   ├── model.tar.gz
│   │   └── deploy.py
│   ├── lab-4.4.4-vertex/
│   │   ├── model/
│   │   └── deploy.py
│   ├── lab-4.4.5-k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml
│   ├── lab-4.4.6-gradio/
│   │   └── app.py
│   └── lab-4.4.7-streamlit/
│       ├── Home.py
│       └── pages/
```

---

## Quick Start Commands

```bash
# Copy-paste to set up everything:
cd /workspace
mkdir -p module-4.4/{lab-4.4.1-docker/app,lab-4.4.2-compose,lab-4.4.3-sagemaker,lab-4.4.4-vertex,lab-4.4.5-k8s,lab-4.4.6-gradio,lab-4.4.7-streamlit/pages}

# Install dependencies
pip install gradio>=4.0.0 streamlit>=1.30.0 fastapi uvicorn

# Verify Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Verify Gradio
python -c "import gradio; print(f'Gradio {gradio.__version__} ready!')"

# Verify Streamlit
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} ready!')"

echo "Setup complete!"
```

---

## Model Preparation

For inference labs, you'll need a model ready:

```bash
# Option 1: Use Ollama (simplest)
ollama pull llama3.1:8b

# Option 2: Download HuggingFace model
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# Option 3: Use your fine-tuned model from Module 3.1
# Copy from /workspace/finetuned_model/
```

---

## Estimated Storage Requirements

| Lab | Storage Needed |
|-----|----------------|
| Docker images | 5-10 GB |
| Model artifacts | 10-20 GB |
| K8s images | 5-10 GB |
| Total | ~30-40 GB |

Ensure you have sufficient disk space before starting.
