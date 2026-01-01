# Module 4.4: Containerization & Cloud Deployment - Quickstart

## Time: ~5 minutes

## What You'll Build

Create a simple Docker container for an ML inference server.

## Before You Start

- [ ] Docker installed and running
- [ ] DGX Spark access

## Let's Go!

### Step 1: Create a Simple Inference Script

```python
# save as serve.py
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "gpu": torch.cuda.is_available()})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simple echo for demo
    return jsonify({"input": data.get("text"), "prediction": "demo_result"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Step 2: Create a Dockerfile

```dockerfile
# save as Dockerfile
FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /app
RUN pip install flask
COPY serve.py /app/

EXPOSE 8000
CMD ["python", "serve.py"]
```

### Step 3: Build the Image

```bash
docker build -t my-inference:latest .
```

### Step 4: Run the Container

```bash
docker run --gpus all -p 8000:8000 my-inference:latest
```

### Step 5: Test the Endpoint

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello world"}'
```

**Expected output**:
```json
{"status": "healthy", "gpu": true}
{"input": "Hello world", "prediction": "demo_result"}
```

## You Did It!

You just containerized an ML inference server! In the full module, you'll learn:

- **Optimized Dockerfiles**: Multi-stage builds, smaller images
- **Docker Compose**: Multi-service ML stacks
- **AWS SageMaker**: Production deployment on AWS
- **GCP Vertex AI**: Production deployment on Google Cloud
- **Kubernetes**: Container orchestration
- **Gradio/Streamlit**: Interactive demo deployment

## Next Steps

1. **Add a real model**: Replace echo with actual inference
2. **Optimize image**: Use multi-stage build
3. **Full tutorial**: Start with [LAB_PREP.md](./LAB_PREP.md)
