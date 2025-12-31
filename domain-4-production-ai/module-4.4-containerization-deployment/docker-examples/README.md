# Docker Examples for ML Deployment

This directory contains production-ready Docker configurations for deploying ML models.

## Directory Structure

```
docker-examples/
├── inference-server/          # Single-service inference server
│   ├── Dockerfile             # Multi-stage optimized Dockerfile
│   ├── requirements.txt       # Python dependencies
│   └── app/
│       └── main.py            # FastAPI inference server
│
├── ml-stack/                  # Complete ML stack with Docker Compose
│   ├── docker-compose.yml     # Multi-service stack
│   └── monitoring/
│       ├── prometheus.yml     # Prometheus configuration
│       └── grafana/           # Grafana dashboards
│
└── gradio-space/              # Hugging Face Space deployment
    ├── app.py                 # Gradio application
    ├── requirements.txt       # Dependencies
    └── README.md              # Space configuration
```

## Quick Start

### Single Inference Server

```bash
cd inference-server

# Build
docker build -t llm-inference:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 \
    -v /path/to/models:/models:ro \
    llm-inference:latest

# Test
curl http://localhost:8000/health
```

### Complete ML Stack

```bash
cd ml-stack

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f inference

# Stop
docker compose down
```

## Services in ML Stack

| Service | Port | Description |
|---------|------|-------------|
| inference | 8000 | LLM inference server |
| chromadb | 8001 | Vector database |
| redis | 6379 | Response cache |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Dashboards |

## GPU Requirements

- NVIDIA Container Toolkit installed
- Docker 19.03+ with GPU support
- CUDA 12.0+ compatible GPU

```bash
# Verify GPU support
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | /models | Path to model files |
| CUDA_VISIBLE_DEVICES | 0 | GPU device(s) to use |
| MAX_NEW_TOKENS | 512 | Max tokens to generate |
| TEMPERATURE | 0.7 | Sampling temperature |

## Best Practices

1. **Use NGC Base Images**: For ARM64 + CUDA compatibility
2. **Multi-Stage Builds**: Smaller production images
3. **Non-Root User**: Security best practice
4. **Health Checks**: Required for orchestration
5. **Resource Limits**: Prevent runaway containers

## DGX Spark Notes

On DGX Spark (ARM64), always use NGC containers:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.12-py3
```

Clear buffer cache before loading large models:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
