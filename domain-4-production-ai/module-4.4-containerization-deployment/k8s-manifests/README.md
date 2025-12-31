# Kubernetes Manifests for ML Deployment

This directory contains production-ready Kubernetes manifests for deploying LLM inference servers.

## Files

- `deployment.yaml` - Complete deployment with:
  - GPU-enabled Deployment
  - LoadBalancer Service
  - Horizontal Pod Autoscaler
  - PersistentVolumeClaim
  - ConfigMap
  - Pod Disruption Budget

## Prerequisites

1. Kubernetes cluster with GPU nodes
2. NVIDIA GPU Operator installed
3. kubectl configured

## Quick Start

```bash
# Apply all manifests
kubectl apply -f deployment.yaml

# Check status
kubectl get pods -l app=llm-inference
kubectl get svc llm-inference-service

# View logs
kubectl logs -l app=llm-inference -f

# Port forward for local testing
kubectl port-forward svc/llm-inference-service 8000:80
```

## GPU Configuration

Ensure your cluster has:
- NVIDIA device plugin installed
- GPU nodes labeled with `accelerator: nvidia-gpu`
- Proper tolerations for GPU taints

```bash
# Check GPU availability
kubectl describe nodes | grep -A5 "nvidia.com/gpu"
```

## Scaling

The HPA automatically scales based on CPU usage:
- Min replicas: 1
- Max replicas: 5
- Target CPU: 70%

Manual scaling:
```bash
kubectl scale deployment llm-inference --replicas=3
```

## Monitoring

Prometheus annotations are included for automatic scraping:
- Metrics endpoint: `/metrics`
- Port: 8000

## Cleanup

```bash
kubectl delete -f deployment.yaml
```
