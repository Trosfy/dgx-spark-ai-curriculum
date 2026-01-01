# Module 4.4: Containerization & Cloud Deployment - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Containerize ML applications** with optimized Docker images
2. **Deploy to cloud platforms** (AWS SageMaker, GCP Vertex AI)
3. **Use Kubernetes** for container orchestration
4. **Build interactive demos** with Gradio and Streamlit

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-4.4.1-docker-ml-image.ipynb | Docker Basics | ~2h | Optimized ML Docker image |
| 2 | lab-4.4.2-docker-compose-stack.ipynb | Docker Compose | ~2h | Multi-service ML stack |
| 3 | lab-4.4.3-aws-sagemaker-deployment.ipynb | AWS SageMaker | ~2h | Working SageMaker endpoint |
| 4 | lab-4.4.4-gcp-vertex-ai-deployment.ipynb | GCP Vertex AI | ~2h | Working Vertex AI endpoint |
| 5 | lab-4.4.5-kubernetes-deployment.ipynb | Kubernetes | ~2h | K8s deployment with HPA |
| 6 | lab-4.4.6-gradio-demo.ipynb | Gradio | ~2h | Demo on HF Spaces |
| 7 | lab-4.4.7-streamlit-dashboard.ipynb | Streamlit | ~2h | Multi-page dashboard |

**Total Time**: ~14 hours

---

## Core Concepts

### Containerization
**What**: Packaging applications with all dependencies into portable, isolated units.
**Why it matters**: "Works on my machine" → "Works everywhere." Critical for production deployment.
**First appears in**: Lab 4.4.1

### Multi-Stage Builds
**What**: Docker technique to separate build and runtime environments, reducing final image size.
**Why it matters**: Smaller images = faster deployment, less storage, fewer vulnerabilities.
**First appears in**: Lab 4.4.1

### Container Orchestration
**What**: Automated management of containerized applications (scaling, healing, networking).
**Why it matters**: Production ML needs auto-scaling, health checks, rolling updates.
**First appears in**: Lab 4.4.5

### Managed ML Platforms
**What**: Cloud services (SageMaker, Vertex AI) that handle infrastructure for ML deployment.
**Why it matters**: Focus on ML, not DevOps. Built-in scaling, monitoring, versioning.
**First appears in**: Labs 4.4.3-4

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 4.3              ──►  Module 4.4           ──►   Module 4.5
MLOps                        Containerization           Demo Building
[Model registry              [Docker, K8s,              [Gradio,
 feeds deployment]            cloud platforms]           Streamlit UI]
```

**Builds on**:
- Model registry from Module 4.3 (versioned models to deploy)
- Safety measures from Module 4.2 (deploy with guardrails)
- Optimized models from Module 3.2 (quantized for efficiency)

**Prepares for**:
- Module 4.5 builds demos on deployed backends
- Module 4.6 capstone requires production deployment

---

## DGX Spark to Cloud Path

| Stage | Local (DGX Spark) | Production (Cloud) |
|-------|-------------------|-------------------|
| Development | NGC container | Same container |
| Testing | Docker Compose | Cloud preview |
| Staging | Local K8s (minikube) | Cloud K8s |
| Production | - | SageMaker/Vertex AI |

---

## Recommended Approach

**Standard Path** (14 hours):
1. Labs 4.4.1-2: Master Docker for ML
2. Labs 4.4.3-4: Deploy to one cloud platform
3. Lab 4.4.5: Kubernetes basics
4. Labs 4.4.6-7: Build demos

**Quick Path** (if Docker-experienced, 8 hours):
1. Skim Lab 4.4.1, focus on GPU-specific parts
2. Lab 4.4.2: Docker Compose for ML
3. One cloud platform (4.4.3 or 4.4.4)
4. Lab 4.4.6: Gradio demo

**Cloud-focused Path**:
1. Labs 4.4.1-2: Docker essentials
2. Both cloud labs 4.4.3-4
3. Lab 4.4.5: Kubernetes
4. Skip local demos

---

## Before You Start

- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success
- See [COMPARISONS.md](./COMPARISONS.md) for platform comparisons
- See [WORKFLOWS.md](./WORKFLOWS.md) for deployment workflows
