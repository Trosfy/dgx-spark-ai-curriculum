# Module 4.4: Containerization & Cloud Deployment - Frequently Asked Questions

## Setup & Environment

### Q: Which cloud platform should I learn first?

**A**: If you have no preference, **AWS SageMaker** has more examples and tutorials. Both SageMaker and Vertex AI are similar in capability.

| Choose AWS if | Choose GCP if |
|---------------|---------------|
| Company uses AWS | Company uses GCP |
| Need HuggingFace integration | Need TPU access |
| Want more tutorials | Prefer GCP console |

---

### Q: Do I need GPU instances for all deployments?

**A**: No! Consider your use case:

| Use Case | Instance |
|----------|----------|
| Inference (small model) | CPU may be enough |
| Inference (LLM) | GPU required |
| Inference (quantized) | GPU (smaller) |
| Training | GPU required |

---

### Q: How do I reduce Docker image size?

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

## Concepts

### Q: What's the difference between Docker Compose and Kubernetes?

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

### Q: When should I use managed services vs Kubernetes?

**A**:

| Use Managed (SageMaker/Vertex) | Use Kubernetes |
|--------------------------------|----------------|
| Small team, no DevOps | Have DevOps team |
| Want quick deployment | Need full control |
| Standard ML workflows | Custom requirements |
| Predictable workloads | Complex scaling needs |

---

### Q: How do I handle secrets in containers?

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

## Troubleshooting

### Q: Docker build fails with "no space left on device"

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

### Q: Container runs but can't access GPU

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

### Q: SageMaker endpoint times out

**A**: Common causes and fixes:

1. **Model loading too slow**: Increase `model_server_timeout` in endpoint config
2. **Container unhealthy**: Add health check endpoint
3. **Memory insufficient**: Use larger instance type
4. **Cold start**: Use provisioned concurrency

---

### Q: Kubernetes pods stuck in Pending

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

## Beyond the Basics

### Q: How do I set up auto-scaling?

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

### Q: How do I do blue-green deployments?

**A**: Basic pattern:

1. Deploy new version alongside old (blue + green)
2. Test new version
3. Switch traffic to new version
4. Keep old version for rollback
5. Delete old version after confidence

**SageMaker**: Use endpoint update with `RetainAllVariantProperties`
**Kubernetes**: Use separate Deployments with Service selector switch

---

### Q: How do I monitor costs?

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

## Still Have Questions?

- Check [WORKFLOWS.md](./WORKFLOWS.md) for step-by-step guides
- Review [COMPARISONS.md](./COMPARISONS.md) for platform decisions
- See module [Resources](./README.md#resources) for official documentation
