# ============================================
# Production Dockerfile for LLM Inference Server
# Optimized for DGX Spark (ARM64 + CUDA)
# ============================================

# ===========================================
# Stage 1: Builder
# ===========================================
FROM nvcr.io/nvidia/pytorch:24.12-py3 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to user directory
COPY requirements.txt /tmp/requirements.txt
RUN pip install --user --no-cache-dir -r /tmp/requirements.txt

# ===========================================
# Stage 2: Production
# ===========================================
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Labels
LABEL maintainer="DGX Spark AI Curriculum"
LABEL description="LLM Inference Server for DGX Spark"
LABEL version="1.0"

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Create non-root user (security best practice)
RUN useradd -m -u 1000 inference && \
    mkdir -p /app /models /data && \
    chown -R inference:inference /app /models /data

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=inference:inference app/ /app/

# Environment variables
ENV MODEL_PATH=/models \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/data/cache \
    HF_HOME=/data/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER inference

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
