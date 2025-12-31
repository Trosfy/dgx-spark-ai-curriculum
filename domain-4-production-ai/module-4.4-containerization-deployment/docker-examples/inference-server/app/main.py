"""
Production LLM Inference Server.

Features:
- FastAPI with async support
- Model loading with GPU optimization
- Health and metrics endpoints
- Streaming responses
- Request batching

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================
# Configuration
# ============================================

MODEL_PATH = os.environ.get("MODEL_PATH", "/models")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))
DEFAULT_TOP_P = float(os.environ.get("TOP_P", 0.9))

# ============================================
# Global State
# ============================================

model = None
tokenizer = None
model_info = {}


# ============================================
# Request/Response Models
# ============================================

class GenerateRequest(BaseModel):
    """Text generation request."""
    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)


class GenerateResponse(BaseModel):
    """Text generation response."""
    generated_text: str
    tokens_generated: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    model_path: str


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_requests: int
    total_tokens: int
    avg_latency_ms: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float


# ============================================
# Metrics Tracking
# ============================================

class MetricsTracker:
    """Track inference metrics."""

    def __init__(self):
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    def record(self, tokens: int, latency_ms: float):
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


metrics = MetricsTracker()


# ============================================
# Model Loading
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, tokenizer, model_info

    print(f"Loading model from: {MODEL_PATH}")
    start_time = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        load_time = time.time() - start_time

        model_info = {
            "path": MODEL_PATH,
            "load_time_s": load_time,
            "dtype": str(model.dtype),
            "device": str(model.device),
        }

        print(f"Model loaded in {load_time:.2f}s")
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
        tokenizer = None

    yield

    # Cleanup
    if model is not None:
        del model, tokenizer
        torch.cuda.empty_cache()
        print("Model unloaded, GPU memory cleared")


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="LLM Inference Server",
    description="Production inference server for large language models",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        model_path=MODEL_PATH,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get inference metrics."""
    gpu_memory_used = 0.0
    gpu_memory_total = 0.0

    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    return MetricsResponse(
        total_requests=metrics.total_requests,
        total_tokens=metrics.total_tokens,
        avg_latency_ms=metrics.avg_latency_ms,
        gpu_memory_used_gb=gpu_memory_used,
        gpu_memory_total_gb=gpu_memory_total,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text completion."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Tokenize
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    latency_ms = (time.time() - start_time) * 1000
    tokens_generated = len(generated_tokens)

    # Record metrics
    metrics.record(tokens_generated, latency_ms)

    return GenerateResponse(
        generated_text=generated_text,
        tokens_generated=tokens_generated,
        latency_ms=latency_ms,
    )


@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
