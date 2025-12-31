"""
Simple LLM Inference Server.

A minimal FastAPI server for serving LLM models.
Can be used as a starting point for more complex deployments.

Usage:
    python serve.py
    # or
    uvicorn serve:app --host 0.0.0.0 --port 8000

Example request:
    curl -X POST http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Hello, how are you?"}'
"""

import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# ============================================
# Configuration
# ============================================

MODEL_PATH = os.environ.get("MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 256))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# Request/Response Models
# ============================================


class GenerateRequest(BaseModel):
    """Generation request."""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7


class GenerateResponse(BaseModel):
    """Generation response."""
    generated_text: str
    tokens: int
    latency_ms: float


# ============================================
# Model Loading
# ============================================

print(f"Loading model: {MODEL_PATH}")
print(f"Device: {DEVICE}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    if DEVICE == "cpu":
        model = model.to(DEVICE)

    print(f"Model loaded successfully!")
    if DEVICE == "cuda":
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    tokenizer = None

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="Simple LLM Server",
    description="Minimal inference server for LLMs",
    version="1.0.0",
)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "device": DEVICE,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text completion."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Tokenize
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens or MAX_TOKENS,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    latency = (time.time() - start) * 1000

    return GenerateResponse(
        generated_text=text,
        tokens=len(generated),
        latency_ms=latency,
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Simple LLM Server",
        "model": MODEL_PATH,
        "endpoints": ["/health", "/generate"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
