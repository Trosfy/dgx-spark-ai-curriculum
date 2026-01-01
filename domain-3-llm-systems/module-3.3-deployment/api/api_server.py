"""
Production-Ready FastAPI Server for LLM Inference

This module provides a production-quality API server that wraps any
inference engine (Ollama, vLLM, TensorRT-LLM) with:
- OpenAI-compatible API endpoints
- Streaming support (SSE)
- Request rate limiting
- Health checks and metrics
- Proper error handling
- Request/response logging

Example:
    # Start the server
    $ python api_server.py --engine ollama --model qwen3:8b --port 8080

    # Or with uvicorn for production
    $ uvicorn api_server:app --host 0.0.0.0 --port 8080 --workers 4

Usage with curl:
    # Non-streaming
    $ curl -X POST http://localhost:8080/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "qwen3:8b", "messages": [{"role": "user", "content": "Hello!"}]}'

    # Streaming
    $ curl -X POST http://localhost:8080/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "qwen3:8b", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Add parent directory for imports
# This allows importing from scripts/ when running from api/ directory
module_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if module_root not in sys.path:
    sys.path.insert(0, module_root)

# Verify the scripts directory exists before attempting import
_scripts_dir = os.path.join(module_root, 'scripts')
_inference_client_path = os.path.join(_scripts_dir, 'inference_client.py')

if not os.path.isdir(_scripts_dir):
    print(f"ERROR: Scripts directory not found at: {_scripts_dir}")
    print("\nThis usually means you're running from the wrong directory.")
    print("To fix this:")
    print("  1. cd domain-3-llm-systems/module-3.3-deployment")
    print("  2. python api/api_server.py")
    raise SystemExit(1)

if not os.path.isfile(_inference_client_path):
    print(f"ERROR: inference_client.py not found at: {_inference_client_path}")
    print("\nEnsure all module files are present.")
    raise SystemExit(1)

try:
    from scripts.inference_client import (
        GenerationConfig,
        UnifiedInferenceClient,
        EngineType,
    )
except ImportError as e:
    # Provide helpful error message
    print(f"ERROR: Failed to import inference_client: {e}")
    print(f"Module root: {module_root}")
    print(f"Scripts directory: {_scripts_dir}")
    print(f"Expected file: {_inference_client_path}")
    print("\nTo fix this:")
    print("  1. Ensure you're running from the module-3.3-deployment directory")
    print("  2. Or run: cd domain-3-llm-systems/module-3.3-deployment && python api/api_server.py")
    print("  3. Check that all dependencies are installed: pip install aiohttp requests")
    raise SystemExit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_server")


# ============================================================================
# Pydantic Models (OpenAI-compatible API schema)
# ============================================================================

class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str = Field(..., description="The role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatCompletionRequest(BaseModel):
    """Request body for chat completion endpoint."""
    model: str = Field(..., description="Model identifier")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(False, description="Enable streaming")
    stop: Optional[list[str]] = Field(None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    n: Optional[int] = Field(1, description="Number of completions (only 1 supported)")
    user: Optional[str] = Field(None, description="User identifier for tracking")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response from chat completion endpoint."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A streaming response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelList(BaseModel):
    """List of available models."""
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine: str
    model: str
    uptime_seconds: float
    total_requests: int
    active_requests: int


class ErrorResponse(BaseModel):
    """Error response structure."""
    error: dict[str, Any]


# ============================================================================
# Metrics and Rate Limiting
# ============================================================================

@dataclass
class ServerMetrics:
    """Tracks server performance metrics."""
    start_time: float = field(default_factory=time.time)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    active_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def avg_latency_ms(self) -> float:
        if not self.request_times:
            return 0.0
        return sum(self.request_times) / len(self.request_times)

    @property
    def requests_per_minute(self) -> float:
        if self.uptime_seconds < 60:
            return self.total_requests * 60 / max(self.uptime_seconds, 1)
        # Count requests in the last minute
        now = time.time()
        recent = sum(1 for t in self.request_times if now - t < 60)
        return recent


class RateLimiter:
    """
    Simple token bucket rate limiter.

    Limits requests per minute per IP address.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests: dict[str, deque] = {}

    def is_allowed(self, client_ip: str) -> bool:
        """Check if a request is allowed for the given IP."""
        now = time.time()

        if client_ip not in self.requests:
            self.requests[client_ip] = deque()

        # Remove requests older than 1 minute
        while self.requests[client_ip] and now - self.requests[client_ip][0] > 60:
            self.requests[client_ip].popleft()

        if len(self.requests[client_ip]) >= self.rpm:
            return False

        self.requests[client_ip].append(now)
        return True


# ============================================================================
# Global State
# ============================================================================

# These will be initialized at startup
inference_client: Optional[UnifiedInferenceClient] = None
model_name: str = ""
engine_name: str = ""
metrics = ServerMetrics()
rate_limiter = RateLimiter(requests_per_minute=60)


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global inference_client, model_name, engine_name

    # Get configuration from environment
    engine = os.getenv("INFERENCE_ENGINE", "ollama")
    model = os.getenv("INFERENCE_MODEL", "qwen3:8b")
    base_url = os.getenv("INFERENCE_URL", "")

    logger.info(f"Starting API server with engine={engine}, model={model}")

    # Initialize inference client
    try:
        if engine == "ollama":
            url = base_url or "http://localhost:11434"
            inference_client = UnifiedInferenceClient.from_ollama(model, url)
        elif engine == "vllm":
            url = base_url or "http://localhost:8000"
            inference_client = UnifiedInferenceClient.from_vllm(model, url)
        elif engine == "sglang":
            url = base_url or "http://localhost:30000"
            inference_client = UnifiedInferenceClient.from_sglang(model, url)
        elif engine == "tensorrt-llm":
            url = base_url or "http://localhost:8000"
            inference_client = UnifiedInferenceClient.from_tensorrt_llm(model, url)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        model_name = model
        engine_name = engine

        if not inference_client.is_healthy():
            logger.warning(f"Inference engine at {url} is not responding")
        else:
            logger.info(f"Successfully connected to {engine} at {url}")

    except Exception as e:
        logger.error(f"Failed to initialize inference client: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down API server")


app = FastAPI(
    title="LLM Inference API",
    description="OpenAI-compatible API for local LLM inference",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware for Logging and Rate Limiting
# ============================================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and apply rate limiting."""
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting (skip for health checks)
    if request.url.path not in ["/health", "/v1/models"]:
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
            )

    # Track request
    start_time = time.time()
    metrics.total_requests += 1
    metrics.active_requests += 1

    try:
        response = await call_next(request)
        metrics.successful_requests += 1
    except Exception as e:
        metrics.failed_requests += 1
        logger.error(f"Request failed: {e}")
        raise
    finally:
        metrics.active_requests -= 1
        latency_ms = (time.time() - start_time) * 1000
        metrics.request_times.append(latency_ms)

        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Client: {client_ip} - "
            f"Status: {response.status_code if 'response' in locals() else 'ERROR'} - "
            f"Latency: {latency_ms:.1f}ms"
        )

    return response


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and get basic metrics."""
    is_healthy = inference_client.is_healthy() if inference_client else False

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        engine=engine_name,
        model=model_name,
        uptime_seconds=metrics.uptime_seconds,
        total_requests=metrics.total_requests,
        active_requests=metrics.active_requests
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    return ModelList(
        data=[ModelInfo(id=model_name, owned_by=engine_name)]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Generate a chat completion.

    Compatible with OpenAI's Chat Completions API.
    Supports both streaming and non-streaming responses.
    """
    if not inference_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized"
        )

    # Convert messages to dict format
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Create generation config
    config = GenerationConfig(
        max_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        stop=request.stop,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0
    )

    if request.stream:
        return await stream_chat_completion(request, messages, config)
    else:
        return await non_streaming_chat_completion(request, messages, config)


async def non_streaming_chat_completion(
    request: ChatCompletionRequest,
    messages: list[dict[str, str]],
    config: GenerationConfig
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    try:
        response = inference_client.chat_with_metadata(messages, config)

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=response.text),
                    finish_reason=response.finish_reason or "stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.tokens_generated,
                total_tokens=response.prompt_tokens + response.tokens_generated
            )
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def stream_chat_completion(
    request: ChatCompletionRequest,
    messages: list[dict[str, str]],
    config: GenerationConfig
) -> StreamingResponse:
    """Handle streaming chat completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    async def generate():
        try:
            # Send initial chunk with role
            initial_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(role="assistant"),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {initial_chunk.model_dump_json()}\n\n"

            # Stream content chunks
            for text_chunk in inference_client.stream_chat(messages, config):
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=text_chunk),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run

            # Send final chunk
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            error_msg = json.dumps({"error": {"message": str(e)}})
            yield f"data: {error_msg}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/metrics")
async def get_metrics():
    """Get detailed server metrics."""
    return {
        "uptime_seconds": metrics.uptime_seconds,
        "total_requests": metrics.total_requests,
        "successful_requests": metrics.successful_requests,
        "failed_requests": metrics.failed_requests,
        "active_requests": metrics.active_requests,
        "avg_latency_ms": metrics.avg_latency_ms,
        "requests_per_minute": metrics.requests_per_minute,
        "total_tokens_generated": metrics.total_tokens_generated,
        "engine": engine_name,
        "model": model_name
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Inference API Server")
    parser.add_argument("--engine", default="ollama", choices=["ollama", "vllm", "sglang", "tensorrt-llm"])
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--url", default="", help="Inference engine URL")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    # Set environment variables for lifespan
    os.environ["INFERENCE_ENGINE"] = args.engine
    os.environ["INFERENCE_MODEL"] = args.model
    if args.url:
        os.environ["INFERENCE_URL"] = args.url

    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║              LLM Inference API Server                       ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Engine: {args.engine:<48} ║
    ║  Model:  {args.model:<48} ║
    ║  URL:    http://{args.host}:{args.port:<37} ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
