"""
Unified Inference Client for Multiple LLM Engines

This module provides a unified interface for interacting with different
LLM inference engines (Ollama, vLLM, TensorRT-LLM, SGLang). It abstracts
away the differences between APIs and provides consistent streaming support.

Example:
    >>> from inference_client import UnifiedInferenceClient
    >>>
    >>> # Connect to Ollama
    >>> client = UnifiedInferenceClient.from_ollama("llama3.1:8b")
    >>>
    >>> # Simple completion
    >>> response = client.complete("Tell me a joke")
    >>> print(response)
    >>>
    >>> # Streaming completion
    >>> for chunk in client.stream("Write a story"):
    ...     print(chunk, end="", flush=True)
    >>>
    >>> # Chat completion
    >>> messages = [
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user", "content": "Hello!"}
    ... ]
    >>> response = client.chat(messages)
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Optional, Union

import aiohttp
import requests


class EngineType(Enum):
    """Supported inference engine types."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt-llm"
    LLAMA_CPP = "llama.cpp"
    OPENAI = "openai"


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling parameter
        stop: List of stop sequences
        presence_penalty: Penalize tokens based on their presence in output
        frequency_penalty: Penalize tokens based on their frequency
        seed: Random seed for reproducibility
    """
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop: Optional[list[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None

    def to_ollama_options(self) -> dict[str, Any]:
        """Convert to Ollama API format."""
        options = {
            "num_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.stop:
            options["stop"] = self.stop
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def to_openai_params(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.stop:
            params["stop"] = self.stop
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.seed is not None:
            params["seed"] = self.seed
        return params


@dataclass
class ChatMessage:
    """A chat message with role and content."""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "ChatMessage":
        return cls(role=d["role"], content=d["content"])


@dataclass
class CompletionResponse:
    """
    Response from a completion request.

    Attributes:
        text: The generated text
        tokens_generated: Number of tokens in the response
        prompt_tokens: Number of tokens in the prompt
        total_time: Total generation time in seconds
        tokens_per_second: Generation speed
        model: Model used for generation
        finish_reason: Why generation stopped ("stop", "length", etc.)
    """
    text: str
    tokens_generated: int = 0
    prompt_tokens: int = 0
    total_time: float = 0.0
    tokens_per_second: float = 0.0
    model: str = ""
    finish_reason: str = ""


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engine implementations."""

    @abstractmethod
    def complete(self, prompt: str, config: GenerationConfig) -> CompletionResponse:
        """Generate a completion for the given prompt."""
        pass

    @abstractmethod
    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream a completion for the given prompt."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig
    ) -> CompletionResponse:
        """Generate a chat completion."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig
    ) -> Iterator[str]:
        """Stream a chat completion."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the inference server is healthy."""
        pass


class OllamaEngine(BaseInferenceEngine):
    """Ollama inference engine implementation."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, prompt: str, config: GenerationConfig) -> CompletionResponse:
        """Generate completion using Ollama."""
        start_time = time.perf_counter()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "options": config.to_ollama_options(),
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        total_time = time.perf_counter() - start_time
        tokens_generated = data.get("eval_count", 0)

        return CompletionResponse(
            text=data.get("response", ""),
            tokens_generated=tokens_generated,
            prompt_tokens=data.get("prompt_eval_count", 0),
            total_time=total_time,
            tokens_per_second=tokens_generated / total_time if total_time > 0 else 0,
            model=self.model,
            finish_reason="stop" if data.get("done") else "unknown"
        )

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream completion from Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "options": config.to_ollama_options(),
                "stream": True
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                text = chunk.get("response", "")
                if text:
                    yield text

    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig
    ) -> CompletionResponse:
        """Generate chat completion using Ollama."""
        start_time = time.perf_counter()

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "options": config.to_ollama_options(),
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        total_time = time.perf_counter() - start_time
        message = data.get("message", {})
        tokens_generated = data.get("eval_count", 0)

        return CompletionResponse(
            text=message.get("content", ""),
            tokens_generated=tokens_generated,
            prompt_tokens=data.get("prompt_eval_count", 0),
            total_time=total_time,
            tokens_per_second=tokens_generated / total_time if total_time > 0 else 0,
            model=self.model,
            finish_reason="stop" if data.get("done") else "unknown"
        )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig
    ) -> Iterator[str]:
        """Stream chat completion from Ollama."""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "options": config.to_ollama_options(),
                "stream": True
            },
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                message = chunk.get("message", {})
                text = message.get("content", "")
                if text:
                    yield text

    def is_healthy(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


class OpenAICompatibleEngine(BaseInferenceEngine):
    """OpenAI-compatible API engine (vLLM, SGLang, TensorRT-LLM)."""

    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def complete(self, prompt: str, config: GenerationConfig) -> CompletionResponse:
        """Generate completion using OpenAI-compatible API."""
        # Use chat completion with a single user message
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, config)

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream completion using OpenAI-compatible API."""
        messages = [{"role": "user", "content": prompt}]
        yield from self.stream_chat(messages, config)

    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig
    ) -> CompletionResponse:
        """Generate chat completion using OpenAI-compatible API."""
        start_time = time.perf_counter()

        params = config.to_openai_params()
        params["model"] = self.model
        params["messages"] = messages
        params["stream"] = False

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=params,
            headers=self.headers,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        total_time = time.perf_counter() - start_time
        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})
        tokens_generated = usage.get("completion_tokens", 0)

        return CompletionResponse(
            text=choice.get("message", {}).get("content", ""),
            tokens_generated=tokens_generated,
            prompt_tokens=usage.get("prompt_tokens", 0),
            total_time=total_time,
            tokens_per_second=tokens_generated / total_time if total_time > 0 else 0,
            model=self.model,
            finish_reason=choice.get("finish_reason", "unknown")
        )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig
    ) -> Iterator[str]:
        """Stream chat completion using OpenAI-compatible API."""
        params = config.to_openai_params()
        params["model"] = self.model
        params["messages"] = messages
        params["stream"] = True

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=params,
            headers=self.headers,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def is_healthy(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


class UnifiedInferenceClient:
    """
    Unified client for interacting with any supported inference engine.

    This class provides a consistent interface regardless of the underlying
    inference engine, making it easy to switch between Ollama, vLLM, etc.

    Example:
        >>> # Create client for Ollama
        >>> client = UnifiedInferenceClient.from_ollama("llama3.1:8b")
        >>>
        >>> # Simple completion
        >>> response = client.complete("What is Python?")
        >>> print(response)
        >>>
        >>> # Chat with custom config
        >>> config = GenerationConfig(temperature=0.3, max_tokens=256)
        >>> messages = [
        ...     {"role": "system", "content": "You are a Python tutor."},
        ...     {"role": "user", "content": "Explain decorators."}
        ... ]
        >>> response = client.chat(messages, config)
        >>> print(response)
        >>>
        >>> # Streaming
        >>> for chunk in client.stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)
    """

    def __init__(self, engine: BaseInferenceEngine, engine_type: EngineType):
        """
        Initialize the unified client.

        Args:
            engine: The underlying inference engine implementation
            engine_type: The type of engine being used
        """
        self._engine = engine
        self._engine_type = engine_type
        self._default_config = GenerationConfig()

    @classmethod
    def from_ollama(
        cls,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434"
    ) -> "UnifiedInferenceClient":
        """
        Create a client connected to Ollama.

        Args:
            model: Model name (e.g., "llama3.1:8b", "mistral", "qwen2.5:32b")
            base_url: Ollama server URL

        Returns:
            UnifiedInferenceClient configured for Ollama
        """
        engine = OllamaEngine(base_url, model)
        return cls(engine, EngineType.OLLAMA)

    @classmethod
    def from_vllm(
        cls,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000",
        api_key: str = ""
    ) -> "UnifiedInferenceClient":
        """
        Create a client connected to vLLM.

        Args:
            model: HuggingFace model ID
            base_url: vLLM server URL
            api_key: Optional API key

        Returns:
            UnifiedInferenceClient configured for vLLM
        """
        engine = OpenAICompatibleEngine(base_url, model, api_key)
        return cls(engine, EngineType.VLLM)

    @classmethod
    def from_sglang(
        cls,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:30000",
        api_key: str = ""
    ) -> "UnifiedInferenceClient":
        """
        Create a client connected to SGLang.

        Args:
            model: Model identifier
            base_url: SGLang server URL
            api_key: Optional API key

        Returns:
            UnifiedInferenceClient configured for SGLang
        """
        engine = OpenAICompatibleEngine(base_url, model, api_key)
        return cls(engine, EngineType.SGLANG)

    @classmethod
    def from_tensorrt_llm(
        cls,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000",
        api_key: str = ""
    ) -> "UnifiedInferenceClient":
        """
        Create a client connected to TensorRT-LLM.

        Args:
            model: Model identifier
            base_url: TensorRT-LLM server URL
            api_key: Optional API key

        Returns:
            UnifiedInferenceClient configured for TensorRT-LLM
        """
        engine = OpenAICompatibleEngine(base_url, model, api_key)
        return cls(engine, EngineType.TENSORRT_LLM)

    @classmethod
    def from_openai(
        cls,
        model: str = "gpt-4",
        api_key: str = "",
        base_url: str = "https://api.openai.com"
    ) -> "UnifiedInferenceClient":
        """
        Create a client connected to OpenAI API.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            base_url: API base URL (for Azure or proxies)

        Returns:
            UnifiedInferenceClient configured for OpenAI
        """
        engine = OpenAICompatibleEngine(base_url, model, api_key)
        return cls(engine, EngineType.OPENAI)

    @property
    def engine_type(self) -> EngineType:
        """Get the type of inference engine being used."""
        return self._engine_type

    def set_default_config(self, config: GenerationConfig) -> None:
        """Set the default generation configuration."""
        self._default_config = config

    def complete(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input text to complete
            config: Optional generation configuration

        Returns:
            The generated text
        """
        cfg = config or self._default_config
        response = self._engine.complete(prompt, cfg)
        return response.text

    def complete_with_metadata(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> CompletionResponse:
        """
        Generate a completion with full metadata.

        Args:
            prompt: The input text to complete
            config: Optional generation configuration

        Returns:
            CompletionResponse with text and timing information
        """
        cfg = config or self._default_config
        return self._engine.complete(prompt, cfg)

    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Stream a completion for the given prompt.

        Args:
            prompt: The input text to complete
            config: Optional generation configuration

        Yields:
            Text chunks as they are generated
        """
        cfg = config or self._default_config
        yield from self._engine.stream(prompt, cfg)

    def chat(
        self,
        messages: list[dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of chat messages (role + content)
            config: Optional generation configuration

        Returns:
            The assistant's response text
        """
        cfg = config or self._default_config
        response = self._engine.chat(messages, cfg)
        return response.text

    def chat_with_metadata(
        self,
        messages: list[dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> CompletionResponse:
        """
        Generate a chat completion with full metadata.

        Args:
            messages: List of chat messages (role + content)
            config: Optional generation configuration

        Returns:
            CompletionResponse with text and timing information
        """
        cfg = config or self._default_config
        return self._engine.chat(messages, cfg)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Stream a chat completion.

        Args:
            messages: List of chat messages (role + content)
            config: Optional generation configuration

        Yields:
            Text chunks as they are generated
        """
        cfg = config or self._default_config
        yield from self._engine.stream_chat(messages, cfg)

    def is_healthy(self) -> bool:
        """Check if the inference server is running and healthy."""
        return self._engine.is_healthy()

    def __repr__(self) -> str:
        return f"UnifiedInferenceClient(engine_type={self._engine_type.value})"


class ConversationManager:
    """
    Manages multi-turn conversations with automatic history tracking.

    Example:
        >>> client = UnifiedInferenceClient.from_ollama("llama3.1:8b")
        >>> conversation = ConversationManager(client, system_prompt="You are a helpful assistant.")
        >>>
        >>> # First turn
        >>> response = conversation.say("Hello!")
        >>> print(response)  # "Hello! How can I help you today?"
        >>>
        >>> # Second turn (history is preserved)
        >>> response = conversation.say("What did I just say?")
        >>> print(response)  # "You said 'Hello!'"
        >>>
        >>> # Reset conversation
        >>> conversation.reset()
    """

    def __init__(
        self,
        client: UnifiedInferenceClient,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize conversation manager.

        Args:
            client: The inference client to use
            system_prompt: Optional system message to set context
            config: Optional default generation configuration
        """
        self.client = client
        self.system_prompt = system_prompt
        self.config = config or GenerationConfig()
        self._history: list[dict[str, str]] = []

        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

    @property
    def history(self) -> list[dict[str, str]]:
        """Get the current conversation history."""
        return self._history.copy()

    def say(self, message: str) -> str:
        """
        Send a message and get a response.

        Args:
            message: The user's message

        Returns:
            The assistant's response
        """
        self._history.append({"role": "user", "content": message})
        response = self.client.chat(self._history, self.config)
        self._history.append({"role": "assistant", "content": response})
        return response

    def stream_say(self, message: str) -> Iterator[str]:
        """
        Send a message and stream the response.

        Args:
            message: The user's message

        Yields:
            Text chunks as they are generated
        """
        self._history.append({"role": "user", "content": message})

        full_response = []
        for chunk in self.client.stream_chat(self._history, self.config):
            full_response.append(chunk)
            yield chunk

        self._history.append({"role": "assistant", "content": "".join(full_response)})

    def reset(self) -> None:
        """Reset the conversation history."""
        self._history = []
        if self.system_prompt:
            self._history.append({"role": "system", "content": self.system_prompt})

    def add_assistant_message(self, content: str) -> None:
        """Manually add an assistant message to history."""
        self._history.append({"role": "assistant", "content": content})

    def add_user_message(self, content: str) -> None:
        """Manually add a user message to history."""
        self._history.append({"role": "user", "content": content})

    def pop_last(self) -> Optional[dict[str, str]]:
        """Remove and return the last message from history."""
        if self._history and (not self.system_prompt or len(self._history) > 1):
            return self._history.pop()
        return None

    def __len__(self) -> int:
        """Get the number of messages in history."""
        return len(self._history)


if __name__ == "__main__":
    # Demo usage
    print("Unified Inference Client Demo")
    print("=" * 50)

    # Try to connect to Ollama
    client = UnifiedInferenceClient.from_ollama("llama3.1:8b")

    if client.is_healthy():
        print("Connected to Ollama!")

        # Simple completion
        print("\n--- Simple Completion ---")
        response = client.complete("What is 2+2? Answer briefly.")
        print(f"Response: {response}")

        # Streaming
        print("\n--- Streaming ---")
        print("Response: ", end="")
        for chunk in client.stream("Tell me a very short joke."):
            print(chunk, end="", flush=True)
        print()

        # Conversation
        print("\n--- Conversation ---")
        convo = ConversationManager(
            client,
            system_prompt="You are a helpful assistant. Keep responses brief."
        )
        print(f"User: Hello!")
        print(f"Assistant: {convo.say('Hello!')}")
        print(f"User: What's your favorite color?")
        print(f"Assistant: {convo.say('What is your favorite color?')}")
    else:
        print("Could not connect to Ollama. Make sure it's running with: ollama serve")
