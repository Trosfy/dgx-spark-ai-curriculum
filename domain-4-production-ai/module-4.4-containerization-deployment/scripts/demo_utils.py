"""
Demo Building Utilities for ML Applications

This module provides utilities for building interactive demos with:
- Gradio interfaces
- Streamlit dashboards
- LLM client wrappers with streaming support

Example usage:
    from demo_utils import create_gradio_chat_interface, StreamingLLMClient

    # Create a streaming LLM client
    client = StreamingLLMClient(model="llama3.1:8b")

    # Create Gradio chat interface
    demo = create_gradio_chat_interface(client)
    demo.launch()
"""

import os
import json
import time
import requests
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Iterator, Callable, Union
from abc import ABC, abstractmethod


@dataclass
class Message:
    """Represents a chat message."""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Response from an LLM."""

    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0
    metadata: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    def stream_chat(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """Stream a chat completion response."""
        pass


class StreamingLLMClient(LLMClient):
    """
    Universal LLM client with streaming support.

    Supports multiple backends:
    - Ollama (local)
    - OpenAI API
    - OpenAI-compatible APIs (vLLM, SGLang, etc.)

    Example:
        >>> client = StreamingLLMClient(model="qwen3:8b", backend="ollama")
        >>> for chunk in client.stream_chat([Message("user", "Hello!")]):
        ...     print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        backend: str = "ollama",  # "ollama", "openai", "openai-compatible"
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name/ID
            backend: Backend type
            base_url: API base URL (auto-detected for common backends)
            api_key: API key (if required)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Default system prompt
        """
        self.model = model
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        # Set base URL based on backend
        if base_url:
            self.base_url = base_url
        elif backend == "ollama":
            self.base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        elif backend == "openai":
            self.base_url = "https://api.openai.com/v1"
        else:
            self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def _build_messages(
        self,
        messages: List[Message],
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """Build message list with optional system prompt."""
        result = []

        if include_system and self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})

        for msg in messages:
            result.append(msg.to_dict())

        return result

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of messages
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            ChatResponse object
        """
        start_time = time.time()
        built_messages = self._build_messages(messages)

        if self.backend == "ollama":
            response = self._ollama_chat(
                built_messages,
                temperature or self.temperature,
            )
        else:
            response = self._openai_chat(
                built_messages,
                temperature or self.temperature,
                max_tokens or self.max_tokens,
            )

        latency = (time.time() - start_time) * 1000
        response.latency_ms = latency

        return response

    def _ollama_chat(
        self,
        messages: List[Dict],
        temperature: float,
    ) -> ChatResponse:
        """Chat using Ollama API."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        return ChatResponse(
            content=data["message"]["content"],
            model=self.model,
            tokens_used=data.get("eval_count", 0),
        )

    def _openai_chat(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        """Chat using OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ChatResponse(
            content=choice["message"]["content"],
            model=self.model,
            tokens_used=usage.get("total_tokens", 0),
        )

    def stream_chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Stream a chat completion response.

        Args:
            messages: List of messages
            temperature: Override temperature

        Yields:
            String chunks of the response
        """
        built_messages = self._build_messages(messages)

        if self.backend == "ollama":
            yield from self._ollama_stream(
                built_messages,
                temperature or self.temperature,
            )
        else:
            yield from self._openai_stream(
                built_messages,
                temperature or self.temperature,
            )

    def _ollama_stream(
        self,
        messages: List[Dict],
        temperature: float,
    ) -> Iterator[str]:
        """Stream using Ollama API."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]

    def _openai_stream(
        self,
        messages: List[Dict],
        temperature: float,
    ) -> Iterator[str]:
        """Stream using OpenAI-compatible API."""
        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without a real model.

    Useful for:
    - UI development
    - Testing
    - Demos without GPU
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        delay_per_token: float = 0.02,
    ):
        """
        Initialize mock client.

        Args:
            responses: List of canned responses (cycles through)
            delay_per_token: Simulated delay per token
        """
        self.responses = responses or [
            "Hello! I'm a mock AI assistant. I'm here to help you test your interface.",
            "That's a great question! Let me think about it...\n\nHere's what I know: Mock responses are useful for testing UI without needing a real model.",
            "I can help with that! Here's a simple example:\n\n```python\nprint('Hello, World!')\n```\n\nThis code prints a greeting.",
        ]
        self.delay = delay_per_token
        self._response_idx = 0

    def _get_next_response(self) -> str:
        response = self.responses[self._response_idx % len(self.responses)]
        self._response_idx += 1
        return response

    def chat(self, messages: List[Message], **kwargs) -> ChatResponse:
        response = self._get_next_response()
        time.sleep(len(response) * self.delay)

        return ChatResponse(
            content=response,
            model="mock-model",
            tokens_used=len(response.split()),
        )

    def stream_chat(self, messages: List[Message], **kwargs) -> Iterator[str]:
        response = self._get_next_response()

        # Simulate token-by-token streaming
        words = response.split()
        for i, word in enumerate(words):
            time.sleep(self.delay)
            if i > 0:
                yield " "
            yield word


def create_gradio_chat_interface(
    client: LLMClient,
    title: str = "AI Chat Demo",
    description: str = "Chat with an AI assistant",
    examples: Optional[List[str]] = None,
    theme: str = "soft",
    enable_file_upload: bool = False,
    custom_css: Optional[str] = None,
) -> Any:
    """
    Create a Gradio chat interface.

    Args:
        client: LLM client instance
        title: Interface title
        description: Interface description
        examples: Example prompts
        theme: Gradio theme name
        enable_file_upload: Enable file upload
        custom_css: Custom CSS styling

    Returns:
        Gradio Interface object
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required. Install with: pip install gradio")

    # Default examples
    examples = examples or [
        "Hello! How are you?",
        "Explain machine learning in simple terms",
        "Write a Python function to calculate fibonacci numbers",
    ]

    def chat_fn(message: str, history: List) -> Iterator[str]:
        """Chat function with streaming."""
        # Convert history to messages
        messages = []
        for h in history:
            if isinstance(h, (list, tuple)) and len(h) == 2:
                messages.append(Message("user", h[0]))
                messages.append(Message("assistant", h[1]))

        messages.append(Message("user", message))

        # Stream response
        full_response = ""
        for chunk in client.stream_chat(messages):
            full_response += chunk
            yield full_response

    # Select theme
    if theme == "soft":
        gr_theme = gr.themes.Soft()
    elif theme == "default":
        gr_theme = gr.themes.Default()
    elif theme == "monochrome":
        gr_theme = gr.themes.Monochrome()
    elif theme == "glass":
        gr_theme = gr.themes.Glass()
    else:
        gr_theme = gr.themes.Soft()

    # Create interface
    demo = gr.ChatInterface(
        fn=chat_fn,
        title=title,
        description=description,
        examples=examples,
        theme=gr_theme,
        css=custom_css,
    )

    return demo


def create_gradio_inference_interface(
    inference_fn: Callable,
    title: str = "ML Inference Demo",
    input_components: Optional[List] = None,
    output_components: Optional[List] = None,
    examples: Optional[List] = None,
) -> Any:
    """
    Create a Gradio interface for general ML inference.

    Args:
        inference_fn: Function that takes inputs and returns outputs
        title: Interface title
        input_components: Gradio input components
        output_components: Gradio output components
        examples: Example inputs

    Returns:
        Gradio Interface object
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required. Install with: pip install gradio")

    # Default components
    inputs = input_components or [gr.Textbox(label="Input")]
    outputs = output_components or [gr.Textbox(label="Output")]

    demo = gr.Interface(
        fn=inference_fn,
        inputs=inputs,
        outputs=outputs,
        title=title,
        examples=examples,
        theme=gr.themes.Soft(),
    )

    return demo


def create_streamlit_dashboard(
    client: LLMClient,
    title: str = "AI Dashboard",
    pages: Optional[Dict[str, Callable]] = None,
) -> str:
    """
    Generate a Streamlit dashboard script.

    This function generates a complete Streamlit script that can be
    saved and run with `streamlit run`.

    Args:
        client: LLM client configuration (for reference)
        title: Dashboard title
        pages: Dictionary of page name -> page function

    Returns:
        Python script as string
    """
    script = f'''"""
Auto-generated Streamlit Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import time
from typing import List, Dict

# ============================================
# Configuration
# ============================================

st.set_page_config(
    page_title="{title}",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Session State Initialization
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# ============================================
# Sidebar Navigation
# ============================================

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Chat", "Metrics", "Settings"]
)

# ============================================
# LLM Client Setup
# ============================================

@st.cache_resource
def get_llm_client():
    """Get or create LLM client."""
    try:
        import requests

        # Try Ollama first
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                return "ollama"
        except:
            pass

        return "mock"
    except:
        return "mock"


def get_response(messages: List[Dict], stream: bool = True):
    """Get response from LLM."""
    client_type = get_llm_client()

    if client_type == "ollama":
        import requests
        import json

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={{
                "model": "qwen3:8b",
                "messages": messages,
                "stream": stream,
            }},
            stream=stream,
        )

        if stream:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        yield data["message"]["content"]
        else:
            yield response.json()["message"]["content"]
    else:
        # Mock response
        mock_response = "Hello! I'm a simulated AI assistant. To use a real model, ensure Ollama is running."
        for word in mock_response.split():
            time.sleep(0.05)
            yield word + " "

# ============================================
# Page: Chat
# ============================================

def page_chat():
    st.title("💬 Chat")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message
        st.session_state.messages.append({{"role": "user", "content": prompt}})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in get_response(st.session_state.messages):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.messages.append({{"role": "assistant", "content": full_response}})
        st.session_state.total_tokens += len(full_response.split())

    # Clear button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ============================================
# Page: Metrics
# ============================================

def page_metrics():
    st.title("📊 Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Messages",
            len(st.session_state.messages),
            delta=None,
        )

    with col2:
        st.metric(
            "Tokens Used",
            st.session_state.total_tokens,
            delta=None,
        )

    with col3:
        client_type = get_llm_client()
        st.metric(
            "Backend",
            client_type.upper(),
            delta=None,
        )

    # Chat history visualization
    if st.session_state.messages:
        st.subheader("Conversation Length Over Time")

        lengths = []
        for i, msg in enumerate(st.session_state.messages):
            lengths.append({{
                "message": i + 1,
                "length": len(msg["content"]),
                "role": msg["role"],
            }})

        import pandas as pd
        df = pd.DataFrame(lengths)
        st.bar_chart(df.set_index("message")["length"])

# ============================================
# Page: Settings
# ============================================

def page_settings():
    st.title("⚙️ Settings")

    st.subheader("Model Configuration")

    model = st.selectbox(
        "Model",
        ["qwen3:8b", "qwen3:32b", "codellama:7b", "mistral:7b"],
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
    )

    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=8192,
        value=2048,
    )

    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant.",
        height=100,
    )

    if st.button("Save Settings"):
        st.success("Settings saved!")

# ============================================
# Main Router
# ============================================

if page == "Chat":
    page_chat()
elif page == "Metrics":
    page_metrics()
elif page == "Settings":
    page_settings()

# ============================================
# Footer
# ============================================

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit + DGX Spark")
'''

    return script


def generate_gradio_space_config(
    title: str,
    sdk: str = "gradio",
    sdk_version: str = "4.0",
    python_version: str = "3.10",
    requirements: Optional[List[str]] = None,
) -> str:
    """
    Generate a Hugging Face Spaces configuration.

    Args:
        title: Space title
        sdk: SDK type (gradio or streamlit)
        sdk_version: SDK version
        python_version: Python version
        requirements: Additional pip requirements

    Returns:
        README.md content with Spaces config
    """
    reqs = requirements or ["transformers", "torch"]
    reqs_str = "\n".join(f"  - {r}" for r in reqs)

    config = f"""---
title: {title}
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: {sdk}
sdk_version: "{sdk_version}"
python_version: "{python_version}"
app_file: app.py
pinned: false
---

# {title}

This is a demo application deployed on Hugging Face Spaces.

## Requirements

{reqs_str}

## Usage

This demo runs a chat interface powered by a language model.

## Built With

- DGX Spark AI Curriculum
- Gradio {sdk_version}
- Python {python_version}
"""

    return config


def create_inference_api_client(
    endpoint_url: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> Callable:
    """
    Create a client function for a deployed inference API.

    Args:
        endpoint_url: API endpoint URL
        api_key: API key for authentication
        timeout: Request timeout in seconds

    Returns:
        Function that can be used with Gradio/Streamlit
    """
    def inference_fn(text: str) -> str:
        """Call the inference API."""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.post(
                endpoint_url,
                json={"inputs": text},
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()

            # Handle common response formats
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return result.get("generated_text", result.get("output", str(result)))
            elif isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", str(result[0]))
            else:
                return str(result)
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    return inference_fn


if __name__ == "__main__":
    # Example usage
    print("=== Demo Utils Examples ===\n")

    # Test mock client
    print("Testing MockLLMClient:")
    mock_client = MockLLMClient()

    messages = [Message("user", "Hello!")]
    response = mock_client.chat(messages)
    print(f"Response: {response.content[:50]}...")

    print("\nStreaming test:")
    for chunk in mock_client.stream_chat(messages):
        print(chunk, end="", flush=True)
    print("\n")

    # Generate Streamlit dashboard script
    print("\n=== Generated Streamlit Dashboard ===\n")
    script = create_streamlit_dashboard(mock_client, "My AI Dashboard")
    print(script[:1000] + "...\n")

    # Generate Spaces config
    print("=== Hugging Face Spaces Config ===\n")
    config = generate_gradio_space_config("My AI Demo")
    print(config)
