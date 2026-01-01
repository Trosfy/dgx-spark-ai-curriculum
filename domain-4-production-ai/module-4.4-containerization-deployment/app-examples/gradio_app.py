"""
Gradio Chat Application.

A production-ready Gradio chat interface for LLMs.
Can connect to Ollama or any OpenAI-compatible API.

Usage:
    python gradio_app.py
    # Opens at http://localhost:7860

Environment variables:
    OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    MODEL_NAME: Model to use (default: qwen3:8b)
"""

import os
import time
from typing import List, Tuple, Iterator

import gradio as gr
import requests

# ============================================
# Configuration
# ============================================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3:8b")

# ============================================
# LLM Client
# ============================================


def check_ollama() -> bool:
    """Check if Ollama is available."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> List[str]:
    """Get list of available models."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except:
        pass
    return ["qwen3:8b", "mistral:7b", "codellama:7b"]  # Defaults


def stream_response(
    message: str,
    history: List[Tuple[str, str]],
    model: str = MODEL_NAME,
    temperature: float = 0.7,
) -> Iterator[str]:
    """Stream response from Ollama."""
    # Build messages
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
            stream=True,
            timeout=120,
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    full_response += data["message"]["content"]
                    yield full_response

    except Exception as e:
        yield f"Error: {str(e)}\n\nMake sure Ollama is running at {OLLAMA_HOST}"


# ============================================
# Gradio Interface
# ============================================

# Check Ollama status
ollama_available = check_ollama()
available_models = get_available_models()

# Custom CSS
custom_css = """
.gradio-container {
    max-width: 900px !important;
}
footer {
    display: none !important;
}
"""

# Build interface
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="green",
        neutral_hue="slate",
    ),
    css=custom_css,
    title="AI Chat",
) as demo:
    gr.Markdown("# AI Chat Demo")
    gr.Markdown(
        f"**Status:** {'Connected to Ollama' if ollama_available else 'Ollama not available (mock mode)'}"
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=MODEL_NAME if MODEL_NAME in available_models else available_models[0],
                label="Model",
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            )

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=lambda msg, hist: stream_response(
                    msg, hist,
                    model=model_dropdown.value,
                    temperature=temperature_slider.value,
                ),
                examples=[
                    "Hello! How are you?",
                    "Explain machine learning in simple terms",
                    "Write a Python function to reverse a string",
                    "What are the benefits of using Docker?",
                ],
                retry_btn="Retry",
                undo_btn="Undo",
                clear_btn="Clear",
            )

    gr.Markdown("---")
    gr.Markdown("Built with Gradio for the DGX Spark AI Curriculum")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
