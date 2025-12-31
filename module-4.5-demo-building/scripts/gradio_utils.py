"""
Gradio Utility Functions for Demo Building.

This module provides reusable utilities for building Gradio demos,
including theme helpers, error handling wrappers, and common patterns.

Author: Professor SPARK
Module: 4.5 - Demo Building & Prototyping
"""

import gradio as gr
from functools import wraps
from typing import Callable, Any, List, Optional
import random
import time


# =============================================================================
# FRIENDLY ERROR MESSAGES
# =============================================================================

FRIENDLY_ERRORS = [
    "Hmm, I'm having trouble with that. Could you try asking differently? 🤔",
    "That's a tricky one! Let me suggest trying a simpler input first.",
    "Oops! Something went sideways. Let's try again with different input.",
    "I encountered an issue. Please try one of the example inputs!",
    "Sorry, I couldn't process that. Could you rephrase it?",
]


def safe_execute(default_return: Any = None):
    """
    Decorator that wraps functions in error handling.

    Returns a friendly error message instead of crashing.

    Example:
        @safe_execute(default_return="Error occurred")
        def my_function(x):
            return risky_operation(x)

    Args:
        default_return: What to return if an error occurs (besides the message)

    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the real error (in production, use proper logging)
                print(f"ERROR in {func.__name__}: {e}")

                # Return friendly message
                friendly_msg = random.choice(FRIENDLY_ERRORS)

                if default_return is not None:
                    return friendly_msg, default_return
                return friendly_msg
        return wrapper
    return decorator


# =============================================================================
# THEME PRESETS
# =============================================================================

def get_nvidia_theme() -> gr.themes.Soft:
    """
    Get a custom NVIDIA-branded theme.

    Returns:
        gr.themes.Soft: A Gradio theme with NVIDIA green styling

    Example:
        with gr.Blocks(theme=get_nvidia_theme()) as demo:
            ...
    """
    return gr.themes.Soft(
        primary_hue="green",
        secondary_hue="gray",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        button_primary_background_fill="#76b900",  # NVIDIA green
        button_primary_background_fill_hover="#5a9000",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="500",
    )


def get_professional_theme(
    primary_color: str = "blue",
    accent_color: str = "indigo"
) -> gr.themes.Soft:
    """
    Get a professional, customizable theme.

    Args:
        primary_color: Primary hue name (e.g., "blue", "green", "purple")
        accent_color: Accent hue name

    Returns:
        gr.themes.Soft: A professional Gradio theme

    Example:
        theme = get_professional_theme(primary_color="purple")
        with gr.Blocks(theme=theme) as demo:
            ...
    """
    return gr.themes.Soft(
        primary_hue=primary_color,
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        button_primary_background_fill=f"*{accent_color}_500",
        button_primary_background_fill_hover=f"*{accent_color}_600",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="500",
        input_background_fill="#f8fafc",
    )


# =============================================================================
# COMMON CSS SNIPPETS
# =============================================================================

PROFESSIONAL_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.source-citation {
    background-color: #f0f9ff;
    border-left: 4px solid #0284c7;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.9em;
}

.success-message {
    background-color: #dcfce7;
    border-left: 4px solid #16a34a;
    padding: 0.75rem;
    border-radius: 0 8px 8px 0;
}

.error-message {
    background-color: #fef2f2;
    border-left: 4px solid #dc2626;
    padding: 0.75rem;
    border-radius: 0 8px 8px 0;
}

.warning-message {
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 0.75rem;
    border-radius: 0 8px 8px 0;
}

footer {display: none !important;}
"""


# =============================================================================
# PROGRESS HELPERS
# =============================================================================

def create_progress_callback(progress: gr.Progress):
    """
    Create a progress callback for long-running operations.

    Args:
        progress: Gradio Progress object

    Returns:
        Callable that updates progress

    Example:
        def long_operation(files, progress=gr.Progress()):
            callback = create_progress_callback(progress)
            for i, file in enumerate(files):
                process(file)
                callback(i + 1, len(files), f"Processing {file.name}")
    """
    def update_progress(current: int, total: int, message: str = ""):
        progress((current / total), desc=message)
    return update_progress


# =============================================================================
# GOLDEN EXAMPLES HELPER
# =============================================================================

class GoldenExamples:
    """
    Manage golden path examples that always work.

    Example:
        examples = GoldenExamples()
        examples.add("What is machine learning?", "ML is...")
        examples.add("Explain neural networks", "Neural networks are...")

        # In Gradio:
        gr.Examples(
            examples=examples.get_inputs(),
            inputs=input_box,
            outputs=output_box,
            fn=examples.get_cached_response
        )
    """

    def __init__(self):
        self._examples = {}

    def add(self, input_text: str, expected_output: str, category: str = "default"):
        """Add a golden example."""
        self._examples[input_text] = {
            "output": expected_output,
            "category": category
        }

    def get_inputs(self) -> List[List[str]]:
        """Get list of input examples for gr.Examples."""
        return [[text] for text in self._examples.keys()]

    def get_cached_response(self, input_text: str) -> str:
        """Get cached response for a golden example."""
        if input_text in self._examples:
            return self._examples[input_text]["output"]
        return None

    def is_golden(self, input_text: str) -> bool:
        """Check if input is a golden example."""
        return input_text in self._examples


# =============================================================================
# DEMO MODE HELPER
# =============================================================================

class DemoMode:
    """
    Helper class for demo mode with cached responses.

    Example:
        demo_mode = DemoMode(enabled=True)
        demo_mode.add_response("hello", "Hello! How can I help?")

        @demo_mode.with_fallback
        def respond(message):
            return expensive_llm_call(message)
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._cached_responses = {}

    def add_response(self, input_text: str, response: str):
        """Add a cached demo response."""
        self._cached_responses[input_text.lower().strip()] = response

    def get_response(self, input_text: str) -> Optional[str]:
        """Get cached response if available."""
        if not self.enabled:
            return None
        return self._cached_responses.get(input_text.lower().strip())

    def with_fallback(self, func: Callable) -> Callable:
        """Decorator to use cached responses when available."""
        @wraps(func)
        def wrapper(input_text: str, *args, **kwargs):
            cached = self.get_response(input_text)
            if cached is not None:
                time.sleep(0.3)  # Simulate processing
                return cached
            return func(input_text, *args, **kwargs)
        return wrapper


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Using the utilities

    # Create a demo with professional theme
    theme = get_professional_theme(primary_color="indigo")

    # Set up golden examples
    examples = GoldenExamples()
    examples.add("What is AI?", "AI is artificial intelligence...")
    examples.add("Explain RAG", "RAG is Retrieval-Augmented Generation...")

    # Create demo mode
    demo_mode = DemoMode(enabled=True)
    demo_mode.add_response("hello", "Hello! I'm your AI assistant.")

    @safe_execute(default_return="")
    @demo_mode.with_fallback
    def respond(message: str) -> str:
        """Example response function with all utilities."""
        return f"Processing: {message}"

    print("Utilities loaded successfully!")
    print(f"Golden examples: {examples.get_inputs()}")
    print(f"Test response: {respond('hello')}")
