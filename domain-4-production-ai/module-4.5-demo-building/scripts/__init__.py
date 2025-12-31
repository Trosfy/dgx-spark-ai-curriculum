"""
Demo Building Utilities

This module provides utilities for building polished demos with Gradio and Streamlit.
"""

from .gradio_utils import (
    create_theme,
    safe_handler,
    add_progress_indicator,
    create_examples_from_files,
)

from .streamlit_utils import (
    init_session_state,
    cached_model_loader,
    render_chat_message,
    create_download_button,
)

from .demo_utils import (
    DemoConfig,
    validate_input,
    format_error_message,
    create_deployment_files,
)

__all__ = [
    # Gradio utilities
    "create_theme",
    "safe_handler",
    "add_progress_indicator",
    "create_examples_from_files",
    # Streamlit utilities
    "init_session_state",
    "cached_model_loader",
    "render_chat_message",
    "create_download_button",
    # Demo utilities
    "DemoConfig",
    "validate_input",
    "format_error_message",
    "create_deployment_files",
]
