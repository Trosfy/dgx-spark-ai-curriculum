"""
Streamlit Utility Functions for Demo Building.

This module provides reusable utilities for building Streamlit demos,
including session state helpers, caching patterns, and common components.

Author: Professor SPARK
Module: 4.5 - Demo Building & Prototyping
"""

import streamlit as st
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps
from datetime import datetime
import time

T = TypeVar('T')


# =============================================================================
# SESSION STATE HELPERS
# =============================================================================

def init_state(key: str, default: Any) -> Any:
    """
    Initialize a session state variable if not present.

    Args:
        key: Session state key
        default: Default value if key doesn't exist

    Returns:
        The current value of the session state variable

    Example:
        messages = init_state("messages", [])
        settings = init_state("settings", {"theme": "light"})
    """
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def init_states(defaults: Dict[str, Any]) -> None:
    """
    Initialize multiple session state variables at once.

    Args:
        defaults: Dictionary of key: default_value pairs

    Example:
        init_states({
            "messages": [],
            "user_name": "Guest",
            "settings": {"theme": "light"}
        })
    """
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_state(key: str) -> None:
    """
    Clear a session state variable.

    Args:
        key: Session state key to clear
    """
    if key in st.session_state:
        del st.session_state[key]


def clear_states(keys: List[str]) -> None:
    """
    Clear multiple session state variables.

    Args:
        keys: List of session state keys to clear
    """
    for key in keys:
        clear_state(key)


# =============================================================================
# CACHING PATTERNS
# =============================================================================

def cached_model_loader(load_fn: Callable[[], T]) -> Callable[[], T]:
    """
    Decorator pattern for caching model loading.

    This wraps your model loading function with st.cache_resource.

    Args:
        load_fn: Function that loads and returns the model

    Returns:
        Cached model loader function

    Example:
        @cached_model_loader
        def load_my_model():
            return HuggingFaceModel("gpt2")

        model = load_my_model()  # Loads once, cached forever
    """
    return st.cache_resource(load_fn)


def cached_data_loader(ttl_seconds: int = 3600):
    """
    Decorator for caching data with time-to-live.

    Args:
        ttl_seconds: How long to cache the data (default: 1 hour)

    Returns:
        Decorator function

    Example:
        @cached_data_loader(ttl_seconds=300)  # 5 minutes
        def fetch_data(query):
            return api.search(query)
    """
    def decorator(func: Callable) -> Callable:
        return st.cache_data(ttl=ttl_seconds)(func)
    return decorator


# =============================================================================
# UI COMPONENTS
# =============================================================================

def show_message(
    message: str,
    type: str = "info",
    icon: Optional[str] = None
) -> None:
    """
    Display a styled message.

    Args:
        message: Message text to display
        type: One of "info", "success", "warning", "error"
        icon: Optional emoji icon

    Example:
        show_message("Settings saved!", type="success", icon="✅")
        show_message("Please wait...", type="warning")
    """
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    final_icon = icon or icons.get(type, "")

    if type == "success":
        st.success(f"{final_icon} {message}")
    elif type == "warning":
        st.warning(f"{final_icon} {message}")
    elif type == "error":
        st.error(f"{final_icon} {message}")
    else:
        st.info(f"{final_icon} {message}")


def show_metrics_row(metrics: Dict[str, Any], columns: int = 4) -> None:
    """
    Display a row of metrics.

    Args:
        metrics: Dictionary of {label: value} or {label: (value, delta)}
        columns: Number of columns (default: 4)

    Example:
        show_metrics_row({
            "Users": 1000,
            "Revenue": ("$50K", "+10%"),
            "Churn": ("5%", "-2%")
        })
    """
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            if isinstance(value, tuple):
                st.metric(label, value[0], value[1])
            else:
                st.metric(label, value)


def page_header(
    title: str,
    description: Optional[str] = None,
    icon: str = "🚀"
) -> None:
    """
    Display a consistent page header.

    Args:
        title: Page title
        description: Optional description
        icon: Emoji icon for the title

    Example:
        page_header("Dashboard", "View your metrics", icon="📊")
    """
    st.title(f"{icon} {title}")
    if description:
        st.markdown(description)
    st.markdown("---")


# =============================================================================
# CHAT HELPERS
# =============================================================================

def chat_message(
    role: str,
    content: str,
    avatar: Optional[str] = None,
    timestamp: Optional[str] = None
) -> None:
    """
    Display a chat message with optional metadata.

    Args:
        role: "user" or "assistant"
        content: Message content
        avatar: Optional avatar image path or emoji
        timestamp: Optional timestamp string

    Example:
        chat_message("user", "Hello!", avatar="👤")
        chat_message("assistant", "Hi there!", avatar="🤖")
    """
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if timestamp:
            st.caption(timestamp)


def chat_history(
    messages: List[Dict[str, Any]],
    show_timestamps: bool = False
) -> None:
    """
    Display a full chat history.

    Args:
        messages: List of message dicts with "role" and "content" keys
        show_timestamps: Whether to show timestamps

    Example:
        chat_history(st.session_state.messages)
    """
    for msg in messages:
        chat_message(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            avatar=msg.get("avatar"),
            timestamp=msg.get("timestamp") if show_timestamps else None
        )


# =============================================================================
# PROGRESS HELPERS
# =============================================================================

def run_with_progress(
    items: List[Any],
    process_fn: Callable[[Any], Any],
    message: str = "Processing..."
) -> List[Any]:
    """
    Process items with a progress bar.

    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        message: Progress bar message

    Returns:
        List of processed results

    Example:
        results = run_with_progress(
            files,
            process_file,
            "Processing files..."
        )
    """
    results = []
    progress_bar = st.progress(0, text=message)

    for i, item in enumerate(items):
        results.append(process_fn(item))
        progress_bar.progress(
            (i + 1) / len(items),
            text=f"{message} ({i + 1}/{len(items)})"
        )

    progress_bar.empty()
    return results


# =============================================================================
# ERROR HANDLING
# =============================================================================

def safe_run(
    func: Callable,
    error_message: str = "An error occurred",
    show_details: bool = False
) -> Callable:
    """
    Decorator to safely run functions with error handling.

    Args:
        func: Function to wrap
        error_message: User-friendly error message
        show_details: Whether to show technical details

    Returns:
        Wrapped function

    Example:
        @safe_run(error_message="Failed to load data")
        def load_data():
            return risky_operation()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ {error_message}")
            if show_details:
                with st.expander("Technical Details"):
                    st.code(str(e))
            return None
    return wrapper


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Streamlit utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - init_state(key, default)")
    print("  - init_states(defaults)")
    print("  - cached_model_loader(load_fn)")
    print("  - cached_data_loader(ttl)")
    print("  - show_message(msg, type)")
    print("  - show_metrics_row(metrics)")
    print("  - page_header(title, desc)")
    print("  - chat_message(role, content)")
    print("  - chat_history(messages)")
    print("  - run_with_progress(items, fn)")
    print("  - safe_run(fn)")
