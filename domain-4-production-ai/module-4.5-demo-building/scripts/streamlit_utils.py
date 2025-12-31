"""
Streamlit Utilities for Demo Building

Production-quality utilities for building polished Streamlit demos.

Example usage:
    >>> from streamlit_utils import init_session_state, cached_model_loader
    >>> init_session_state({"messages": [], "user": None})
    >>> model = cached_model_loader("llama3.2:3b")
"""

import streamlit as st
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps
import json
import time
from datetime import datetime

T = TypeVar('T')


def init_session_state(defaults: Dict[str, Any]) -> None:
    """
    Initialize session state with default values.

    Only sets values that don't already exist, preserving
    existing session state across reruns.

    Args:
        defaults: Dictionary of default values for session state

    Example:
        >>> init_session_state({
        ...     "messages": [],
        ...     "user_name": "Guest",
        ...     "settings": {"theme": "light"}
        ... })
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def cached_model_loader(
    model_name: str,
    loader_fn: Callable[[], T] = None,
    **loader_kwargs
) -> T:
    """
    Load a model with caching using st.cache_resource.

    This ensures the model is only loaded once per app lifetime,
    not on every rerun.

    Args:
        model_name: Name/identifier for the model
        loader_fn: Optional custom loader function
        **loader_kwargs: Arguments passed to the loader

    Returns:
        The loaded model/resource

    Example:
        >>> # With Ollama
        >>> client = cached_model_loader("ollama")
        >>>
        >>> # With custom loader
        >>> model = cached_model_loader(
        ...     "my_model",
        ...     loader_fn=lambda: MyModel.load("path/to/model")
        ... )
    """
    @st.cache_resource
    def _load_model(name: str):
        if loader_fn is not None:
            return loader_fn(**loader_kwargs)

        # Default loaders for common frameworks
        if name == "ollama":
            import ollama
            return ollama.Client()
        elif name.startswith("openai"):
            from openai import OpenAI
            return OpenAI()
        else:
            raise ValueError(f"No default loader for {name}. Provide loader_fn.")

    return _load_model(model_name)


def render_chat_message(
    role: str,
    content: str,
    tool_calls: List[Dict] = None,
    thinking: str = None,
    timestamp: datetime = None
) -> None:
    """
    Render a chat message with optional tool calls and thinking.

    Args:
        role: "user" or "assistant"
        content: Message content
        tool_calls: Optional list of tool call dictionaries
        thinking: Optional thinking/reasoning text
        timestamp: Optional message timestamp

    Example:
        >>> render_chat_message(
        ...     role="assistant",
        ...     content="The answer is 42.",
        ...     tool_calls=[{"tool": "calculator", "result": "42"}],
        ...     thinking="I need to use the calculator..."
        ... )
    """
    with st.chat_message(role):
        # Main content
        st.write(content)

        # Timestamp
        if timestamp:
            st.caption(timestamp.strftime("%H:%M:%S"))

        # Tool calls
        if tool_calls:
            with st.expander("🔧 Tool Calls", expanded=False):
                for tc in tool_calls:
                    st.markdown(f"**{tc.get('tool', 'Unknown')}**")
                    if "args" in tc:
                        st.code(json.dumps(tc["args"], indent=2), language="json")
                    if "result" in tc:
                        st.success(f"Result: {tc['result']}")

        # Thinking
        if thinking:
            with st.expander("💭 Thinking", expanded=False):
                st.markdown(thinking)


def create_download_button(
    data: Any,
    filename: str,
    file_type: str = "json",
    button_label: str = "Download"
) -> None:
    """
    Create a download button for various data types.

    Args:
        data: Data to download
        filename: Name of the downloaded file
        file_type: Type of file ("json", "csv", "txt", "md")
        button_label: Text shown on the button

    Example:
        >>> create_download_button(
        ...     data={"results": [1, 2, 3]},
        ...     filename="results.json",
        ...     file_type="json"
        ... )
    """
    if file_type == "json":
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, indent=2)
        mime = "application/json"
    elif file_type == "csv":
        if hasattr(data, "to_csv"):
            content = data.to_csv(index=False)
        else:
            content = str(data)
        mime = "text/csv"
    elif file_type in ["txt", "md"]:
        content = str(data)
        mime = "text/plain" if file_type == "txt" else "text/markdown"
    else:
        content = str(data)
        mime = "application/octet-stream"

    st.download_button(
        label=button_label,
        data=content,
        file_name=filename,
        mime=mime
    )


def with_loading_indicator(message: str = "Loading..."):
    """
    Decorator that adds a loading spinner to a function.

    Args:
        message: Message to display during loading

    Example:
        >>> @with_loading_indicator("Fetching data...")
        ... def fetch_data():
        ...     time.sleep(2)
        ...     return {"data": "here"}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner(message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def create_metric_row(metrics: Dict[str, Any], columns: int = 4) -> None:
    """
    Create a row of metrics.

    Args:
        metrics: Dictionary of {label: value} or {label: (value, delta)}
        columns: Number of columns

    Example:
        >>> create_metric_row({
        ...     "Users": 1234,
        ...     "Revenue": ("$5,678", "+12%"),
        ...     "Accuracy": "95%"
        ... })
    """
    cols = st.columns(columns)

    for i, (label, value) in enumerate(metrics.items()):
        col = cols[i % columns]

        if isinstance(value, tuple):
            col.metric(label, value[0], value[1])
        else:
            col.metric(label, value)


def create_sidebar_nav(pages: Dict[str, str]) -> str:
    """
    Create a sidebar navigation menu.

    Args:
        pages: Dictionary of {page_name: description}

    Returns:
        Selected page name

    Example:
        >>> selected = create_sidebar_nav({
        ...     "Home": "Main dashboard",
        ...     "Settings": "Configure options",
        ...     "About": "Learn more"
        ... })
        >>> if selected == "Home":
        ...     show_home()
    """
    with st.sidebar:
        st.markdown("### Navigation")

        selected = st.radio(
            "Go to",
            options=list(pages.keys()),
            format_func=lambda x: f"{x}",
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.caption(pages.get(selected, ""))

    return selected


def safe_run(func: Callable, error_message: str = None) -> Any:
    """
    Run a function with error handling for Streamlit.

    Args:
        func: Function to run
        error_message: Custom error message to display

    Returns:
        Function result or None if error occurred

    Example:
        >>> result = safe_run(
        ...     lambda: api.fetch_data(),
        ...     error_message="Failed to fetch data"
        ... )
    """
    try:
        return func()
    except Exception as e:
        if error_message:
            st.error(error_message)
        else:
            st.error(f"An error occurred: {str(e)}")
        return None


def create_form(
    fields: Dict[str, Dict],
    submit_label: str = "Submit"
) -> Optional[Dict]:
    """
    Create a form with various field types.

    Args:
        fields: Dictionary defining form fields
        submit_label: Text for submit button

    Returns:
        Dictionary of field values if submitted, None otherwise

    Example:
        >>> result = create_form({
        ...     "name": {"type": "text", "label": "Your Name"},
        ...     "age": {"type": "number", "label": "Age", "min": 0, "max": 120},
        ...     "agree": {"type": "checkbox", "label": "I agree"}
        ... })
        >>> if result:
        ...     process_form(result)
    """
    with st.form("dynamic_form"):
        values = {}

        for field_name, config in fields.items():
            field_type = config.get("type", "text")
            label = config.get("label", field_name)

            if field_type == "text":
                values[field_name] = st.text_input(
                    label,
                    value=config.get("default", "")
                )
            elif field_type == "textarea":
                values[field_name] = st.text_area(
                    label,
                    value=config.get("default", "")
                )
            elif field_type == "number":
                values[field_name] = st.number_input(
                    label,
                    min_value=config.get("min"),
                    max_value=config.get("max"),
                    value=config.get("default", 0)
                )
            elif field_type == "slider":
                values[field_name] = st.slider(
                    label,
                    min_value=config.get("min", 0),
                    max_value=config.get("max", 100),
                    value=config.get("default", 50)
                )
            elif field_type == "checkbox":
                values[field_name] = st.checkbox(
                    label,
                    value=config.get("default", False)
                )
            elif field_type == "select":
                values[field_name] = st.selectbox(
                    label,
                    options=config.get("options", []),
                    index=config.get("default", 0)
                )
            elif field_type == "multiselect":
                values[field_name] = st.multiselect(
                    label,
                    options=config.get("options", []),
                    default=config.get("default", [])
                )

        submitted = st.form_submit_button(submit_label)

        if submitted:
            return values
        return None


def display_code_with_copy(code: str, language: str = "python") -> None:
    """
    Display code with a copy button.

    Args:
        code: Code to display
        language: Programming language for syntax highlighting

    Example:
        >>> display_code_with_copy(
        ...     "print('Hello, World!')",
        ...     language="python"
        ... )
    """
    st.code(code, language=language)


def create_tabs_with_icons(tabs: Dict[str, str]) -> str:
    """
    Create tabs with emoji icons.

    Args:
        tabs: Dictionary of {tab_name: emoji}

    Returns:
        Selected tab name

    Example:
        >>> tab = create_tabs_with_icons({
        ...     "Chat": "💬",
        ...     "Settings": "⚙️",
        ...     "History": "📊"
        ... })
    """
    tab_labels = [f"{emoji} {name}" for name, emoji in tabs.items()]
    selected = st.tabs(tab_labels)
    return selected


if __name__ == "__main__":
    # Demo the utilities
    print("Streamlit Utilities")
    print("=" * 40)
    print("These utilities are designed to be used in Streamlit apps.")
    print("Import them in your app.py or Home.py file.")
    print()
    print("Example:")
    print("  from streamlit_utils import init_session_state, render_chat_message")
    print("  init_session_state({'messages': []})")
