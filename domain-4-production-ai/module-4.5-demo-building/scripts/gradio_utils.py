"""
Gradio Utilities for Demo Building

Production-quality utilities for building polished Gradio demos.

Example usage:
    >>> from gradio_utils import create_theme, safe_handler
    >>> theme = create_theme(primary_color="blue")
    >>> @safe_handler
    ... def my_function(input):
    ...     return process(input)
"""

import gradio as gr
from functools import wraps
from typing import Callable, Any, List, Optional, Dict, Union
from pathlib import Path
import time
import traceback


def create_theme(
    primary_color: str = "blue",
    font: str = "Inter",
    dark_mode: bool = False
) -> gr.themes.Base:
    """
    Create a customized Gradio theme.

    Args:
        primary_color: Primary color name (blue, green, red, purple, orange)
        font: Google font name to use
        dark_mode: Whether to use dark mode

    Returns:
        A configured Gradio theme

    Example:
        >>> theme = create_theme(primary_color="purple", font="Roboto")
        >>> with gr.Blocks(theme=theme) as demo:
        ...     gr.Markdown("# My Demo")
    """
    color_map = {
        "blue": gr.themes.colors.blue,
        "green": gr.themes.colors.green,
        "red": gr.themes.colors.red,
        "purple": gr.themes.colors.purple,
        "orange": gr.themes.colors.orange,
        "slate": gr.themes.colors.slate,
    }

    primary_hue = color_map.get(primary_color, gr.themes.colors.blue)

    base_theme = gr.themes.Soft if not dark_mode else gr.themes.Default

    theme = base_theme(
        primary_hue=primary_hue,
        secondary_hue="slate",
        font=gr.themes.GoogleFont(font),
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
        block_title_text_weight="600",
    )

    return theme


def safe_handler(func: Callable) -> Callable:
    """
    Decorator that wraps a function with error handling for demos.

    Catches exceptions and returns user-friendly error messages
    instead of exposing raw Python tracebacks.

    Args:
        func: The function to wrap

    Returns:
        Wrapped function with error handling

    Example:
        >>> @safe_handler
        ... def process_input(text):
        ...     result = model.generate(text)
        ...     return result
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError:
            return "Unable to connect to the service. Please try again in a moment."
        except TimeoutError:
            return "Request timed out. Please try with a shorter input."
        except ValueError as e:
            return f"Invalid input: {str(e)}"
        except Exception as e:
            # Log the actual error for debugging
            print(f"Error in {func.__name__}: {e}")
            traceback.print_exc()
            # Return friendly message
            return "Something went wrong. Please try again or try a different input."

    return wrapper


def add_progress_indicator(
    func: Callable,
    message: str = "Processing..."
) -> Callable:
    """
    Decorator that adds progress indication for long-running functions.

    Note: This works with gr.Progress() in the function signature.

    Args:
        func: The function to wrap
        message: Message to display during processing

    Returns:
        Wrapped function with progress indication

    Example:
        >>> @add_progress_indicator(message="Generating response...")
        ... def generate(text, progress=gr.Progress()):
        ...     for i in progress.tqdm(range(100)):
        ...         do_work(i)
        ...     return result
    """
    @wraps(func)
    def wrapper(*args, progress=None, **kwargs):
        if progress is not None:
            progress(0, desc=message)

        result = func(*args, progress=progress, **kwargs)

        if progress is not None:
            progress(1, desc="Done!")

        return result

    return wrapper


def create_examples_from_files(
    directory: Union[str, Path],
    extensions: List[str] = None,
    max_examples: int = 5
) -> List[List[str]]:
    """
    Create Gradio examples from files in a directory.

    Args:
        directory: Path to directory containing example files
        extensions: List of file extensions to include (e.g., [".txt", ".jpg"])
        max_examples: Maximum number of examples to return

    Returns:
        List of examples for gr.Examples

    Example:
        >>> examples = create_examples_from_files(
        ...     "examples/",
        ...     extensions=[".jpg", ".png"],
        ...     max_examples=3
        ... )
        >>> gr.Examples(examples=examples, inputs=[image_input])
    """
    directory = Path(directory)
    extensions = extensions or [".txt", ".jpg", ".png", ".pdf", ".md"]

    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist")
        return []

    examples = []
    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() in extensions:
            examples.append([str(file_path)])
            if len(examples) >= max_examples:
                break

    return examples


def create_chat_interface(
    respond_fn: Callable,
    title: str = "Chat Demo",
    description: str = "",
    examples: List[str] = None,
    **kwargs
) -> gr.Blocks:
    """
    Create a standardized chat interface.

    Args:
        respond_fn: Function that takes (message, history) and returns response
        title: Title for the chat interface
        description: Description text
        examples: List of example prompts
        **kwargs: Additional arguments passed to gr.Blocks

    Returns:
        A configured Gradio Blocks interface

    Example:
        >>> def respond(message, history):
        ...     return f"Echo: {message}"
        >>> demo = create_chat_interface(
        ...     respond,
        ...     title="My Chatbot",
        ...     examples=["Hello", "Help me with..."]
        ... )
        >>> demo.launch()
    """
    theme = kwargs.pop("theme", create_theme())

    with gr.Blocks(theme=theme, **kwargs) as demo:
        gr.Markdown(f"# {title}")
        if description:
            gr.Markdown(description)

        chatbot = gr.Chatbot(height=400)

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message...",
                scale=4,
                show_label=False
            )
            send = gr.Button("Send", variant="primary", scale=1)

        clear = gr.Button("Clear Chat")

        if examples:
            gr.Examples(
                examples=[[ex] for ex in examples],
                inputs=[msg],
                label="Try these"
            )

        def user_message(message, history):
            if not message:
                return "", history
            return "", history + [[message, None]]

        def bot_response(history):
            if not history or history[-1][1] is not None:
                return history

            message = history[-1][0]
            history_context = history[:-1]

            response = respond_fn(message, history_context)
            history[-1][1] = response

            return history

        msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot], [chatbot]
        )
        send.click(user_message, [msg, chatbot], [msg, chatbot]).then(
            bot_response, [chatbot], [chatbot]
        )
        clear.click(lambda: [], outputs=[chatbot])

    return demo


def create_file_processor(
    process_fn: Callable,
    title: str = "File Processor",
    file_types: List[str] = None,
    output_type: str = "text",
    **kwargs
) -> gr.Blocks:
    """
    Create a standardized file processing interface.

    Args:
        process_fn: Function that takes file path and returns result
        title: Title for the interface
        file_types: Allowed file types (e.g., [".pdf", ".txt"])
        output_type: Type of output ("text", "image", "json")
        **kwargs: Additional arguments passed to gr.Blocks

    Returns:
        A configured Gradio Blocks interface

    Example:
        >>> def process_pdf(file_path):
        ...     return extract_text(file_path)
        >>> demo = create_file_processor(
        ...     process_pdf,
        ...     title="PDF Extractor",
        ...     file_types=[".pdf"]
        ... )
        >>> demo.launch()
    """
    theme = kwargs.pop("theme", create_theme())
    file_types = file_types or [".txt", ".pdf", ".docx"]

    output_components = {
        "text": gr.Textbox(label="Output", lines=10),
        "image": gr.Image(label="Output"),
        "json": gr.JSON(label="Output"),
    }

    with gr.Blocks(theme=theme, **kwargs) as demo:
        gr.Markdown(f"# {title}")

        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload File",
                    file_types=file_types
                )
                process_btn = gr.Button("Process", variant="primary")

            with gr.Column():
                output = output_components.get(output_type, gr.Textbox(label="Output"))

        @safe_handler
        def handle_file(file):
            if file is None:
                return "Please upload a file first."
            return process_fn(file.name)

        process_btn.click(handle_file, [file_input], [output])

    return demo


# CSS Snippets for common styling needs
CUSTOM_CSS = {
    "centered": """
        .gradio-container {
            max-width: 1000px !important;
            margin: auto !important;
        }
    """,
    "full_width": """
        .gradio-container {
            max-width: 100% !important;
        }
    """,
    "source_citation": """
        .source-box {
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
    """,
    "dark_chat": """
        .chatbot {
            background-color: #1a1a1a;
        }
        .chatbot .user {
            background-color: #2d2d2d;
        }
        .chatbot .bot {
            background-color: #0d6efd;
        }
    """,
}


def get_css(style_names: List[str]) -> str:
    """
    Get combined CSS for specified styles.

    Args:
        style_names: List of style names from CUSTOM_CSS

    Returns:
        Combined CSS string

    Example:
        >>> css = get_css(["centered", "source_citation"])
        >>> with gr.Blocks(css=css) as demo:
        ...     ...
    """
    return "\n".join(
        CUSTOM_CSS.get(name, "")
        for name in style_names
    )


if __name__ == "__main__":
    # Demo the utilities
    print("Gradio Utilities Demo")
    print("=" * 40)

    # Create a simple demo
    @safe_handler
    def echo(text):
        if "error" in text.lower():
            raise ValueError("You asked for an error!")
        return f"You said: {text}"

    demo = create_chat_interface(
        respond_fn=lambda msg, hist: echo(msg),
        title="Echo Bot",
        description="A simple echo bot to demo the utilities",
        examples=["Hello!", "Tell me a joke", "error"]
    )

    print("Demo created successfully!")
    print("Run with: demo.launch()")
