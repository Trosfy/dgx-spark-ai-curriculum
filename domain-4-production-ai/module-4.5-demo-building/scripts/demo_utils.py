"""
General Demo Utilities

Shared utilities for both Gradio and Streamlit demos.

Example usage:
    >>> from demo_utils import DemoConfig, validate_input, create_deployment_files
    >>> config = DemoConfig(title="My Demo", framework="gradio")
    >>> validated = validate_input(user_text, max_length=1000)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json


@dataclass
class DemoConfig:
    """
    Configuration for a demo application.

    Attributes:
        title: Demo title
        description: One-line description
        framework: "gradio" or "streamlit"
        theme: Theme configuration
        features: List of key features
        examples: List of example inputs
        github_url: Link to GitHub repository
        author: Author name

    Example:
        >>> config = DemoConfig(
        ...     title="My RAG Demo",
        ...     description="Chat with your documents",
        ...     framework="gradio",
        ...     features=["PDF support", "Source citations", "Multi-model"]
        ... )
    """
    title: str = "Demo"
    description: str = ""
    framework: str = "gradio"
    theme: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    github_url: str = ""
    author: str = ""
    version: str = "1.0.0"

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "framework": self.framework,
            "theme": self.theme,
            "features": self.features,
            "examples": self.examples,
            "github_url": self.github_url,
            "author": self.author,
            "version": self.version,
        }

    def to_json(self, filepath: str = None) -> str:
        """
        Export config as JSON.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, json_str: str) -> 'DemoConfig':
        """Create config from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


def validate_input(
    text: str,
    max_length: int = None,
    min_length: int = None,
    required: bool = True,
    allowed_chars: str = None,
    blocked_patterns: List[str] = None
) -> tuple[bool, str]:
    """
    Validate user input with various rules.

    Args:
        text: Input text to validate
        max_length: Maximum allowed length
        min_length: Minimum required length
        required: Whether input is required
        allowed_chars: Regex pattern for allowed characters
        blocked_patterns: List of regex patterns to block

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = validate_input(
        ...     user_input,
        ...     max_length=1000,
        ...     min_length=10,
        ...     blocked_patterns=[r"<script>"]
        ... )
        >>> if not is_valid:
        ...     show_error(error)
    """
    # Check required
    if required and (not text or not text.strip()):
        return False, "Input is required."

    if not text:
        return True, ""

    text = text.strip()

    # Check length
    if max_length and len(text) > max_length:
        return False, f"Input is too long. Maximum {max_length} characters allowed."

    if min_length and len(text) < min_length:
        return False, f"Input is too short. Minimum {min_length} characters required."

    # Check allowed characters
    if allowed_chars:
        if not re.match(f"^[{allowed_chars}]+$", text):
            return False, "Input contains invalid characters."

    # Check blocked patterns
    if blocked_patterns:
        for pattern in blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Input contains blocked content."

    return True, ""


def format_error_message(
    error: Exception,
    user_friendly: bool = True,
    include_type: bool = False
) -> str:
    """
    Format an exception into a user-friendly message.

    Args:
        error: The exception to format
        user_friendly: Whether to use friendly language
        include_type: Whether to include error type

    Returns:
        Formatted error message

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     message = format_error_message(e)
        ...     display_error(message)
    """
    error_type = type(error).__name__
    error_message = str(error)

    # Map common errors to friendly messages
    friendly_messages = {
        "ConnectionError": "Unable to connect to the service. Please check your connection and try again.",
        "TimeoutError": "The request took too long. Please try again with a smaller input.",
        "ValueError": f"Invalid input: {error_message}",
        "KeyError": "A required piece of data was missing. Please try again.",
        "FileNotFoundError": "The requested file could not be found.",
        "PermissionError": "You don't have permission to perform this action.",
        "MemoryError": "Not enough memory. Please try with smaller data.",
    }

    if user_friendly and error_type in friendly_messages:
        message = friendly_messages[error_type]
    else:
        message = error_message or f"An error occurred: {error_type}"

    if include_type and not user_friendly:
        message = f"[{error_type}] {message}"

    return message


def create_deployment_files(
    output_dir: Union[str, Path],
    config: DemoConfig,
    dependencies: List[str] = None
) -> Dict[str, str]:
    """
    Create deployment files for a demo.

    Creates:
    - requirements.txt
    - README.md (for HF Spaces or Streamlit Cloud)
    - .gitignore

    Args:
        output_dir: Directory to create files in
        config: Demo configuration
        dependencies: List of pip dependencies

    Returns:
        Dictionary of {filename: content}

    Example:
        >>> config = DemoConfig(title="My Demo", framework="gradio")
        >>> files = create_deployment_files(
        ...     "deploy/",
        ...     config,
        ...     dependencies=["gradio>=4.0.0", "chromadb"]
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dependencies = dependencies or []
    files = {}

    # Add default dependencies based on framework
    if config.framework == "gradio" and "gradio" not in str(dependencies):
        dependencies.insert(0, "gradio>=4.0.0")
    elif config.framework == "streamlit" and "streamlit" not in str(dependencies):
        dependencies.insert(0, "streamlit>=1.30.0")

    # requirements.txt
    requirements_content = "\n".join(dependencies)
    files["requirements.txt"] = requirements_content
    with open(output_dir / "requirements.txt", "w") as f:
        f.write(requirements_content)

    # README.md for HF Spaces
    if config.framework == "gradio":
        readme_content = f"""---
title: {config.title}
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# {config.title}

{config.description}

## Features
{chr(10).join(f"- {f}" for f in config.features)}

## Examples
{chr(10).join(f"- {e}" for e in config.examples)}

## Author
{config.author}

## Links
- [GitHub]({config.github_url})
"""
    else:  # Streamlit
        readme_content = f"""# {config.title}

{config.description}

## Features
{chr(10).join(f"- {f}" for f in config.features)}

## Usage

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Examples
{chr(10).join(f"- {e}" for e in config.examples)}

## Author
{config.author}

## Links
- [GitHub]({config.github_url})
"""

    files["README.md"] = readme_content
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.env

# IDE
.vscode/
.idea/

# Misc
.DS_Store
*.log
"""
    files[".gitignore"] = gitignore_content
    with open(output_dir / ".gitignore", "w") as f:
        f.write(gitignore_content)

    return files


def estimate_memory_usage(
    model_size_b: float,
    precision: str = "fp16",
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Estimate memory usage for a model.

    Args:
        model_size_b: Model size in billions of parameters
        precision: Precision ("fp32", "fp16", "int8", "int4")
        batch_size: Batch size for inference

    Returns:
        Dictionary with memory estimates in GB

    Example:
        >>> mem = estimate_memory_usage(7, precision="int4")
        >>> print(f"Model needs ~{mem['total']:.1f} GB")
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
        "nvfp4": 0.5,
    }

    bytes_pp = bytes_per_param.get(precision, 2)

    # Model weights
    model_memory = model_size_b * bytes_pp

    # KV cache estimate (roughly 10-20% of model size for inference)
    kv_cache = model_memory * 0.15 * batch_size

    # Activations and overhead (~10%)
    overhead = model_memory * 0.1

    total = model_memory + kv_cache + overhead

    return {
        "model_weights": model_memory,
        "kv_cache": kv_cache,
        "overhead": overhead,
        "total": total,
        "fits_dgx_spark": total < 120,  # Leave some headroom
    }


def format_number(n: Union[int, float], precision: int = 1) -> str:
    """
    Format a number with K/M/B suffixes.

    Args:
        n: Number to format
        precision: Decimal places

    Returns:
        Formatted string

    Example:
        >>> format_number(1234567)
        '1.2M'
        >>> format_number(1500, precision=0)
        '2K'
    """
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.{precision}f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.{precision}f}M"
    elif n >= 1_000:
        return f"{n/1_000:.{precision}f}K"
    else:
        return str(n)


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        name: Original string

    Returns:
        Safe filename string

    Example:
        >>> sanitize_filename("My Project: v2.0")
        'my_project_v2.0'
    """
    # Remove or replace unsafe characters
    name = name.lower()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')

    return name[:100]  # Limit length


if __name__ == "__main__":
    # Demo the utilities
    print("Demo Utilities")
    print("=" * 40)

    # Test DemoConfig
    config = DemoConfig(
        title="RAG Chat Demo",
        description="Chat with your documents",
        framework="gradio",
        features=["PDF support", "Source citations"],
        examples=["What is the main topic?", "Summarize the document"],
        author="Professor SPARK"
    )

    print(f"Config: {config.title}")
    print(f"JSON:\n{config.to_json()}")

    # Test validate_input
    is_valid, error = validate_input("Hello world", max_length=100)
    print(f"\nValidation: valid={is_valid}, error='{error}'")

    # Test memory estimation
    mem = estimate_memory_usage(70, precision="int4")
    print(f"\n70B model (int4): {mem['total']:.1f} GB")
    print(f"Fits DGX Spark: {mem['fits_dgx_spark']}")
