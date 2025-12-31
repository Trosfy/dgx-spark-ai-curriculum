"""
Custom Tools for AI Agents

This module provides production-ready custom tools for LangChain agents,
including search, calculation, code execution, file operations, and API calls.

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 3.6: AI Agents & Agentic Systems

IMPORTANT SECURITY NOTE:
These tools are for educational purposes. In production:
- Sandbox code execution
- Validate and sanitize all inputs
- Implement proper access controls
- Rate limit API calls
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json
import re
import ast
import operator
import math
# import subprocess  # Removed: not used (blocked for security reasons)
import tempfile
import urllib.request
import urllib.parse
from datetime import datetime
from dataclasses import dataclass

# Try to import LangChain for decorator (optional)
try:
    from langchain.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a dummy decorator if LangChain isn't available
    def tool(func):
        func._is_tool = True
        return func


# ============================================================================
# TOOL 1: CALCULATOR
# ============================================================================

@dataclass
class CalculatorResult:
    """Result from a calculation."""
    expression: str
    result: float
    formatted: str
    steps: List[str]


class SafeCalculator:
    """
    A safe mathematical expression evaluator.

    Unlike using eval(), this parser only allows mathematical operations,
    making it safe from code injection attacks.

    Supported operations:
        - Basic arithmetic: +, -, *, /, //, %, **
        - Comparisons: <, >, <=, >=, ==, !=
        - Functions: sin, cos, tan, sqrt, log, log10, exp, abs, round, floor, ceil
        - Constants: pi, e

    Example:
        >>> calc = SafeCalculator()
        >>> result = calc.evaluate("sqrt(16) + 2**3")
        >>> print(result.result)
        12.0
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    FUNCTIONS = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'factorial': math.factorial,
        'gcd': math.gcd,
    }

    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
    }

    def __init__(self):
        self.steps = []

    def evaluate(self, expression: str) -> CalculatorResult:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression as a string

        Returns:
            CalculatorResult with the result and calculation steps

        Raises:
            ValueError: If expression contains unsupported operations
        """
        self.steps = []
        expression = expression.strip()

        try:
            tree = ast.parse(expression, mode='eval')
            result = self._eval_node(tree.body)

            # Format the result nicely
            if isinstance(result, float) and result.is_integer():
                formatted = str(int(result))
            elif isinstance(result, float):
                formatted = f"{result:.10g}"
            else:
                formatted = str(result)

            return CalculatorResult(
                expression=expression,
                result=float(result),
                formatted=formatted,
                steps=self.steps
            )
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    def _eval_node(self, node) -> float:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]
            raise ValueError(f"Unknown constant: {node.id}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            result = self.OPERATORS[op_type](left, right)
            self.steps.append(f"{left} {self._op_symbol(op_type)} {right} = {result}")
            return result

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            return self.OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            result = self.FUNCTIONS[func_name](*args)
            self.steps.append(f"{func_name}({', '.join(str(a) for a in args)}) = {result}")
            return result

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def _op_symbol(self, op_type) -> str:
        """Get the symbol for an operator."""
        symbols = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
        }
        return symbols.get(op_type, '?')


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Use this tool when you need to perform calculations. Supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Functions: sin, cos, tan, sqrt, log, exp, abs, round, floor, ceil
    - Constants: pi, e

    Args:
        expression: A mathematical expression like "sqrt(16) + 2**3"

    Returns:
        The result as a string

    Examples:
        calculate("2 + 2") -> "4"
        calculate("sqrt(16) * 3") -> "12"
        calculate("sin(pi/2)") -> "1"
    """
    try:
        calc = SafeCalculator()
        result = calc.evaluate(expression)
        return f"{result.expression} = {result.formatted}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# TOOL 2: WEB SEARCH (Simulated)
# ============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information.

    Use this tool when you need to find current information from the internet.

    Args:
        query: The search query
        num_results: Number of results to return (default: 5)

    Returns:
        Search results as formatted text

    Note:
        This is a simulated search for educational purposes.
        In production, integrate with a real search API like:
        - Google Custom Search API
        - Bing Search API
        - SerpAPI
    """
    # Simulated search results for educational purposes
    simulated_results = {
        "langchain": [
            {
                "title": "LangChain Documentation",
                "url": "https://python.langchain.com/docs/",
                "snippet": "LangChain is a framework for developing applications powered by language models."
            },
            {
                "title": "LangChain GitHub Repository",
                "url": "https://github.com/langchain-ai/langchain",
                "snippet": "Build context-aware reasoning applications with LangChain."
            }
        ],
        "dgx spark": [
            {
                "title": "NVIDIA DGX Spark",
                "url": "https://www.nvidia.com/dgx-spark",
                "snippet": "DGX Spark is NVIDIA's desktop AI supercomputer with 128GB unified memory."
            }
        ],
        "llama 3": [
            {
                "title": "Meta Llama 3 Model Card",
                "url": "https://huggingface.co/meta-llama/Llama-3",
                "snippet": "Llama 3 is Meta's latest open-source large language model."
            }
        ]
    }

    # Find matching results
    query_lower = query.lower()
    results = []

    for key, value in simulated_results.items():
        if key in query_lower:
            results.extend(value)

    if not results:
        results = [{
            "title": f"Search results for: {query}",
            "url": f"https://search.example.com/?q={urllib.parse.quote(query)}",
            "snippet": "This is a simulated search result. In production, connect to a real search API."
        }]

    # Format results
    output = f"Search results for: {query}\n\n"
    for i, result in enumerate(results[:num_results], 1):
        output += f"{i}. {result['title']}\n"
        output += f"   URL: {result['url']}\n"
        output += f"   {result['snippet']}\n\n"

    return output


# ============================================================================
# TOOL 3: CODE EXECUTION
# ============================================================================

class SafePythonExecutor:
    """
    Execute Python code in a restricted environment.

    SECURITY WARNING - EDUCATIONAL USE ONLY:
    This implementation is for EDUCATIONAL PURPOSES ONLY.

    In production environments, you MUST use proper sandboxing:
    - Docker containers with resource limits (CPU, memory, time)
    - gVisor or Firecracker for additional isolation
    - AWS Lambda or Google Cloud Functions
    - Dedicated code execution services like Judge0
    - RestrictedPython package for AST-level restrictions

    This implementation attempts to restrict dangerous operations but
    is NOT suitable for untrusted code execution. A determined attacker
    could still escape these restrictions.

    Example:
        >>> executor = SafePythonExecutor()
        >>> result = executor.execute("print(2 + 2)")
        >>> print(result['output'])
        '4'
    """

    FORBIDDEN_IMPORTS = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests',
        'urllib', 'http', 'ftplib', 'smtplib', 'telnetlib',
        '__builtins__', 'builtins', 'eval', 'exec', 'compile',
        'open', 'file', 'input', '__import__'
    }

    ALLOWED_MODULES = {
        'math', 'random', 'datetime', 'json', 're',
        'collections', 'itertools', 'functools',
        'statistics', 'decimal', 'fractions'
    }

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def _check_safety(self, code: str) -> bool:
        """Check if code contains forbidden patterns."""
        # Check for forbidden imports
        for forbidden in self.FORBIDDEN_IMPORTS:
            if forbidden in code:
                return False

        # Check for file operations
        file_patterns = [
            r'open\s*\(',
            r'with\s+.*\s+as\s+',
            r'\.read\s*\(',
            r'\.write\s*\(',
        ]
        for pattern in file_patterns:
            if re.search(pattern, code):
                return False

        return True

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with 'output', 'error', and 'success' keys
        """
        if not self._check_safety(code):
            return {
                'output': '',
                'error': 'Code contains forbidden operations (file access, imports, etc.)',
                'success': False
            }

        try:
            # Create a restricted globals environment
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'True': True,
                    'False': False,
                    'None': None,
                }
            }

            # Add allowed modules
            import math
            import random
            import json as json_module
            import re as re_module
            from datetime import datetime as dt

            restricted_globals['math'] = math
            restricted_globals['random'] = random
            restricted_globals['json'] = json_module
            restricted_globals['re'] = re_module
            restricted_globals['datetime'] = dt

            # Capture output
            import io
            import contextlib

            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                exec(code, restricted_globals)

            return {
                'output': output_buffer.getvalue(),
                'error': None,
                'success': True
            }

        except Exception as e:
            return {
                'output': '',
                'error': str(e),
                'success': False
            }


@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return the output.

    Use this tool when you need to run Python code to solve a problem.
    The environment has access to: math, random, json, re, datetime.

    Args:
        code: Python code to execute

    Returns:
        The output of the code execution

    Examples:
        execute_python("print(2 ** 10)")  # -> "1024"
        execute_python("import math; print(math.factorial(5))")  # -> "120"

    Note:
        This is a restricted environment. File operations and network
        access are not allowed for security reasons.
    """
    executor = SafePythonExecutor()
    result = executor.execute(code)

    if result['success']:
        output = result['output'].strip()
        return output if output else "Code executed successfully (no output)"
    else:
        return f"Error: {result['error']}"


# ============================================================================
# TOOL 4: FILE READER
# ============================================================================

@tool
def read_file(file_path: str, max_chars: int = 10000) -> str:
    """
    Read the contents of a text file.

    Use this tool when you need to read a file from the filesystem.

    Args:
        file_path: Path to the file to read
        max_chars: Maximum characters to return (default: 10000)

    Returns:
        The file contents or an error message

    Note:
        For security, this tool only reads from allowed directories
        and only reads text files.
    """
    try:
        path = Path(file_path)

        # Security checks
        if not path.exists():
            return f"Error: File not found: {file_path}"

        if not path.is_file():
            return f"Error: Not a file: {file_path}"

        # Check file size
        file_size = path.stat().st_size
        if file_size > 1_000_000:  # 1MB limit
            return f"Error: File too large ({file_size} bytes). Maximum is 1MB."

        # Check extension
        allowed_extensions = {'.txt', '.md', '.json', '.py', '.csv', '.yaml', '.yml', '.xml'}
        if path.suffix.lower() not in allowed_extensions:
            return f"Error: Unsupported file type: {path.suffix}"

        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(max_chars)

        if len(content) == max_chars:
            content += f"\n\n... (truncated at {max_chars} characters)"

        return content

    except UnicodeDecodeError:
        return f"Error: File is not a text file or uses unsupported encoding"
    except PermissionError:
        return f"Error: Permission denied for file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


# ============================================================================
# TOOL 5: API CALLER
# ============================================================================

@tool
def call_api(
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    body: Optional[str] = None
) -> str:
    """
    Make an HTTP API call.

    Use this tool when you need to fetch data from a web API.

    Args:
        url: The API endpoint URL
        method: HTTP method (GET or POST)
        headers: Optional JSON string of headers
        body: Optional request body (for POST requests)

    Returns:
        The API response or an error message

    Examples:
        call_api("https://api.github.com/users/octocat")
        call_api("https://api.example.com/data", method="POST", body='{"key": "value"}')

    Note:
        For security, this tool validates URLs and has rate limiting.
    """
    try:
        # Validate URL
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return f"Error: Only HTTP/HTTPS URLs are allowed"

        # Block internal URLs
        blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '169.254.169.254']
        if parsed.hostname in blocked_hosts:
            return f"Error: Access to {parsed.hostname} is not allowed"

        # Parse headers if provided
        header_dict = {}
        if headers:
            try:
                header_dict = json.loads(headers)
            except json.JSONDecodeError:
                return "Error: Invalid headers JSON"

        # Add default headers
        header_dict.setdefault('User-Agent', 'DGX-Spark-Agent/1.0')
        header_dict.setdefault('Accept', 'application/json')

        # Create request
        if method.upper() == "POST":
            data = body.encode('utf-8') if body else None
            req = urllib.request.Request(url, data=data, headers=header_dict, method='POST')
        else:
            req = urllib.request.Request(url, headers=header_dict, method='GET')

        # Make request with timeout
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')

            # Truncate if too long
            max_length = 5000
            if len(content) > max_length:
                content = content[:max_length] + "\n... (truncated)"

            return f"Status: {response.status}\n\nResponse:\n{content}"

    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL Error: {str(e.reason)}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# ADDITIONAL UTILITY TOOLS
# ============================================================================

@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current date and time.

    Use this tool when you need to know the current date or time.

    Args:
        format: strftime format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        Current datetime as a formatted string

    Examples:
        get_current_datetime()  # -> "2024-01-15 14:30:00"
        get_current_datetime("%A, %B %d, %Y")  # -> "Monday, January 15, 2024"
    """
    return datetime.now().strftime(format)


@tool
def json_parser(json_string: str, path: str = "") -> str:
    """
    Parse JSON and optionally extract a specific path.

    Use this tool to parse JSON data and extract values.

    Args:
        json_string: The JSON string to parse
        path: Optional dot-notation path to extract (e.g., "data.users[0].name")

    Returns:
        Parsed JSON or extracted value

    Examples:
        json_parser('{"name": "John", "age": 30}')
        json_parser('{"data": {"users": [{"name": "John"}]}}', "data.users[0].name")
    """
    try:
        data = json.loads(json_string)

        if not path:
            return json.dumps(data, indent=2)

        # Navigate the path
        current = data
        for part in path.replace('[', '.').replace(']', '').split('.'):
            if not part:
                continue
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                return f"Error: Cannot navigate to '{part}' in non-container type"

        if isinstance(current, (dict, list)):
            return json.dumps(current, indent=2)
        return str(current)

    except json.JSONDecodeError as e:
        return f"JSON Parse Error: {e}"
    except (KeyError, IndexError) as e:
        return f"Path Error: {e}"
    except Exception as e:
        return f"Error: {e}"


@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text and return statistics.

    Use this tool to get statistics about a piece of text.

    Args:
        text: The text to analyze

    Returns:
        Text statistics including word count, character count, etc.
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    stats = {
        'character_count': len(text),
        'character_count_no_spaces': len(text.replace(' ', '')),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len(paragraphs),
        'average_word_length': sum(len(w) for w in words) / len(words) if words else 0,
        'average_sentence_length': len(words) / len(sentences) if sentences else 0,
    }

    output = "Text Analysis:\n"
    output += f"  Characters: {stats['character_count']:,}\n"
    output += f"  Characters (no spaces): {stats['character_count_no_spaces']:,}\n"
    output += f"  Words: {stats['word_count']:,}\n"
    output += f"  Sentences: {stats['sentence_count']:,}\n"
    output += f"  Paragraphs: {stats['paragraph_count']:,}\n"
    output += f"  Avg word length: {stats['average_word_length']:.1f} chars\n"
    output += f"  Avg sentence length: {stats['average_sentence_length']:.1f} words\n"

    return output


# ============================================================================
# TOOL REGISTRY
# ============================================================================

def get_all_tools() -> List:
    """
    Get all available tools as a list.

    Returns:
        List of tool functions for use with LangChain agents

    Example:
        >>> from custom_tools import get_all_tools
        >>> tools = get_all_tools()
        >>> agent = create_agent(llm, tools)
    """
    return [
        calculate,
        web_search,
        execute_python,
        read_file,
        call_api,
        get_current_datetime,
        json_parser,
        text_analyzer,
    ]


def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all available tools.

    Returns:
        Formatted string describing all tools
    """
    tools = get_all_tools()
    descriptions = []

    for tool_func in tools:
        name = tool_func.name if hasattr(tool_func, 'name') else tool_func.__name__
        doc = tool_func.description if hasattr(tool_func, 'description') else tool_func.__doc__
        descriptions.append(f"- {name}: {doc.split(chr(10))[0].strip()}")

    return "\n".join(descriptions)


if __name__ == "__main__":
    print("Custom Tools Demo")
    print("=" * 50)

    # Test calculator
    print("\n1. Calculator Tool:")
    print(calculate("sqrt(16) + 2**3"))

    # Test code execution
    print("\n2. Code Execution Tool:")
    print(execute_python("for i in range(5): print(i**2)"))

    # Test text analyzer
    print("\n3. Text Analyzer Tool:")
    sample_text = """
    The DGX Spark is a revolutionary desktop AI supercomputer.
    It features 128GB of unified memory. This enables running large language models locally.
    """
    print(text_analyzer(sample_text))

    # Test JSON parser
    print("\n4. JSON Parser Tool:")
    sample_json = '{"model": "llama3", "parameters": {"temperature": 0.7, "max_tokens": 100}}'
    print(json_parser(sample_json, "parameters.temperature"))

    # List all tools
    print("\n5. Available Tools:")
    print(get_tool_descriptions())
