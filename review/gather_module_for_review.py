#!/usr/bin/env python3
"""
Module Content Gatherer for Review

Collects all files from a module directory and formats them
for pasting into the Content Reviewer prompt.

Usage:
    python gather_module_for_review.py module-01-dgx-spark-platform/
    python gather_module_for_review.py module-01-dgx-spark-platform/ --output review_input.txt
    python gather_module_for_review.py module-01-dgx-spark-platform/ --notebooks-only

Example:
    python gather_module_for_review.py phase-1-foundations/module-01-dgx-spark-platform/
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional


# File extensions to include
INCLUDE_EXTENSIONS = {
    '.ipynb': 'json',  # Jupyter notebooks
    '.py': 'python',
    '.md': 'markdown',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.txt': 'text',
    '.csv': 'csv',
    '.sh': 'bash',
}

# Directories to skip
SKIP_DIRS = {
    '__pycache__',
    '.ipynb_checkpoints',
    '.git',
    'node_modules',
    '.venv',
    'venv',
}

# Files to skip
SKIP_FILES = {
    '.DS_Store',
    'Thumbs.db',
}


def get_language(filepath: Path) -> str:
    """Get the language identifier for a file."""
    return INCLUDE_EXTENSIONS.get(filepath.suffix.lower(), 'text')


def should_include(filepath: Path) -> bool:
    """Check if a file should be included."""
    # Check extension
    if filepath.suffix.lower() not in INCLUDE_EXTENSIONS:
        return False
    
    # Check filename
    if filepath.name in SKIP_FILES:
        return False
    
    # Check parent directories
    for part in filepath.parts:
        if part in SKIP_DIRS:
            return False
    
    return True


def read_file_content(filepath: Path) -> Optional[str]:
    """Read file content, handling different encodings."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # For notebooks, pretty-print JSON
        if filepath.suffix == '.ipynb':
            try:
                notebook = json.loads(content)
                content = json.dumps(notebook, indent=2)
            except json.JSONDecodeError:
                pass  # Keep original content if JSON parsing fails
        
        return content
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            return f"[Error reading file: {e}]"
    except Exception as e:
        return f"[Error reading file: {e}]"


def gather_files(module_path: Path, notebooks_only: bool = False) -> List[tuple]:
    """
    Gather all relevant files from a module directory.
    
    Returns:
        List of (relative_path, language, content) tuples
    """
    files = []
    
    for filepath in sorted(module_path.rglob('*')):
        if not filepath.is_file():
            continue
        
        if not should_include(filepath):
            continue
        
        if notebooks_only and filepath.suffix != '.ipynb':
            continue
        
        relative_path = filepath.relative_to(module_path)
        language = get_language(filepath)
        content = read_file_content(filepath)
        
        if content:
            files.append((str(relative_path), language, content))
    
    return files


def format_for_review(files: List[tuple], module_name: str) -> str:
    """Format gathered files for the review prompt."""
    output = []
    output.append(f"# Module Content for Review: {module_name}")
    output.append(f"# Files: {len(files)}")
    output.append("")
    
    for relative_path, language, content in files:
        output.append("---")
        output.append(f"## FILE: {relative_path}")
        output.append(f"```{language}")
        output.append(content)
        output.append("```")
        output.append("")
    
    return "\n".join(output)


def generate_summary(files: List[tuple]) -> str:
    """Generate a summary of files gathered."""
    summary = []
    summary.append("=" * 60)
    summary.append("GATHERING SUMMARY")
    summary.append("=" * 60)
    
    # Count by type
    by_type = {}
    for path, lang, _ in files:
        ext = Path(path).suffix
        by_type[ext] = by_type.get(ext, 0) + 1
    
    summary.append(f"Total files: {len(files)}")
    summary.append("")
    summary.append("By type:")
    for ext, count in sorted(by_type.items()):
        summary.append(f"  {ext}: {count}")
    
    summary.append("")
    summary.append("Files included:")
    for path, _, _ in files:
        summary.append(f"  - {path}")
    
    summary.append("=" * 60)
    return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Gather module files for review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python gather_module_for_review.py module-01-dgx-spark-platform/
    python gather_module_for_review.py module-01-dgx-spark-platform/ --output review.txt
    python gather_module_for_review.py module-01-dgx-spark-platform/ --notebooks-only
        """
    )
    
    parser.add_argument("module_path", type=Path, help="Path to module directory")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")
    parser.add_argument("--notebooks-only", "-n", action="store_true", 
                       help="Only include .ipynb files")
    parser.add_argument("--summary", "-s", action="store_true",
                       help="Print summary only (don't output full content)")
    parser.add_argument("--max-size", "-m", type=int, default=0,
                       help="Maximum file size in KB (0 = no limit)")
    
    args = parser.parse_args()
    
    # Validate path
    if not args.module_path.exists():
        print(f"❌ Error: Path does not exist: {args.module_path}")
        return 1
    
    if not args.module_path.is_dir():
        print(f"❌ Error: Path is not a directory: {args.module_path}")
        return 1
    
    # Gather files
    print(f"📁 Gathering files from: {args.module_path}")
    files = gather_files(args.module_path, args.notebooks_only)
    
    if not files:
        print("⚠️ No files found matching criteria")
        return 1
    
    # Filter by size if specified
    if args.max_size > 0:
        max_bytes = args.max_size * 1024
        files = [(p, l, c) for p, l, c in files if len(c.encode('utf-8')) <= max_bytes]
        print(f"📏 Filtered to files under {args.max_size}KB: {len(files)} files")
    
    # Print summary
    summary = generate_summary(files)
    print(summary)
    
    if args.summary:
        return 0
    
    # Format content
    module_name = args.module_path.name
    content = format_for_review(files, module_name)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Output written to: {args.output}")
        print(f"   Size: {len(content) / 1024:.1f} KB")
        print(f"   Estimated tokens: ~{len(content) // 4}")
    else:
        print("\n" + "=" * 60)
        print("CONTENT (paste into review prompt)")
        print("=" * 60 + "\n")
        print(content)
    
    return 0


if __name__ == "__main__":
    exit(main())
