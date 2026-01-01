#!/usr/bin/env python3
"""
Module Content Validator

Automated validation of module content for common issues.
Run this BEFORE using the AI reviewer prompt to catch obvious problems.

Usage:
    python validate_module.py module-1.1-dgx-spark-platform/
    python validate_module.py module-1.1-dgx-spark-platform/ --fix
    python validate_module.py module-1.1-dgx-spark-platform/ --verbose

Example:
    python validate_module.py domain-1-platform-foundations/module-1.1-dgx-spark-platform/
"""

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Issue:
    """Represents a validation issue."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    file: str
    location: str  # Line number or cell number
    category: str
    message: str
    fix: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    module_path: str
    files_checked: int = 0
    issues: List[Issue] = field(default_factory=list)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "CRITICAL")
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "HIGH")
    
    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "MEDIUM")
    
    @property
    def low_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "LOW")
    
    @property
    def passed(self) -> bool:
        return self.critical_count == 0


# Standard library modules (common ones)
STDLIB_MODULES = {
    'os', 'sys', 're', 'json', 'time', 'datetime', 'pathlib', 'typing',
    'collections', 'itertools', 'functools', 'math', 'random', 'copy',
    'subprocess', 'threading', 'multiprocessing', 'argparse', 'logging',
    'io', 'pickle', 'csv', 'dataclasses', 'abc', 'contextlib', 'gc',
    'warnings', 'statistics', 'hashlib', 'base64', 'struct', 'array'
}

# Common third-party modules for AI/ML
THIRDPARTY_MODULES = {
    'torch', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
    'transformers', 'datasets', 'accelerate', 'peft', 'bitsandbytes',
    'requests', 'tqdm', 'PIL', 'cv2', 'scipy', 'plotly', 'gradio',
    'langchain', 'chromadb', 'faiss', 'einops', 'safetensors'
}


def extract_imports_from_python(code: str) -> Set[str]:
    """Extract imported module names from Python code."""
    imports = set()
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get root module name
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except SyntaxError:
        # If AST parsing fails, use regex
        import_pattern = r'^(?:from|import)\s+(\w+)'
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            imports.add(match.group(1))
    
    return imports


def extract_used_names(code: str) -> Set[str]:
    """Extract names used in code (potential undefined references)."""
    used = set()
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
    except SyntaxError:
        pass
    
    return used


def check_notebook_json(filepath: Path) -> List[Issue]:
    """Check if notebook is valid JSON."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        notebook = json.loads(content)
        
        # Check for required keys
        if 'cells' not in notebook:
            issues.append(Issue(
                severity="CRITICAL",
                file=str(filepath),
                location="root",
                category="Structure",
                message="Notebook missing 'cells' key - invalid notebook format"
            ))
        
        if 'nbformat' not in notebook:
            issues.append(Issue(
                severity="MEDIUM",
                file=str(filepath),
                location="root",
                category="Structure",
                message="Notebook missing 'nbformat' key"
            ))
            
    except json.JSONDecodeError as e:
        issues.append(Issue(
            severity="CRITICAL",
            file=str(filepath),
            location=f"char {e.pos}",
            category="JSON",
            message=f"Invalid JSON: {e.msg}"
        ))
    except Exception as e:
        issues.append(Issue(
            severity="CRITICAL",
            file=str(filepath),
            location="N/A",
            category="File",
            message=f"Cannot read file: {e}"
        ))
    
    return issues


def check_notebook_cells(filepath: Path) -> List[Issue]:
    """Check notebook cells for common issues."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except:
        return issues  # Already caught by JSON check
    
    cells = notebook.get('cells', [])
    all_imports = set()
    defined_names = set()
    
    for i, cell in enumerate(cells):
        cell_num = i + 1
        cell_type = cell.get('cell_type', '')
        source = ''.join(cell.get('source', []))
        
        if cell_type == 'code':
            # Check for syntax errors
            try:
                ast.parse(source)
            except SyntaxError as e:
                issues.append(Issue(
                    severity="CRITICAL",
                    file=str(filepath),
                    location=f"Cell {cell_num}, Line {e.lineno}",
                    category="Syntax",
                    message=f"Syntax error: {e.msg}"
                ))
                continue
            
            # Collect imports
            cell_imports = extract_imports_from_python(source)
            all_imports.update(cell_imports)
            
            # Check for common issues
            
            # Issue: Using torch without import
            if 'torch.' in source or 'torch(' in source:
                if 'torch' not in all_imports and 'import torch' not in source:
                    # Check if torch is imported in this cell
                    if 'import torch' not in source and 'from torch' not in source:
                        issues.append(Issue(
                            severity="HIGH",
                            file=str(filepath),
                            location=f"Cell {cell_num}",
                            category="Import",
                            message="Using 'torch' but no import statement found in preceding cells",
                            fix="Add 'import torch' at the beginning of the notebook"
                        ))
            
            # Issue: Hardcoded paths
            hardcoded_paths = re.findall(r'["\']\/home\/\w+[^"\']*["\']', source)
            for path in hardcoded_paths:
                issues.append(Issue(
                    severity="HIGH",
                    file=str(filepath),
                    location=f"Cell {cell_num}",
                    category="Path",
                    message=f"Hardcoded absolute path: {path}",
                    fix="Use os.path.expanduser('~') or relative paths"
                ))
            
            # Issue: pip install in code cell (won't work on DGX Spark for PyTorch)
            if 'pip install torch' in source.lower():
                issues.append(Issue(
                    severity="CRITICAL",
                    file=str(filepath),
                    location=f"Cell {cell_num}",
                    category="DGX Spark",
                    message="'pip install torch' will not work on DGX Spark (ARM64+CUDA)",
                    fix="Use NGC container: nvcr.io/nvidia/pytorch:25.11-py3"
                ))
            
            # Issue: Missing memory cleanup at end
            if i == len(cells) - 1:  # Last cell
                if 'torch.cuda' in source and 'empty_cache' not in source:
                    issues.append(Issue(
                        severity="LOW",
                        file=str(filepath),
                        location=f"Cell {cell_num} (last cell)",
                        category="Memory",
                        message="Last cell uses CUDA but doesn't clean up memory",
                        fix="Add torch.cuda.empty_cache() and gc.collect() for cleanup"
                    ))
        
        elif cell_type == 'markdown':
            # Check for broken markdown
            if source.count('```') % 2 != 0:
                issues.append(Issue(
                    severity="MEDIUM",
                    file=str(filepath),
                    location=f"Cell {cell_num}",
                    category="Markdown",
                    message="Unmatched code fence (```) in markdown cell"
                ))
    
    return issues


def check_python_script(filepath: Path) -> List[Issue]:
    """Check Python script for common issues."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        issues.append(Issue(
            severity="CRITICAL",
            file=str(filepath),
            location="N/A",
            category="File",
            message=f"Cannot read file: {e}"
        ))
        return issues
    
    # Check syntax
    try:
        ast.parse(content)
    except SyntaxError as e:
        issues.append(Issue(
            severity="CRITICAL",
            file=str(filepath),
            location=f"Line {e.lineno}",
            category="Syntax",
            message=f"Syntax error: {e.msg}"
        ))
        return issues  # Can't do more checks if syntax is broken
    
    # Check for hardcoded paths
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if re.search(r'["\']\/home\/\w+', line):
            issues.append(Issue(
                severity="HIGH",
                file=str(filepath),
                location=f"Line {i}",
                category="Path",
                message=f"Hardcoded absolute path found",
                fix="Use os.path.expanduser('~') or Path.home()"
            ))
    
    # Check for missing docstring
    tree = ast.parse(content)
    if tree.body:
        first_node = tree.body[0]
        if not (isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant)):
            issues.append(Issue(
                severity="LOW",
                file=str(filepath),
                location="Line 1",
                category="Documentation",
                message="Missing module docstring"
            ))
    
    # Check functions for docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not (node.body and isinstance(node.body[0], ast.Expr) 
                    and isinstance(node.body[0].value, ast.Constant)):
                issues.append(Issue(
                    severity="LOW",
                    file=str(filepath),
                    location=f"Line {node.lineno}",
                    category="Documentation",
                    message=f"Function '{node.name}' missing docstring"
                ))
    
    return issues


def check_cross_references(module_path: Path) -> List[Issue]:
    """Check for cross-file reference issues."""
    issues = []
    
    # Gather all files
    notebooks = list(module_path.glob('notebooks/*.ipynb'))
    scripts = list(module_path.glob('scripts/*.py'))
    data_files = list(module_path.glob('data/*'))
    
    # Extract imports from notebooks that reference local scripts
    for nb_path in notebooks:
        try:
            with open(nb_path, 'r') as f:
                notebook = json.load(f)
            
            for i, cell in enumerate(notebook.get('cells', [])):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    
                    # Check for imports from scripts/
                    script_imports = re.findall(r'from\s+scripts\.(\w+)\s+import', source)
                    for script_name in script_imports:
                        script_path = module_path / 'scripts' / f'{script_name}.py'
                        if not script_path.exists():
                            issues.append(Issue(
                                severity="CRITICAL",
                                file=str(nb_path),
                                location=f"Cell {i+1}",
                                category="Cross-Reference",
                                message=f"Imports 'scripts.{script_name}' but scripts/{script_name}.py does not exist"
                            ))
                    
                    # Check for data file references
                    data_refs = re.findall(r'["\'](?:\.\.\/)?data\/([^"\']+)["\']', source)
                    for data_ref in data_refs:
                        data_path = module_path / 'data' / data_ref
                        if not data_path.exists():
                            issues.append(Issue(
                                severity="HIGH",
                                file=str(nb_path),
                                location=f"Cell {i+1}",
                                category="Cross-Reference",
                                message=f"References 'data/{data_ref}' but file does not exist"
                            ))
        except:
            pass  # JSON errors already caught elsewhere
    
    return issues


def validate_module(module_path: Path, verbose: bool = False) -> ValidationReport:
    """Run all validations on a module."""
    report = ValidationReport(module_path=str(module_path))
    
    # Find all files
    notebooks = list(module_path.rglob('*.ipynb'))
    scripts = list(module_path.rglob('*.py'))
    
    # Exclude checkpoints
    notebooks = [p for p in notebooks if '.ipynb_checkpoints' not in str(p)]
    
    report.files_checked = len(notebooks) + len(scripts)
    
    if verbose:
        print(f"Checking {len(notebooks)} notebooks and {len(scripts)} Python files...")
    
    # Check notebooks
    for nb_path in notebooks:
        if verbose:
            print(f"  Checking: {nb_path.name}")
        
        # JSON validity
        issues = check_notebook_json(nb_path)
        report.issues.extend(issues)
        
        # Cell checks (only if JSON is valid)
        if not any(i.severity == "CRITICAL" and i.category == "JSON" for i in issues):
            report.issues.extend(check_notebook_cells(nb_path))
    
    # Check Python scripts
    for script_path in scripts:
        if verbose:
            print(f"  Checking: {script_path.name}")
        report.issues.extend(check_python_script(script_path))
    
    # Check cross-references
    if verbose:
        print("  Checking cross-references...")
    report.issues.extend(check_cross_references(module_path))
    
    return report


def print_report(report: ValidationReport):
    """Print formatted validation report."""
    print("\n" + "=" * 70)
    print("MODULE VALIDATION REPORT")
    print("=" * 70)
    print(f"Module: {report.module_path}")
    print(f"Files checked: {report.files_checked}")
    print()
    
    # Summary
    status = "✅ PASSED" if report.passed else "❌ FAILED"
    print(f"Status: {status}")
    print(f"  🔴 Critical: {report.critical_count}")
    print(f"  🟠 High: {report.high_count}")
    print(f"  🟡 Medium: {report.medium_count}")
    print(f"  🟢 Low: {report.low_count}")
    print()
    
    # Group issues by severity
    for severity, emoji in [("CRITICAL", "🔴"), ("HIGH", "🟠"), ("MEDIUM", "🟡"), ("LOW", "🟢")]:
        severity_issues = [i for i in report.issues if i.severity == severity]
        if severity_issues:
            print(f"\n{emoji} {severity} ISSUES ({len(severity_issues)})")
            print("-" * 50)
            for i, issue in enumerate(severity_issues, 1):
                print(f"\n{i}. [{issue.category}] {issue.message}")
                print(f"   File: {issue.file}")
                print(f"   Location: {issue.location}")
                if issue.fix:
                    print(f"   Fix: {issue.fix}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Validate module content")
    parser.add_argument("module_path", type=Path, help="Path to module directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--fail-on", choices=["critical", "high", "medium", "low"],
                       default="critical", help="Exit code 1 if issues of this level or higher")
    
    args = parser.parse_args()
    
    if not args.module_path.exists():
        print(f"❌ Error: Path does not exist: {args.module_path}")
        return 1
    
    report = validate_module(args.module_path, args.verbose)
    
    if args.json:
        import json
        output = {
            "module_path": report.module_path,
            "files_checked": report.files_checked,
            "passed": report.passed,
            "summary": {
                "critical": report.critical_count,
                "high": report.high_count,
                "medium": report.medium_count,
                "low": report.low_count
            },
            "issues": [
                {
                    "severity": i.severity,
                    "file": i.file,
                    "location": i.location,
                    "category": i.category,
                    "message": i.message,
                    "fix": i.fix
                }
                for i in report.issues
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report)
    
    # Determine exit code
    fail_levels = {
        "critical": ["CRITICAL"],
        "high": ["CRITICAL", "HIGH"],
        "medium": ["CRITICAL", "HIGH", "MEDIUM"],
        "low": ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    }
    
    fail_severities = fail_levels[args.fail_on]
    should_fail = any(i.severity in fail_severities for i in report.issues)
    
    return 1 if should_fail else 0


if __name__ == "__main__":
    exit(main())
