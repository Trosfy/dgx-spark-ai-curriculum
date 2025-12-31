"""
Pytest configuration for MicroGrad+ tests.

This module sets up the Python path to enable imports from the micrograd_plus
package regardless of how pytest is invoked.
"""

import sys
from pathlib import Path


def _find_module_root() -> str:
    """Find the module root directory containing micrograd_plus."""
    # Start from the tests directory
    current = Path(__file__).parent.absolute()
    for parent in [current] + list(current.parents):
        if (parent / 'micrograd_plus' / '__init__.py').exists():
            return str(parent)
    # Fallback to parent of tests directory
    return str(Path(__file__).parent.parent)


# Add module root to path for imports
module_root = _find_module_root()
if module_root not in sys.path:
    sys.path.insert(0, module_root)
