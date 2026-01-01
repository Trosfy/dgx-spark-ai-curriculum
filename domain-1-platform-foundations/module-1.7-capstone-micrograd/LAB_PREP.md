# Module 1.7: MicroGrad+ Lab Preparation

Complete this setup before starting the lab notebooks.

---

## Environment Setup

### Step 1: Launch the NGC Container

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Required flags explained:**
| Flag | Purpose |
|------|---------|
| `--gpus all` | Enable GPU access |
| `-it` | Interactive terminal |
| `--rm` | Remove container on exit |
| `-v $HOME/workspace:/workspace` | Persist your work |
| `-v $HOME/.cache/huggingface:/root/.cache/huggingface` | Cache downloads |
| `--ipc=host` | Required for PyTorch DataLoader |
| `-p 8888:8888` | Expose Jupyter port |

### Step 2: Access JupyterLab

1. Look for the URL in terminal output:
   ```
   http://127.0.0.1:8888/lab?token=<your-token>
   ```
2. Open in your browser
3. Navigate to the module directory

---

## Directory Structure

Ensure your workspace has this structure:

```
module-1.7-capstone-micrograd/
├── micrograd_plus/           # The library you'll explore and use
│   ├── __init__.py
│   ├── tensor.py            # Core Tensor with autograd
│   ├── layers.py            # Neural network layers
│   ├── losses.py            # Loss functions
│   ├── optimizers.py        # Optimization algorithms
│   ├── nn.py                # Sequential and training utilities
│   └── utils.py             # Helper functions
├── labs/                     # Lab notebooks
│   ├── lab-1.7.1-core-tensor-implementation.ipynb
│   ├── lab-1.7.2-neural-network-layers.ipynb
│   ├── lab-1.7.3-loss-functions-optimizers.ipynb
│   ├── lab-1.7.4-training-loop-integration.ipynb
│   ├── lab-1.7.5-mnist-example.ipynb
│   └── lab-1.7.6-documentation-benchmarks.ipynb
├── tests/                    # Unit tests
│   ├── test_tensor.py
│   ├── test_layers.py
│   └── test_optimizers.py
├── data/                     # Dataset storage
│   └── README.md
├── solutions/                # Reference solutions
└── docs/                     # Additional documentation
    ├── API.md
    └── TUTORIAL.md
```

---

## Dependency Verification

Run this cell at the start of any lab notebook:

```python
# Verify environment
import sys
from pathlib import Path

def verify_environment():
    """Check all requirements are met."""
    checks = []

    # Python version
    py_version = sys.version_info
    checks.append(("Python 3.10+", py_version >= (3, 10)))

    # NumPy
    try:
        import numpy as np
        checks.append((f"NumPy {np.__version__}", True))
    except ImportError:
        checks.append(("NumPy", False))

    # Matplotlib
    try:
        import matplotlib
        checks.append((f"Matplotlib {matplotlib.__version__}", True))
    except ImportError:
        checks.append(("Matplotlib", False))

    # MicroGrad+ accessibility
    try:
        # Find module root
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / 'micrograd_plus' / '__init__.py').exists():
                sys.path.insert(0, str(parent))
                break

        from micrograd_plus import Tensor
        checks.append(("MicroGrad+ import", True))
    except ImportError as e:
        checks.append((f"MicroGrad+ import ({e})", False))

    # Print results
    print("Environment Check:")
    print("-" * 40)
    all_pass = True
    for name, status in checks:
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {name}")
        if not status:
            all_pass = False
    print("-" * 40)

    if all_pass:
        print("✅ All checks passed! Ready to start.")
    else:
        print("❌ Some checks failed. See above.")

    return all_pass

verify_environment()
```

**Expected output:**
```
Environment Check:
----------------------------------------
  ✅ Python 3.10+
  ✅ NumPy 1.24.0
  ✅ Matplotlib 3.7.1
  ✅ MicroGrad+ import
----------------------------------------
✅ All checks passed! Ready to start.
```

---

## Running Tests

Before each lab, run the test suite to ensure the library works:

```bash
# From module root directory
cd /workspace/module-1.7-capstone-micrograd

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tensor.py -v

# Run with coverage
python -m pytest tests/ -v --cov=micrograd_plus
```

**Expected output:**
```
tests/test_tensor.py::TestTensorCreation::test_from_list PASSED
tests/test_tensor.py::TestTensorCreation::test_from_numpy PASSED
tests/test_tensor.py::TestTensorOperations::test_addition PASSED
...
```

---

## MNIST Data Download

Lab 1.7.5 requires MNIST data. Pre-download it:

```python
import os
import gzip
import socket
from urllib import request
from pathlib import Path

def download_mnist(path='./data', timeout=30):
    """Download MNIST dataset."""
    os.makedirs(path, exist_ok=True)

    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
        'http://yann.lecun.com/exdb/mnist/',
    ]

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    for f in files:
        filepath = os.path.join(path, f)
        if os.path.exists(filepath):
            print(f"✅ {f} already exists")
            continue

        for base_url in urls:
            try:
                print(f"📥 Downloading {f}...")
                socket.setdefaulttimeout(timeout)
                request.urlretrieve(base_url + f, filepath)
                print(f"✅ Downloaded {f}")
                break
            except Exception as e:
                print(f"⚠️ {base_url} failed: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)

# Run from module root
download_mnist()
```

---

## Memory Considerations

MicroGrad+ is designed for learning, not performance. Keep these limits in mind:

| Operation | Recommended Limit | Notes |
|-----------|------------------|-------|
| Batch size | 32-128 | Larger batches use more memory |
| Model layers | 3-5 | Deep models are slower |
| Training samples | 10,000 | Full MNIST (60k) is slow |
| Epochs | 10-20 | More than enough to converge |

### Memory Cleanup

Use this pattern at the end of each notebook:

```python
# Standard cleanup
from micrograd_plus.utils import cleanup_notebook
cleanup_notebook(globals())
```

Or manually:

```python
import gc

# Delete large variables
del model, X_train, y_train, optimizer
gc.collect()

print("Memory cleaned!")
```

---

## Lab Workflow

For each lab notebook:

1. **Start fresh:**
   ```python
   # Run verification
   verify_environment()
   ```

2. **Read the objectives** at the top of the notebook

3. **Run cells in order** - cells often depend on previous ones

4. **Complete exercises** before checking solutions

5. **Clean up** at the end:
   ```python
   cleanup_notebook(globals())
   ```

---

## Troubleshooting Setup Issues

### Can't import micrograd_plus

```python
# Add to the start of your notebook
import sys
from pathlib import Path

def find_module_root():
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / 'micrograd_plus' / '__init__.py').exists():
            return str(parent)
    raise FileNotFoundError("Could not find micrograd_plus directory")

sys.path.insert(0, find_module_root())
```

### Tests fail

1. Ensure you're in the correct directory:
   ```bash
   pwd  # Should show module-1.7-capstone-micrograd
   ```

2. Check Python path:
   ```python
   import sys
   print(sys.path)  # Should include module root
   ```

### Jupyter can't find kernel

Restart the container and ensure you're using the correct Python:
```bash
which python  # Should be /usr/bin/python or similar in container
```

---

## Quick Start Checklist

- [ ] NGC container launched with all required flags
- [ ] JupyterLab accessible in browser
- [ ] `verify_environment()` passes all checks
- [ ] `python -m pytest tests/ -v` passes
- [ ] MNIST data downloaded (for Lab 1.7.5)
- [ ] Understand memory limits (batch size 32-128)

---

→ Ready? Start with [Lab 1.7.1: Core Tensor Implementation](./labs/lab-1.7.1-core-tensor-implementation.ipynb)
