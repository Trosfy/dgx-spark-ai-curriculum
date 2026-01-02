
#!/usr/bin/env python3
"""
DGX Spark GPU Verification Script
Run this inside the NGC container to verify GPU access.
"""

import sys

def check_torch():
    """Check PyTorch GPU access."""
    print("=" * 60)
    print("PyTorch GPU Verification")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")

            # Memory info
            props = torch.cuda.get_device_properties(0)
            print(f"Total memory: {props.total_memory / 1e9:.1f} GB")

            # Test tensor operation
            print("
Testing tensor operations...")
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            print(f"✅ Matrix multiplication successful!")
            print(f"   Result shape: {z.shape}")
            print(f"   Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

            return True
        else:
            print("❌ CUDA not available!")
            return False

    except ImportError:
        print("❌ PyTorch not installed!")
        return False

def check_cudnn():
    """Check cuDNN status."""
    print("
" + "-" * 40)
    print("cuDNN Status")
    print("-" * 40)

    try:
        import torch
        print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_bfloat16():
    """Check bfloat16 support (important for Blackwell)."""
    print("
" + "-" * 40)
    print("BFloat16 Support (Blackwell Optimized)")
    print("-" * 40)

    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(100, 100, dtype=torch.bfloat16, device="cuda")
            y = torch.randn(100, 100, dtype=torch.bfloat16, device="cuda")
            z = torch.matmul(x, y)
            print(f"✅ BFloat16 operations work!")
            return True
    except Exception as e:
        print(f"❌ BFloat16 error: {e}")
        return False

if __name__ == "__main__":
    all_passed = True
    all_passed &= check_torch()
    all_passed &= check_cudnn()
    all_passed &= check_bfloat16()

    print("
" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! Your DGX Spark is ready for AI!")
    else:
        print("❌ Some checks failed. Please review the output above.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
