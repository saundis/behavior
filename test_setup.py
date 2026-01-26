"""
Test script to verify the setup is working correctly.

Run this after installing dependencies to ensure everything is configured.
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"  [OK] torch {torch.__version__}")
    except ImportError as e:
        print(f"  [FAIL] torch: {e}")
        return False

    try:
        import transformers
        print(f"  [OK] transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  [FAIL] transformers: {e}")
        return False

    try:
        import accelerate
        print(f"  [OK] accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"  [FAIL] accelerate: {e}")
        return False

    try:
        import matplotlib
        print(f"  [OK] matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  [FAIL] matplotlib: {e}")
        return False

    try:
        import pandas
        print(f"  [OK] pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  [FAIL] pandas: {e}")
        return False

    try:
        import tqdm
        print(f"  [OK] tqdm {tqdm.__version__}")
    except ImportError as e:
        print(f"  [FAIL] tqdm: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    import torch

    if torch.cuda.is_available():
        print(f"  [OK] CUDA is available")
        print(f"  [OK] Device: {torch.cuda.get_device_name(0)}")
        print(f"  [OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("  [WARN] CUDA is not available, will use CPU (slower)")
        return True  # Not a failure, just a warning


def test_local_modules():
    """Test that our local modules can be imported."""
    print("\nTesting local modules...")

    # Add src to path
    sys.path.insert(0, 'src')

    try:
        from models import load_model, load_tokenizer, OLMO_MODELS
        print("  [OK] models.py")
    except ImportError as e:
        print(f"  [FAIL] models.py: {e}")
        return False

    try:
        from logits import get_next_token_logits, compare_logit_distributions
        print("  [OK] logits.py")
    except ImportError as e:
        print(f"  [FAIL] logits.py: {e}")
        return False

    try:
        from interpolation import interpolate_logits, LogitInterpolator
        print("  [OK] interpolation.py")
    except ImportError as e:
        print(f"  [FAIL] interpolation.py: {e}")
        return False

    try:
        from generation import generate_from_interpolated_logits, sample_token
        print("  [OK] generation.py")
    except ImportError as e:
        print(f"  [FAIL] generation.py: {e}")
        return False

    try:
        from evaluation import classify_refusal, compute_refusal_rate
        print("  [OK] evaluation.py")
    except ImportError as e:
        print(f"  [FAIL] evaluation.py: {e}")
        return False

    return True


def test_interpolation_math():
    """Test that interpolation math is correct."""
    print("\nTesting interpolation math...")

    import torch
    sys.path.insert(0, 'src')
    from interpolation import interpolate_logits

    # Create dummy logits
    logits_m = torch.tensor([1.0, 2.0, 3.0])
    logits_n = torch.tensor([4.0, 5.0, 6.0])

    # Test alpha = 0 (should return logits_n)
    result = interpolate_logits(logits_m, logits_n, 0.0)
    assert torch.allclose(result, logits_n), "alpha=0 should return logits_n"
    print("  [OK] alpha=0 returns logits_n")

    # Test alpha = 1 (should return logits_m)
    result = interpolate_logits(logits_m, logits_n, 1.0)
    assert torch.allclose(result, logits_m), "alpha=1 should return logits_m"
    print("  [OK] alpha=1 returns logits_m")

    # Test alpha = 0.5 (should return average)
    result = interpolate_logits(logits_m, logits_n, 0.5)
    expected = (logits_m + logits_n) / 2
    assert torch.allclose(result, expected), "alpha=0.5 should return average"
    print("  [OK] alpha=0.5 returns average")

    # Test extrapolation (alpha = 2)
    result = interpolate_logits(logits_m, logits_n, 2.0)
    expected = 2.0 * logits_m + (-1.0) * logits_n
    assert torch.allclose(result, expected), "extrapolation formula incorrect"
    print("  [OK] extrapolation (alpha=2) works correctly")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("BEHAVIOR EXPERIMENTS - SETUP TEST")
    print("=" * 50)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_cuda()
    all_passed &= test_local_modules()
    all_passed &= test_interpolation_math()

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. Run: python -c \"from src.models import load_model; m = load_model('allenai/OLMo-1B')\"")
        print("  2. This will download OLMo-1B (~2GB) and verify it loads")
    else:
        print("SOME TESTS FAILED - please fix before proceeding")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
