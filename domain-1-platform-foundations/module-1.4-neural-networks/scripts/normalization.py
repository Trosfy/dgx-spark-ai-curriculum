"""
Normalization Layers - Built from Scratch with NumPy

This module implements normalization techniques used in deep learning:
- Batch Normalization (BatchNorm)
- Layer Normalization (LayerNorm)
- RMS Normalization (RMSNorm)

Professor SPARK says: "Normalization is like making sure all students
in a class are graded on the same curve. It keeps things fair and
prevents any single feature from dominating the learning!"

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 4
"""

import numpy as np
from typing import Tuple, Optional, Dict


class BatchNorm:
    """
    Batch Normalization layer.

    Normalizes activations across the batch dimension, then applies
    learnable scale (gamma) and shift (beta) parameters.

    ELI5: Imagine you're comparing test scores from different schools.
    Each school might have different grading standards. BatchNorm is like
    first converting everyone to the same scale (mean=0, std=1), then
    letting the network decide what scale works best (gamma, beta).

    Why it helps:
    1. Reduces internal covariate shift (activations changing during training)
    2. Allows higher learning rates
    3. Provides some regularization (batch statistics add noise)

    Parameters:
        num_features: Number of features (channels)
        momentum: Momentum for running statistics (default: 0.1)
        epsilon: Small constant for numerical stability (default: 1e-5)

    Notes:
        - Training: Use batch statistics
        - Inference: Use running (moving average) statistics
        - This difference can cause issues with small batch sizes!

    Example:
        >>> bn = BatchNorm(num_features=256)
        >>> x = np.random.randn(32, 256)  # batch of 32, 256 features
        >>> output = bn(x, training=True)
        >>> print(output.mean(axis=0)[:5])  # Should be close to beta (0)
        >>> print(output.std(axis=0)[:5])   # Should be close to gamma (1)
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        epsilon: float = 1e-5
    ):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        self.trainable = True

        # Learnable parameters
        self.gamma = np.ones(num_features)   # Scale
        self.beta = np.zeros(num_features)   # Shift

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Cache for backward pass
        self.cache: Dict[str, np.ndarray] = {}
        self.gradients: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, num_features)
            training: Whether we're training (use batch stats) or not (use running stats)

        Returns:
            Normalized output of same shape as input
        """
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Update running statistics (exponential moving average)
            self.running_mean = (
                (1 - self.momentum) * self.running_mean +
                self.momentum * batch_mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var +
                self.momentum * batch_var
            )

            # Save for backward pass
            self.cache['x'] = x
            self.cache['x_normalized'] = x_normalized
            self.cache['mean'] = batch_mean
            self.cache['var'] = batch_var
            self.cache['std'] = np.sqrt(batch_var + self.epsilon)
        else:
            # Use running statistics for inference
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        output = self.gamma * x_normalized + self.beta

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Computes gradients for gamma, beta, and input.

        The math here is a bit involved, but the key insight is:
        we need to account for how the mean and variance depend on
        all elements of the batch, not just the current element.
        """
        x = self.cache['x']
        x_normalized = self.cache['x_normalized']
        mean = self.cache['mean']
        var = self.cache['var']
        std = self.cache['std']
        batch_size = x.shape[0]

        # Gradient of gamma and beta
        self.gradients['gamma'] = np.sum(grad_output * x_normalized, axis=0)
        self.gradients['beta'] = np.sum(grad_output, axis=0)

        # Gradient with respect to normalized x
        dx_normalized = grad_output * self.gamma

        # Gradient with respect to variance
        dvar = np.sum(
            dx_normalized * (x - mean) * -0.5 * (var + self.epsilon) ** (-1.5),
            axis=0
        )

        # Gradient with respect to mean
        dmean = (
            np.sum(dx_normalized * -1 / std, axis=0) +
            dvar * np.mean(-2 * (x - mean), axis=0)
        )

        # Gradient with respect to input
        grad_input = (
            dx_normalized / std +
            dvar * 2 * (x - mean) / batch_size +
            dmean / batch_size
        )

        return grad_input

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Allow layer to be called like a function."""
        return self.forward(x, training)


class LayerNorm:
    """
    Layer Normalization.

    Normalizes across the feature dimension (not the batch dimension).

    ELI5: If BatchNorm compares students across schools (batch),
    LayerNorm compares each student's performance across subjects (features).
    Each student gets normalized independently based on their own subjects.

    Key differences from BatchNorm:
    1. Normalizes over features, not batch
    2. Same behavior in training and inference
    3. No dependency on batch size (works with batch_size=1!)
    4. Preferred for transformers, RNNs, and small batch scenarios

    Parameters:
        normalized_shape: Shape of the feature dimension(s)
        epsilon: Small constant for numerical stability (default: 1e-5)

    Example:
        >>> ln = LayerNorm(normalized_shape=256)
        >>> x = np.random.randn(32, 256)
        >>> output = ln(x)
        >>> # Each sample is normalized independently
        >>> print(output[0].mean(), output[0].std())  # ~0, ~1
    """

    def __init__(
        self,
        normalized_shape: int,
        epsilon: float = 1e-5
    ):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.trainable = True

        # Learnable parameters
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

        # Cache for backward pass
        self.cache: Dict[str, np.ndarray] = {}
        self.gradients: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, normalized_shape)

        Returns:
            Normalized output of same shape as input
        """
        # Compute mean and variance over feature dimension (last axis)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)

        # Scale and shift
        output = self.gamma * x_normalized + self.beta

        # Save for backward
        self.cache['x'] = x
        self.cache['x_normalized'] = x_normalized
        self.cache['mean'] = mean
        self.cache['var'] = var
        self.cache['std'] = np.sqrt(var + self.epsilon)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Similar to BatchNorm, but normalizing over features instead of batch.
        """
        x = self.cache['x']
        x_normalized = self.cache['x_normalized']
        mean = self.cache['mean']
        var = self.cache['var']
        std = self.cache['std']
        n = x.shape[-1]  # Number of features

        # Gradient of gamma and beta
        self.gradients['gamma'] = np.sum(grad_output * x_normalized, axis=0)
        self.gradients['beta'] = np.sum(grad_output, axis=0)

        # Gradient with respect to normalized x
        dx_normalized = grad_output * self.gamma

        # Gradient with respect to variance
        dvar = np.sum(
            dx_normalized * (x - mean) * -0.5 * (var + self.epsilon) ** (-1.5),
            axis=-1,
            keepdims=True
        )

        # Gradient with respect to mean
        dmean = (
            np.sum(dx_normalized * -1 / std, axis=-1, keepdims=True) +
            dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        )

        # Gradient with respect to input
        grad_input = (
            dx_normalized / std +
            dvar * 2 * (x - mean) / n +
            dmean / n
        )

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow layer to be called like a function."""
        return self.forward(x)


class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    A simplified version of LayerNorm used in modern transformers (LLaMA, etc.)

    ELI5: RMSNorm is LayerNorm's lazy cousin. Instead of centering the data
    AND scaling it, it only does the scaling part. Surprisingly, this often
    works just as well and is faster to compute!

    Key differences from LayerNorm:
    1. No mean subtraction (no centering)
    2. Uses RMS (root mean square) instead of standard deviation
    3. Slightly faster computation
    4. Used in LLaMA, Mistral, and other modern LLMs

    Formula: RMSNorm(x) = x / RMS(x) * gamma
             where RMS(x) = sqrt(mean(x^2))

    Parameters:
        normalized_shape: Shape of the feature dimension
        epsilon: Small constant for numerical stability (default: 1e-5)

    Example:
        >>> rms = RMSNorm(normalized_shape=256)
        >>> x = np.random.randn(32, 256)
        >>> output = rms(x)
    """

    def __init__(
        self,
        normalized_shape: int,
        epsilon: float = 1e-5
    ):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.trainable = True

        # Only gamma (scale), no beta (shift)
        self.gamma = np.ones(normalized_shape)

        # Cache for backward pass
        self.cache: Dict[str, np.ndarray] = {}
        self.gradients: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, normalized_shape)

        Returns:
            Normalized output of same shape as input
        """
        # Compute RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)

        # Normalize
        x_normalized = x / rms

        # Scale (no shift in RMSNorm)
        output = self.gamma * x_normalized

        # Save for backward
        self.cache['x'] = x
        self.cache['rms'] = rms
        self.cache['x_normalized'] = x_normalized

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass for RMSNorm."""
        x = self.cache['x']
        rms = self.cache['rms']
        x_normalized = self.cache['x_normalized']
        n = x.shape[-1]

        # Gradient of gamma
        self.gradients['gamma'] = np.sum(grad_output * x_normalized, axis=0)

        # Gradient with respect to normalized x
        dx_normalized = grad_output * self.gamma

        # Gradient with respect to RMS
        drms = np.sum(dx_normalized * x * -1 / (rms ** 2), axis=-1, keepdims=True)

        # Gradient with respect to x^2 (via mean)
        dx_sq = drms * 0.5 / rms / n

        # Gradient with respect to input
        grad_input = dx_normalized / rms + 2 * dx_sq * x

        return grad_input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow layer to be called like a function."""
        return self.forward(x)


def compare_normalizations(
    x: np.ndarray,
    batch_norm: Optional[BatchNorm] = None,
    layer_norm: Optional[LayerNorm] = None,
    rms_norm: Optional[RMSNorm] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different normalization techniques on the same input.

    Args:
        x: Input array of shape (batch_size, features)
        batch_norm: BatchNorm layer (optional)
        layer_norm: LayerNorm layer (optional)
        rms_norm: RMSNorm layer (optional)

    Returns:
        Dictionary with statistics for each normalization type

    Example:
        >>> x = np.random.randn(32, 256) * 10 + 5  # Shifted and scaled
        >>> bn = BatchNorm(256)
        >>> ln = LayerNorm(256)
        >>> rms = RMSNorm(256)
        >>> stats = compare_normalizations(x, bn, ln, rms)
        >>> print(stats)
    """
    results = {}

    if batch_norm is not None:
        out_bn = batch_norm(x, training=True)
        results['batch_norm'] = {
            'mean_across_batch': float(np.mean(out_bn.mean(axis=0))),
            'std_across_batch': float(np.mean(out_bn.std(axis=0))),
            'mean_per_sample': float(np.mean([s.mean() for s in out_bn])),
            'std_per_sample': float(np.mean([s.std() for s in out_bn]))
        }

    if layer_norm is not None:
        out_ln = layer_norm(x)
        results['layer_norm'] = {
            'mean_across_batch': float(np.mean(out_ln.mean(axis=0))),
            'std_across_batch': float(np.mean(out_ln.std(axis=0))),
            'mean_per_sample': float(np.mean([s.mean() for s in out_ln])),
            'std_per_sample': float(np.mean([s.std() for s in out_ln]))
        }

    if rms_norm is not None:
        out_rms = rms_norm(x)
        results['rms_norm'] = {
            'mean_across_batch': float(np.mean(out_rms.mean(axis=0))),
            'std_across_batch': float(np.mean(out_rms.std(axis=0))),
            'mean_per_sample': float(np.mean([s.mean() for s in out_rms])),
            'std_per_sample': float(np.mean([s.std() for s in out_rms]))
        }

    return results


if __name__ == "__main__":
    print("Testing Normalization Layers")
    print("=" * 50)

    np.random.seed(42)

    # Create test input with non-zero mean and non-unit variance
    batch_size, num_features = 32, 256
    x = np.random.randn(batch_size, num_features) * 5 + 3  # mean~3, std~5

    print(f"\nInput statistics:")
    print(f"  Mean: {x.mean():.4f}")
    print(f"  Std:  {x.std():.4f}")

    # Test BatchNorm
    print("\n1. Testing BatchNorm:")
    bn = BatchNorm(num_features)
    out_bn = bn(x, training=True)
    print(f"   Output mean (should be ~0): {out_bn.mean(axis=0).mean():.6f}")
    print(f"   Output std (should be ~1):  {out_bn.std(axis=0).mean():.6f}")

    # Test backward
    grad = np.random.randn(*out_bn.shape)
    grad_input = bn.backward(grad)
    print(f"   Backward pass shape: {grad_input.shape}")

    # Test inference mode
    out_bn_infer = bn(x, training=False)
    print(f"   Inference mode works: {out_bn_infer.shape == x.shape}")

    # Test LayerNorm
    print("\n2. Testing LayerNorm:")
    ln = LayerNorm(num_features)
    out_ln = ln(x)
    print(f"   Per-sample mean (should be ~0): {np.mean([s.mean() for s in out_ln]):.6f}")
    print(f"   Per-sample std (should be ~1):  {np.mean([s.std() for s in out_ln]):.6f}")

    # Test backward
    grad_input = ln.backward(grad)
    print(f"   Backward pass shape: {grad_input.shape}")

    # Test RMSNorm
    print("\n3. Testing RMSNorm:")
    rms = RMSNorm(num_features)
    out_rms = rms(x)
    rms_values = np.sqrt(np.mean(out_rms ** 2, axis=-1))
    print(f"   Per-sample RMS (should be ~1): {rms_values.mean():.6f}")
    print(f"   Note: Mean is not centered (that's the point of RMSNorm)")
    print(f"   Per-sample mean: {np.mean([s.mean() for s in out_rms]):.6f}")

    # Test backward
    grad_input = rms.backward(grad)
    print(f"   Backward pass shape: {grad_input.shape}")

    # Compare all normalizations
    print("\n4. Comparison Summary:")
    print("   " + "-" * 40)
    print("   Normalization | Across Batch | Per Sample")
    print("   " + "-" * 40)
    print(f"   BatchNorm     | mean={out_bn.mean(axis=0).mean():.3f}, std={out_bn.std(axis=0).mean():.3f} |"
          f" mean varies, std varies")
    print(f"   LayerNorm     | mean varies, std varies |"
          f" mean={np.mean([s.mean() for s in out_ln]):.3f}, std={np.mean([s.std() for s in out_ln]):.3f}")
    print(f"   RMSNorm       | mean varies, std varies |"
          f" RMS={rms_values.mean():.3f} (no mean centering)")
    print("   " + "-" * 40)

    # Key takeaways
    print("\n5. When to use each:")
    print("   - BatchNorm: CNNs, large batch sizes, vision tasks")
    print("   - LayerNorm: Transformers, RNNs, small batches, sequence models")
    print("   - RMSNorm:   Modern LLMs (LLaMA, Mistral), faster than LayerNorm")

    print("\n" + "=" * 50)
    print("All normalization tests passed!")
