"""
Tests for automatic differentiation (gradient verification).

These tests verify that analytical gradients match numerical gradients
computed via finite differences.

Run with: python -m pytest tests/test_autograd.py -v
"""

import numpy as np
import pytest

from micrograd_plus import Tensor, Linear, ReLU, Sequential
from micrograd_plus.utils import numerical_gradient


def gradient_check(f, x, eps=1e-5, atol=1e-4, rtol=1e-3):
    """
    Verify analytical gradient matches numerical gradient.

    Args:
        f: Function taking Tensor, returning scalar Tensor
        x: Input tensor with requires_grad=True
        eps: Perturbation for numerical gradient
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        (passed, max_error)
    """
    # Analytical gradient
    x.zero_grad()
    y = f(x)
    y.backward()
    analytical = x.grad.copy()

    # Numerical gradient
    def numpy_f(arr):
        return f(Tensor(arr)).data.item()

    numerical = numerical_gradient(numpy_f, x.data.copy(), eps)

    # Compare
    max_error = np.max(np.abs(analytical - numerical))
    passed = np.allclose(analytical, numerical, atol=atol, rtol=rtol)

    return passed, max_error, analytical, numerical


class TestBasicOperationGradients:
    """Test gradients for basic arithmetic operations."""

    def test_addition(self):
        """Test gradient of addition."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: (t + 5).sum(), x)
        assert passed, f"Addition gradient error: {error:.2e}"

    def test_subtraction(self):
        """Test gradient of subtraction."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: (t - 5).sum(), x)
        assert passed, f"Subtraction gradient error: {error:.2e}"

    def test_multiplication(self):
        """Test gradient of element-wise multiplication."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Tensor([4.0, 5.0, 6.0])
        passed, error, _, _ = gradient_check(lambda t: (t * y).sum(), x)
        assert passed, f"Multiplication gradient error: {error:.2e}"

    def test_division(self):
        """Test gradient of division."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: (t / 2).sum(), x)
        assert passed, f"Division gradient error: {error:.2e}"

    def test_power(self):
        """Test gradient of power."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: (t ** 2).sum(), x)
        assert passed, f"Power gradient error: {error:.2e}"

    def test_power_fractional(self):
        """Test gradient of fractional power (sqrt)."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: (t ** 0.5).sum(), x)
        assert passed, f"Fractional power gradient error: {error:.2e}"

    def test_negation(self):
        """Test gradient of negation."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: (-t).sum(), x)
        assert passed, f"Negation gradient error: {error:.2e}"


class TestMatrixOperationGradients:
    """Test gradients for matrix operations."""

    def test_matmul_2d(self):
        """Test gradient of 2D matrix multiplication."""
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(4, 2).astype(np.float32))
        passed, error, _, _ = gradient_check(lambda t: (t @ w).sum(), x)
        assert passed, f"Matmul gradient error: {error:.2e}"

    def test_matmul_batched(self):
        """Test gradient of batched matrix multiplication."""
        np.random.seed(42)
        x = Tensor(np.random.randn(5, 3).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(3, 4).astype(np.float32))
        passed, error, _, _ = gradient_check(lambda t: (t @ w).sum(), x)
        assert passed, f"Batched matmul gradient error: {error:.2e}"

    def test_transpose(self):
        """Test gradient of transpose."""
        np.random.seed(42)
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.T.sum(), x)
        assert passed, f"Transpose gradient error: {error:.2e}"


class TestReductionGradients:
    """Test gradients for reduction operations."""

    def test_sum_all(self):
        """Test gradient of sum over all elements."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.sum(), x)
        assert passed, f"Sum gradient error: {error:.2e}"

    def test_sum_axis(self):
        """Test gradient of sum over axis."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.sum(axis=0).sum(), x)
        assert passed, f"Sum axis gradient error: {error:.2e}"

    def test_mean_all(self):
        """Test gradient of mean over all elements."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.mean(), x)
        assert passed, f"Mean gradient error: {error:.2e}"

    def test_max(self):
        """Test gradient of max."""
        x = Tensor([1.0, 5.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.max(), x)
        assert passed, f"Max gradient error: {error:.2e}"


class TestActivationGradients:
    """Test gradients for activation functions."""

    def test_relu(self):
        """Test gradient of ReLU (away from zero)."""
        x = Tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.relu().sum(), x)
        assert passed, f"ReLU gradient error: {error:.2e}"

    def test_sigmoid(self):
        """Test gradient of sigmoid."""
        x = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.sigmoid().sum(), x)
        assert passed, f"Sigmoid gradient error: {error:.2e}"

    def test_tanh(self):
        """Test gradient of tanh."""
        x = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.tanh().sum(), x)
        assert passed, f"Tanh gradient error: {error:.2e}"

    def test_softmax(self):
        """Test gradient of softmax."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        # Use log after softmax for more meaningful gradient
        passed, error, _, _ = gradient_check(
            lambda t: (t.softmax() * Tensor([[1, 2, 3]])).sum(), x
        )
        assert passed, f"Softmax gradient error: {error:.2e}"

    def test_log_softmax(self):
        """Test gradient of log-softmax."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.log_softmax().sum(), x)
        assert passed, f"Log-softmax gradient error: {error:.2e}"


class TestMathFunctionGradients:
    """Test gradients for mathematical functions."""

    def test_exp(self):
        """Test gradient of exp."""
        x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.exp().sum(), x)
        assert passed, f"Exp gradient error: {error:.2e}"

    def test_log(self):
        """Test gradient of log."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.log().sum(), x)
        assert passed, f"Log gradient error: {error:.2e}"

    def test_sqrt(self):
        """Test gradient of sqrt."""
        x = Tensor([1.0, 4.0, 9.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.sqrt().sum(), x)
        assert passed, f"Sqrt gradient error: {error:.2e}"

    def test_abs(self):
        """Test gradient of abs (away from zero)."""
        x = Tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.abs().sum(), x)
        assert passed, f"Abs gradient error: {error:.2e}"


class TestShapeOperationGradients:
    """Test gradients for shape operations."""

    def test_reshape(self):
        """Test gradient of reshape."""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.reshape(3, 2).sum(), x)
        assert passed, f"Reshape gradient error: {error:.2e}"

    def test_flatten(self):
        """Test gradient of flatten."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.flatten().sum(), x)
        assert passed, f"Flatten gradient error: {error:.2e}"

    def test_squeeze(self):
        """Test gradient of squeeze."""
        x = Tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.squeeze().sum(), x)
        assert passed, f"Squeeze gradient error: {error:.2e}"

    def test_unsqueeze(self):
        """Test gradient of unsqueeze."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: t.unsqueeze(0).sum(), x)
        assert passed, f"Unsqueeze gradient error: {error:.2e}"


class TestBroadcastingGradients:
    """Test gradients with broadcasting."""

    def test_broadcast_add(self):
        """Test gradient of broadcast addition."""
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = Tensor([10.0])
        passed, error, _, _ = gradient_check(lambda t: (t + y).sum(), x)
        assert passed, f"Broadcast add gradient error: {error:.2e}"

    def test_broadcast_mul(self):
        """Test gradient of broadcast multiplication."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([2.0, 3.0])
        passed, error, _, _ = gradient_check(lambda t: (t * y).sum(), x)
        assert passed, f"Broadcast mul gradient error: {error:.2e}"


class TestComplexExpressionGradients:
    """Test gradients for complex expressions."""

    def test_chain_of_operations(self):
        """Test gradient through chain of operations."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        passed, error, _, _ = gradient_check(
            lambda t: ((t * 2 + 1) ** 2).sum(), x
        )
        assert passed, f"Chain gradient error: {error:.2e}"

    def test_linear_layer(self):
        """Test gradient through Linear layer."""
        np.random.seed(42)
        layer = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: layer(t).sum(), x)
        assert passed, f"Linear layer gradient error: {error:.2e}"

    def test_mlp(self):
        """Test gradient through MLP."""
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 3)
        )
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        passed, error, _, _ = gradient_check(lambda t: model(t).sum(), x)
        assert passed, f"MLP gradient error: {error:.2e}"


class TestSpecialCases:
    """Test gradient handling for special cases."""

    def test_multiple_uses(self):
        """Test gradient when variable is used multiple times."""
        x = Tensor([2.0], requires_grad=True)
        passed, error, anal, num = gradient_check(
            lambda t: (t * t).sum(), x  # x^2
        )
        # Gradient should be 2x = 4
        assert np.allclose(anal, [4.0])
        assert passed, f"Multiple uses gradient error: {error:.2e}"

    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        x = Tensor([3.0], requires_grad=True)
        passed, error, anal, num = gradient_check(
            lambda t: (t + t + t).sum(), x  # 3x
        )
        # Gradient should be 3
        assert np.allclose(anal, [3.0])
        assert passed, f"Accumulation gradient error: {error:.2e}"

    def test_detach_breaks_gradient(self):
        """Test that detach breaks gradient flow."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.detach()
        z = (y ** 2).sum()
        z.backward()

        # x should have no gradient (detach broke the connection)
        assert np.allclose(x.grad, 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
