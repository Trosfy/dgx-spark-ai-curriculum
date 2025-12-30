"""
Unit tests for the Tensor class.

Run with: python -m pytest tests/test_tensor.py -v
"""

import numpy as np
import pytest

from micrograd_plus import Tensor
from micrograd_plus.utils import numerical_gradient


class TestTensorCreation:
    """Tests for tensor creation and basic properties."""

    def test_from_list(self):
        """Test tensor creation from Python list."""
        t = Tensor([1, 2, 3])
        assert t.shape == (3,)
        assert np.allclose(t.data, [1, 2, 3])

    def test_from_nested_list(self):
        """Test tensor creation from nested list."""
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)

    def test_from_numpy(self):
        """Test tensor creation from numpy array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        t = Tensor(arr)
        assert t.shape == (2, 3)
        assert np.allclose(t.data, arr)

    def test_from_scalar(self):
        """Test tensor creation from scalar."""
        t = Tensor(5.0)
        assert t.shape == ()
        assert t.item() == 5.0

    def test_requires_grad(self):
        """Test requires_grad flag."""
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([1, 2, 3], requires_grad=False)

        assert t1.requires_grad == True
        assert t1.grad is not None
        assert t2.requires_grad == False
        assert t2.grad is None

    def test_dtype(self):
        """Test default dtype is float32."""
        t = Tensor([1, 2, 3])
        assert t.dtype == np.float32


class TestTensorOperations:
    """Tests for tensor arithmetic operations."""

    def test_addition(self):
        """Test element-wise addition."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        assert np.allclose(c.data, [5, 7, 9])

    def test_addition_scalar(self):
        """Test addition with scalar."""
        a = Tensor([1, 2, 3])
        c = a + 5
        assert np.allclose(c.data, [6, 7, 8])

    def test_subtraction(self):
        """Test element-wise subtraction."""
        a = Tensor([5, 6, 7])
        b = Tensor([1, 2, 3])
        c = a - b
        assert np.allclose(c.data, [4, 4, 4])

    def test_multiplication(self):
        """Test element-wise multiplication."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a * b
        assert np.allclose(c.data, [4, 10, 18])

    def test_division(self):
        """Test element-wise division."""
        a = Tensor([4, 6, 8])
        b = Tensor([2, 2, 2])
        c = a / b
        assert np.allclose(c.data, [2, 3, 4])

    def test_power(self):
        """Test power operation."""
        a = Tensor([1, 2, 3])
        c = a ** 2
        assert np.allclose(c.data, [1, 4, 9])

    def test_negation(self):
        """Test negation."""
        a = Tensor([1, -2, 3])
        c = -a
        assert np.allclose(c.data, [-1, 2, -3])

    def test_matmul(self):
        """Test matrix multiplication."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a @ b
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(c.data, expected)

    def test_broadcasting(self):
        """Test broadcasting in operations."""
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([10, 20, 30])
        c = a + b
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        assert np.allclose(c.data, expected)


class TestTensorReductions:
    """Tests for reduction operations."""

    def test_sum_all(self):
        """Test sum over all elements."""
        t = Tensor([[1, 2], [3, 4]])
        assert np.allclose(t.sum().data, 10)

    def test_sum_axis(self):
        """Test sum over specific axis."""
        t = Tensor([[1, 2], [3, 4]])
        assert np.allclose(t.sum(axis=0).data, [4, 6])
        assert np.allclose(t.sum(axis=1).data, [3, 7])

    def test_mean_all(self):
        """Test mean over all elements."""
        t = Tensor([[1, 2], [3, 4]])
        assert np.allclose(t.mean().data, 2.5)

    def test_max(self):
        """Test max operation."""
        t = Tensor([[1, 5], [3, 2]])
        assert np.allclose(t.max().data, 5)
        assert np.allclose(t.max(axis=0).data, [3, 5])


class TestTensorShapeOps:
    """Tests for shape operations."""

    def test_reshape(self):
        """Test reshape operation."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        r = t.reshape(3, 2)
        assert r.shape == (3, 2)

    def test_transpose(self):
        """Test transpose operation."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        r = t.T
        assert r.shape == (3, 2)
        assert np.allclose(r.data, [[1, 4], [2, 5], [3, 6]])

    def test_flatten(self):
        """Test flatten operation."""
        t = Tensor([[1, 2], [3, 4]])
        r = t.flatten()
        assert r.shape == (4,)


class TestTensorActivations:
    """Tests for activation functions."""

    def test_relu(self):
        """Test ReLU activation."""
        t = Tensor([-2, -1, 0, 1, 2])
        r = t.relu()
        assert np.allclose(r.data, [0, 0, 0, 1, 2])

    def test_sigmoid(self):
        """Test sigmoid activation."""
        t = Tensor([0])
        r = t.sigmoid()
        assert np.allclose(r.data, [0.5])

    def test_tanh(self):
        """Test tanh activation."""
        t = Tensor([0])
        r = t.tanh()
        assert np.allclose(r.data, [0])

    def test_softmax(self):
        """Test softmax activation."""
        t = Tensor([[1, 2, 3]])
        r = t.softmax()
        assert np.allclose(r.data.sum(), 1.0)
        assert all(r.data[0] > 0)


class TestTensorGradients:
    """Tests for automatic differentiation."""

    def test_addition_gradient(self):
        """Test gradient of addition."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        assert np.allclose(a.grad, [1, 1, 1])
        assert np.allclose(b.grad, [1, 1, 1])

    def test_multiplication_gradient(self):
        """Test gradient of multiplication."""
        a = Tensor([2, 3], requires_grad=True)
        b = Tensor([4, 5], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        assert np.allclose(a.grad, b.data)
        assert np.allclose(b.grad, a.data)

    def test_power_gradient(self):
        """Test gradient of power."""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        assert np.allclose(x.grad, [2, 4, 6])  # dy/dx = 2x

    def test_matmul_gradient(self):
        """Test gradient of matrix multiplication."""
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
        c = (a @ b).sum()
        c.backward()

        # Verify against numerical gradient
        def f_a(arr):
            return (Tensor(arr) @ b).sum().data.item()
        num_grad = numerical_gradient(f_a, a.data.copy())
        assert np.allclose(a.grad, num_grad, atol=1e-4)

    def test_relu_gradient(self):
        """Test gradient of ReLU."""
        x = Tensor([-2, -1, 1, 2], requires_grad=True)
        y = x.relu().sum()
        y.backward()
        assert np.allclose(x.grad, [0, 0, 1, 1])

    def test_sigmoid_gradient(self):
        """Test gradient of sigmoid."""
        x = Tensor([0], requires_grad=True)
        y = x.sigmoid().sum()
        y.backward()
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert np.allclose(x.grad, [0.25])

    def test_chain_rule(self):
        """Test chain rule with multiple operations."""
        x = Tensor([2], requires_grad=True)
        y = ((x * 3) + 2) ** 2  # y = (3x + 2)^2
        y.backward()
        # dy/dx = 2 * (3x + 2) * 3 = 6 * (3*2 + 2) = 6 * 8 = 48
        assert np.allclose(x.grad, [48])

    def test_gradient_accumulation(self):
        """Test that gradients accumulate when variable used multiple times."""
        x = Tensor([2], requires_grad=True)
        y = x + x  # y = 2x
        z = y.sum()
        z.backward()
        assert np.allclose(x.grad, [2])  # dz/dx = 2


class TestTensorUtilities:
    """Tests for utility methods."""

    def test_zero_grad(self):
        """Test gradient zeroing."""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        assert not np.allclose(x.grad, [0, 0, 0])

        x.zero_grad()
        assert np.allclose(x.grad, [0, 0, 0])

    def test_detach(self):
        """Test detaching from computation graph."""
        x = Tensor([1, 2, 3], requires_grad=True)
        y = x.detach()
        assert y.requires_grad == False
        assert np.allclose(y.data, x.data)

    def test_item(self):
        """Test extracting scalar value."""
        x = Tensor([5.0])
        assert x.item() == 5.0

    def test_numpy(self):
        """Test conversion to numpy."""
        x = Tensor([1, 2, 3])
        arr = x.numpy()
        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, [1, 2, 3])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
