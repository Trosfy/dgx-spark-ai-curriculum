"""
Unit tests for neural network layers.

Run with: python -m pytest tests/test_layers.py -v
"""

import numpy as np
import pytest

from micrograd_plus import Tensor, Linear, ReLU, Sigmoid, Softmax, Dropout, Sequential
from micrograd_plus.layers import BatchNorm, LayerNorm, Tanh, Embedding


class TestLinearLayer:
    """Tests for the Linear layer."""

    def test_output_shape(self):
        """Test output shape is correct."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(3, 10).astype(np.float32))
        y = layer(x)
        assert y.shape == (3, 5)

    def test_single_sample(self):
        """Test with single sample (no batch dimension)."""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(1, 10).astype(np.float32))
        y = layer(x)
        assert y.shape == (1, 5)

    def test_parameters(self):
        """Test parameter shapes and count."""
        layer = Linear(10, 5)
        params = layer.parameters()
        assert len(params) == 2  # weight and bias
        assert params[0].shape == (10, 5)  # weight
        assert params[1].shape == (5,)  # bias

    def test_no_bias(self):
        """Test layer without bias."""
        layer = Linear(10, 5, bias=False)
        params = layer.parameters()
        assert len(params) == 1  # only weight
        assert layer.bias is None

    def test_gradient_flow(self):
        """Test gradients flow through layer."""
        np.random.seed(42)
        layer = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None

        # Check gradient shapes
        assert layer.weight.grad.shape == (4, 3)
        assert layer.bias.grad.shape == (3,)
        assert x.grad.shape == (2, 4)

    def test_zero_grad(self):
        """Test zero_grad clears gradients."""
        layer = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))

        y = layer(x).sum()
        y.backward()

        assert not np.allclose(layer.weight.grad, 0)

        layer.zero_grad()
        assert np.allclose(layer.weight.grad, 0)
        assert np.allclose(layer.bias.grad, 0)


class TestActivationLayers:
    """Tests for activation function layers."""

    def test_relu_forward(self):
        """Test ReLU forward pass."""
        relu = ReLU()
        x = Tensor([-2, -1, 0, 1, 2])
        y = relu(x)
        assert np.allclose(y.data, [0, 0, 0, 1, 2])

    def test_relu_backward(self):
        """Test ReLU backward pass."""
        relu = ReLU()
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        y = relu(x).sum()
        y.backward()
        # Gradient is 1 where input > 0, else 0
        # Note: gradient at exactly 0 is technically undefined
        assert np.allclose(x.grad, [0, 0, 0, 1, 1])

    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        sigmoid = Sigmoid()
        x = Tensor([0])
        y = sigmoid(x)
        assert np.allclose(y.data, [0.5])

    def test_sigmoid_backward(self):
        """Test Sigmoid backward pass."""
        sigmoid = Sigmoid()
        x = Tensor([0], requires_grad=True)
        y = sigmoid(x).sum()
        y.backward()
        assert np.allclose(x.grad, [0.25])  # sigmoid(0) * (1 - sigmoid(0))

    def test_tanh_forward(self):
        """Test Tanh forward pass."""
        tanh = Tanh()
        x = Tensor([0])
        y = tanh(x)
        assert np.allclose(y.data, [0])

    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        softmax = Softmax()
        x = Tensor([[1, 2, 3]])
        y = softmax(x)
        assert np.allclose(y.data.sum(), 1.0)
        assert all(y.data[0] > 0)

    def test_softmax_backward(self):
        """Test Softmax backward pass."""
        softmax = Softmax()
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = softmax(x).sum()
        y.backward()
        # Sum of softmax is always 1, so gradient should be 0
        assert np.allclose(x.grad, 0, atol=1e-5)


class TestDropoutLayer:
    """Tests for Dropout layer."""

    def test_training_mode_drops(self):
        """Test Dropout drops values in training mode."""
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout.train()

        x = Tensor(np.ones((100, 100)))
        y = dropout(x)

        # Some values should be zero
        assert np.sum(y.data == 0) > 0

    def test_training_mode_scaling(self):
        """Test Dropout scales remaining values."""
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout.train()

        x = Tensor(np.ones((1000, 100)))
        y = dropout(x)

        # Non-zero values should be scaled by 1/(1-p) = 2
        non_zero = y.data[y.data != 0]
        assert np.allclose(non_zero, 2.0)

    def test_eval_mode_identity(self):
        """Test Dropout is identity in eval mode."""
        dropout = Dropout(p=0.5)
        dropout.eval()

        x = Tensor(np.ones((10, 10)))
        y = dropout(x)

        assert np.allclose(y.data, x.data)

    def test_gradient_flow(self):
        """Test gradients flow through dropout."""
        np.random.seed(42)
        dropout = Dropout(p=0.3)
        dropout.train()

        x = Tensor(np.ones((5, 5)), requires_grad=True)
        y = dropout(x).sum()
        y.backward()

        # Gradient should exist
        assert x.grad is not None
        # Gradient should have some zeros (where dropout occurred)
        assert np.sum(x.grad == 0) > 0


class TestSequential:
    """Tests for Sequential container."""

    def test_forward_pass(self):
        """Test forward pass through sequential."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        )

        x = Tensor(np.random.randn(3, 10).astype(np.float32))
        y = model(x)

        assert y.shape == (3, 2)

    def test_parameter_collection(self):
        """Test all parameters are collected."""
        model = Sequential(
            Linear(10, 5),  # 10*5 + 5 = 55 params
            ReLU(),  # 0 params
            Linear(5, 2)  # 5*2 + 2 = 12 params
        )

        params = model.parameters()
        total = sum(p.size for p in params)

        assert total == 67

    def test_train_eval_modes(self):
        """Test train/eval mode propagation."""
        model = Sequential(
            Linear(10, 5),
            Dropout(0.5),
            Linear(5, 2)
        )

        model.train()
        assert model.layers[1].training == True

        model.eval()
        assert model.layers[1].training == False

    def test_gradient_flow(self):
        """Test gradients flow through all layers."""
        model = Sequential(
            Linear(4, 3),
            ReLU(),
            Linear(3, 2)
        )

        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        y = model(x).sum()
        y.backward()

        # Check gradients exist for all linear layers
        for layer in model.layers:
            if hasattr(layer, 'weight'):
                assert layer.weight.grad is not None

    def test_indexing(self):
        """Test layer indexing."""
        linear1 = Linear(10, 5)
        relu = ReLU()
        linear2 = Linear(5, 2)

        model = Sequential(linear1, relu, linear2)

        assert model[0] is linear1
        assert model[1] is relu
        assert model[2] is linear2

    def test_len(self):
        """Test len returns number of layers."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        )
        assert len(model) == 3


class TestBatchNorm:
    """Tests for BatchNorm layer."""

    def test_output_shape(self):
        """Test output shape matches input."""
        bn = BatchNorm(64)
        x = Tensor(np.random.randn(32, 64).astype(np.float32))
        y = bn(x)
        assert y.shape == x.shape

    def test_training_normalization(self):
        """Test normalization in training mode."""
        bn = BatchNorm(10)
        bn.train()

        x = Tensor(np.random.randn(100, 10).astype(np.float32) * 10 + 5)
        y = bn(x)

        # Output should have approximately zero mean and unit variance
        assert np.abs(y.data.mean()) < 0.5
        assert np.abs(y.data.std() - 1.0) < 0.5

    def test_parameters(self):
        """Test gamma and beta parameters."""
        bn = BatchNorm(10)
        params = bn.parameters()
        assert len(params) == 2  # gamma and beta


class TestEmbedding:
    """Tests for Embedding layer."""

    def test_output_shape(self):
        """Test output shape."""
        emb = Embedding(100, 32)  # vocab=100, dim=32
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))  # (2, 3)
        y = emb(x)
        assert y.shape == (2, 3, 32)

    def test_lookup(self):
        """Test embedding lookup."""
        emb = Embedding(10, 4)

        # Set known values
        emb.weight.data[5] = [1, 2, 3, 4]

        x = Tensor(np.array([5]))
        y = emb(x)

        assert np.allclose(y.data[0], [1, 2, 3, 4])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
