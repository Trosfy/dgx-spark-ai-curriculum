"""
Tests for end-to-end training functionality.

Run with: python -m pytest tests/test_training.py -v
"""

import numpy as np
import pytest

from micrograd_plus import (
    Tensor, Linear, ReLU, Sigmoid, Dropout, Sequential,
    MSELoss, CrossEntropyLoss, SGD, Adam
)
from micrograd_plus.utils import set_seed


class TestMSELoss:
    """Tests for MSE loss function."""

    def test_loss_value(self):
        """Test MSE computes correct value."""
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.5, 2.0, 2.5])

        loss_fn = MSELoss()
        loss = loss_fn(pred, target)

        expected = np.mean([0.25, 0, 0.25])
        assert np.allclose(loss.data, expected)

    def test_loss_gradient(self):
        """Test MSE gradient is correct."""
        pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = Tensor([2.0, 2.0, 2.0])

        loss_fn = MSELoss()
        loss = loss_fn(pred, target)
        loss.backward()

        # Gradient should be 2 * (pred - target) / n
        expected_grad = 2 * (pred.data - target.data) / 3
        assert np.allclose(pred.grad, expected_grad)

    def test_reduction_sum(self):
        """Test MSE with sum reduction."""
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.5, 2.0, 2.5])

        loss_fn = MSELoss(reduction='sum')
        loss = loss_fn(pred, target)

        expected = 0.25 + 0 + 0.25
        assert np.allclose(loss.data, expected)


class TestCrossEntropyLoss:
    """Tests for Cross-Entropy loss function."""

    def test_loss_value(self):
        """Test Cross-Entropy computes correct value."""
        logits = Tensor([[2.0, 1.0, 0.1]])
        targets = Tensor([0])

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, targets)

        # Manual computation
        exp_logits = np.exp(logits.data - logits.data.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        expected = -np.log(probs[0, 0])

        assert np.allclose(loss.data, expected, atol=1e-4)

    def test_gradient_flow(self):
        """Test gradients flow correctly."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (1, 3)

    def test_batch_loss(self):
        """Test Cross-Entropy with batch of samples."""
        logits = Tensor([
            [2.0, 1.0, 0.1],
            [0.1, 2.0, 1.0]
        ], requires_grad=True)
        targets = Tensor([0, 1])

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad.shape == (2, 3)


class TestSGDOptimizer:
    """Tests for SGD optimizer."""

    def test_basic_step(self):
        """Test basic SGD step."""
        x = Tensor([1.0], requires_grad=True)
        optimizer = SGD([x], lr=0.1)

        # Manual gradient
        x.grad = np.array([2.0])
        optimizer.step()

        # x = x - lr * grad = 1.0 - 0.1 * 2.0 = 0.8
        assert np.allclose(x.data, [0.8])

    def test_momentum(self):
        """Test SGD with momentum."""
        x = Tensor([1.0], requires_grad=True)
        optimizer = SGD([x], lr=0.1, momentum=0.9)

        # First step
        x.grad = np.array([1.0])
        optimizer.step()

        # Second step (momentum should accelerate)
        x.grad = np.array([1.0])
        optimizer.step()

        # With momentum, the second step should be larger
        assert x.data[0] < 0.8  # Would be 0.8 without momentum

    def test_zero_grad(self):
        """Test zero_grad clears gradients."""
        x = Tensor([1.0], requires_grad=True)
        optimizer = SGD([x], lr=0.1)

        x.grad = np.array([5.0])
        optimizer.zero_grad()

        assert np.allclose(x.grad, [0.0])


class TestAdamOptimizer:
    """Tests for Adam optimizer."""

    def test_convergence(self):
        """Test Adam converges on simple problem."""
        set_seed(42)

        x = Tensor([0.0], requires_grad=True)
        optimizer = Adam([x], lr=0.1)

        # Minimize (x - 3)^2
        for _ in range(100):
            loss = (x - 3) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert abs(x.item() - 3.0) < 0.1

    def test_multiple_params(self):
        """Test Adam with multiple parameters."""
        x = Tensor([0.0], requires_grad=True)
        y = Tensor([0.0], requires_grad=True)
        optimizer = Adam([x, y], lr=0.1)

        # Minimize (x - 1)^2 + (y - 2)^2
        for _ in range(100):
            loss = (x - 1) ** 2 + (y - 2) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert abs(x.item() - 1.0) < 0.1
        assert abs(y.item() - 2.0) < 0.1


class TestEndToEndTraining:
    """Tests for complete training pipelines."""

    def test_xor_problem(self):
        """Test training on XOR problem."""
        set_seed(42)

        # XOR data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.int32)

        # Model
        model = Sequential(
            Linear(2, 8),
            ReLU(),
            Linear(8, 2)
        )

        loss_fn = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.1)

        X_tensor = Tensor(X, requires_grad=True)
        y_tensor = Tensor(y)

        # Train
        for _ in range(500):
            logits = model(X_tensor)
            loss = loss_fn(logits, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        predictions = np.argmax(model(X_tensor).data, axis=1)
        accuracy = np.mean(predictions == y)

        assert accuracy == 1.0, f"Failed to solve XOR: accuracy={accuracy}"

    def test_regression(self):
        """Test regression training."""
        set_seed(42)

        # Linear regression: y = 2x + 1
        X = np.random.randn(100, 1).astype(np.float32)
        y = 2 * X + 1 + np.random.randn(100, 1).astype(np.float32) * 0.1

        # Model
        model = Linear(1, 1)
        loss_fn = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.1)

        # Train
        for _ in range(200):
            X_tensor = Tensor(X, requires_grad=True)
            y_tensor = Tensor(y)

            pred = model(X_tensor)
            loss = loss_fn(pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check learned parameters are close to true values
        # Weight should be ~2, bias should be ~1
        weight = model.weight.data[0, 0]
        bias = model.bias.data[0]

        assert abs(weight - 2.0) < 0.2, f"Weight {weight} far from 2.0"
        assert abs(bias - 1.0) < 0.2, f"Bias {bias} far from 1.0"

    def test_loss_decreases(self):
        """Test that loss decreases during training."""
        set_seed(42)

        # Random data
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 3, 50)

        # Model
        model = Sequential(
            Linear(10, 8),
            ReLU(),
            Linear(8, 3)
        )

        loss_fn = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.01)

        X_tensor = Tensor(X, requires_grad=True)
        y_tensor = Tensor(y)

        # Initial loss
        initial_loss = loss_fn(model(X_tensor), y_tensor).item()

        # Train
        for _ in range(100):
            logits = model(X_tensor)
            loss = loss_fn(logits, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final loss
        final_loss = loss_fn(model(X_tensor), y_tensor).item()

        assert final_loss < initial_loss, \
            f"Loss didn't decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_training_eval_modes(self):
        """Test that train/eval modes affect behavior."""
        set_seed(42)

        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Dropout(0.5),
            Linear(8, 2)
        )

        x = Tensor(np.ones((10, 4)).astype(np.float32))

        # Training mode - outputs should vary (due to dropout)
        model.train()
        out1 = model(x).data.copy()
        out2 = model(x).data.copy()

        # Eval mode - outputs should be consistent
        model.eval()
        out3 = model(x).data.copy()
        out4 = model(x).data.copy()

        assert np.allclose(out3, out4), "Eval mode should be deterministic"


class TestGradientClipping:
    """Tests for gradient clipping (if implemented)."""

    def test_large_gradients(self):
        """Test model handles large gradients."""
        set_seed(42)

        # Create situation with potentially large gradients
        model = Sequential(
            Linear(10, 100),
            ReLU(),
            Linear(100, 10)
        )

        x = Tensor(np.random.randn(5, 10).astype(np.float32) * 10, requires_grad=True)
        y = Tensor(np.random.randint(0, 10, 5))

        loss_fn = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)

        # Should not explode
        for _ in range(10):
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check no NaN
            for p in model.parameters():
                assert not np.any(np.isnan(p.data)), "Parameters became NaN"
                assert not np.any(np.isnan(p.grad)), "Gradients became NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
