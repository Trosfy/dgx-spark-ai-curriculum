"""
Tensor Module - The Core of MicroGrad+.

This module implements a Tensor class with automatic differentiation capabilities.
It supports basic operations needed for building and training neural networks.

The autograd engine uses reverse-mode automatic differentiation, which computes
gradients by traversing the computational graph backwards from the output.

ELI5 Explanation:
    Imagine you're baking a cake (the output) using multiple ingredients (inputs).
    If the cake tastes bad, you want to know which ingredient caused the problem.
    Autograd is like a magical recipe book that can trace back through every step
    and tell you exactly how much each ingredient contributed to the final taste.
    That's what gradients tell us - how much each input affects the output!

Example:
    >>> a = Tensor([2.0, 3.0], requires_grad=True)
    >>> b = Tensor([4.0, 5.0], requires_grad=True)
    >>> c = a * b + a
    >>> c.sum().backward()
    >>> print(a.grad)  # [5. 6.] because dc/da = b + 1
    >>> print(b.grad)  # [2. 3.] because dc/db = a
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Union, Callable, Set
import numpy as np


class Tensor:
    """
    A multi-dimensional array with automatic differentiation support.

    This class wraps a numpy array and tracks operations performed on it,
    enabling automatic computation of gradients through backpropagation.

    Attributes:
        data (np.ndarray): The actual numerical data stored as numpy array.
        grad (np.ndarray | None): Gradient of the loss with respect to this tensor.
        requires_grad (bool): Whether to compute gradients for this tensor.
        shape (tuple): Shape of the tensor.
        dtype: Data type of the tensor elements.

    Example:
        >>> x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> y = x * 2
        >>> z = y.sum()
        >>> z.backward()
        >>> print(x.grad)  # All 2's because dz/dx = 2
    """

    def __init__(
        self,
        data: Union[np.ndarray, List, float, int],
        requires_grad: bool = False,
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
        dtype: np.dtype = np.float32
    ):
        """
        Initialize a new Tensor.

        Args:
            data: The numerical data (list, numpy array, or scalar).
            requires_grad: If True, gradients will be computed for this tensor.
            _children: Tuple of parent tensors (used internally for autograd).
            _op: String describing the operation that created this tensor.
            dtype: Data type for the tensor (default: float32).

        Example:
            >>> t1 = Tensor([1, 2, 3])  # From list
            >>> t2 = Tensor(np.array([[1, 2], [3, 4]]))  # From numpy
            >>> t3 = Tensor(5.0, requires_grad=True)  # Scalar with grad
        """
        # Convert to numpy array if needed
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=dtype)

        # Gradient storage
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        if requires_grad:
            self.grad = np.zeros_like(self.data, dtype=dtype)

        # Autograd graph tracking
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return self.data.dtype

    @property
    def T(self) -> 'Tensor':
        """Return the transpose of the tensor."""
        return self.transpose()

    # ==================== Utility Methods ====================

    def numpy(self) -> np.ndarray:
        """Convert tensor to numpy array."""
        return self.data.copy()

    def item(self) -> float:
        """Return the scalar value for single-element tensors."""
        if self.size != 1:
            raise ValueError(f"Can only convert tensors with 1 element to scalar, got {self.size}")
        return float(self.data.flat[0])

    def zero_grad(self) -> None:
        """Reset gradient to zeros."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def detach(self) -> 'Tensor':
        """Return a new tensor detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self) -> 'Tensor':
        """Return a copy of the tensor that shares the computation graph."""
        out = Tensor(
            self.data.copy(),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='clone'
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
        out._backward = _backward

        return out

    # ==================== Arithmetic Operations ====================

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise addition: self + other."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='+'
        )

        def _backward():
            if self.requires_grad:
                # Handle broadcasting
                grad = out.grad
                if self.shape != out.shape:
                    grad = _unbroadcast(grad, self.shape)
                self.grad += grad
            if other.requires_grad:
                grad = out.grad
                if other.shape != out.shape:
                    grad = _unbroadcast(grad, other.shape)
                other.grad += grad
        out._backward = _backward

        return out

    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        """Right addition: other + self."""
        return self + other

    def __neg__(self) -> 'Tensor':
        """Negation: -self."""
        return self * -1

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise subtraction: self - other."""
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> 'Tensor':
        """Right subtraction: other - self."""
        return (-self) + other

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise multiplication: self * other."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='*'
        )

        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                if self.shape != out.shape:
                    grad = _unbroadcast(grad, self.shape)
                self.grad += grad
            if other.requires_grad:
                grad = self.data * out.grad
                if other.shape != out.shape:
                    grad = _unbroadcast(grad, other.shape)
                other.grad += grad
        out._backward = _backward

        return out

    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        """Right multiplication: other * self."""
        return self * other

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise division: self / other."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='/'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad / other.data
                if self.shape != out.shape:
                    grad = _unbroadcast(grad, self.shape)
                self.grad += grad
            if other.requires_grad:
                grad = -self.data / (other.data ** 2) * out.grad
                if other.shape != out.shape:
                    grad = _unbroadcast(grad, other.shape)
                other.grad += grad
        out._backward = _backward

        return out

    def __rtruediv__(self, other: Union[float, int]) -> 'Tensor':
        """Right division: other / self."""
        return Tensor(other) / self

    def __pow__(self, power: Union[float, int]) -> 'Tensor':
        """Power operation: self ** power."""
        assert isinstance(power, (int, float)), "Power must be a scalar"
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'**{power}'
        )

        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication: self @ other."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='@'
        )

        def _backward():
            if self.requires_grad:
                # Gradient for self: out.grad @ other.T
                if self.ndim == 1 and other.ndim == 2:
                    self.grad += out.grad @ other.data.T
                elif self.ndim == 2 and other.ndim == 1:
                    self.grad += np.outer(out.grad, other.data)
                else:
                    self.grad += out.grad @ other.data.T
            if other.requires_grad:
                # Gradient for other: self.T @ out.grad
                if self.ndim == 1 and other.ndim == 2:
                    other.grad += np.outer(self.data, out.grad)
                elif self.ndim == 2 and other.ndim == 1:
                    other.grad += self.data.T @ out.grad
                else:
                    other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
        """Right matrix multiplication: other @ self."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other @ self

    # ==================== Comparison Operations ====================

    def __eq__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Element-wise equality comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data == other_data

    def __ne__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Element-wise inequality comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data != other_data

    def __lt__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Element-wise less than comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data < other_data

    def __le__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Element-wise less than or equal comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data <= other_data

    def __gt__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Element-wise greater than comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data > other_data

    def __ge__(self, other: Union['Tensor', float, int]) -> np.ndarray:
        """Element-wise greater than or equal comparison."""
        other_data = other.data if isinstance(other, Tensor) else other
        return self.data >= other_data

    # ==================== Reduction Operations ====================

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Sum of elements over given axis.

        Args:
            axis: Axis or axes along which to sum. None sums all elements.
            keepdims: Whether to keep the summed dimensions.

        Returns:
            Tensor with summed values.

        Example:
            >>> t = Tensor([[1, 2], [3, 4]])
            >>> t.sum()  # Tensor(10.)
            >>> t.sum(axis=0)  # Tensor([4., 6.])
            >>> t.sum(axis=1)  # Tensor([3., 7.])
        """
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    # Expand dimensions that were summed
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, ax)
                # Broadcast to original shape
                self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward

        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """
        Mean of elements over given axis.

        Args:
            axis: Axis or axes along which to compute mean. None computes over all elements.
            keepdims: Whether to keep the reduced dimensions.

        Returns:
            Tensor with mean values.
        """
        n = self.data.size if axis is None else np.prod([self.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis=axis, keepdims=keepdims) / n

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """
        Maximum of elements over given axis.

        Args:
            axis: Axis along which to find maximum. None finds global maximum.
            keepdims: Whether to keep the reduced dimensions.

        Returns:
            Tensor with maximum values.
        """
        out = Tensor(
            np.max(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='max'
        )

        def _backward():
            if self.requires_grad:
                if axis is None:
                    # Global max
                    mask = (self.data == self.data.max()).astype(np.float32)
                    self.grad += mask * out.grad / mask.sum()
                else:
                    # Max along axis
                    expanded_out = np.expand_dims(out.data, axis) if not keepdims else out.data
                    mask = (self.data == expanded_out).astype(np.float32)
                    mask_sum = mask.sum(axis=axis, keepdims=True)
                    expanded_grad = np.expand_dims(out.grad, axis) if not keepdims else out.grad
                    self.grad += mask * expanded_grad / mask_sum
        out._backward = _backward

        return out

    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Minimum of elements over given axis."""
        return (-self).max(axis=axis, keepdims=keepdims) * -1

    # ==================== Shape Operations ====================

    def reshape(self, *shape: int) -> 'Tensor':
        """
        Reshape the tensor to a new shape.

        Args:
            *shape: The new shape dimensions.

        Returns:
            Reshaped tensor.

        Example:
            >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
            >>> t.reshape(3, 2)  # Shape: (3, 2)
            >>> t.reshape(-1)  # Shape: (6,)
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])

        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='reshape'
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    def flatten(self) -> 'Tensor':
        """Flatten the tensor to 1D."""
        return self.reshape(-1)

    def transpose(self, *axes: int) -> 'Tensor':
        """
        Transpose the tensor.

        Args:
            *axes: Permutation of axes. If not provided, reverses axes.

        Returns:
            Transposed tensor.
        """
        if not axes:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])

        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='T'
        )

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += np.transpose(out.grad)
                else:
                    # Inverse permutation
                    inv_axes = [0] * len(axes)
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    self.grad += np.transpose(out.grad, inv_axes)
        out._backward = _backward

        return out

    def unsqueeze(self, axis: int) -> 'Tensor':
        """Add a dimension at the specified axis."""
        out = Tensor(
            np.expand_dims(self.data, axis),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='unsqueeze'
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.squeeze(out.grad, axis)
        out._backward = _backward

        return out

    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        out = Tensor(
            np.squeeze(self.data, axis),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='squeeze'
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    # ==================== Activation Functions ====================

    def relu(self) -> 'Tensor':
        """
        Rectified Linear Unit activation.

        Returns max(0, x) element-wise.

        Example:
            >>> t = Tensor([-1, 0, 1, 2])
            >>> t.relu()  # Tensor([0., 0., 1., 2.])
        """
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='relu'
        )

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self) -> 'Tensor':
        """
        Sigmoid activation: 1 / (1 + exp(-x)).

        Example:
            >>> t = Tensor([0, 1, -1])
            >>> t.sigmoid()  # Tensor([0.5, 0.731, 0.269])
        """
        sig = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))  # Clip to avoid overflow
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sigmoid'
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self) -> 'Tensor':
        """
        Hyperbolic tangent activation.

        Example:
            >>> t = Tensor([0, 1, -1])
            >>> t.tanh()  # Tensor([0., 0.762, -0.762])
        """
        out = Tensor(
            np.tanh(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='tanh'
        )

        def _backward():
            if self.requires_grad:
                self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out

    def softmax(self, axis: int = -1) -> 'Tensor':
        """
        Softmax activation: exp(x) / sum(exp(x)).

        Args:
            axis: Axis along which to compute softmax.

        Returns:
            Tensor with softmax probabilities (sum to 1 along axis).

        Example:
            >>> t = Tensor([[1, 2, 3]])
            >>> t.softmax()  # Tensor([[0.09, 0.24, 0.67]])
        """
        # Numerically stable softmax
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        softmax_out = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

        out = Tensor(
            softmax_out,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='softmax'
        )

        def _backward():
            if self.requires_grad:
                # Softmax gradient: s * (grad - sum(s * grad))
                s = out.data
                sum_sg = np.sum(s * out.grad, axis=axis, keepdims=True)
                self.grad += s * (out.grad - sum_sg)
        out._backward = _backward

        return out

    def log_softmax(self, axis: int = -1) -> 'Tensor':
        """
        Log-softmax: log(softmax(x)) computed in a numerically stable way.

        Args:
            axis: Axis along which to compute log-softmax.

        Returns:
            Tensor with log-softmax values.
        """
        # log_softmax = x - max(x) - log(sum(exp(x - max(x))))
        max_val = np.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - max_val
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

        out = Tensor(
            shifted - log_sum_exp,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='log_softmax'
        )

        def _backward():
            if self.requires_grad:
                softmax = np.exp(out.data)
                self.grad += out.grad - softmax * np.sum(out.grad, axis=axis, keepdims=True)
        out._backward = _backward

        return out

    def exp(self) -> 'Tensor':
        """Exponential function: e^x."""
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='exp'
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self) -> 'Tensor':
        """Natural logarithm."""
        out = Tensor(
            np.log(self.data + 1e-8),  # Small epsilon for numerical stability
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='log'
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / (self.data + 1e-8)
        out._backward = _backward

        return out

    def sqrt(self) -> 'Tensor':
        """Square root."""
        return self ** 0.5

    def abs(self) -> 'Tensor':
        """Absolute value."""
        out = Tensor(
            np.abs(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='abs'
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad
        out._backward = _backward

        return out

    # ==================== Backward Pass ====================

    def backward(self, gradient: Optional[np.ndarray] = None) -> None:
        """
        Compute gradients via reverse-mode automatic differentiation.

        This method computes gradients for all tensors in the computation graph
        that have requires_grad=True, using the chain rule.

        Args:
            gradient: Optional initial gradient. If None, uses ones for scalar
                      outputs or requires explicit gradient for non-scalars.

        Raises:
            RuntimeError: If called on non-scalar without providing gradient.

        Example:
            >>> x = Tensor([1., 2., 3.], requires_grad=True)
            >>> y = (x ** 2).sum()
            >>> y.backward()
            >>> print(x.grad)  # [2., 4., 6.] because dy/dx = 2x
        """
        if gradient is None:
            if self.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError(
                    "Gradient must be specified for non-scalar outputs. "
                    f"Output shape: {self.shape}. "
                    "Use .sum() or .mean() to reduce to a scalar first, or provide a gradient tensor."
                )

        # Set the gradient for this tensor
        self.grad = gradient.astype(np.float32)

        # Build topological ordering
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backpropagate in reverse topological order
        for v in reversed(topo):
            v._backward()

    # ==================== String Representations ====================

    def __repr__(self) -> str:
        """String representation for debugging."""
        grad_info = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_info})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return str(self.data)

    def __len__(self) -> int:
        """Return length of first dimension."""
        return len(self.data)

    def __getitem__(self, idx) -> 'Tensor':
        """Index into the tensor."""
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='getitem'
        )

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad += grad
        out._backward = _backward

        return out

    def __setitem__(self, idx, value) -> None:
        """Set values at index (no gradient tracking)."""
        if isinstance(value, Tensor):
            value = value.data
        self.data[idx] = value


# ==================== Helper Functions ====================

def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reverse broadcasting by summing along broadcasted dimensions.

    When we have y = x + broadcasted_tensor, the gradient needs to be
    summed along the dimensions that were broadcasted.
    """
    # Sum along dimensions that were added (leading dimensions)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Sum along dimensions that were broadcasted (size 1 -> size n)
    for i, (grad_dim, shape_dim) in enumerate(zip(grad.shape, shape)):
        if shape_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# ==================== Tensor Creation Functions ====================

def zeros(*shape: int, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with zeros."""
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with ones."""
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape: int, requires_grad: bool = False) -> Tensor:
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def rand(*shape: int, requires_grad: bool = False) -> Tensor:
    """Create a tensor with random uniform values in [0, 1)."""
    return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)


def arange(start: int, stop: Optional[int] = None, step: int = 1, requires_grad: bool = False) -> Tensor:
    """Create a tensor with evenly spaced values."""
    if stop is None:
        stop = start
        start = 0
    return Tensor(np.arange(start, stop, step).astype(np.float32), requires_grad=requires_grad)


def eye(n: int, requires_grad: bool = False) -> Tensor:
    """Create an identity matrix."""
    return Tensor(np.eye(n).astype(np.float32), requires_grad=requires_grad)


def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Create a tensor from a numpy array."""
    return Tensor(arr, requires_grad=requires_grad)
