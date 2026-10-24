"""Small CPU reverse-mode autodiff engine, implemented independently for TraceGrad.

NumPy provides array arithmetic and BLAS. This module owns the computation graph,
local vector-Jacobian products (VJPs), traversal, accumulation and graph lifetime.
Only first-order differentiation and 2-D matrix multiplication are supported.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import numbers
from typing import Callable

import numpy as np


_GRAD_ENABLED: ContextVar[bool] = ContextVar("tracegrad_grad_enabled", default=True)
_FLOAT_DTYPES = (np.dtype("float32"), np.dtype("float64"))


@contextmanager
def no_grad():
    """Disable graph construction in this context, typically during evaluation."""
    token = _GRAD_ENABLED.set(False)
    try:
        yield
    finally:
        _GRAD_ENABLED.reset(token)


def _array(data, dtype=None) -> np.ndarray:
    raw = np.asarray(data)
    if dtype is None:
        if raw.dtype.kind in "iub":
            dtype = np.float64
        elif raw.dtype in _FLOAT_DTYPES:
            dtype = raw.dtype
        else:
            raise TypeError("Tensor supports only float32/float64 or real integer inputs")
    if np.dtype(dtype) not in _FLOAT_DTYPES or raw.dtype.kind not in "fiub":
        raise TypeError("Tensor dtype must be float32 or float64 and input must be real")
    result = np.array(raw, dtype=dtype, copy=True)
    _validate(result)
    result.flags.writeable = False
    return result


def _validate(array: np.ndarray):
    if array.size == 0:
        raise ValueError("empty tensors are outside the supported shape contract")
    if array.dtype not in _FLOAT_DTYPES:
        raise TypeError("operation produced an unsupported dtype")
    if not np.isfinite(array).all():
        raise ValueError("Tensor values and operation results must be finite")


def _finite_mean(values):
    """Mean with a rescaled fallback when summing finite entries overflows."""
    with np.errstate(over='ignore', invalid='ignore'):
        value = np.mean(values)
    if not np.isfinite(value) and np.isfinite(values).all():
        scale = np.max(np.abs(values))
        if scale > 0:
            value = np.mean(values / scale) * scale
    if not np.isfinite(value):
        raise ValueError('mean result must be finite')
    return value


def _axes(axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    items = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    normalized = []
    for value in items:
        if not isinstance(value, (int, np.integer)):
            raise TypeError("reduction axes must be integers")
        value = int(value)
        if value < -ndim or value >= ndim:
            raise ValueError("reduction axis out of range")
        value %= ndim
        if value in normalized:
            raise ValueError("duplicate reduction axes are not supported")
        normalized.append(value)
    return tuple(sorted(normalized))


def _unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Adjoint of broadcasting: sum added axes and axes expanded from size one.

    Input gradient has the broadcast output shape; output has exactly ``shape``.
    Work is O(number of gradient entries), and the result is an ndarray.
    """
    gradient = np.asarray(gradient)
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    axes = tuple(i for i, (old, new) in enumerate(zip(shape, gradient.shape))
                 if old == 1 and new != 1)
    if axes:
        gradient = gradient.sum(axis=axes, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    """Owned finite float array plus an optional first-order computation graph.

    Public ``data`` is read-only. Leaf ``grad`` is an ndarray or None; intermediate
    gradients are temporary during backward. Integer construction defaults to
    float64. A floating ndarray retains float32/float64 unless dtype is explicit.
    """

    __array_priority__ = 1000

    def __init__(self, data, requires_grad: bool = False, dtype=None, name=None):
        self._data = _array(data, dtype)
        self.requires_grad = bool(requires_grad)
        self.name = name
        self.grad: np.ndarray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._versions: tuple[int, ...] = ()
        self._vjp: Callable | None = None
        self._saved_arrays: tuple[np.ndarray, ...] = ()
        self._version = 0
        self._is_leaf = True
        self._freed = False
        self._op_name = "leaf"

    @classmethod
    def _operation(cls, data, parents, vjp, name, saved_arrays=()):
        array = np.asarray(data)
        _validate(array)
        array.flags.writeable = False
        result = cls.__new__(cls)
        result._data = array
        result.requires_grad = _GRAD_ENABLED.get() and any(p.requires_grad for p in parents)
        result.name = None
        result.grad = None
        result._version = 0
        result._freed = False
        result._op_name = name
        result._is_leaf = not result.requires_grad
        if result.requires_grad:
            for parent in parents:
                if parent.requires_grad and parent._freed:
                    raise RuntimeError("cannot reuse a released graph; use detach or retain_graph=True")
            result._parents = tuple(parents)
            result._versions = tuple(p._version for p in parents)
            result._vjp = vjp
            result._saved_arrays = tuple(saved_arrays)
        else:
            result._parents = ()
            result._versions = ()
            result._vjp = None
            result._saved_arrays = ()
        return result

    @property
    def data(self):
        view = self._data.view()
        view.flags.writeable = False
        return view

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def T(self):
        return self.transpose()

    def item(self):
        return self._data.item()

    def numpy(self):
        """Return an independent writable copy, never a mutable saved operand."""
        return self._data.copy()

    def detach(self):
        return Tensor(self._data, name=self.name)

    def zero_grad(self):
        self.grad = None

    def _assign(self, data):
        """Replace a leaf's values for an optimizer; O(size), exact shape required.

        The version increments so backward can reject stale forward operands.
        This internal update does not expose a general in-place operation API.
        """
        if not self._is_leaf:
            raise RuntimeError("only leaf tensors may be assigned")
        new = _array(data, self.dtype)
        if new.shape != self.shape:
            raise ValueError("assignment must preserve tensor shape")
        self._data = new
        self._version += 1

    def _coerce(self, other):
        if isinstance(other, Tensor):
            return other
        # A Python scalar is a constant in the receiving tensor's dtype.
        return Tensor(other, dtype=self.dtype if np.isscalar(other) else None)

    def __repr__(self):
        return f"Tensor({self._data!r}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = self._coerce(other)
        return Tensor._operation(self._data + other._data, (self, other),
            lambda g: (_unbroadcast(g, self.shape), _unbroadcast(g, other.shape)), "add")

    __radd__ = __add__

    def __neg__(self):
        return Tensor._operation(-self._data, (self,), lambda g: (-g,), "neg")

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        return Tensor._operation(self._data * other._data, (self, other),
            lambda g: (_unbroadcast(g * other._data, self.shape),
                       _unbroadcast(g * self._data, other.shape)), "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        if np.any(other._data == 0):
            raise ValueError("division by zero")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            data = self._data / other._data
        return Tensor._operation(data, (self, other),
            lambda g: (_unbroadcast(g / other._data, self.shape),
                       _unbroadcast(-g * data / other._data, other.shape)), "div", (data,))

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, power):
        if not isinstance(power, numbers.Real) or not np.isfinite(power):
            raise TypeError("power must be a finite real scalar")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            data = self._data ** power
        def vjp(g):
            if power == 0:
                return (np.zeros_like(self._data),)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return (g * power * self._data ** (power - 1),)
        return Tensor._operation(data, (self,), vjp, "pow")

    def __matmul__(self, other):
        """2-D matrix product; forward O(mkn), VJP uses g@B.T and A.T@g."""
        other = self._coerce(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul currently supports exactly two 2-D matrices")
        if self.shape[1] != other.shape[0]:
            raise ValueError("matmul inner dimensions do not match")
        def vjp(g):
            # Inputs such as images usually require no gradient. Avoid a full
            # matrix product whose result backward would immediately discard.
            return (g @ other._data.T if self.requires_grad else None,
                    self._data.T @ g if other.requires_grad else None)
        return Tensor._operation(self._data @ other._data, (self, other), vjp, "matmul")

    def __rmatmul__(self, other):
        return self._coerce(other) @ self

    def sum(self, axis=None, keepdims=False):
        """Sum selected axes; backward broadcasts the seed, O(number of entries)."""
        axes = _axes(axis, self.ndim)
        data = self._data.sum(axis=axes, keepdims=keepdims)
        def vjp(g):
            if not keepdims:
                for ax in axes:
                    g = np.expand_dims(g, ax)
            return (np.broadcast_to(g, self.shape),)
        return Tensor._operation(data, (self,), vjp, "sum")

    def mean(self, axis=None, keepdims=False):
        axes = _axes(axis, self.ndim)
        count = int(np.prod([self.shape[a] for a in axes], dtype=np.int64))
        return self.sum(axis=axes, keepdims=keepdims) / count

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        data = self._data.reshape(*shape) if shape else self._data.reshape(())
        return Tensor._operation(data, (self,), lambda g: (g.reshape(self.shape),), "reshape")

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes or axes == (None,):
            axes = tuple(reversed(range(self.ndim)))
        if len(axes) != self.ndim or set(axes) != set(range(self.ndim)):
            raise ValueError("transpose requires a permutation of nonnegative axes")
        inverse = np.argsort(axes)
        data = self._data.transpose(axes)
        return Tensor._operation(data, (self,), lambda g: (g.transpose(inverse),), "transpose")

    def relu(self):
        mask = self._data > 0
        return Tensor._operation(np.maximum(self._data, 0), (self,),
                                 lambda g: (g * mask,), "relu", (mask,))

    def exp(self):
        with np.errstate(over="ignore", invalid="ignore"):
            data = np.exp(self._data)
        return Tensor._operation(data, (self,), lambda g: (g * data,), "exp", (data,))

    def log(self):
        if np.any(self._data <= 0):
            raise ValueError("log requires strictly positive inputs")
        return Tensor._operation(np.log(self._data), (self,),
                                 lambda g: (g / self._data,), "log")

    def logsumexp(self, axis=None, keepdims=False):
        """Stable log(sum(exp(x))) and softmax VJP; O(size) time and saved space."""
        axes = _axes(axis, self.ndim)
        maxima = self._data.max(axis=axes, keepdims=True)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            exponential = np.exp(self._data - maxima)
        denominator = exponential.sum(axis=axes, keepdims=True)
        probabilities = exponential / denominator
        output = maxima + np.log(denominator)
        if not keepdims:
            output = np.squeeze(output, axis=axes)
        def vjp(g):
            if not keepdims:
                for ax in axes:
                    g = np.expand_dims(g, ax)
            return (g * probabilities,)
        return Tensor._operation(output, (self,), vjp, "logsumexp", (probabilities,))

    def backward(self, gradient=None, retain_graph=False):
        """Compute first-order VJPs and accumulate only leaf gradients.

        A scalar output permits an implicit seed of one. Every nonscalar output
        requires a finite seed with exactly its shape. Traversal is iterative
        O(V+E), while VJP arithmetic costs depend on operators. Local temporary
        gradients are fresh on each pass. Saved operand versions are checked
        before any leaf gradients are changed. Graph edges and saved arrays are
        freed by default; retain_graph=True explicitly permits another pass.
        """
        if not self.requires_grad:
            raise RuntimeError("backward requires a tensor with requires_grad=True")
        if self._freed:
            raise RuntimeError("graph has been released; use retain_graph=True on the earlier backward")
        if gradient is None:
            if self.ndim != 0:
                raise ValueError("nonscalar backward requires an explicit matching seed")
            seed = np.ones_like(self._data)
        else:
            seed = np.asarray(gradient, dtype=self.dtype)
            if seed.shape != self.shape:
                raise ValueError("backward seed shape must exactly match output shape")
            if not np.isfinite(seed).all():
                raise ValueError("backward seed must be finite")

        order = _topological(self)
        for node in order:
            if node.requires_grad and node._freed:
                raise RuntimeError("backward encountered a released ancestor graph")
            for parent, version in zip(node._parents, node._versions):
                if parent._version != version:
                    raise RuntimeError("saved operand was modified after forward; rerun the forward pass")

        gradients = {id(self): seed}
        leaves = []
        for node in reversed(order):
            grad = gradients.pop(id(node), None)
            if grad is None or not node.requires_grad:
                continue
            grad = np.asarray(grad, dtype=node.dtype)
            if grad.shape != node.shape or not np.isfinite(grad).all():
                raise ValueError("VJP produced an invalid shape or nonfinite gradient")
            if node._is_leaf:
                candidate = grad.copy() if node.grad is None else node.grad + grad
                if not np.isfinite(candidate).all():
                    raise ValueError("accumulated leaf gradient is nonfinite")
                leaves.append((node, candidate))
            elif node._vjp is not None:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    contributions = node._vjp(grad)
                if len(contributions) != len(node._parents):
                    raise RuntimeError("internal VJP arity does not match parent count")
                for parent, contribution in zip(node._parents, contributions):
                    if not parent.requires_grad:
                        continue
                    contribution = np.asarray(contribution, dtype=parent.dtype)
                    if contribution.shape != parent.shape or not np.isfinite(contribution).all():
                        raise ValueError("VJP produced an invalid shape or nonfinite gradient")
                    previous = gradients.get(id(parent))
                    gradients[id(parent)] = contribution if previous is None else previous + contribution
        for leaf, gradient_value in leaves:
            leaf.grad = gradient_value
        if not retain_graph:
            for node in order:
                if not node._is_leaf:
                    node._parents = ()
                    node._versions = ()
                    node._vjp = None
                    node._saved_arrays = ()
                    node._freed = True


def _topological(output: Tensor) -> list[Tensor]:
    """Postorder without Python recursion, including shared operands once."""
    order = []
    seen = set()
    stack = [(output, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
        elif id(node) not in seen:
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((parent, False) for parent in reversed(node._parents))
    return order


def _labels(logits: Tensor, labels) -> np.ndarray:
    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ValueError("cross_entropy expects a 2-D Tensor of logits")
    labels = np.asarray(labels)
    if labels.shape != (logits.shape[0],) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array matching batch size")
    labels = labels.astype(np.int64, copy=True)
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("class label is outside the logit class range")
    return labels


def cross_entropy(logits: Tensor, labels) -> Tensor:
    """Mean multiclass CE with a fused stable forward and analytic VJP.

    logits: (N,C) finite float Tensor; labels: (N,) integers in [0,C).
    Returns a scalar of the logit dtype. Time and softmax saved space are O(NC).
    Gradient is (softmax(logits)-one_hot(labels))/N times the upstream scalar.
    """
    labels = _labels(logits, labels)
    values = logits._data
    maxima = values.max(axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        shifted = values - maxima
        exponential = np.exp(shifted)
    denominators = exponential.sum(axis=1, keepdims=True)
    probabilities = exponential / denominators
    with np.errstate(over="ignore", invalid="ignore"):
        loss = _finite_mean(np.log(denominators[:, 0]) - shifted[np.arange(len(labels)), labels])
    def vjp(g):
        gradient = probabilities.copy()
        gradient[np.arange(len(labels)), labels] -= 1
        return (gradient * (g / len(labels)),)
    return Tensor._operation(loss, (logits,), vjp, "cross_entropy", (probabilities, labels))


def cross_entropy_composed(logits: Tensor, labels) -> Tensor:
    """Same mean CE expressed as stable graph operators, a fusion baseline.

    Subtract a detached row maximum before both terms to avoid cancellation of
    huge common offsets. Treating the shift as constant preserves CE's gradient:
    the objective is invariant to row shifts and each row gradient sums to zero.
    """
    labels = _labels(logits, labels)
    shifted = logits - Tensor(logits._data.max(axis=1, keepdims=True))
    one_hot = np.zeros(logits.shape, dtype=logits.dtype)
    one_hot[np.arange(len(labels)), labels] = 1
    correct = (shifted * Tensor(one_hot)).sum(axis=1)
    return (shifted.logsumexp(axis=1) - correct).mean()


def graph_statistics(output: Tensor) -> dict:
    """Count reachable graph nodes/edges and unique retained ndarray buffers.

    Counts data and explicit VJP saved arrays, deduplicating shared base buffers.
    This is a graph-buffer accounting metric, not peak process memory/RSS, BLAS
    workspace, interpreter overhead or transient operator workspace. Call before
    backward frees the graph. Traversal and accounting cost O(V+E).
    """
    if not isinstance(output, Tensor):
        raise TypeError("graph_statistics expects a Tensor")
    nodes = _topological(output)
    arrays = {}
    for node in nodes:
        for array in (node._data,) + node._saved_arrays:
            base = array
            while isinstance(base.base, np.ndarray):
                base = base.base
            arrays[id(base)] = base.nbytes
    return {"nodes": len(nodes), "edges": sum(len(n._parents) for n in nodes),
            "tracked_nodes": sum(n.requires_grad for n in nodes),
            "unique_array_bytes": int(sum(arrays.values()))}
