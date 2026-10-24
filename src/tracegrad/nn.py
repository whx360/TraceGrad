"""Small classifiers whose backward pass is built entirely by Tensor operations."""
from __future__ import annotations

from collections.abc import Mapping
import numbers
import numpy as np

from .tensor import Tensor


def _dimension(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class Classifier:
    """Linear softmax logits or one-hidden-layer ReLU MLP.

    Input: a Tensor (batch,input_dim) with the model dtype. Output: (batch,classes)
    logits; no softmax is applied here, since stable CE consumes logits directly.
    hidden_dim=0 selects a linear model. Parameters use (in,out) matrix layout.

    The MLP forward costs O(batch*(input_dim*hidden_dim+hidden_dim*classes));
    linear forward costs O(batch*input_dim*classes), backed by NumPy/BLAS.
    Initialization uses a local NumPy Generator, never mutating global RNG state.
    """

    def __init__(self, input_dim=784, hidden_dim=64, classes=10, seed=17, dtype=np.float32):
        self.input_dim = _dimension(input_dim, "input_dim")
        self.hidden_dim = _dimension(hidden_dim, "hidden_dim", 0)
        self.classes = _dimension(classes, "classes", 2)
        self.seed = _dimension(seed, "seed", 0)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("classifier dtype must be float32 or float64")
        rng = np.random.default_rng(self.seed)

        def parameter(name, array):
            return Tensor(np.asarray(array, dtype=self.dtype), requires_grad=True, name=name)

        if self.hidden_dim == 0:
            self._parameters = {
                "weight": parameter("weight", rng.normal(0, .01, (self.input_dim, self.classes))),
                "bias": parameter("bias", np.zeros(self.classes)),
            }
            self.weight_names = ("weight",)
        else:
            self._parameters = {
                "w1": parameter("w1", rng.normal(0, np.sqrt(2/self.input_dim),
                                                 (self.input_dim, self.hidden_dim))),
                "b1": parameter("b1", np.zeros(self.hidden_dim)),
                "w2": parameter("w2", rng.normal(0, 1/np.sqrt(self.hidden_dim),
                                                 (self.hidden_dim, self.classes))),
                "b2": parameter("b2", np.zeros(self.classes)),
            }
            self.weight_names = ("w1", "w2")

    def __call__(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("classifier input must be a Tensor")
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("classifier input shape must be (batch,input_dim)")
        if inputs.dtype != self.dtype:
            raise TypeError("classifier input dtype must match parameter dtype")
        p = self._parameters
        if self.hidden_dim == 0:
            return inputs @ p["weight"] + p["bias"]
        hidden = (inputs @ p["w1"] + p["b1"]).relu()
        return hidden @ p["w2"] + p["b2"]

    def named_parameters(self) -> dict[str, Tensor]:
        """Return a new name mapping to owned live leaf parameters, O(param count)."""
        return dict(self._parameters)

    def parameters(self) -> list[Tensor]:
        return list(self._parameters.values())

    def state_dict(self) -> dict[str, np.ndarray]:
        """Independent copies of all parameter arrays; O(number of parameters)."""
        return {name: parameter.numpy() for name, parameter in self._parameters.items()}

    def load_state_dict(self, state):
        """Strict all-or-nothing validation of names, shapes, dtypes and finiteness.

        This restores model values only; it is not an optimizer/RNG training
        checkpoint. Assignment increments Tensor versions to reject stale graphs.
        """
        if not isinstance(state, Mapping) or set(state) != set(self._parameters):
            raise ValueError("model state must have exactly the model parameter names")
        prepared = {}
        for name, parameter in self._parameters.items():
            value = np.asarray(state[name])
            if value.shape != parameter.shape:
                raise ValueError(f"state shape mismatch for {name}")
            if value.dtype != parameter.dtype:
                raise TypeError(f"state dtype mismatch for {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite state parameter {name}")
            prepared[name] = value.copy()
        for name, value in prepared.items():
            self._parameters[name]._assign(value)


def l2_penalty(model: Classifier, coefficient: float) -> Tensor:
    """Return coefficient/2 * sum(weight**2), excluding every bias.

    coefficient must be finite and nonnegative. The scalar zero case has the
    model dtype and no graph. Positive coefficient uses O(weight count) work;
    autodiff therefore contributes coefficient*weight to each weight gradient.
    """
    if not isinstance(model, Classifier):
        raise TypeError("l2_penalty expects a Classifier")
    if not isinstance(coefficient, numbers.Real) or not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("L2 coefficient must be finite and nonnegative")
    if coefficient == 0:
        return Tensor(0., dtype=model.dtype)
    penalty = Tensor(0., dtype=model.dtype)
    parameters = model.named_parameters()
    for name in model.weight_names:
        weight = parameters[name]
        penalty = penalty + (weight * weight).sum()
    return penalty * (coefficient/2)
