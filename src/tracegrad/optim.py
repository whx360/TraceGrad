"""SGD/momentum and Adam with explicit state and validated atomic updates.

There is no optimizer weight decay. L2 belongs in the explicit training objective.
Each update costs O(total active parameter entries) in time and state storage.
"""
from __future__ import annotations

from collections.abc import Mapping
import numbers
import numpy as np

from .tensor import Tensor


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real")
    return float(value)


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must satisfy 0 <= value < 1")
    return float(value)


class _Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer needs at least one parameter")
        if any(not isinstance(p, Tensor) or not p.requires_grad or not p.is_leaf for p in self.params):
            raise ValueError("optimizer parameters must be trainable leaf Tensors")
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError("duplicate optimizer parameters are not allowed")
        self.lr = _positive(lr, "lr")
        self.step_count = 0

    def zero_grad(self):
        for parameter in self.params:
            parameter.zero_grad()

    def _gradients(self):
        active = []
        # Validate every gradient before computing or assigning any update.
        for index, parameter in enumerate(self.params):
            if parameter.grad is None:
                continue
            raw = np.asarray(parameter.grad)
            if raw.shape != parameter.shape or raw.dtype.kind not in "fiu":
                raise ValueError("gradient shape or real numeric dtype is invalid")
            gradient = np.array(raw, dtype=parameter.dtype, copy=True)
            if not np.isfinite(gradient).all():
                raise ValueError("optimizer gradient must be finite")
            active.append((index, parameter, gradient))
        return active

    @staticmethod
    def _validate_update(array):
        if not np.isfinite(array).all():
            raise ValueError("optimizer update produced nonfinite values")

    def _check_state_header(self, state, kind, configuration):
        if not isinstance(state, Mapping) or state.get("type") != kind:
            raise ValueError("optimizer state type mismatch")
        if state.get("configuration") != configuration:
            raise ValueError("optimizer hyperparameters do not match state")
        expected = [(p.shape, p.dtype.str) for p in self.params]
        if state.get("parameters") != expected:
            raise ValueError("optimizer parameter shapes/dtypes do not match state")
        count = state.get("step_count")
        if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
            raise ValueError("optimizer step_count must be a nonnegative integer")
        return int(count)

    def _state_array(self, value, index):
        array = np.asarray(value)
        parameter = self.params[index]
        if array.shape != parameter.shape or array.dtype != parameter.dtype or not np.isfinite(array).all():
            raise ValueError("optimizer state array shape/dtype/finiteness mismatch")
        return array.copy()

    def _header(self, kind, configuration):
        return {"type": kind, "configuration": configuration,
                "parameters": [(p.shape, p.dtype.str) for p in self.params],
                "step_count": self.step_count}


class SGD(_Optimizer):
    """SGD with optional undampened momentum: v=m*v+g; p=p-lr*v.

    grad=None parameters are skipped. An all-None step is a no-op. The first
    momentum buffer equals its gradient, matching ordinary SGD momentum rules.
    """
    def __init__(self, params, lr, momentum=0):
        super().__init__(params, lr)
        self.momentum = _fraction(momentum, "momentum")
        self._velocity = [None] * len(self.params)

    def step(self):
        active = self._gradients()
        proposed = []
        with np.errstate(over="ignore", invalid="ignore"):
            for index, parameter, gradient in active:
                old = self._velocity[index]
                velocity = gradient if old is None else self.momentum * old + gradient
                value = parameter.data - self.lr * velocity
                self._validate_update(value)
                self._validate_update(velocity)
                proposed.append((index, parameter, value, velocity))
        for index, parameter, value, velocity in proposed:
            parameter._assign(value)
            self._velocity[index] = velocity if self.momentum else None
        if proposed:
            self.step_count += 1

    def state_dict(self):
        result = self._header("SGD", {"lr": self.lr, "momentum": self.momentum})
        result["velocity"] = [None if x is None else x.copy() for x in self._velocity]
        return result

    def load_state_dict(self, state):
        count = self._check_state_header(state, "SGD", {"lr": self.lr, "momentum": self.momentum})
        values = state.get("velocity")
        if not isinstance(values, list) or len(values) != len(self.params):
            raise ValueError("SGD velocity state length mismatch")
        prepared = [None if value is None else self._state_array(value, i)
                    for i, value in enumerate(values)]
        self._velocity = prepared
        self.step_count = count


class Adam(_Optimizer):
    """Bias-corrected Adam; epsilon is outside sqrt(v_hat), no weight decay.

    Each active parameter has its own update count, so parameters with grad=None
    do not age their moments. step_count counts calls having at least one update.
    State is copied/validated before mutation; invalid later gradients cannot
    partially change an earlier parameter or optimizer moment.
    """
    def __init__(self, params, lr=.001, beta1=.9, beta2=.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = _fraction(beta1, "beta1")
        self.beta2 = _fraction(beta2, "beta2")
        self.eps = _positive(eps, "eps")
        self._m = [None] * len(self.params)
        self._v = [None] * len(self.params)
        self._steps = [0] * len(self.params)

    def _configuration(self):
        return {"lr": self.lr, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps}

    def step(self):
        active = self._gradients()
        proposed = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for index, parameter, gradient in active:
                old_m, old_v = self._m[index], self._v[index]
                m = (1-self.beta1) * gradient if old_m is None else self.beta1*old_m + (1-self.beta1)*gradient
                v = (1-self.beta2) * gradient**2 if old_v is None else self.beta2*old_v + (1-self.beta2)*gradient**2
                step = self._steps[index] + 1
                mhat = m / (1-self.beta1**step)
                vhat = v / (1-self.beta2**step)
                value = parameter.data - self.lr*mhat/(np.sqrt(vhat)+self.eps)
                for array in (m, v, value):
                    self._validate_update(array)
                proposed.append((index, parameter, value, m, v, step))
        for index, parameter, value, m, v, step in proposed:
            parameter._assign(value)
            self._m[index], self._v[index], self._steps[index] = m, v, step
        if proposed:
            self.step_count += 1

    def state_dict(self):
        result = self._header("Adam", self._configuration())
        result.update({"m": [None if x is None else x.copy() for x in self._m],
                       "v": [None if x is None else x.copy() for x in self._v],
                       "steps": list(self._steps)})
        return result

    def load_state_dict(self, state):
        count = self._check_state_header(state, "Adam", self._configuration())
        moments, variances, steps = state.get("m"), state.get("v"), state.get("steps")
        if not all(isinstance(x, list) and len(x) == len(self.params) for x in (moments, variances, steps)):
            raise ValueError("Adam state list length mismatch")
        new_m, new_v, new_steps = [], [], []
        for index, (m, v, step) in enumerate(zip(moments, variances, steps)):
            if isinstance(step, bool) or not isinstance(step, numbers.Integral) or not 0 <= step <= count:
                raise ValueError("Adam per-parameter step is invalid")
            if step == 0:
                if m is not None or v is not None:
                    raise ValueError("unused Adam parameter must have no moments")
                new_m.append(None); new_v.append(None)
            else:
                if m is None or v is None:
                    raise ValueError("used Adam parameter must have both moments")
                new_m.append(self._state_array(m, index))
                new_v.append(self._state_array(v, index))
                if np.any(new_v[-1] < 0):
                    raise ValueError("Adam second moment must be nonnegative")
            new_steps.append(int(step))
        self._m, self._v, self._steps = new_m, new_v, new_steps
        self.step_count = count
