"""TraceGrad: a small, independent NumPy CPU autodiff training engine."""
from .tensor import Tensor, cross_entropy, cross_entropy_composed, graph_statistics, no_grad

__all__ = ["Tensor", "cross_entropy", "cross_entropy_composed", "graph_statistics", "no_grad"]
__version__ = "0.1.0"
