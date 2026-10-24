# Architecture and boundaries

TraceGrad implements first-order reverse-mode differentiation over a dynamic DAG. A node stores finite immutable data, parent references, saved versions and a local vector-Jacobian product(VJP). NumPy/BLAS supplies numerical arrays and matrix multiplication.

| Module | Responsibility |
|---|---|
| tensor.py | operators, broadcasting adjoints, iterative traversal, graph release, stable losses |
| nn.py | linear logits, single-hidden-layer MLP, explicit L2 |
| optim.py | validated SGD/momentum and Adam; parameter-wise state |
| data.py | author-checked IDX, deterministic split, pixel scaling |
| training.py | shuffling, objectives, validation selection, inference checkpoints, metrics |
| verification.py | independent PyTorch forward/gradient/update comparisons |
| benchmarks.py | eager/selective VJP, composed/fused CE, profiles, paired timing |
| experiments.py | fixed variants/seeds, raw records, aggregation |
| demo.py | zero-download gradient and synthetic learning demonstration |
| reporting.py | local offline figures/report from completed runs |

## Reverse pass

1. Forward creates a graph when grad mode is enabled and at least one parent requires gradients.
2. Iterative DFS builds postorder, including shared nodes once.
3. Reverse order aggregates incoming cotangents before invoking each localVJP.
4. Leaf gradients accumulate; intermediate gradients are fresh on every pass.
5. Saved operand versions are checked before leaf gradients are committed.
6. Successful backward releases nonleaf edges and saved buffers unless retained.

Traversal is O(V+E), plus operator arithmetic. For A(m,k)@B(k,n), gradients are GB^T and A^TG; each product is computed only for a required parent.

## Interfaces

`Tensor(data,requires_grad=False,dtype=None)` supports scalar/nonempty float32/64. Integer data defaults to float64. Elementwise operators broadcast; matmul is2-D. `.numpy()`/`.detach()` make copies. Scalar backward defaults to1; vector seeds must be exactly shaped and finite.

`Classifier(input_dim,hidden_dim,classes,seed,dtype)` uses hidden_dim0 for linear logits, otherwise affine-ReLU-affine. Weights use(in,out), biases(out,). A local RNG preserves global NumPy state.

`SGD(params,lr,momentum=0)` uses undampened momentum without Nesterov. `Adam(params,lr=.001,beta1=.9,beta2=.999,eps=1e-8)` puts epsilon outside sqrt(vhat), counting actual updates per parameter. All proposed updates are validated before assignments. None gradients are skipped. In-memory optimizer state is positional with strict shape/dtype/hyperparameter matching.

NPZ files contain selected model arrays and JSON inference metadata, loaded with allow_pickle=False. CLI training resume is not implemented.

## Scope

Dense CPU first-order differentiation and declared operator subset. No arbitrary in-place mutation, higher derivatives, batched matmul, device abstraction, graph compiler, convolutions or distributed runtime.
