# Derivatives, edge cases and evidence

For Y=f(X) and incoming G, reverse mode computes J_f(X)^T G without a full Jacobian.

| Operator | VJP |
|---|---|
| add | G to both parents, then unbroadcast |
| multiply | G*other, then unbroadcast |
| AB | GB^T and A^TG |
| sum/mean | restore reduced axes, broadcast; mean divides by element count |
| reshape/transpose | inverse shape/permutation |
| ReLU | G*(X>0); chosen derivative0 at0 |
| log/exp | G/X and G*exp(X) |
| logsumexp | restored G*softmax(X) |
| mean softmaxCE | G*(P-one_hot(y))/batch |

## Stable loss

Per row let c=max(logits),q=logits-c. CE=log(sum(exp(q)))-q_target. Fused backward saves probabilities. The composed baseline uses a detached row shift, one-hot selection and graph operators; the fixed shift is valid because CE is row-shift invariant.

Computing logsumexp(logits)-target directly can lose log2 at a common1e16 offset. Shift before the final subtraction. A second safeguard rescales only when ordinary averaging of finite per-example losses overflows: two1e308 losses have representable mean1e308 but an unrepresentable sum.

Division's denominatorVJP uses -G*(X/Y)/Y. Computing Y² first can overflow/underflow although the derivative is representable. Regressions cover1e200/1e200 and1e-300/1e-200.

Nonfinite results are rejected. Finite inputs do not guarantee every intermediate expression is representable. float64 central differences avoid ReLU's nondifferentiable point; that point is checked against its explicit convention.

## Verification layers

- Analytic cases: scalar chains, repeated variables, broadcasting, matrix gradients and mean scaling.
- Central differences: independently perturb coordinates instead of reusing the analyticalVJP.
- PyTorch: float32/64×SGD/Adam, five steps each with identical initialization, data and explicitL2.
- State/boundary checks: invalid inputs, graph release, operand version changes, duplicated parameters and atomic optimizer updates.
- Integration: fitting, safe checkpoint roundtrip and prediction equality.

Recorded fixed-reference float64 parameter-gradient error≤2.220446e-16. Numerical evidence on these inputs is not proof for every possible program. See the executable tests and `benchmarks/reference/numerical_reference.json`.
