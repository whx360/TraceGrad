"""Controlled CPU performance experiments with raw repeated samples.

The eager matrix VJP below is retained solely as a measured historical baseline;
runtime model code uses Tensor's selective VJP. Timings contain forward/backward
and gradient clearing, but no optimizer updates or data loading. Graph bytes count
retained NumPy buffers, never process peak RSS or transient BLAS workspace.
"""
from __future__ import annotations

import cProfile
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import pstats
import shutil
import sys
from time import perf_counter_ns

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .tensor import Tensor, cross_entropy, cross_entropy_composed, graph_statistics
from .nn import Classifier


def _eager_matmul(left: Tensor, right: Tensor) -> Tensor:
    """Pre-optimization 2-D matmul rule: eagerly calculate both parent VJPs.

    The helper is only used in this benchmark, not behind a runtime feature flag.
    It deliberately preserves the work which backward discarded for constants.
    """
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("eager matmul baseline requires compatible 2-D tensors")
    return Tensor._operation(left._data @ right._data, (left, right),
        lambda g: (g @ right._data.T, left._data.T @ g), "eager_matmul_baseline")


def _eager_classifier(model: Classifier, inputs: Tensor) -> Tensor:
    p = model.named_parameters()
    if model.hidden_dim == 0:
        return _eager_matmul(inputs, p["weight"]) + p["bias"]
    hidden = (_eager_matmul(inputs, p["w1"]) + p["b1"]).relu()
    return _eager_matmul(hidden, p["w2"]) + p["b2"]


def _paired_measure(baseline, optimized, repeats=15, inner=10, warmup=3):
    """Alternate paired execution order and keep every per-call wall-time sample.

    Both callables must perform equivalent work. No speed threshold is assumed.
    The returned seconds divide one inner-loop wall time by its call count;
    they are not an independently measured per-request latency distribution.
    """
    if not all(isinstance(x, int) and x >= 1 for x in (repeats, inner)):
        raise ValueError("repeats and inner must be positive integers")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be nonnegative")
    functions = {"baseline": baseline, "optimized": optimized}
    for _ in range(warmup):
        baseline(); optimized()
    records = []
    for sample in range(repeats):
        order = ("baseline", "optimized") if sample % 2 == 0 else ("optimized", "baseline")
        for position, variant in enumerate(order):
            start = perf_counter_ns()
            for _ in range(inner):
                functions[variant]()
            elapsed = (perf_counter_ns() - start) / 1e9
            records.append({"sample": sample, "variant": variant,
                            "order_position": position, "inner_calls": inner,
                            "seconds_per_call": elapsed / inner})
    return records


def _summary(records, variant):
    values = np.asarray([x["seconds_per_call"] for x in records if x["variant"] == variant])
    return {"samples": int(values.size), "median_seconds": float(np.median(values)),
            "p95_of_inner_mean_seconds": float(np.quantile(values, .95)),
            "min_seconds": float(values.min()), "max_seconds": float(values.max())}


def _comparison(identifier, shape, dtype, baseline_name, optimized_name,
                baseline, optimized, correctness, baseline_graph, optimized_graph):
    records = _paired_measure(baseline, optimized)
    base_summary, opt_summary = _summary(records, "baseline"), _summary(records, "optimized")
    paired = []
    for sample in range(15):
        pair = {r["variant"]: r["seconds_per_call"] for r in records if r["sample"] == sample}
        paired.append(pair["baseline"]/pair["optimized"])
    result = {"id": identifier, "shape": list(shape), "dtype": dtype,
              "baseline_name": baseline_name, "optimized_name": optimized_name,
              "baseline": base_summary, "optimized": opt_summary,
              "ratio_of_median_times": base_summary["median_seconds"]/opt_summary["median_seconds"],
              "median_paired_ratio": float(np.median(paired)),
              "correctness": correctness, "baseline_graph": baseline_graph,
              "optimized_graph": optimized_graph}
    for record in records:
        record["comparison"] = identifier
    return result, records


def _profile(function, path: Path, loops=50):
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(loops):
        function()
    profiler.disable()
    profiler.dump_stats(str(path.with_suffix(".prof")))
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)
    path.with_suffix(".txt").write_text(buffer.getvalue(), encoding="utf-8")
    rows = []
    for (file, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({"file": file, "line": line, "function": name,
                     "primitive_calls": primitive, "calls": calls,
                     "self_seconds": own, "cumulative_seconds": cumulative})
    rows.sort(key=lambda x: x["cumulative_seconds"], reverse=True)
    return {"loops": loops, "total_profile_seconds": stats.total_tt, "top": rows[:30],
            "binary": f"profiles/{path.name}.prof", "text": f"profiles/{path.name}.txt"}


def _ce_case(values, labels):
    baseline_input = Tensor(values, requires_grad=True)
    optimized_input = Tensor(values, requires_grad=True)
    baseline_loss = cross_entropy_composed(baseline_input, labels)
    optimized_loss = cross_entropy(optimized_input, labels)
    baseline_graph, optimized_graph = graph_statistics(baseline_loss), graph_statistics(optimized_loss)
    baseline_loss.backward(); optimized_loss.backward()
    np.testing.assert_allclose(baseline_loss.data, optimized_loss.data, rtol=5e-12, atol=5e-12)
    np.testing.assert_allclose(baseline_input.grad, optimized_input.grad, rtol=5e-12, atol=5e-12)
    correctness = {"loss_absolute_error": abs(baseline_loss.item()-optimized_loss.item()),
                   "max_gradient_absolute_error": float(np.max(np.abs(baseline_input.grad-optimized_input.grad))),
                   "atol": 5e-12, "rtol": 5e-12, "passed": True}

    def baseline():
        baseline_input.zero_grad()
        loss = cross_entropy_composed(baseline_input, labels)
        loss.backward()
        return loss.item()

    def optimized():
        optimized_input.zero_grad()
        loss = cross_entropy(optimized_input, labels)
        loss.backward()
        return loss.item()

    return baseline, optimized, correctness, baseline_graph, optimized_graph


def _mlp_case(values, labels):
    baseline_model = Classifier(values.shape[1], 64, 10, seed=17, dtype=np.float32)
    optimized_model = Classifier(values.shape[1], 64, 10, seed=17, dtype=np.float32)
    inputs = Tensor(values)

    def baseline():
        for p in baseline_model.parameters(): p.zero_grad()
        loss = cross_entropy(_eager_classifier(baseline_model, inputs), labels)
        loss.backward()
        return loss.item()

    def optimized():
        for p in optimized_model.parameters(): p.zero_grad()
        loss = cross_entropy(optimized_model(inputs), labels)
        loss.backward()
        return loss.item()

    base_loss = cross_entropy(_eager_classifier(baseline_model, inputs), labels)
    opt_loss = cross_entropy(optimized_model(inputs), labels)
    base_graph, opt_graph = graph_statistics(base_loss), graph_statistics(opt_loss)
    base_loss.backward(); opt_loss.backward()
    np.testing.assert_allclose(base_loss.data, opt_loss.data, rtol=3e-5, atol=1e-6)
    max_error = 0.
    for name, base_parameter in baseline_model.named_parameters().items():
        optimized_parameter = optimized_model.named_parameters()[name]
        np.testing.assert_array_equal(base_parameter.data, optimized_parameter.data)
        np.testing.assert_allclose(base_parameter.grad, optimized_parameter.grad, rtol=3e-5, atol=1e-6)
        max_error = max(max_error, float(np.max(np.abs(base_parameter.grad-optimized_parameter.grad))))
    correctness = {"loss_absolute_error": abs(base_loss.item()-opt_loss.item()),
                   "max_parameter_gradient_absolute_error": max_error,
                   "atol": 1e-6, "rtol": 3e-5, "passed": True}
    return baseline, optimized, correctness, base_graph, opt_graph


def run_benchmarks(root, output=None) -> dict:
    """Run the fixed CPU protocol and save JSON, raw samples and real profiles.

    root is the independent TraceGrad directory. Output defaults to
    results/performance. An existing nonempty output is rejected before any
    experiment, preserving prior formal evidence. This function does not train a
    model, access data files or make network requests.
    """
    root = Path(root).resolve()
    destination = Path(output).resolve() if output is not None else root / "results" / "performance"
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("benchmark output already contains files; choose a fresh output directory")
    destination.mkdir(parents=True, exist_ok=True)
    profiles = destination / "profiles"
    profiles.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    results, records = [], []
    with threadpool_limits(limits=1, user_api="blas"):
        values = rng.normal(0, .5, (256, 784)).astype(np.float32)
        labels = rng.integers(0, 10, size=256)
        base, opt, correctness, base_graph, opt_graph = _mlp_case(values, labels)
        # Baseline profile precedes paired timing; historical unchanged-runtime
        # profile was captured before the tensor rule was edited.
        profile_records = {"mlp_eager": _profile(base, profiles/"mlp_eager"),
                           "mlp_selective": _profile(opt, profiles/"mlp_selective")}
        result, rows = _comparison("mlp_forward_backward", (256,784,64,10), "float32",
            "eager_two_parent_vjp", "selective_parent_vjp", base, opt, correctness, base_graph, opt_graph)
        result["optimizer_update"] = False
        result["avoided_input_gradient_shape"] = [256,784]
        result["avoided_matmul_flops_per_call"] = 2*256*64*784
        results.append(result); records.extend(rows)

        for shape in [(128,10),(512,10),(256,100),(256,1000)]:
            values = rng.normal(0, 3, shape).astype(np.float64)
            labels = rng.integers(0, shape[1], size=shape[0])
            base, opt, correctness, base_graph, opt_graph = _ce_case(values, labels)
            if shape == (256,1000):
                profile_records["ce_composed_256x1000"] = _profile(base, profiles/"ce_composed_256x1000")
                profile_records["ce_fused_256x1000"] = _profile(opt, profiles/"ce_fused_256x1000")
            result, rows = _comparison(f"ce_{shape[0]}x{shape[1]}", shape, "float64",
                "stable_graph_composed_ce", "fused_ce", base, opt, correctness, base_graph, opt_graph)
            results.append(result); records.extend(rows)

        environment = {"python": sys.version, "numpy": np.__version__,
                       "platform": platform.platform(), "processor": platform.processor(),
                       "threadpools_during_measurement": threadpool_info()}

    original = root/"artifacts"/"perf_preoptimization"
    historical = None
    if original.is_dir():
        for name in ("mlp_eager_original.prof", "mlp_eager_original.txt", "summary.json"):
            source = original/name
            if source.is_file():
                shutil.copy2(source, profiles/f"preoptimization_{name}")
        if (profiles/"preoptimization_summary.json").is_file():
            historical = json.loads((profiles/"preoptimization_summary.json").read_text(encoding="utf-8"))
            historical["source_scope"] = "captured from unchanged runtime before editing Tensor.__matmul__"

    with (destination/"raw_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison","sample","variant","order_position","inner_calls","seconds_per_call"])
        writer.writeheader(); writer.writerows(records)
    payload = {"schema_version": 1, "status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"seed": 2026, "blas_threads": 1, "warmup_calls_per_variant": 3,
                     "samples_per_variant": 15, "inner_calls_per_sample": 10,
                     "order": "alternate baseline-first and optimized-first by paired sample",
                     "clock": "perf_counter_ns wall time / inner calls",
                     "ce_dtype": "float64", "mlp_dtype": "float32",
                     "optimizer_updates": False},
        "environment": environment, "comparisons": results, "profiles": profile_records,
        "historical_preoptimization_profile": historical,
        "raw_samples": "raw_samples.csv", "raw_row_count": len(records),
        "limits": ["CE kernel ratios are not whole-training acceleration.",
                   "MLP comparison measures forward/backward only; no data loading, validation or optimizer step.",
                   "unique_array_bytes counts retained graph ndarray buffers, not process peak RSS or temporary workspaces.",
                   "Composed CE includes explicit detached row shift and one-hot selection; no claim against every possible manual CE implementation.",
                   "Ratios are measured on fixed shapes/hardware; regressions and timing variability remain in the raw records."]}
    (destination/"benchmark.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return payload
