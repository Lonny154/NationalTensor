"""Paired, preallocated CUDA-event benchmark for a matmul launcher.

The launcher contract is ``launch(a, b, out, config)``. Correctness checks and
result persistence belong to the caller. Torch is imported only for the default
CUDA output allocator, synchronization, and timer, so orchestration can be tested
without a GPU or third-party packages.
"""

import math
import random
import statistics


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _statistics(values):
    median = statistics.median(values)
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "median": median,
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "mad": statistics.median(abs(value - median) for value in values),
        "stdev": statistics.pstdev(values),
        "cv": statistics.pstdev(values) / mean,
    }


def _cuda_event_ms_per_launch(function, iterations):
    """Time one block with CUDA events, returning milliseconds per launch."""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _cuda_output(a, b):
    import torch

    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


def _cuda_synchronize():
    import torch

    torch.cuda.synchronize()


def run_paired(
    launch, a, b, baseline_config, candidate_config, *, rounds=8,
    warmups=10, iterations=100, seed=42, timer=None, output_factory=None,
    synchronize=None, mode="comparison", prepare=None, timer_by_side=None,
):
    """Measure matched baseline/candidate blocks with balanced execution order.

    ``timer(function, iterations)`` must return completed-block milliseconds per
    launch. Inject it, ``output_factory(a, b)``, and ``synchronize()`` in CPU tests.
    The two outputs are allocated once and kept distinct, including in A/A mode.
    ``prepare(outputs)`` runs after both sides are primed and initially warmed,
    before timed rounds. ``timer_by_side(name, function, iterations)`` supports
    a prepared timing path such as CUDA Graph replay without changing warmups.
    """
    for name, value, minimum in (
        ("rounds", rounds, 2), ("warmups", warmups, 0),
        ("iterations", iterations, 1), ("seed", seed, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if rounds % 2:
        raise ValueError("rounds must be even to balance execution order")
    if mode not in {"comparison", "aa"}:
        raise ValueError("mode must be comparison or aa")
    if mode == "aa" and baseline_config != candidate_config:
        raise ValueError("A/A mode requires identical configurations")
    if timer is not None and timer_by_side is not None:
        raise ValueError("provide timer or timer_by_side, not both")

    timer = timer or _cuda_event_ms_per_launch
    output_factory = output_factory or _cuda_output
    synchronize = synchronize or _cuda_synchronize

    outputs = {name: output_factory(a, b) for name in ("baseline", "candidate")}
    if outputs["baseline"] is outputs["candidate"]:
        raise ValueError("output_factory must return distinct outputs")
    configs = {"baseline": baseline_config, "candidate": candidate_config}
    functions = {
        name: (lambda name=name: launch(a, b, outputs[name], configs[name]))
        for name in configs
    }

    # Compile, prime, and warm both paths before any timed round. For A/A these
    # are separate output buffers, even though the compiled kernel is identical.
    for name in ("baseline", "candidate"):
        functions[name]()
        synchronize()
        for _ in range(warmups):
            functions[name]()
        synchronize()
    if prepare is not None:
        prepare(outputs)

    rng = random.Random(seed)
    orders = []
    for _ in range(rounds // 2):
        pair = [("baseline", "candidate"), ("candidate", "baseline")]
        rng.shuffle(pair)
        orders.extend(pair)

    rows = []
    samples = {"baseline": [], "candidate": []}
    for round_number, order in enumerate(orders, 1):
        row = {"round": round_number, "order": list(order)}
        for name in order:
            for _ in range(warmups):
                functions[name]()
            synchronize()
            elapsed_ms = float(
                timer_by_side(name, functions[name], iterations)
                if timer_by_side is not None else timer(functions[name], iterations)
            )
            synchronize()
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise ValueError(f"{name} timer returned a nonfinite or nonpositive latency")
            row[f"{name}_ms"] = elapsed_ms
            samples[name].append(elapsed_ms)
        ratio = row["baseline_ms"] / row["candidate_ms"]
        if not math.isfinite(ratio):
            raise ValueError("paired latency ratio is nonfinite")
        row["baseline_over_candidate"] = ratio
        rows.append(row)

    ratios = [row["baseline_over_candidate"] for row in rows]
    result = {
        "mode": mode,
        "method": {
            "clock": (
                "cuda_event" if timer_by_side is None and timer is _cuda_event_ms_per_launch
                else "injected_timer"
            ),
            "timing_boundary": (
                "CUDA-event block divided by launch count"
                if timer_by_side is None and timer is _cuda_event_ms_per_launch
                else "defined by injected timer"
            ),
            "preallocated_outputs": True,
            "prime_calls_per_side": 1,
            "initial_warmups_per_side": warmups,
            "warmups_per_side_per_round": warmups,
            "iterations_per_side_per_round": iterations,
            "rounds": rounds,
            "order_seed": seed,
            "balanced_order": True,
        },
        "rounds": rows,
        "samples_ms": samples,
        "baseline_ms": _statistics(samples["baseline"]),
        "candidate_ms": _statistics(samples["candidate"]),
        "paired_ratios": _statistics(ratios),
        "ratio_of_medians": (
            statistics.median(samples["baseline"])
            / statistics.median(samples["candidate"])
        ),
    }
    if mode == "aa":
        deviations = [abs(ratio - 1.0) for ratio in ratios]
        result["aa_variability"] = {
            "median_absolute_ratio_deviation": statistics.median(deviations),
            "max_absolute_ratio_deviation": max(deviations),
        }
    return result


def run_aa_calibration(launch, a, b, config, **kwargs):
    """Measure the same configuration through both sides of the paired harness."""
    return run_paired(launch, a, b, config, config, mode="aa", **kwargs)
