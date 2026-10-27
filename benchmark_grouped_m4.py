"""Standalone direct-2D versus grouped-M4 GEMM experiment; no Muse calls.

Run: python benchmark_grouped_m4.py [--output PATH]
Results live outside the autonomous optimizer's runs/ history.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import torch
import triton

import benchmark_core
import reference_precision
from benchmark_proposal import launch as launch_direct, matmul_kernel
from grouped_matmul import (
    DIRECT_2D, GROUPED_M4, GROUP_M, grouped_matmul_kernel,
    launch_grouped, launch_scheduled,
)
from optimize import (
    BASELINE, CONFIRM_ITERATIONS, CONFIRM_WARMUPS,
    DEFAULT_CONFIRMATION_PAIRS, DEFAULT_MIN_CONFIRMATION_SPEEDUP,
    SCREEN_ITERATIONS, SCREEN_WARMUPS, benchmark_method,
    confirmation_decision,
)
from reference_precision import fp16_mm_reference


SHAPES = ((512, 3072, 768), (384, 3072, 768))  # M, N, K
INPUT_SEED = 42
INPUT_SCALE = 1.0
SCREEN_ROUNDS = 8
CORRECTNESS_RTOL = 0.02
CORRECTNESS_ATOL = 0.02


def sha256_source(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def source_fingerprints():
    return {
        "direct_kernel": sha256_source(matmul_kernel.fn),
        "direct_launcher": sha256_source(launch_direct),
        "grouped_kernel": sha256_source(grouped_matmul_kernel.fn),
        "grouped_launcher": sha256_source(launch_grouped),
        "schedule_dispatcher": sha256_source(launch_scheduled),
        "benchmark_core_file": hashlib.sha256(
            Path(benchmark_core.__file__).read_bytes()
        ).hexdigest(),
        "reference_precision_file": hashlib.sha256(
            Path(reference_precision.__file__).read_bytes()
        ).hexdigest(),
    }


def screening_summary(raw):
    """Use the optimizer's selection metric, retaining both speedup views."""
    paired_speedup = raw["paired_ratios"]["median"]
    return {
        "ratio_of_medians": raw["ratio_of_medians"],
        "median_paired_speedup": paired_speedup,
        "selection_metric": "median_paired_speedup",
        "selected_for_confirmation": paired_speedup > 1.0,
    }


def confirm_raw(raw):
    """Adapt benchmark_core's field name to the optimizer's existing rule."""
    return confirmation_decision(
        {"paired_rounds": raw["rounds"]},
        required_pairs=DEFAULT_CONFIRMATION_PAIRS,
        min_speedup=DEFAULT_MIN_CONFIRMATION_SPEEDUP,
    )


def validate(a, b, reference, config):
    output = torch.empty_like(reference)
    launch_scheduled(a, b, output, config)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, reference, rtol=CORRECTNESS_RTOL, atol=CORRECTNESS_ATOL
    )
    return output


def print_rounds(raw, label):
    print(f"  {label} raw paired rounds (microseconds per launch):")
    for row in raw["rounds"]:
        print(
            f"    {row['round']:2d} {' -> '.join(row['order']):22s} "
            f"baseline={row['baseline_ms'] * 1000:.3f} "
            f"grouped={row['candidate_ms'] * 1000:.3f} "
            f"paired={row['baseline_over_candidate']:.4f}x"
        )


def run_shape(shape, shape_index, direct, grouped):
    m, n, k = shape
    torch.manual_seed(INPUT_SEED + shape_index)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    reference, reference_policy = fp16_mm_reference(a, b)

    # Both launchers compile and both outputs pass the same numerical check
    # before either schedule enters a measured block.
    baseline_out = validate(a, b, reference, direct)
    grouped_out = validate(a, b, reference, grouped)
    torch.testing.assert_close(grouped_out, baseline_out, rtol=0, atol=0)
    result = {
        "shape_mnk": list(shape),
        "input_seed": INPUT_SEED + shape_index,
        "input_scale": INPUT_SCALE,
        "reference_precision": reference_policy,
        "correctness": {DIRECT_2D: "PASS", GROUPED_M4: "PASS"},
    }
    print(f"M={m}, N={n}, K={k}: both schedules pass correctness")

    outputs = iter((baseline_out, grouped_out))
    screening = benchmark_core.run_paired(
        launch_scheduled, a, b, direct, grouped,
        rounds=SCREEN_ROUNDS, warmups=SCREEN_WARMUPS,
        iterations=SCREEN_ITERATIONS, seed=42 + shape_index,
        output_factory=lambda _a, _b: next(outputs),
    )
    summary = screening_summary(screening)
    result["screening"] = {"summary": summary, "paired": screening}
    print_rounds(screening, "screening")
    print(
        f"  ratio of medians={summary['ratio_of_medians']:.4f}x; "
        f"median paired speedup={summary['median_paired_speedup']:.4f}x "
        "(selection metric)"
    )

    if summary["selected_for_confirmation"]:
        baseline_out = validate(a, b, reference, direct)
        grouped_out = validate(a, b, reference, grouped)
        outputs = iter((baseline_out, grouped_out))
        confirmation = benchmark_core.run_paired(
            launch_scheduled, a, b, direct, grouped,
            rounds=2 * DEFAULT_CONFIRMATION_PAIRS,
            warmups=CONFIRM_WARMUPS, iterations=CONFIRM_ITERATIONS,
            seed=2026 + shape_index,
            output_factory=lambda _a, _b: next(outputs),
        )
        decision = confirm_raw(confirmation)
        result["confirmation"] = {"status": "confirmed" if decision["confirmed"] else "unconfirmed",
                                  "decision": decision, "paired": confirmation}
        print_rounds(confirmation, "confirmation")
        print(
            f"  confirmation: {result['confirmation']['status']}; "
            f"{decision['pairs_favoring_candidate']}/{DEFAULT_CONFIRMATION_PAIRS} "
            "verification pairs favor grouped"
        )
    else:
        result["confirmation"] = {"status": "not_run", "reason": "screening selection metric <= 1.0"}
        print("  confirmation: not run; grouped did not win screening")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="JSON output path (default: unique evaluations/grouped-m4 timestamp)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run in the GPU-enabled WSL environment")
    gpu = torch.cuda.get_device_properties(0)
    output_path = args.output or Path("evaluations/grouped-m4") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        parser.error(f"output already exists: {output_path}")

    direct = {**BASELINE, "schedule": DIRECT_2D}
    grouped = {**BASELINE, "schedule": GROUPED_M4}
    artifact = {
        "schema_version": 1,
        "status": "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": gpu.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "torch_cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_sha256": source_fingerprints(),
        "configurations": {DIRECT_2D: direct, GROUPED_M4: grouped},
        "input_generation": {"distribution": "torch.randn", "seed": INPUT_SEED,
                             "seed_increments_per_shape": 1, "scale": INPUT_SCALE},
        "correctness_tolerance": {"rtol": CORRECTNESS_RTOL, "atol": CORRECTNESS_ATOL},
        "reference_precision_policy": "fp16_torch_mm_without_reduced_precision_reduction",
        "group_size_m": GROUP_M,
        "method": benchmark_method(
            SCREEN_ROUNDS, DEFAULT_CONFIRMATION_PAIRS,
            DEFAULT_MIN_CONFIRMATION_SPEEDUP,
        ),
        "selection_metric": "median_paired_speedup",
        "limitations": [
            "CUDA-event blocks include host submission and GPU scheduling gaps.",
            "Idle-associated first-block anomalies may occur; no settling blocks or outlier removal are used.",
            "The confirmation rule is a heuristic, not a statistical significance test.",
        ],
        "shapes": [],
    }
    start = time.perf_counter()
    try:
        for index, shape in enumerate(SHAPES):
            artifact["shapes"].append(run_shape(shape, index, direct, grouped))
        artifact["status"] = "complete"
    except Exception as exc:
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        artifact["wall_seconds"] = time.perf_counter() - start
        with output_path.open("x") as stream:
            json.dump(artifact, stream, indent=2)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
