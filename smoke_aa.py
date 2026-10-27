"""Compare A/A variability for eager Python launches and CUDA Graph replay.

Run from the project root: .venv/bin/python smoke_aa.py
Use --mode-order graph-first in a separate capture to counterbalance mode order.
The JSON result includes every paired timing and its execution order.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from benchmark_core import run_aa_calibration


M, N, K = 512, 3072, 768
BASELINE = {
    "block_m": 64,
    "block_n": 128,
    "block_k": 32,
    "num_warps": 4,
    "num_stages": 3,
}
INPUT_SEED = 42
ORDER_SEED = 42
ROUNDS = 20
WARMUPS = 20
ITERATIONS = 100


class DiagnosticTimer:
    """Keep CUDA-event timing while recording two separate host intervals."""

    def __init__(self, event_factory=None, clock_ns=None):
        self.event_factory = event_factory
        self.clock_ns = clock_ns or time.perf_counter_ns
        self.blocks = []

    def __call__(self, function, iterations):
        event_factory = self.event_factory
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)

        start = event_factory()
        end = event_factory()
        wall_start_ns = self.clock_ns()
        start.record()
        submit_start_ns = self.clock_ns()
        for _ in range(iterations):
            function()
        submit_end_ns = self.clock_ns()
        end.record()
        end.synchronize()
        wall_end_ns = self.clock_ns()

        event_ms = float(start.elapsed_time(end))
        block = {
            "cuda_event_ms_per_block": event_ms,
            "cuda_event_ms_per_launch": event_ms / iterations,
            "host_submission_ms_per_block": (submit_end_ns - submit_start_ns) / 1e6,
            "host_wall_ms_per_block": (wall_end_ns - wall_start_ns) / 1e6,
        }
        self.blocks.append(block)
        return block["cuda_event_ms_per_launch"]


class GraphDiagnosticTimer(DiagnosticTimer):
    """Time one replay of a graph containing ``iterations`` GEMM launches."""

    def __init__(self, graphs, event_factory=None, clock_ns=None):
        super().__init__(event_factory=event_factory, clock_ns=clock_ns)
        self.graphs = graphs

    def __call__(self, name, _function, iterations):
        event_factory = self.event_factory
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)

        start = event_factory()
        end = event_factory()
        wall_start_ns = self.clock_ns()
        start.record()
        submit_start_ns = self.clock_ns()
        self.graphs[name].replay()
        submit_end_ns = self.clock_ns()
        end.record()
        end.synchronize()
        wall_end_ns = self.clock_ns()

        event_ms = float(start.elapsed_time(end))
        block = {
            "cuda_event_ms_per_block": event_ms,
            "cuda_event_ms_per_launch": event_ms / iterations,
            "host_submission_ms_per_block": (submit_end_ns - submit_start_ns) / 1e6,
            "host_wall_ms_per_block": (wall_end_ns - wall_start_ns) / 1e6,
        }
        self.blocks.append(block)
        return block["cuda_event_ms_per_launch"]


def capture_graphs(
    launch, a, b, config, outputs, iterations, *,
    graph_factory=None, capture_context=None, synchronize=None,
):
    """Capture and prime one fixed-address graph per preallocated output."""
    if graph_factory is None or capture_context is None or synchronize is None:
        import torch
        graph_factory = graph_factory or torch.cuda.CUDAGraph
        capture_context = capture_context or torch.cuda.graph
        synchronize = synchronize or torch.cuda.synchronize

    graphs = {}
    for name, output in outputs.items():
        graph = graph_factory()
        with capture_context(graph):
            for _ in range(iterations):
                launch(a, b, output, config)
        graphs[name] = graph
        graph.replay()
        synchronize()
    return graphs


def attach_diagnostics(result, blocks, execution_mode="python_launch"):
    """Join timer calls to saved rounds using the recorded execution order."""
    if execution_mode not in {"python_launch", "cuda_graph"}:
        raise ValueError("unknown execution mode")
    if len(blocks) != 2 * len(result["rounds"]):
        raise ValueError("diagnostic block count does not match paired rounds")
    block_iter = iter(blocks)
    for row in result["rounds"]:
        row["blocks"] = {}
        for name in row["order"]:
            block = next(block_iter)
            if not math.isclose(row[f"{name}_ms"], block["cuda_event_ms_per_launch"]):
                raise ValueError("diagnostic block does not match its paired timing")
            row["blocks"][name] = block
    result["method"].update({
        "execution_mode": execution_mode,
        "clock": "cuda_event",
        "timing_boundary": (
            "CUDA events around 100 Python launches; duration divided by 100"
            if execution_mode == "python_launch"
            else "CUDA events around one 100-launch graph replay; duration divided by 100"
        ),
        "host_submission_boundary": (
            "perf_counter_ns around Python launch loop only"
            if execution_mode == "python_launch"
            else "perf_counter_ns around one graph.replay() call only"
        ),
        "host_wall_boundary": (
            "perf_counter_ns before start-event record through end-event synchronization; "
            "the subsequent device-wide synchronization is outside this interval"
        ),
    })


def _correlation(pairs):
    if len(pairs) < 2:
        return None
    try:
        return statistics.correlation(*zip(*pairs))
    except statistics.StatisticsError:
        return None


def diagnostic_summary(result):
    """Descriptive host/event correlations and prior-trace latency bands."""
    rows = result["rounds"]
    blocks = [(name, row["blocks"][name]) for row in rows for name in row["order"]]
    pairs = [
        (block["host_submission_ms_per_block"], block["cuda_event_ms_per_launch"])
        for _, block in blocks
    ]
    by_side = {
        name: _correlation([
            (block["host_submission_ms_per_block"], block["cuda_event_ms_per_launch"])
            for label, block in blocks if label == name
        ])
        for name in ("baseline", "candidate")
    }
    paired_deltas = [
        (
            row["blocks"]["baseline"]["host_submission_ms_per_block"]
            - row["blocks"]["candidate"]["host_submission_ms_per_block"],
            row["baseline_ms"] - row["candidate_ms"],
        )
        for row in rows
    ]
    bands = []
    for label, lower, upper in (
        ("<61 us", None, 61.0),
        ("61-<66 us", 61.0, 66.0),
        (">=66 us", 66.0, None),
    ):
        selected = [
            block for _, block in blocks
            if (lower is None or block["cuda_event_ms_per_launch"] * 1000 >= lower)
            and (upper is None or block["cuda_event_ms_per_launch"] * 1000 < upper)
        ]
        bands.append({
            "label": label,
            "count": len(selected),
            "event_median_us_per_launch": (
                statistics.median(b["cuda_event_ms_per_launch"] * 1000 for b in selected)
                if selected else None
            ),
            "submission_median_ms_per_block": (
                statistics.median(b["host_submission_ms_per_block"] for b in selected)
                if selected else None
            ),
            "wall_median_ms_per_block": (
                statistics.median(b["host_wall_ms_per_block"] for b in selected)
                if selected else None
            ),
        })
    pooled_median_us = statistics.median(
        block["cuda_event_ms_per_launch"] * 1000 for _, block in blocks
    )
    relative_bands = {
        "below_97_percent_of_mode_median": 0,
        "within_3_percent_of_mode_median": 0,
        "above_103_percent_of_mode_median": 0,
    }
    for _, block in blocks:
        value_us = block["cuda_event_ms_per_launch"] * 1000
        if value_us < 0.97 * pooled_median_us:
            relative_bands["below_97_percent_of_mode_median"] += 1
        elif value_us > 1.03 * pooled_median_us:
            relative_bands["above_103_percent_of_mode_median"] += 1
        else:
            relative_bands["within_3_percent_of_mode_median"] += 1
    return {
        "correlation_method": "Pearson; descriptive only",
        "submission_vs_event_all_blocks": _correlation(pairs),
        "submission_vs_event_by_side": by_side,
        "submission_vs_event_within_round_deltas": _correlation(paired_deltas),
        "latency_bands_from_prior_aa_trace": bands,
        "pooled_median_event_us_per_launch": pooled_median_us,
        "relative_latency_band_counts": relative_bands,
    }


def comparison_summary(modes):
    """Compare descriptive A/A variability without treating it as significance."""
    metrics = {}
    for name, capture in modes.items():
        benchmark = capture["benchmark"]
        diagnostics = capture["diagnostic_summary"]
        metrics[name] = {
            "median_paired_ratio": benchmark["paired_ratios"]["median"],
            "median_absolute_ratio_deviation": benchmark["aa_variability"][
                "median_absolute_ratio_deviation"
            ],
            "baseline_cv": benchmark["baseline_ms"]["cv"],
            "candidate_cv": benchmark["candidate_ms"]["cv"],
            "absolute_latency_band_counts": {
                band["label"]: band["count"]
                for band in diagnostics["latency_bands_from_prior_aa_trace"]
            },
            "relative_latency_band_counts": diagnostics["relative_latency_band_counts"],
        }
    python_deviation = metrics["python_launch"]["median_absolute_ratio_deviation"]
    graph_deviation = metrics["cuda_graph"]["median_absolute_ratio_deviation"]
    return {
        "mode_order": list(modes),
        "metrics": metrics,
        "graph_minus_python_median_absolute_ratio_deviation": (
            graph_deviation - python_deviation
        ),
        "observed_graph_aa_deviation": (
            "lower" if graph_deviation < python_deviation
            else "higher" if graph_deviation > python_deviation else "equal"
        ),
        "interpretation": (
            "Descriptive difference in this sequential capture only; mode order, "
            "GPU state, and different launch mechanisms can confound it."
        ),
    }


def main():
    import torch
    import triton
    from benchmark_proposal import launch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON output path; must not exist")
    parser.add_argument("--mode-order", choices=("python-first", "graph-first"),
                        default="python-first",
                        help="Execution order for the two A/A timing modes")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    output = args.output or Path("runs") / f"aa_modes_{started:%Y%m%dT%H%M%S%fZ}.json"
    if output.exists():
        parser.error(f"output already exists: {output}")

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(INPUT_SEED)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16, generator=generator)

    mode_order = (["python_launch", "cuda_graph"] if args.mode_order == "python-first"
                  else ["cuda_graph", "python_launch"])
    modes = {}
    with torch.inference_mode():
        for mode in mode_order:
            if mode == "python_launch":
                timer = DiagnosticTimer()
                result = run_aa_calibration(
                    launch, a, b, BASELINE, rounds=ROUNDS, warmups=WARMUPS,
                    iterations=ITERATIONS, seed=ORDER_SEED, timer=timer,
                )
            else:
                graphs = {}
                timer = GraphDiagnosticTimer(graphs)

                def prepare_graphs(outputs):
                    graphs.update(capture_graphs(
                        launch, a, b, BASELINE, outputs, ITERATIONS))

                result = run_aa_calibration(
                    launch, a, b, BASELINE, rounds=ROUNDS, warmups=WARMUPS,
                    iterations=ITERATIONS, seed=ORDER_SEED,
                    prepare=prepare_graphs, timer_by_side=timer,
                )
            attach_diagnostics(result, timer.blocks, mode)
            if mode == "cuda_graph":
                result["method"].update({
                    "graph_capture_launches_per_side": ITERATIONS,
                    "untimed_graph_warmup_replays_per_side": 1,
                    "timed_graph_replays_per_block": 1,
                    "per_round_warmups": "20 eager launches per side, as in Python mode",
                })
            modes[mode] = {
                "benchmark": result,
                "diagnostic_summary": diagnostic_summary(result),
            }

    comparison = comparison_summary(modes)

    captured = datetime.now(timezone.utc)
    output.parent.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(0)
    record = {
        "schema_version": 2,
        "started_utc": started.isoformat(),
        "captured_utc": captured.isoformat(),
        "workload": {"M": M, "N": N, "K": K, "dtype": "fp16"},
        "configuration_a": BASELINE,
        "configuration_b": BASELINE,
        "input_seed": INPUT_SEED,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
        },
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "modes": modes,
        "comparison": comparison,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")

    for name, capture in modes.items():
        benchmark = capture["benchmark"]
        diagnostics = capture["diagnostic_summary"]
        print(f"\n{name}: {len(benchmark['rounds'])} balanced A/A pairs")
        print(f"  median paired ratio: {benchmark['paired_ratios']['median']:.6f}x")
        print("  median absolute paired-ratio deviation from 1: "
              f"{benchmark['aa_variability']['median_absolute_ratio_deviation']:.3%}")
        print(f"  CV: A={benchmark['baseline_ms']['cv']:.3%}, "
              f"B={benchmark['candidate_ms']['cv']:.3%}")
        print("  submission/event correlation: "
              f"{diagnostics['submission_vs_event_all_blocks']}")
        print("  absolute latency bands (<61, 61-<66, >=66 us): "
              + ", ".join(str(band["count"])
                          for band in diagnostics["latency_bands_from_prior_aa_trace"]))
        print("  relative bands (<97%, within 3%, >103% of mode median): "
              + ", ".join(str(count) for count in
                          diagnostics["relative_latency_band_counts"].values()))
    print("\nGraph minus Python median absolute A/A deviation: "
          f"{comparison['graph_minus_python_median_absolute_ratio_deviation']:+.3%}")
    print("Observed graph A/A deviation in this capture: "
          f"{comparison['observed_graph_aa_deviation']}")
    print("This is a descriptive sequential comparison; no clock correction or "
          "significance claim is applied.")
    print(f"Saved separate mode traces and execution orders to {output}")


if __name__ == "__main__":
    main()
