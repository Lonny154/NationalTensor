"""Benchmark-only tests of idle gaps and one untimed settling block per side.

Run from the project root: .venv/bin/python idle_gap_diagnostic.py --settling
The settling mode measures each idle/preparation combination twice while
retaining the normal eight-round paired Python-launch timing method.
No Muse client, optimization history, or confirmation decision is involved.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import time

from benchmark_core import run_aa_calibration, run_paired
import optimize
from smoke_aa import DiagnosticTimer, attach_diagnostics


CANDIDATE = dict(block_m=128, block_n=128, block_k=32,
                 num_warps=8, num_stages=3)
INPUT_SEED = 42
ORDER_SEED = 20261026
IDLE_SECONDS = 10
AA_ROUNDS = 20
AA_WARMUPS = 10
AA_ITERATIONS = 100
SCREEN_ROUNDS = 8
FIRST_BLOCK_LIMIT_US = 100


def condition_schedule(seed=ORDER_SEED):
    """Two matched condition pairs with the second pair in reverse order."""
    first = ["immediate", "idle"]
    if random.Random(seed).getrandbits(1):
        first.reverse()
    return [{"pair": pair, "condition": condition}
            for pair, order in ((1, first), (2, list(reversed(first))))
            for condition in order]


def settling_schedule(seed=ORDER_SEED):
    """Two reversed passes through all four idle/preparation combinations."""
    first = [("immediate", False), ("idle", False),
             ("idle", True), ("immediate", True)]
    if random.Random(seed).getrandbits(1):
        first.reverse()
    prep_first = ["baseline", "candidate"]
    if random.Random(seed + 1).getrandbits(1):
        prep_first.reverse()
    schedule = []
    for pair, order in ((1, first), (2, list(reversed(first)))):
        prep_order = prep_first if pair == 1 else list(reversed(prep_first))
        for condition, prepared in order:
            schedule.append({
                "pair": pair, "condition": condition, "settling": prepared,
                "preparation_order": list(prep_order),
                "screen_order_seed": seed + pair,
            })
    return schedule


def package_measurement(timing, blocks):
    """Attach host intervals in actual launch order without changing raw ratios."""
    rows = timing["paired_rounds"] if "paired_rounds" in timing else timing["rounds"]
    attach_diagnostics({"rounds": rows, "method": timing["method"]}, blocks)
    return {"benchmark": timing, "first_two_rounds": deepcopy(rows[:2])}


def analyze(comparisons, aa):
    """Descriptive results only; retain every timed block in the artifact."""
    conditions = {}
    for name in ("immediate", "idle"):
        matching = [entry for entry in comparisons if entry["condition"] == name]
        first_baseline_us = [
            entry["measurement"]["first_two_rounds"][0]["baseline_ms"] * 1000
            for entry in matching
        ]
        first_baseline_submission_ms = [
            entry["measurement"]["first_two_rounds"][0]["blocks"]["baseline"]
            ["host_submission_ms_per_block"] for entry in matching
        ]
        later_baseline_median_us = [
            statistics.median(row["baseline_ms"] * 1000 for row in
                              entry["measurement"]["benchmark"]["paired_rounds"][2:])
            for entry in matching
        ]
        paired = [entry["measurement"]["benchmark"]["median_paired_speedup"]
                  for entry in matching]
        conditions[name] = {
            "comparisons": len(matching),
            "first_baseline_us": first_baseline_us,
            "first_baseline_host_submission_ms_per_block": first_baseline_submission_ms,
            "later_baseline_median_us": later_baseline_median_us,
            "first_baseline_over_100us_count": sum(
                value > FIRST_BLOCK_LIMIT_US for value in first_baseline_us),
            "paired_speedups": paired,
            "median_paired_speedup_across_comparisons": (
                statistics.median(paired) if paired else None),
        }
    pairwise = []
    for pair in sorted({entry["pair"] for entry in comparisons}):
        entries = {entry["condition"]: entry for entry in comparisons
                   if entry["pair"] == pair}
        if set(entries) == {"immediate", "idle"}:
            pairwise.append({
                "pair": pair,
                "idle_minus_immediate_paired_speedup": (
                    entries["idle"]["measurement"]["benchmark"]["median_paired_speedup"]
                    - entries["immediate"]["measurement"]["benchmark"]
                    ["median_paired_speedup"]),
            })
    return {
        "conditions": conditions,
        "pairwise_differences": pairwise,
        "idle_minus_immediate_median_paired_speedup": (
            conditions["idle"]["median_paired_speedup_across_comparisons"]
            - conditions["immediate"]["median_paired_speedup_across_comparisons"]),
        "aa": {
            stage: {
                "median_paired_ratio": capture["benchmark"]["paired_ratios"]["median"],
                "median_absolute_ratio_deviation": capture["benchmark"][
                    "aa_variability"]["median_absolute_ratio_deviation"],
            }
            for stage, capture in aa.items()
        },
        "interpretation": (
            "Descriptive comparison of two repetitions per condition. "
            "No outliers were removed or timing corrections applied."
        ),
    }


def analyze_settling(comparisons, aa):
    """Compare all four cells without removing or correcting any block."""
    cells = {}
    for condition in ("immediate", "idle"):
        for prepared in (False, True):
            matching = [entry for entry in comparisons
                        if entry["condition"] == condition
                        and entry["settling"] == prepared]
            first_baseline = [entry["measurement"]["first_two_rounds"][0]
                              ["baseline_ms"] * 1000 for entry in matching]
            first_timed_block = [
                row[f"{row['order'][0]}_ms"] * 1000
                for entry in matching
                for row in entry["measurement"]["first_two_rounds"][:1]
            ]
            first_preparation = [
                entry["measurement"]["preparation"]["blocks"][0]
                ["timing"]["cuda_event_ms_per_launch"] * 1000
                for entry in matching if prepared
            ]
            preparation_by_side = {
                side: [next(block["timing"]["cuda_event_ms_per_launch"] * 1000
                            for block in entry["measurement"]["preparation"]["blocks"]
                            if block["side"] == side)
                       for entry in matching if prepared]
                for side in ("baseline", "candidate")
            }
            preparation_any_high = sum(
                any(block["timing"]["cuda_event_ms_per_launch"] * 1000
                    > FIRST_BLOCK_LIMIT_US
                    for block in entry["measurement"]["preparation"]["blocks"])
                for entry in matching if prepared
            )
            speedups = [entry["measurement"]["benchmark"]["paired_ratios"]
                        ["median"] for entry in matching]
            cells[f"{condition}_{'settled' if prepared else 'original'}"] = {
                "count": len(matching),
                "first_timed_block_us": first_timed_block,
                "first_timed_block_over_100us_count": sum(
                    value > FIRST_BLOCK_LIMIT_US for value in first_timed_block),
                "first_timed_baseline_us": first_baseline,
                "first_timed_baseline_over_100us_count": sum(
                    value > FIRST_BLOCK_LIMIT_US for value in first_baseline),
                "first_preparation_block_us": first_preparation,
                "first_preparation_block_over_100us_count": sum(
                    value > FIRST_BLOCK_LIMIT_US for value in first_preparation),
                "preparation_us_by_side": preparation_by_side,
                "preparation_any_block_over_100us_count": preparation_any_high,
                "median_paired_speedups": speedups,
                "median_of_paired_speedups": (
                    statistics.median(speedups) if speedups else None),
            }
    return {
        "cells": cells,
        "idle_settling_moved_anomaly_in_this_capture": (
            cells["idle_original"]["first_timed_block_over_100us_count"]
            == cells["idle_original"]["count"]
            and
            cells["idle_settled"]["preparation_any_block_over_100us_count"]
            == cells["idle_settled"]["count"]
            and cells["idle_settled"]["first_timed_block_over_100us_count"] == 0
            and cells["idle_settled"]["first_timed_baseline_over_100us_count"] == 0
        ),
        "aa": {
            stage: {
                "median_paired_ratio": capture["benchmark"]["paired_ratios"]["median"],
                "median_absolute_ratio_deviation": capture["benchmark"]
                ["aa_variability"]["median_absolute_ratio_deviation"],
            } for stage, capture in aa.items()
        },
        "interpretation": (
            "Descriptive comparisons, two repetitions per cell. All raw blocks "
            "are retained; no outlier removal or clock correction."
        ),
    }


def measure_aa(a, b, seed):
    timer = DiagnosticTimer()
    timing = run_aa_calibration(
        optimize.launch, a, b, optimize.BASELINE,
        rounds=AA_ROUNDS, warmups=AA_WARMUPS,
        iterations=AA_ITERATIONS, seed=seed, timer=timer)
    return package_measurement(timing, timer.blocks)


def measure_comparison(a, b, reference, base_out, seed):
    # Correctness immediately before each screening mirrors the pilot path.
    started = time.perf_counter()
    candidate_out = optimize.correct_output(a, b, reference, CANDIDATE)
    correctness_wall_s = time.perf_counter() - started
    timer = DiagnosticTimer()
    started = time.perf_counter()
    timing = optimize.compare(
        a, b, base_out, candidate_out, CANDIDATE,
        SCREEN_ROUNDS, seed, warmups=optimize.SCREEN_WARMUPS,
        iterations=optimize.SCREEN_ITERATIONS, timer=timer)
    screening_wall_s = time.perf_counter() - started
    return {"correctness": "PASS", "correctness_wall_s": correctness_wall_s,
            "screening_wall_s": screening_wall_s,
            **package_measurement(timing, timer.blocks)}


def run_settling_blocks(a, b, outputs, order, timer):
    """Run and record one untimed 100-launch block for each side."""
    if set(order) != {"baseline", "candidate"} or len(order) != 2:
        raise ValueError("preparation order must contain each side once")
    configs = {"baseline": optimize.BASELINE, "candidate": CANDIDATE}
    recorded = []
    for side in order:
        output = outputs[side]
        config = configs[side]
        timer(lambda: optimize.launch(a, b, output, config),
              optimize.SCREEN_ITERATIONS)
        recorded.append({"side": side, "timing": timer.blocks[-1]})
    return recorded


def measure_settling_comparison(a, b, reference, base_out, seed,
                                settled, preparation_order):
    """Use the same core screening path, with only prepare changing by cell."""
    started = time.perf_counter()
    candidate_out = optimize.correct_output(a, b, reference, CANDIDATE)
    correctness_wall_s = time.perf_counter() - started
    timer = DiagnosticTimer()
    preparation_timer = DiagnosticTimer()
    preparation_blocks = []

    def prepare(outputs):
        preparation_blocks.extend(run_settling_blocks(
            a, b, outputs, preparation_order, preparation_timer))

    outputs = iter((base_out, candidate_out))
    started = time.perf_counter()
    timing = run_paired(
        optimize.launch, a, b, optimize.BASELINE, CANDIDATE,
        rounds=SCREEN_ROUNDS, warmups=optimize.SCREEN_WARMUPS,
        iterations=optimize.SCREEN_ITERATIONS, seed=seed,
        output_factory=lambda _a, _b: next(outputs), timer=timer,
        prepare=prepare if settled else None)
    screening_wall_s = time.perf_counter() - started
    return {
        "correctness": "PASS", "correctness_wall_s": correctness_wall_s,
        "screening_wall_s": screening_wall_s,
        "preparation": {
            "enabled": settled, "order": list(preparation_order) if settled else [],
            "iterations_per_side": optimize.SCREEN_ITERATIONS if settled else 0,
            "blocks": preparation_blocks,
        },
        **package_measurement(timing, timer.blocks),
    }


def run(output, seed, *, settling=False):
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    schedule = settling_schedule(seed) if settling else condition_schedule(seed)
    gpu = torch.cuda.get_device_name(0)
    torch.manual_seed(INPUT_SEED)
    a = torch.randn((optimize.M, optimize.K), device="cuda", dtype=torch.float16)
    b = torch.randn((optimize.K, optimize.N), device="cuda", dtype=torch.float16)
    reference = torch.mm(a, b)

    # Both configurations compile and pass numerical validation before either
    # condition or A/A calibration begins.
    base_out = optimize.correct_output(a, b, reference, optimize.BASELINE)
    optimize.correct_output(a, b, reference, CANDIDATE)

    artifact = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "gpu": gpu,
        "environment": {"torch": torch.__version__, "triton": triton.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "compute_capability": torch.cuda.get_device_capability(0)},
        "workload": optimize.run_identity(gpu),
        "candidate": CANDIDATE,
        "diagnostic_mode": "settling_factorial" if settling else "original_idle_gap",
        "input_seed": INPUT_SEED,
        "condition_order_seed": seed,
        "condition_schedule": schedule,
        "screen_order_seeds": ([item["screen_order_seed"] for item in schedule]
                               if settling else [seed + item["pair"] for item in schedule]),
        "idle_seconds_requested": IDLE_SECONDS,
        "method": {
            "idle_position": (
                "Before candidate correctness validation and benchmark warmups, "
                "matching the pilot's post-API sequence"
            ),
            "screening": {"rounds": SCREEN_ROUNDS,
                          "warmups_per_side_per_round": optimize.SCREEN_WARMUPS,
                          "iterations_per_side_per_round": optimize.SCREEN_ITERATIONS},
            "aa": {"rounds": AA_ROUNDS, "warmups_per_side_per_round": AA_WARMUPS,
                   "iterations_per_side_per_round": AA_ITERATIONS},
            "event": "CUDA events around Python launch loop; block duration divided by launches",
            "host_submission": "perf_counter_ns around Python launch loop, excluding synchronization",
            "host_wall": "perf_counter_ns before start-event record through end-event synchronization",
            "first_block_limit_us": FIRST_BLOCK_LIMIT_US,
            "no_clock_correction": True,
            "settling": (
                "When enabled, one 100-launch block per side runs via "
                "run_paired.prepare after initial warmups. Its duration is "
                "recorded diagnostically and excluded from the unchanged "
                "eight-round speedup statistic"
            ) if settling else None,
        },
        "precompiled_and_validated": True,
        "aa": {},
        "comparisons": [],
    }
    try:
        artifact["aa"]["before"] = measure_aa(a, b, seed + 100)
        for index, item in enumerate(schedule):
            idle_start = time.perf_counter()
            if item["condition"] == "idle":
                time.sleep(IDLE_SECONDS)
            actual_idle_s = time.perf_counter() - idle_start
            if settling:
                measurement = measure_settling_comparison(
                    a, b, reference, base_out,
                    artifact["screen_order_seeds"][index],
                    item["settling"], item["preparation_order"])
            else:
                measurement = measure_comparison(
                    a, b, reference, base_out,
                    artifact["screen_order_seeds"][index])
            artifact["comparisons"].append({
                **item, "position": index + 1,
                "actual_prebenchmark_host_wait_s": actual_idle_s,
                "screen_order_seed": artifact["screen_order_seeds"][index],
                "measurement": measurement,
            })
        artifact["aa"]["after"] = measure_aa(a, b, seed + 101)
        artifact["analysis"] = (
            analyze_settling(artifact["comparisons"], artifact["aa"])
            if settling else analyze(artifact["comparisons"], artifact["aa"]))
        artifact["status"] = "complete"
    except BaseException as exc:
        artifact["status"] = "incomplete"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as file:
            json.dump(artifact, file, indent=2)
            file.write("\n")
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON artifact path")
    parser.add_argument("--seed", type=int, default=ORDER_SEED,
                        help="Condition and paired-round order seed")
    parser.add_argument("--settling", action="store_true",
                        help="Test immediate/idle with and without one untimed block per side")
    args = parser.parse_args(argv)
    output = args.output or Path("evaluations/idle-gap") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    artifact = run(output, args.seed, settling=args.settling)
    print(f"Saved raw diagnostic: {output}")
    if args.settling:
        for name, result in artifact["analysis"]["cells"].items():
            print(f"{name}: first timed block >100 us in "
                  f"{result['first_timed_block_over_100us_count']}/{result['count']}; "
                  f"first baseline us {result['first_timed_baseline_us']}; "
                  f"first preparation block us {result['first_preparation_block_us']}; "
                  f"paired speedups {result['median_paired_speedups']}")
        print("Idle anomaly moved into preparation in this capture: "
              f"{artifact['analysis']['idle_settling_moved_anomaly_in_this_capture']}")
    else:
        for name, result in artifact["analysis"]["conditions"].items():
            print(f"{name}: first baseline >100 us in "
                  f"{result['first_baseline_over_100us_count']}/{result['comparisons']}; "
                  f"first baseline us {result['first_baseline_us']}; "
                  f"paired speedups {result['paired_speedups']}")
    print("A/A median ratios: " + ", ".join(
        f"{stage} {result['median_paired_ratio']:.6f}x"
        for stage, result in artifact["analysis"]["aa"].items()))
    print("Descriptive only; raw outliers retained and no timing correction applied.")


if __name__ == "__main__":
    main()
