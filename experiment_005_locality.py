"""Read-only Experiment 005 locality analysis from validated artifacts.

This module never starts a GPU process. A profile attempt is recorded separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from hilbert_matmul import tile_order

import experiment_005_evidence as evidence

RUN_ID = "20261030T013245142673Z"
SCREEN_ID = "20261030T013442437146Z"
SHAPE = "2048x3072x768"
METRICS = (
    "L2 cache hit rate", "L2 sector requests and traffic", "DRAM read bytes",
    "DRAM write bytes", "DRAM throughput", "Global-memory throughput",
    "Executed instructions", "Achieved occupancy", "Major warp-stall categories",
)


def summarize(root=evidence.ROOT, *, run_id=RUN_ID, screen_id=SCREEN_ID):
    """Return only observations bound to the saved verify and screening manifests."""
    _, verified = evidence.validate_verify(root, run_id)
    _, screening = evidence.validate_timing(root, run_id, "screening", screen_id)
    comparisons = {row["opponent"]: row for row in screening["comparisons"]
                   if row["shape"] == SHAPE}
    if set(comparisons) != {"row_major", "grouped_m4"}:
        raise ValueError("representative workload comparisons unavailable")
    compiler = verified["compiler"][SHAPE]
    grouped, hilbert = (compiler[name] for name in ("grouped_m4", "hilbert"))
    grid_m = (2048 + evidence.CONFIG["block_m"] - 1) // evidence.CONFIG["block_m"]
    grid_n = (3072 + evidence.CONFIG["block_n"] - 1) // evidence.CONFIG["block_n"]
    return {
        "schema": "experiment_005_locality_summary_v1", "run_id": run_id,
        "screen_id": screen_id, "shape_mnk": [2048, 3072, 768],
        "environment": verified["environment"],
        "verify_manifest_sha256": evidence.sha((Path(root) / evidence.SECTION / run_id / "verify/manifest.json").read_bytes()),
        "screen_manifest_sha256": evidence.sha((Path(root) / evidence.SECTION / run_id / "screening" / screen_id / "manifest.json").read_bytes()),
        "source_hashes": verified["source_hashes"],
        "wild_lab_accepted_sha256": verified["wild_lab_accepted_sha256"],
        "correctness": "PASS for all three strategies, two seeds and all four verified workloads",
        "screening": {opponent: {
            "median_paired_ratio_opponent_over_hilbert": row["summary"]["median_paired_ratio"],
            "paired_ratio_mad": row["summary"]["paired_ratio_mad"],
            "bootstrap_median_ratio_95pct": row["summary"]["bootstrap_median_ratio_95pct"],
            "rounds": row["summary"]["rounds"],
        } for opponent, row in comparisons.items()},
        "compiler": {schedule: {
            "executable_sections": report["executable_sections"],
            "sass_counts": report["sass_counts"],
            "registers_per_thread": report["registers_per_thread"],
            "shared_memory_bytes": report["shared_memory_bytes"],
            "compiler_reported_spills": report["spills"],
        } for schedule, report in (("grouped_m4", grouped), ("hilbert", hilbert))},
        "mapping": {"tile_grid": [grid_m, grid_n], "table_entries": grid_m * grid_n,
                    "table_bytes": grid_m * grid_n * 2 * 4,
                    "device_lookup": "Two scalar int32 tile-coordinate loads per Hilbert CTA; included in kernel timing.",
                    "kernel_timing_excludes_one_time_map_preparation": True,
                    "verify_setup_ms": json.loads((Path(root) / evidence.SECTION / run_id / "verify" / SHAPE / "hilbert_order.json").read_bytes())["setup_ms"],
                    "screen_setup_ms": comparisons["grouped_m4"]["one_time_map_setup_ms"]},
        "small_workload_setup_ms": {
            "verify": json.loads((Path(root) / evidence.SECTION / run_id / "verify/128x3072x768/hilbert_order.json").read_bytes())["setup_ms"],
            "screening": next(row["one_time_map_setup_ms"] for row in screening["comparisons"]
                              if row["shape"] == "128x3072x768")},
        "interpretation": {
            "screening": "No convincing Hilbert performance advantage; all relevant bootstrap intervals include 1 and screening is unconfirmed.",
            "instruction_overhead": "Hilbert has fewer static whole-kernel SASS instructions than grouped-M4 here. This does not measure executed instructions or isolate lookup cost.",
            "setup_outlier": "The first map setup is much larger in both saved stages than later setups. A first-use initialization effect is plausible but unverified; isolated GPU setup repeats were unavailable.",
            "cache_causation": "No cache counters are present in verification or screening. Cache-locality benefit and cause of near-parity remain unresolved.",
        },
    }


def profile_status(summary, *, version, attempt_command, exit_code, error):
    """Mark every requested counter unavailable when the profiler cannot connect."""
    if exit_code == 0:
        raise ValueError("successful profiles require metric parsing before interpretation")
    reason = "Nsight Compute could not connect to the CUDA driver; no kernel was profiled."
    if "failed to connect to the CUDA driver" not in error:
        reason = "Nsight Compute profile failed before metrics were validated."
    return {**summary, "profiling": {
        "status": "unavailable", "ncu_version": version,
        "attempted_strategy": "grouped_m4", "attempted_command": attempt_command,
        "attempt_exit_code": exit_code, "attempt_stderr": error,
        "hilbert_attempted": False,
        "metrics": {name: {"value": None, "unit": None, "status": "unavailable", "reason": reason}
                    for name in METRICS},
        "recommendation": "Stop active Hilbert tuning. The screening gives no convincing advantage; revisit one targeted counter comparison only if driver access is restored and a human approves it.",
        "limitations": ["nvidia-smi can enumerate the GPU, but PyTorch CUDA was unavailable in this session.",
                        "No L2, DRAM, instruction-execution, occupancy or stall counter was collected.",
                        "Static SASS counts and register allocations cannot substitute for runtime counters.",
                        "The saved eight-pair screening bootstrap interval is descriptive, not independent confirmation."]}}


def cpu_mapping_samples():
    """A small CPU-only isolation check; excludes allocation, transfer, and sync."""
    rows = {}
    for shape, grid in (("128x3072x768", (2, 24)), (SHAPE, (32, 24))):
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            order = tile_order("hilbert", *grid)
            samples.append((time.perf_counter() - start) * 1000)
            if len(order) != grid[0] * grid[1]:
                raise ValueError("CPU Hilbert map coverage changed")
        rows[shape] = {"samples_ms": samples, "scope": "CPU tile_order only; no CUDA allocation or transfer"}
    return rows


def save_diagnostic(folder, report, *, source_root=evidence.ROOT):
    """Seal a separate diagnostic folder without touching verify or screening."""
    folder = Path(folder)
    expected_parent = Path(source_root).resolve() / evidence.SECTION / RUN_ID / "locality"
    if (folder.parent.resolve() != expected_parent or folder.is_symlink()
            or not folder.is_dir() or (folder / "report.json").exists()
            or (folder / "manifest.json").exists()):
        raise ValueError("invalid or previously written locality diagnostic folder")
    (folder / "report.json").write_bytes((json.dumps(report, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n").encode())
    (folder / "manifest.json").write_bytes((json.dumps(evidence.manifest(
        folder, schema="experiment_005_locality_v1"), indent=2, sort_keys=True) + "\n").encode())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
