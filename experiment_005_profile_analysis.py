"""Parse the bounded Experiment 005 Nsight Compute locality reports, without GPU work."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import experiment_005_locality as locality

FULL_METRICS = (
    "lts__t_sector_hit_rate.pct", "lts__t_sectors.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum", "dram__bytes_read.sum",
    "dram__bytes_write.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "launch__registers_per_thread", "launch__shared_mem_per_block_static",
    "gpu__time_duration.sum", "profiler__replayer_passes",
)
RAW_METRICS = (
    "lts__t_sectors_lookup_hit.sum", "lts__t_sectors.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum", "dram__bytes_read.sum",
    "dram__bytes_write.sum", "gpu__time_duration.sum",
    "profiler__replayer_passes",
)


def parse_ncu_csv(path, required):
    """One filtered kernel, names and units retained exactly as exported by ncu."""
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
        raise ValueError("missing or oversized Nsight Compute CSV")
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 3 or len(set(rows[0])) != len(rows[0]) or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Nsight Compute CSV must contain one complete kernel")
    headers, units, values = rows
    columns = {name: (units[index], values[index]) for index, name in enumerate(headers)}
    if (columns.get("Kernel Name", (None, None))[1] != "matmul_kernel"
            or columns.get("Grid Size", (None, None))[1] != "(768, 1, 1)"
            or columns.get("Block Size", (None, None))[1] != "(128, 1, 1)"):
        raise ValueError("profiled kernel or launch geometry mismatch")
    metrics = {}
    for name in required:
        unit, raw = columns.get(name, (None, None))
        if unit is None or raw is None or raw == "":
            metrics[name] = {"value": None, "unit": unit, "status": "unavailable"}
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"non-numeric Nsight metric {name}") from exc
        if not math.isfinite(value):
            raise ValueError(f"nonfinite Nsight metric {name}")
        metrics[name] = {"value": value, "unit": unit, "status": "collected"}
    return {"kernel": columns["Kernel Name"][1],
            "grid": columns["Grid Size"][1], "block": columns["Block Size"][1],
            "device": columns.get("Device", (None, None))[1],
            "compute_capability": columns.get("CC", (None, None))[1],
            "metrics": metrics}


def _value(profile, metric):
    item = profile["metrics"][metric]
    if item["status"] != "collected":
        raise ValueError(f"required metric unavailable: {metric}")
    return item["value"]


def compare_profiles(folder):
    """Preserve raw metrics and avoid interpreting impossible derived percentages."""
    folder = Path(folder)
    profiles = {}
    for strategy in ("grouped_m4", "hilbert"):
        profiles[strategy] = {
            "full": parse_ncu_csv(folder / f"{strategy}.metrics.csv", FULL_METRICS),
            "l2check": parse_ncu_csv(folder / f"{strategy}_l2check.metrics.csv", RAW_METRICS),
        }
    computed = {}
    for strategy, pair in profiles.items():
        full, raw = pair["full"], pair["l2check"]
        hits = _value(raw, "lts__t_sectors_lookup_hit.sum")
        total = _value(raw, "lts__t_sectors.sum")
        if total <= 0 or hits < 0 or hits > total:
            raise ValueError("raw L2 hit and total counters contradict each other")
        derived = _value(full, "lts__t_sector_hit_rate.pct")
        computed[strategy] = {
            "raw_l2_hit_fraction": hits / total,
            "raw_l2_hit_percent": 100 * hits / total,
            "derived_l2_hit_percent": derived,
            "derived_l2_hit_percent_valid": 0 <= derived <= 100,
            "l2_total_sectors": total,
            "l2_read_sectors_from_l1tex": _value(raw, "lts__t_sectors_srcunit_tex_op_read.sum"),
            "profiler_replay_passes_full": _value(full, "profiler__replayer_passes"),
            "profiler_replay_passes_l2check": _value(raw, "profiler__replayer_passes"),
        }
    grouped, hilbert = computed["grouped_m4"], computed["hilbert"]
    return {"profiles": profiles, "derived": computed,
            "comparison": {
                "raw_l2_hit_percentage_point_difference_hilbert_minus_grouped":
                    hilbert["raw_l2_hit_percent"] - grouped["raw_l2_hit_percent"],
                "l2_sector_difference_hilbert_minus_grouped":
                    hilbert["l2_total_sectors"] - grouped["l2_total_sectors"],
                "l2_read_sector_difference_hilbert_minus_grouped":
                    hilbert["l2_read_sectors_from_l1tex"] - grouped["l2_read_sectors_from_l1tex"],
                "cache_locality_observation":
                    "No observed Hilbert L2 advantage in this bounded pair; grouped-M4 has a slightly higher raw hit fraction and fewer sectors.",
                "measurement_limits": [
                    "Each strategy was profiled once per metric set; Nsight used kernel replay passes.",
                    "A derived L2 hit percentage above 100% is invalid and excluded from inference.",
                    "DRAM read/write totals changed across profiling passes, so isolated byte differences are not a stable scheduling conclusion.",
                    "Profiler kernel duration is not a replacement for paired CUDA-event screening.",
                    "Counter correlation does not establish a causal mechanism or physical CTA execution order.",
                ]}}


def report(folder):
    return {"schema": "experiment_005_locality_profile_v1",
            "validated_source": locality.summarize(),
            "profiling": compare_profiles(folder),
            "recommendation": "Close the Hilbert traversal direction for the tested workload and hardware: screening has no convincing advantage and this bounded profile shows no L2 locality gain. No confirmation or tuning is warranted.",
            "profile_files": [f"{strategy}{suffix}" for strategy in ("grouped_m4", "hilbert")
                              for suffix in (".ncu-rep", ".metrics.csv", "_l2check.ncu-rep", "_l2check.metrics.csv")],
            "limitations": ["Only one profiled launch per strategy and metric set; replay and cache state may affect counters.",
                            "The recorded setup outlier remains unexplained by this profiling probe."]}
