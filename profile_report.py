"""CPU-only validation and reporting for profiling experiment 001.

Raw Nsight Compute CSVs are evidence, never latency measurements. This module
does not import Torch or launch kernels.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shlex

import planned_profile as protocol


RUN_ID = "20261028T004249599584Z"
MANIFEST_SHA = "4dcd96e93e64b25a2550fd2b43c5d0620d7e0483857dd4816dc3e0244e436120"
SECTION = "evaluations/planned-profiling"
MAX_RAW_BYTES = 100_000
EXTRA_METRICS = {
    "sm__maximum_warps_per_active_cycle_pct": "%",
    "launch__shared_mem_per_block_dynamic": "byte/block",
    "launch__shared_mem_per_block_driver": "byte/block",
    "profiler__replayer_passes": "pass",
}


def _bounded(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RAW_BYTES:
        raise ValueError(f"missing, linked, or oversized artifact: {path.name}")
    return path.read_bytes()


def _number(raw: str, name: str) -> float:
    try:
        value = float(raw.replace(",", ""))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"missing or invalid profiler metric: {name}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid profiler metric: {name}")
    return value


def _option(args: list[str], name: str) -> str:
    if args.count(name) != 1:
        raise ValueError(f"missing or repeated profiler option: {name}")
    index = args.index(name)
    if index + 1 >= len(args):
        raise ValueError(f"missing profiler option value: {name}")
    return args[index + 1]


def validate_capture(folder: Path, manifest: dict, item: dict, root: Path = protocol.ROOT) -> dict:
    """Require exactly one expected GEMM record and complete named counters."""
    key = item["kernel"]
    stem = f"{key}-{item['capture']}"
    if (key not in protocol.KERNELS or item["capture"] not in (1, 2)
            or item["raw_csv"] != stem + ".csv"
            or item["command_log"] != stem + ".log"):
        raise ValueError("capture identity differs from frozen manifest")
    csv_path = folder / item["raw_csv"]
    log_path = folder / item["command_log"]
    raw_csv = _bounded(csv_path)
    raw_log = _bounded(log_path)
    log = raw_log.decode("utf-8")
    lines = log.splitlines()
    if len(lines) != 2:
        raise ValueError("capture log has unexpected lines or errors")
    args = shlex.split(lines[0])
    if (not args or args[0] != manifest["profiler"]["executable"]
            or _option(args, "--target-processes") != "application-only"
            or _option(args, "--profile-from-start") != "off"
            or _option(args, "--launch-count") != "1"
            or _option(args, "--cache-control") != "none"
            or _option(args, "--clock-control") != "none"
            or _option(args, "--kernel-name") != "regex:.*matmul_kernel.*"
            or _option(args, "--metrics").split(",") != manifest["profiler"]["metrics"]
            or _option(args, "--manifest-sha256") != MANIFEST_SHA
            or Path(_option(args, "--manifest")).resolve()
                != (protocol.OUTPUT_BASE / RUN_ID / "manifest.json").resolve()
            or _option(args, "--kernel") != key
            or args.count("--worker") != 1 or args.count("--csv") != 1
            or _option(args, "--page") != "raw"):
        raise ValueError("capture command does not match frozen protocol")
    marker = json.loads(lines[1])
    if (marker.get("marker") != "PROFILE_TARGET" or marker.get("kernel") != key
            or marker.get("shape_mnk") != protocol.SHAPE or marker.get("seed") != 42
            or marker.get("correctness") != "PASS"):
        raise ValueError("capture correctness or selection marker mismatch")
    text = raw_csv.decode("utf-8")
    warnings = [line for line in text.splitlines() if line.startswith("==WARNING==")]
    if (text.count("==PROF== Connected") != 1
            or text.count("==PROF== Disconnected") != 1
            or any(line.startswith("==ERROR==") for line in text.splitlines())):
        raise ValueError("profiler process record is missing or failed")
    rows = list(csv.reader(io.StringIO("\n".join(
        line for line in text.splitlines() if not line.startswith("==")))))
    if len(rows) != 3 or len({len(row) for row in rows}) != 1:
        raise ValueError("expected one Nsight Compute GEMM record plus header and units")
    header, unit_row, values = rows
    if len(set(header)) != len(header):
        raise ValueError("duplicate Nsight Compute metric name")
    data, units = dict(zip(header, values)), dict(zip(header, unit_row))
    if (data.get("ID") != "0" or data.get("Kernel Name") != "matmul_kernel"
            or data.get("Block Size") != "(128, 1, 1)"
            or data.get("Grid Size") != "(8, 24, 1)"
            or data.get("CC") != "8.9"
            or data.get("device__attribute_display_name") != protocol.GPU):
        raise ValueError("profiled GEMM identity, hardware, or launch geometry mismatch")
    selected = {}
    for name in manifest["profiler"]["metrics"]:
        expected_unit = manifest["profiler"]["metric_units"].get(name)
        if name not in data or units.get(name) != expected_unit:
            raise ValueError(f"missing metric or mismatched unit: {name}")
        selected[name] = {"value": _number(data[name], name), "unit": units[name]}
    extra = {}
    for name, expected_unit in EXTRA_METRICS.items():
        if name not in data or units.get(name) != expected_unit:
            raise ValueError(f"missing required auxiliary metric: {name}")
        extra[name] = {"value": _number(data[name], name), "unit": units[name]}
    if extra["profiler__replayer_passes"]["value"] < 1:
        raise ValueError("profiler replay count is invalid")
    return {"kernel": key, "capture": item["capture"], "status": "verified",
            "correctness": "PASS on FC1 seed 42 before profiling",
            "measurement_method": "Nsight Compute kernel replay; one selected GEMM",
            "artifacts": {"csv": {"path": str(csv_path.relative_to(root)),
                                   "sha256": protocol.sha(raw_csv)},
                          "log": {"path": str(log_path.relative_to(root)),
                                  "sha256": protocol.sha(raw_log)}},
            "worker_environment": {"torch": marker.get("torch"),
                                   "triton": marker.get("triton")},
            "launch": {"kernel_name": data["Kernel Name"], "grid": data["Grid Size"],
                       "block": data["Block Size"], "gpu": protocol.GPU,
                       "compute_capability": data["CC"]},
            "profiler_warnings": warnings, "metrics": selected,
            "auxiliary_metrics": extra}


def analyze(folder: Path = protocol.OUTPUT_BASE / RUN_ID, *, root: Path = protocol.ROOT) -> dict:
    folder = Path(folder).resolve()
    root = Path(root).resolve()
    if ((root / "evaluations").is_symlink() or (root / SECTION).is_symlink()
            or folder.name != RUN_ID or not folder.is_relative_to((root / SECTION).resolve())):
        raise ValueError("unexpected profiling experiment path")
    manifest_path = folder / "manifest.json"
    manifest_raw = _bounded(manifest_path)
    if protocol.sha(manifest_raw) != MANIFEST_SHA:
        raise ValueError("frozen profiling manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    verified = protocol.verify_inputs(root)
    if (manifest.get("status") != "prepared_only"
            or manifest.get("plan_run_id") != protocol.PLAN_RUN
            or manifest.get("plan_sha256") != protocol.PLAN_SHA
            or manifest.get("runner_sha256") != protocol.sha((root / "planned_profile.py").read_bytes())
            or manifest.get("source_identities") != verified["identities"]
            or manifest.get("workload", {}).get("shape_mnk") != protocol.SHAPE
            or manifest.get("workload", {}).get("gpu") != protocol.GPU
            or manifest.get("workload", {}).get("numeric_config") != protocol.CONFIG
            or manifest.get("workload", {}).get("input_seed") != 42):
        raise ValueError("manifest plan, workload, or source provenance mismatch")
    captures = manifest.get("captures")
    expected_order = [(key, number) for number in (1, 2) for key in protocol.KERNELS]
    if (not isinstance(captures, list) or len(captures) != 6
            or [(row.get("kernel"), row.get("capture")) for row in captures] != expected_order):
        raise ValueError("six-capture order or identity mismatch")
    rows = [validate_capture(folder, manifest, item, root) for item in captures]
    by_kernel = {key: [row for row in rows if row["kernel"] == key]
                 for key in protocol.KERNELS}
    summary = {}
    for key, pair in by_kernel.items():
        summary[key] = {}
        for name in manifest["profiler"]["metrics"] + list(EXTRA_METRICS):
            source = "metrics" if name in manifest["profiler"]["metrics"] else "auxiliary_metrics"
            values = [row[source][name]["value"] for row in pair]
            summary[key][name] = {"captures": values, "min": min(values),
                                  "max": max(values), "unit": pair[0][source][name]["unit"]}
    return {
        "schema_version": 1, "status": "profiling_complete_verified",
        "study_id": f"profile:{RUN_ID}", "experiment_kind": "profiling_study",
        "plan_run_id": protocol.PLAN_RUN, "plan_sha256": protocol.PLAN_SHA,
        "plan_path": manifest["plan_path"], "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": MANIFEST_SHA,
        "runner_sha256": manifest["runner_sha256"],
        "source_identities": manifest["source_identities"],
        "baseline_launcher_sha256": manifest["baseline_launcher_sha256"],
        "workload": manifest["workload"], "capture_order": manifest["capture_order"],
        "profiler_version": manifest["profiler"]["version"],
        "metric_definitions": manifest["metrics"],
        "method": "Nsight Compute kernel replay; profiler start/stop around one warmed-up GEMM per separate process",
        "captures": rows, "summary": summary,
        "observations": [
            "Two-slab allocated 49,152 bytes dynamic shared memory versus 24,576 for baseline and mask hoisting; the shared-memory resident-block limit fell from four to two and theoretical occupancy from 33.33% to 16.67%.",
            "Achieved occupancy was 15.86-15.88% for two-slab, 26.22-26.30% for baseline, and 26.20-26.24% for mask hoisting.",
            "Selected mask-hoisting and baseline profiler counters are broadly similar; DRAM throughput varies materially between repeated baseline captures.",
        ],
        "interpretation": "The profiling data support a lower occupancy ceiling for this two-slab implementation. They do not establish a cause of the earlier CUDA-event screening latency difference.",
        "speculation": None, "confirmed_speedup": False,
        "limitations": [
            "Two captures per kernel are descriptive, not statistical proof.",
            "Nsight Compute replay used uncontrolled caches and unmodified clocks; counter collection may perturb execution.",
            "Profiler counters are separate from paired Python-launch CUDA-event latency measurements.",
            "Selected stall counters do not constitute a complete bottleneck analysis.",
            "The accepted Muse plan is advisory; this study profiles reviewed kernels, not generated source.",
        ],
    }


def markdown(report: dict) -> str:
    summary = report["summary"]
    def pair(kernel, name, digits=2):
        return "/".join(f"{value:.{digits}f}" for value in summary[kernel][name]["captures"])
    occ = "sm__warps_active.avg.pct_of_peak_sustained_active"
    ceiling = "sm__maximum_warps_per_active_cycle_pct"
    regs = "launch__registers_per_thread"
    shared = "launch__shared_mem_per_block_dynamic"
    reg_limit = "launch__occupancy_limit_registers"
    smem_limit = "launch__occupancy_limit_shared_mem"
    lines = ["# AI-planned profiling experiment 001", "",
        "**Status:** verified profiler observations; no speedup confirmation.", "",
        f"Plan: `{report['plan_path']}` (`{report['plan_sha256']}`).  ",
        f"Manifest: `{report['manifest_path']}` (`{report['manifest_sha256']}`).", "",
        "Workload: FP16 GEMM 512×3072×768, RTX 4060 Ti, seed 42; 64×128×32, four warps, three stages.", "",
        "| Reviewed kernel | Achieved occupancy % (captures 1/2) | Theoretical ceiling % | Registers/thread | Dynamic shared memory bytes/block | Register limit blocks/SM | Shared-memory limit blocks/SM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("baseline", "mask_hoisting", "two_slab"):
        lines.append(f"| {key} | {pair(key, occ)} | {pair(key, ceiling)} | {pair(key, regs, 0)} | {pair(key, shared, 0)} | {pair(key, reg_limit, 0)} | {pair(key, smem_limit, 0)} |")
    lines += ["", "Selected counters (capture 1/2; stall values are average stalled warps per active cycle, tensor activity is percent of sustained peak):", "",
        "| Kernel | Long scoreboard | Short scoreboard | Barrier | Tensor pipe % | DRAM GB/s | L2 GB/s | L1 GB/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    metrics = ["smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active",
               "smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active",
               "smsp__warps_issue_stalled_barrier.avg.per_cycle_active",
               "smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active"]
    for key in ("baseline", "mask_hoisting", "two_slab"):
        parts = [pair(key, name) for name in metrics]
        for name in ("dram__bytes.sum.per_second", "lts__t_bytes.sum.per_second",
                     "l1tex__t_bytes.sum.per_second"):
            parts.append("/".join(f"{value / 1e9:.1f}" for value in summary[key][name]["captures"]))
        lines.append(f"| {key} | " + " | ".join(parts) + " |")
    lines += ["", "Two-slab's higher shared-memory allocation lowers the resident-block limit and theoretical occupancy ceiling. Its achieved occupancy was also lower. Baseline and mask hoisting have similar selected counters. DRAM throughput varies across the repeats, especially baseline; two captures do not establish a distribution.", "",
        "The selected counters do not establish why the earlier two-slab screening was slower. Nsight replay with uncontrolled caches and clocks is a different method from paired CUDA-event latency; stall selections are not a complete bottleneck analysis.", "",
        "Each capture has one verified `matmul_kernel` row, a correctness-PASS marker, raw CSV and log hashes in `report.json`. Generated Muse kernels were not profiled. The source identities are in `report.json`.", ""]
    return "\n".join(lines)


def write_report(folder: Path = protocol.OUTPUT_BASE / RUN_ID) -> tuple[Path, Path]:
    report = analyze(folder)
    folder = Path(folder).resolve()
    report_path, markdown_path, metadata_path = (folder / "report.json", folder / "summary.md",
                                                folder / "report_metadata.json")
    if any(path.exists() for path in (report_path, markdown_path, metadata_path)):
        raise FileExistsError("profiling report already exists; raw evidence and reports are immutable")
    report_raw = protocol.json_bytes(report)
    markdown_raw = markdown(report).encode("utf-8")
    metadata = {"status": "profiling_complete_verified", "created_utc": datetime.now(timezone.utc).isoformat(),
                "report_sha256": protocol.sha(report_raw), "summary_sha256": protocol.sha(markdown_raw),
                "manifest_sha256": MANIFEST_SHA, "plan_sha256": protocol.PLAN_SHA}
    report_path.write_bytes(report_raw)
    markdown_path.write_bytes(markdown_raw)
    metadata_path.write_bytes(protocol.json_bytes(metadata))
    return report_path, markdown_path


def load_verified_report(root: Path = protocol.ROOT) -> tuple[dict | None, list[str]]:
    root = Path(root).resolve()
    if (root / "evaluations").is_symlink() or (root / SECTION).is_symlink():
        return None, ["Profiling artifact directory is a symlink"]
    folder = root / SECTION / RUN_ID
    if not folder.is_dir() or folder.is_symlink():
        return None, []
    try:
        metadata = json.loads(_bounded(folder / "report_metadata.json"))
        report_raw = _bounded(folder / "report.json")
        summary_raw = _bounded(folder / "summary.md")
        if (metadata.get("status") != "profiling_complete_verified"
                or metadata.get("manifest_sha256") != MANIFEST_SHA
                or metadata.get("plan_sha256") != protocol.PLAN_SHA
                or metadata.get("report_sha256") != protocol.sha(report_raw)
                or metadata.get("summary_sha256") != protocol.sha(summary_raw)):
            raise ValueError("profiling report hash or provenance mismatch")
        saved = json.loads(report_raw)
        if saved != analyze(folder, root=root):
            raise ValueError("profiling report differs from verified raw captures")
        return saved, []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, [f"Profiling report unavailable or invalid: {type(exc).__name__}: {str(exc)[:140]}"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate saved profiler captures without GPU work")
    parser.add_argument("--write-report", action="store_true", required=True)
    args = parser.parse_args(argv)
    report, summary = write_report()
    print(f"Saved verified profiling report: {report}\nSummary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
