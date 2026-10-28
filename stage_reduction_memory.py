"""Read the saved Experiment 002 screening without importing or running its kernel."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


RUN = "20261028T140056186673Z"
PLAN = "evaluations/experiment-planner/20261028T010924452160Z/plan.json"
PRIOR = "evaluations/trusted-two-slab/20261027T161358957319Z/check.json"
ARTIFACT = f"evaluations/stage-reduction/{RUN}/result.json"
MAX_BYTES = 4_000_000
METRICS = (
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread", "launch__registers_per_thread_allocated",
    "launch__shared_mem_per_block", "launch__shared_mem_per_block_allocated",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


def _read(root, relative):
    path = root / relative
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
            or path.stat().st_size > MAX_BYTES):
        raise ValueError(f"missing, linked, or oversized artifact: {relative}")
    return path.read_bytes()


def _json(root, relative):
    value = json.loads(_read(root, relative))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {relative}")
    return value


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _paired(record, *, mode, minimum):
    if (record.get("mode") != mode or record.get("method", {}).get("clock") != "cuda_event"
            or record["method"].get("balanced_order") is not True):
        raise ValueError("Experiment 002 timing method changed")
    rows = record.get("rounds")
    if not isinstance(rows, list) or len(rows) < minimum or len(rows) % 2:
        raise ValueError("Experiment 002 timing rounds are incomplete")
    ratios = []
    for row in rows:
        if (set(row.get("order", [])) != {"baseline", "candidate"}
                or any(not _number(row.get(key)) or row[key] <= 0 for key in
                       ("baseline_ms", "candidate_ms", "baseline_over_candidate"))
                or not math.isclose(row["baseline_ms"] / row["candidate_ms"],
                                    row["baseline_over_candidate"], rel_tol=1e-10)):
            raise ValueError("Experiment 002 paired timing row is invalid")
        ratios.append(row["baseline_over_candidate"])
    if not math.isclose(statistics.median(ratios), record["paired_ratios"]["median"],
                        rel_tol=1e-10):
        raise ValueError("Experiment 002 paired median changed")
    return record


def load(root):
    """Return one verified memory entry and warnings; absent evidence is optional."""
    root = Path(root).resolve()
    if not (root / ARTIFACT).exists():
        return None, []
    try:
        result = _json(root, ARTIFACT)
        folder = f"evaluations/stage-reduction/{RUN}/"
        profile = _json(root, folder + "profile_summary.json")
        prior = _json(root, PRIOR)
        plan = _json(root, PLAN)
        identity = result["source_identity"]
        if (result["status"] != "screening_complete_unconfirmed"
                or result["experiment"] != "002_pipeline_stage_reduction"
                or result["plan_path"] != PLAN
                or result["plan_sha256"] != _sha(_read(root, PLAN))
                or result["prior_check_path"] != PRIOR
                or result["prior_check_sha256"] != _sha(_read(root, PRIOR))
                or result["runner_sha256"] != _sha(_read(root, "stage_reduction_experiment.py"))
                or identity["trusted_module_sha256"] != _sha(_read(root, "trusted_two_slab_kernel.py"))
                or identity["trusted_implementation_sha256"] != prior["trusted_implementation_sha256"]
                or identity["trusted_module_sha256"] != prior["trusted_module_sha256"]
                or identity["original_muse_candidate_sha256"] != prior["original_muse_candidate_sha256"]
                or prior["status"] != "correctness_and_compiler_captured"
                or plan["experiment_kind"] != "numeric_tuning"
                or plan["target_shape_mnk"] != result["workload"]["shape_mnk"]):
            raise ValueError("Experiment 002 source, plan, or prior-check identity mismatch")
        configs = result["configurations"]
        old, new = configs["stages_3"], configs["stages_2"]
        if (old != prior["numeric_config"] or new != {**old, "num_stages": 2}
                or old["num_stages"] != 3 or result["workload"]["dtype"] != "float16"
                or result["workload"]["seed"] != 42):
            raise ValueError("Experiment 002 configuration or workload changed")
        checks = result["correctness"]
        expected = {(name, tuple(shape), seed) for name in configs
                    for shape in prior["correctness_shapes_mnk"]
                    for seed in prior["correctness_seeds"]}
        observed = {(row["configuration"], tuple(row["shape_mnk"]), row["seed"])
                    for row in checks if row.get("status") == "PASS"
                    and row.get("reference_precision", {}).get("policy")
                    == "fp16_torch_mm_without_reduced_precision_reduction"}
        if len(checks) != 12 or observed != expected:
            raise ValueError("Experiment 002 correctness checks are incomplete")
        comparison = _paired(result["comparison"], mode="comparison", minimum=8)
        aa3 = _paired(result["aa_stages_3"], mode="aa", minimum=8)
        aa2 = _paired(result["aa_stages_2"], mode="aa", minimum=8)
        if (comparison["method"]["iterations_per_side_per_round"] != 100
                or comparison["method"]["warmups_per_side_per_round"] != 10):
            raise ValueError("Experiment 002 measurement protocol changed")
        compiler = {}
        for name in configs:
            capture = result["compiler"][name]
            for key in ("registers_per_thread", "spills", "shared_memory_bytes"):
                if type(capture[key]) is not int or capture[key] < 0:
                    raise ValueError("Experiment 002 compiler resources are incomplete")
            for kind, artifact in capture["artifacts"].items():
                if kind not in {"ttir", "ttgir", "llir", "ptx", "sass", "cubin"}:
                    raise ValueError("unexpected Experiment 002 compiler artifact")
                filename = f"{name}.{kind}"
                if artifact["path"] != filename or artifact["sha256"] != _sha(_read(root, folder + filename)):
                    raise ValueError("Experiment 002 compiler artifact hash mismatch")
            compiler[name] = {key: capture[key] for key in
                              ("registers_per_thread", "spills", "shared_memory_bytes")}
        if profile["status"] != "descriptive_profile_complete":
            raise ValueError("Experiment 002 profiler status changed")
        profiles = {}
        for name in configs:
            saved = profile["profiles"][name]
            if (saved["selected_launches"] != 1 or saved["kernel_name"] != "matmul_kernel"
                    or saved["csv"] != f"{name}.ncu.csv"
                    or saved["log"] != f"{name}.ncu.log"):
                raise ValueError("Experiment 002 profiler capture identity changed")
            raw = _read(root, folder + saved["csv"])
            if (saved["csv_sha256"] != _sha(raw)
                    or saved["log_sha256"] != _sha(_read(root, folder + saved["log"]))):
                raise ValueError("Experiment 002 profiler capture hash mismatch")
            rows = list(csv.reader(raw.decode().splitlines()))
            header = next(row for row in rows if row and row[0] == "ID")
            data = [dict(zip(header, row)) for row in rows if row and row[0].isdigit()]
            if len(data) != 1 or data[0]["Kernel Name"] != "matmul_kernel":
                raise ValueError("Experiment 002 profiler selected more than one launch")
            for metric in METRICS:
                if not math.isclose(float(data[0][metric]), saved["metrics"][metric], rel_tol=1e-10):
                    raise ValueError("Experiment 002 profiler metric mismatch")
            profiles[name] = saved["metrics"]
        latency = {name: comparison[key]["median"] for name, key in
                   (("stages_3", "baseline_ms"), ("stages_2", "candidate_ms"))}
        ratio = comparison["paired_ratios"]["median"]
        if (not latency["stages_2"] > latency["stages_3"] or not ratio < 1
                or not compiler["stages_2"]["shared_memory_bytes"]
                    < compiler["stages_3"]["shared_memory_bytes"]
                or not profiles["stages_2"]["sm__maximum_warps_per_active_cycle_pct"]
                    > profiles["stages_3"]["sm__maximum_warps_per_active_cycle_pct"]):
            raise ValueError("Experiment 002 recorded conclusion contradicts measurements")
        entry = {
            "id": f"stage_reduction:{RUN}", "kind": "numeric_tuning_screening",
            "created_utc": result["created_utc"],
            "shape_mnk": result["workload"]["shape_mnk"], "dtype": "fp16",
            "gpu": result["environment"]["gpu"]["name"],
            "configuration": old, "candidate_configuration": new,
            "hypothesis": plan["technical_hypothesis"],
            "mechanism": plan["optimization_mechanism"],
            "families": ["k_loop_unrolling"],
            "original_muse_source_sha256": identity["original_muse_candidate_sha256"],
            "trusted_implementation_sha256": identity["trusted_implementation_sha256"],
            "baseline_sha256": identity["baseline_kernel_sha256"],
            "launcher_sha256": identity["baseline_launcher_sha256"],
            "compilation": "both configurations compiled", "correctness": "PASS",
            "correctness_cases": [len(checks), len(checks)],
            "screening": {
                "measurement_method": "python_launch_cuda_event",
                "baseline_median_ms": latency["stages_3"],
                "candidate_median_ms": latency["stages_2"],
                "median_paired_speedup": ratio,
                "paired_ratio_mad": comparison["paired_ratios"]["mad"],
                "aa_stages_3_median_absolute_ratio_deviation": aa3["aa_variability"]["median_absolute_ratio_deviation"],
                "aa_stages_2_median_absolute_ratio_deviation": aa2["aa_variability"]["median_absolute_ratio_deviation"],
                "comparison_rounds": len(comparison["rounds"]),
                "screening_only": True,
                "noise_interpretation": "Ratio deficit exceeds A/A median deviations and paired MAD under a heuristic; A/A maxima are larger. No significance claim.",
            },
            "confirmation": "not confirmed", "compiler": {"resources": compiler},
            "outcome": "two_stage_screening_loss",
            "performance_interpretation": "Two stages slower in bounded screening",
            "profiling_evidence": {
                "method": profile["measurement_method"],
                "selected_launches_per_configuration": 1,
                "metrics": profiles, "limitations": profile["limitations"],
                "interpretation": "Lower staging improved occupancy headroom, but the CUDA-event latency worsened; occupancy is not a causal latency explanation.",
            },
            "provenance": "Hash-verified numeric tuning of unchanged reviewed two-slab source; original Muse source unevaluated",
            "exact_generated_source_evaluated": False,
            "artifact_path": ARTIFACT,
        }
        return entry, []
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration, OverflowError) as exc:
        return None, [f"Experiment 002 artifact rejected: {type(exc).__name__}: {exc}"]
