"""Rebuild small, evidence-backed experiment memory from saved artifacts.

No index is stored: each read derives entries from the immutable experiment
files through dashboard_data's bounded, path-checked loader. Generated source
is display and comparison data only; it is never imported or executed here.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

from dashboard_data import ROOT, load_history
from dashboard_interpretation import status_for


FAMILY_NAMES = {
    "grouped_scheduling": "Grouped scheduling",
    "split_k": "Split-K",
    "mask_hoisting": "Mask hoisting",
    "explicit_pipeline_hint": "Explicit pipeline hint",
    "k_loop_unrolling": "K-loop unrolling",
    "k_bound_specialization": "K-bound specialization",
}


def classify_families(hypothesis="", mechanism="", source="", strategy="") -> list[str]:
    """Conservative labels from the proposal itself, never its feedback text."""
    if strategy == "grouped_m4":
        return ["grouped_scheduling"]
    if strategy == "split_k":
        return ["split_k"]
    if strategy == "direct_2d":
        return []
    text = f"{hypothesis} {mechanism} {strategy}".lower()
    source = source if isinstance(source, str) else ""
    families = []
    if ("grouped_m4" in text or "grouped-m4" in text
            or "grouped scheduling" in text or "grouped tile" in text
            or "group_m=4" in text or "l2 swizzle" in text):
        families.append("grouped_scheduling")
    if "split_k" in text or "split-k" in text:
        families.append("split_k")
    # A full-slab loop with a separately masked K tail is primarily a
    # K-bound change, even when the proposal also hoists M/N masks.
    if ("k-bound" in text or "k bound" in text
            or "full bk slab" in text or "full-tile k" in text):
        families.append("k_bound_specialization")
    if "hoist" in text and "mask" in text:
        families.append("mask_hoisting")
    if ("tl.range" in source and "num_stages" in source
            or "explicit" in text and ("pipeline" in text or "staging" in text)):
        families.append("explicit_pipeline_hint")
    if ("two-slab" in text or "two slab" in text or "k-loop unroll" in text
            or "num_pairs" in source and source.count("tl.dot(") >= 3):
        families.append("k_loop_unrolling")
    return families


def _structure(source):
    if not isinstance(source, str):
        return None
    try:
        return ast.dump(ast.parse(source), include_attributes=False)
    except (SyntaxError, TypeError, ValueError):
        return None


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _screening(record, key):
    timing = record.get(key)
    if not isinstance(timing, dict):
        return None
    aa = record.get("aa")
    deviation = ((aa.get("aa_variability") or {}).get(
        "median_absolute_ratio_deviation") if isinstance(aa, dict) else None)
    return {
        "measurement_method": record.get("measurement_method"),
        "median_paired_speedup": _number(timing.get("median_paired_speedup")),
        "baseline_median_ms": _number(timing.get("baseline_ms")),
        "candidate_median_ms": _number(timing.get("candidate_ms")),
        "paired_ratio_mad": _number((timing.get("statistics") or {}).get(
            "paired_ratios", {}).get("mad")),
        "aa_median_absolute_ratio_deviation": _number(deviation),
        "screening_only": True,
    }


def _confirmed(record):
    decision = record.get("confirmation_decision")
    return (record.get("confirmation_artifact_verified") is True
            and isinstance(decision, dict) and decision.get("confirmed") is True)


def from_history(history: dict, *, profile_reports=()) -> dict:
    """Build entries from one already-loaded artifact snapshot."""
    entries = []
    proposals = {item["id"]: item for item in history.get("proposals", [])
                 if item.get("kind") == "kernel"}
    for record in history.get("proposals", []):
        kind = record.get("kind")
        if kind == "kernel":
            # Require both source hash verification and a matching accepted audit.
            if (record.get("source_hash_valid") is not True
                    or not isinstance(record.get("raw_response"), str)):
                continue
            entries.append({
                "id": record["id"], "kind": "generated_proposal",
                "created_utc": record.get("created_utc") or "",
                "shape_mnk": record.get("shape"), "dtype": record.get("dtype") or "fp16",
                "gpu": record.get("gpu"), "configuration": record.get("config"),
                "hypothesis": record.get("hypothesis") or "",
                "mechanism": record.get("expected_mechanism") or "",
                "families": classify_families(record.get("hypothesis"),
                    record.get("expected_mechanism"), record.get("source")),
                "original_muse_source_sha256": record.get("source_sha256"),
                "source": record.get("source"),
                "trusted_implementation_sha256": None,
                "baseline_sha256": record.get("baseline_sha256"),
                "launcher_sha256": record.get("launcher_sha256"),
                "compilation": "not evaluated", "correctness": "not evaluated",
                "screening": None, "confirmation": "not confirmed",
                "compiler": None, "outcome": "awaiting review",
                "performance_interpretation": status_for(record)["performance"]["label"],
                "provenance": "Muse-generated source; exact source unevaluated",
                "artifact_path": record.get("artifact_path"),
            })
        elif kind == "structural":
            entries.append({
                "id": record["id"], "kind": "reviewed_structural_strategy",
                "created_utc": record.get("created_utc") or "",
                "shape_mnk": record.get("shape"), "dtype": record.get("dtype") or "fp16",
                "gpu": record.get("gpu"), "configuration": record.get("config"),
                "hypothesis": record.get("hypothesis") or "",
                "mechanism": record.get("expected_mechanism") or "",
                "families": classify_families(record.get("hypothesis"),
                    record.get("expected_mechanism"), strategy=record.get("strategy")),
                "original_muse_source_sha256": None,
                "trusted_implementation_sha256": None,
                "source_fingerprints": record.get("source_fingerprints"),
                "baseline_sha256": None, "launcher_sha256": None,
                "compilation": "not separately recorded",
                "correctness": record.get("correctness") or "unknown",
                "screening": _screening(record, "timing"),
                "confirmation": "confirmed" if _confirmed(record) else "not confirmed",
                "compiler": None, "outcome": record.get("outcome") or "unknown",
                "performance_interpretation": status_for(record)["performance"]["label"],
                "provenance": "Muse-selected reviewed strategy",
                "artifact_path": record.get("artifact_path"),
            })
    for record in history.get("trusted", []):
        proposal = proposals.get(record.get("related_proposal"))
        if (proposal is None or proposal.get("source_hash_valid") is not True
                or proposal.get("source_sha256") != record.get("generated_sha256")
                or record.get("provenance")
                != "model_suggested_human_reviewed_implementation"
                or record.get("exact_generated_source_evaluated") is not False
                or not record.get("trusted_sha256")
                or record.get("trusted_sha256") == record.get("generated_sha256")):
            continue
        entries.append({
            "id": record["id"], "kind": "trusted_implementation",
            "created_utc": record.get("created_utc") or "",
            "shape_mnk": record.get("shape"), "dtype": proposal.get("dtype") or "fp16",
            "gpu": record.get("gpu"), "configuration": proposal.get("config"),
            "hypothesis": proposal.get("hypothesis") or "",
            "mechanism": proposal.get("expected_mechanism") or "",
            "families": classify_families(proposal.get("hypothesis"),
                proposal.get("expected_mechanism"), proposal.get("source")),
            "original_muse_source_sha256": record.get("generated_sha256"),
            "trusted_implementation_sha256": record.get("trusted_sha256"),
            "baseline_sha256": proposal.get("baseline_sha256"),
            "launcher_sha256": proposal.get("launcher_sha256"),
            "compilation": "compiled" if record.get("compiler_review") else "unavailable",
            "correctness": record.get("correctness") or "unknown",
            "correctness_cases": [record.get("correctness_passed"),
                                  record.get("correctness_total")],
            "screening": _screening(record, "screening"),
            "confirmation": "confirmed" if _confirmed(record) else "not confirmed",
            "compiler": {"review": record.get("compiler_review") or {},
                         "resources": record.get("compiler_resources") or {},
                         "dynamic_profile": record.get("dynamic_profile")},
            "outcome": record.get("status") or "unknown",
            "performance_interpretation": status_for(record)["performance"]["label"],
            "provenance": record.get("provenance"),
            "exact_generated_source_evaluated": False,
            "related_proposal_id": proposal["id"],
            "artifact_path": record.get("artifact_path"),
        })
    for report in profile_reports:
        if report.get("status") != "profiling_complete_verified":
            continue
        summaries = report["summary"]
        def pairs(metric):
            return {key: summaries[key][metric]["captures"]
                    for key in ("baseline", "mask_hoisting", "two_slab")}
        entries.append({
            "id": report["study_id"], "kind": "profiling_study",
            "created_utc": "2026-10-28T00:42:49.599584+00:00",
            "shape_mnk": report["workload"]["shape_mnk"],
            "dtype": report["workload"]["dtype"],
            "gpu": report["workload"]["gpu"],
            "configuration": report["workload"]["numeric_config"],
            "hypothesis": "Which reproducibly measured profiler bottleneck distinguishes the reviewed FC1 implementations?",
            "mechanism": "Profile reviewed baseline, mask-hoisting, and two-slab kernels; no source optimization was evaluated.",
            "families": [], "original_muse_source_sha256": None,
            "trusted_implementation_sha256": None,
            "source_identities": report["source_identities"],
            "baseline_sha256": report["source_identities"]["baseline"]["kernel_sha256"],
            "launcher_sha256": report["baseline_launcher_sha256"],
            "compilation": "reviewed kernels compiled for captures",
            "correctness": "PASS on FC1 seed 42 before each capture",
            "screening": None, "confirmation": "not a performance experiment",
            "compiler": None, "outcome": "profiling_complete_verified",
            "performance_interpretation": "No latency conclusion",
            "profiling_evidence": {
                "method": report["method"],
                "achieved_occupancy_pct": pairs("sm__warps_active.avg.pct_of_peak_sustained_active"),
                "theoretical_occupancy_pct": pairs("sm__maximum_warps_per_active_cycle_pct"),
                "shared_memory_bytes_per_block": pairs("launch__shared_mem_per_block_dynamic"),
                "registers_per_thread": pairs("launch__registers_per_thread"),
                "shared_memory_block_limit": pairs("launch__occupancy_limit_shared_mem"),
                "register_block_limit": pairs("launch__occupancy_limit_registers"),
                "long_scoreboard_warps_per_cycle": pairs("smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active"),
                "short_scoreboard_warps_per_cycle": pairs("smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active"),
                "barrier_warps_per_cycle": pairs("smsp__warps_issue_stalled_barrier.avg.per_cycle_active"),
                "tensor_pipe_pct_peak": pairs("smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active"),
                "dram_bytes_per_second": pairs("dram__bytes.sum.per_second"),
                "interpretation": report["interpretation"],
                "limitations": report["limitations"],
                "capture_count_per_kernel": 2,
                "manifest_sha256": report["manifest_sha256"],
            },
            "provenance": "Hash-verified Nsight Compute captures of reviewed implementations; not CUDA-event latency",
            "artifact_path": f"evaluations/planned-profiling/{report['study_id'].split(':', 1)[1]}/report.json",
        })
    for entry in entries:
        entry["primary_family"] = entry["families"][0] if entry["families"] else None
        entry["secondary_families"] = entry["families"][1:]
    entries.sort(key=lambda item: (item["created_utc"], item["id"]))
    return {"entries": entries, "warnings": list(history.get("warnings", []))}


def reconstruct(root: Path = ROOT) -> dict:
    """Reload artifacts on every call, including after process restart."""
    from profile_report import load_verified_report
    report, warnings = load_verified_report(root)
    memory = from_history(load_history(root), profile_reports=[report] if report else ())
    memory["warnings"].extend(warnings)
    from stage_reduction_memory import load as load_stage_reduction
    stage_entry, stage_warnings = load_stage_reduction(root)
    if stage_entry:
        stage_entry["primary_family"] = stage_entry["families"][0]
        stage_entry["secondary_families"] = stage_entry["families"][1:]
        memory["entries"].append(stage_entry)
        memory["entries"].sort(key=lambda item: (item["created_utc"], item["id"]))
    memory["warnings"].extend(stage_warnings)
    from grouped_confirmation_memory import load as load_grouped_confirmations
    grouped_entry, grouped_warnings = load_grouped_confirmations(root)
    if grouped_entry:
        memory["entries"].append(grouped_entry)
        memory["entries"].sort(key=lambda item: (item["created_utc"], item["id"]))
    memory["warnings"].extend(grouped_warnings)
    from mask_hoisting_memory import load as load_mask_diagnostic
    diagnostic_entry, diagnostic_warnings = load_mask_diagnostic(root)
    if diagnostic_entry:
        memory["entries"].append(diagnostic_entry)
        memory["entries"].sort(key=lambda item: (item["created_utc"], item["id"]))
    memory["warnings"].extend(diagnostic_warnings)
    from workflow_artifacts import load_completed as load_workflow_results
    workflow_entries, workflow_warnings = load_workflow_results(root)
    memory["entries"].extend(workflow_entries)
    memory["entries"].sort(key=lambda item: (item["created_utc"], item["id"]))
    memory["warnings"].extend(workflow_warnings)
    from advisory_decisions import load as load_completed_decisions
    decisions, decision_warnings = load_completed_decisions(root, memory["entries"])
    memory["completed_advisory_decisions"] = decisions
    memory["warnings"].extend(decision_warnings)
    return memory


def compatibility(entry: dict, *, shape, dtype, gpu, method,
                  baseline_sha256=None, launcher_sha256=None,
                  numeric_config=None) -> dict:
    same_shape = entry.get("shape_mnk") == list(shape)
    same_dtype = entry.get("dtype") == dtype
    same_gpu = entry.get("gpu") == gpu
    recorded_method = ("nsight_compute_kernel_replay"
                       if entry.get("kind") == "profiling_study"
                       else (entry.get("screening") or {}).get("measurement_method"))
    same_method = recorded_method == method if recorded_method else None
    prior_base, prior_launcher = (entry.get("baseline_sha256"),
                                  entry.get("launcher_sha256"))
    same_source_version = (prior_base == baseline_sha256 and prior_launcher == launcher_sha256
                           if prior_base and prior_launcher and baseline_sha256
                           and launcher_sha256 else None)
    config = entry.get("configuration")
    same_config = (all(config.get(key) == value for key, value in numeric_config.items())
                   if isinstance(config, dict) and isinstance(numeric_config, dict)
                   else None)
    return {"workload": "same" if same_shape and same_dtype else "different_or_unknown",
            "gpu": "same" if same_gpu else "different_or_unknown",
            "measurement_method": ("same" if same_method else "different"
                                   if same_method is False else "unmeasured"),
            "source_version": ("same" if same_source_version else "different"
                               if same_source_version is False else "unknown"),
            "numeric_configuration": ("same" if same_config else "different"
                                      if same_config is False else "unknown"),
            "protocol_and_config_match": bool(
                same_shape and same_dtype and same_gpu and same_method
                and same_source_version and same_config
                and entry.get("kind") != "reviewed_structural_strategy"),
            "timing_directly_comparable": False,
            "comparison_scope": "Each screening ratio belongs to its saved paired run; absolute times are not pooled across runs.",
    }


def retrieve(memory: dict, *, shape, dtype, gpu, method, baseline_sha256=None,
             launcher_sha256=None, numeric_config=None, anchor_families=(), exclude_id=None,
             limit=6) -> list[dict]:
    """Return bounded evidence summaries; no source or raw compiler files."""
    candidates = []
    for entry in memory["entries"]:
        if entry.get("workflow_validation"):
            continue  # Replay verifies the workflow, not an additional independent research result.
        if entry["id"] == exclude_id:
            continue
        match = compatibility(entry, shape=shape, dtype=dtype, gpu=gpu,
                              method=method, baseline_sha256=baseline_sha256,
                              launcher_sha256=launcher_sha256,
                              numeric_config=numeric_config)
        score = (100 * (match["workload"] == "same")
                 + 30 * (match["gpu"] == "same")
                 + 95 * (entry["kind"] == "profiling_study")
                 + 80 * (entry["kind"] == "numeric_tuning_screening")
                 + 80 * (entry["kind"] == "compiler_diagnostic")
                 + 75 * (entry["kind"] == "confirmation_history")
                 + 45 * (entry["kind"] == "trusted_implementation")
                 + 30 * (entry["kind"] == "reviewed_structural_strategy")
                 + 15 * bool(set(entry["families"]) & set(anchor_families))
                 + 10 * bool(entry.get("compiler")))
        candidates.append((score, entry["created_utc"], entry, match))
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    results = []
    for _, _, entry, match in candidates[:max(0, min(limit, 8))]:
        screening = entry.get("screening") or {}
        compiler = entry.get("compiler") or {}
        review = compiler.get("review") or {}
        results.append({
            "id": entry["id"], "kind": entry["kind"],
            "families": entry["families"],
            "primary_family": entry["primary_family"],
            "secondary_families": entry["secondary_families"],
            "shape_mnk": entry["shape_mnk"], "gpu": entry["gpu"],
            "hypothesis": entry["hypothesis"][:190],
            "mechanism": entry["mechanism"][:190],
            "original_muse_source_sha256": entry["original_muse_source_sha256"],
            "trusted_implementation_sha256": entry["trusted_implementation_sha256"],
            "correctness": entry["correctness"], "compilation": entry["compilation"],
            "confirmation": entry["confirmation"], "outcome": entry["outcome"],
            "confirmation_evidence": entry.get("confirmation_evidence"),
            "review_status": entry.get("review_status"),
            "diagnostic_evidence": entry.get("diagnostic_evidence"),
            "performance_interpretation": entry["performance_interpretation"],
            "screening": screening, "compatibility": match,
            "candidate_configuration": entry.get("candidate_configuration"),
            "compiler_evidence": {key: str(review[key])[:220]
                                  for key in ("ttir", "ttgir", "ptx", "sass", "resources")
                                  if key in review and isinstance(review[key], str)},
            "compiler_resources": compiler.get("resources"),
            "profiling_evidence": entry.get("profiling_evidence"),
            "provenance": entry["provenance"], "artifact_path": entry["artifact_path"],
        })
    return results


def compare_candidate(memory: dict, *, source_sha256, hypothesis, mechanism,
                      source, shape, gpu, limit=5) -> dict:
    """Only an identical source hash rejects; families are review hints."""
    exact = next((entry for entry in memory["entries"]
                  if entry["kind"] == "generated_proposal"
                  and entry["original_muse_source_sha256"] == source_sha256), None)
    families = classify_families(hypothesis, mechanism, source)
    candidate_structure = _structure(source)
    related = []
    for entry in reversed(memory["entries"]):
        overlap = sorted(set(families) & set(entry["families"]))
        if entry.get("shape_mnk") != list(shape):
            continue
        same_structure = bool(candidate_structure and entry.get("source") and
                              candidate_structure == _structure(entry["source"]))
        if not overlap and not same_structure:
            continue
        related.append({"id": entry["id"], "families": overlap,
                        "reason": ("AST structure matches an earlier generated source; "
                                   "compiled equivalence is untested." if same_structure
                                   else "Shared deterministic optimization family; "
                                   "implementation and compiled behavior may differ."),
                        "source_structure_match": same_structure,
                        "gpu_match": entry.get("gpu") == gpu,
                        "kind": entry["kind"], "outcome": entry["outcome"],
                        "performance_interpretation":
                            entry["performance_interpretation"],
                        "screening": entry.get("screening"),
                        "artifact_path": entry.get("artifact_path")})
        if len(related) >= limit:
            break
    return {"families": families,
            "primary_family": families[0] if families else None,
            "secondary_families": families[1:],
            "exact_duplicate_of": exact["id"] if exact else None,
            "related": related}


def related_for(memory: dict, experiment_id: str, *, limit=5) -> list[dict]:
    current = next((item for item in memory["entries"] if item["id"] == experiment_id), None)
    if current is None or current["kind"] != "generated_proposal":
        return []
    related = []
    current_structure = _structure(current.get("source"))
    for item in reversed(memory["entries"]):
        if (item["id"] == experiment_id or item["created_utc"] > current["created_utc"]
                or item.get("shape_mnk") != current["shape_mnk"]):
            continue
        families = sorted(set(item["families"]) & set(current["families"]))
        same_structure = bool(current_structure and item.get("source") and
                              current_structure == _structure(item["source"]))
        if not families and not same_structure:
            continue
        related.append({"id": item["id"], "kind": item["kind"],
                        "families": families, "gpu_match": item["gpu"] == current["gpu"],
                        "family_labels": [FAMILY_NAMES[name] for name in families],
                        "correctness": item["correctness"], "outcome": item["outcome"],
                        "performance_interpretation": item["performance_interpretation"],
                        "screening": item["screening"],
                        "confirmation": item["confirmation"],
                        "reason": ("Matching AST structure; compiled behavior is unverified."
                                   if same_structure else
                                   "Shared known family; source and compiled behavior may differ.")})
        if len(related) >= limit:
            break
    return related
