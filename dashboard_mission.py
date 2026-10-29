"""Small, read-only payload for the cinematic view of validated diagnostics."""

import math

from dashboard_evidence import evidence_summary


HISTORICAL_ID = "diagnostic:mask_hoisting_m128:20261028T145501296300Z"


def classify_ending(entry, runtime_measured):
    """Classify only recorded evidence; a screening gain is never confirmation."""
    if (entry.get("outcome") == "compiler_equivalent_screening_skipped"
            and entry.get("correctness") == "PASS"
            and (entry.get("diagnostic_evidence") or {}).get("executable_section", {}).get("identical") is True
            and runtime_measured is False):
        return "equivalent"
    screening = entry.get("screening") or {}
    ratio = screening.get("median_paired_speedup")
    valid_ratio = (isinstance(ratio, (float, int)) and not isinstance(ratio, bool)
                   and math.isfinite(ratio) and ratio > 0)
    if runtime_measured is not True or entry.get("correctness") != "PASS" or not valid_ratio:
        return "inconclusive"
    if entry.get("confirmed") is True and ratio > 1:
        return "improvement"
    if ratio < 1:
        return "slowdown"
    return "inconclusive"


def mission_evidence(root, entry):
    summary = evidence_summary(root, entry)
    diagnostic = entry.get("diagnostic_evidence") or {}
    executable = summary["executable"]
    return {
        "id": entry["id"], "gpu": entry.get("gpu"),
        "shape": entry.get("shape_mnk"), "dtype": entry.get("dtype"),
        "hypothesis": summary["hypothesis"], "outcome": entry.get("outcome"),
        "correctness": entry.get("correctness"),
        "correctness_cases": summary["correctness_cases"],
        "executable_identical": executable.get("identical"),
        "executable_bytes": executable.get("size_bytes"),
        "metadata_identical": summary["constant_metadata_identical"],
        "runtime_measured": summary["runtime_measured"],
        "skip_reason": summary["skip_reason"],
        "performance": entry.get("performance_interpretation"),
        "screening": entry.get("screening"),
        "confirmed": entry.get("confirmed") is True,
        "ending": classify_ending(entry, summary["runtime_measured"]),
        "workflow_validation": summary["workflow_validation"],
        "warnings": summary["warnings"],
        "evidence_url": "/research/evidence/" + entry["id"],
        "compiler_note": ("PTX differences do not survive into the final executable SASS."
                          if summary["diagnostic"].get("ptx_order_and_operands_identical_ignoring_debug") is False
                          and executable.get("identical") is True else
                          "Complete executable sections and metadata were compared."),
        "scientific_conclusion": diagnostic.get("performance_conclusion"),
    }
