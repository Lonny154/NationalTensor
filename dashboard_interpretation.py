"""Display-only interpretations of saved Day 2 records.

This module never changes experiment artifacts or timing calculations. Its
labels describe screening evidence; they are not significance tests.
"""

from __future__ import annotations

import math
import statistics
import textwrap


KNOWN_SUMMARIES = {
    "20261027T144337580507Z": (
        "c9f81c34d23919359ca67ea2a071ed13a176203e783e6acccca77d702841b3ad",
        "Grouped tile-order proposal",
        "Muse proposed a Grouped-M4 tile-order swizzle. CPU mapping review found full tile coverage, but the saved 2D flattening does not establish the intended launch locality."),
    "20261027T150316362990Z": (
        "38b23fad21dc8bb6d194239c7ac0d0a20b1c8566de4055442e3e927d8f97e59a",
        "Mask hoisting and pipeline hint",
        "Muse proposed hoisting row and column masks, sharing a K offset, and adding an explicit three-stage loop hint. A separate human-written variant was later evaluated."),
    "20261027T160014125560Z": (
        "22ce595f5eb134442dd186de8a3db1b25c9413391daa230dd9e9f87781da43fc",
        "Two-slab K-loop proposal",
        "Muse proposed two loads and dots per K-loop trip to reduce loop overhead. A separate human-written variant was later evaluated."),
    "20261027T163630745455Z": (
        "0dadc8e70d770e88347fefb616b1a6f96c0781274b0fb74b927c48a3174ed974",
        "K-bound-specialized loop",
        "Muse proposed a full-slab loop without repeated K-bound masks, plus a masked tail for partial K. The generated source awaits review and has no compiler or benchmark result."),
}

KNOWN_INCONCLUSIVE = {
    "structural:20261027T135435023171Z:2": (
        "Small-M Grouped-M4 has a 1.01378× paired screening ratio, but a 0.270 ms raw block and a 5.19× pair make the direction unreliable."),
    "structural:20261027T140513987643Z:1": (
        "Square Grouped-M4 has a 0.99181× paired ratio with mixed pairs and a 0.670 ms raw block."),
    "structural:20261027T141453253695Z:1": (
        "FC1 Grouped-M4 has a 1.00520× paired ratio and raw blocks near 0.751 ms; no reliable gain was established."),
    "trusted:trusted-compute:20261027T154233319712Z": (
        "The trusted mask/pipeline variant screened at 1.00239× with mixed pairs; A/A median absolute ratio deviation was 3.84%."),
}

STUDIES = {
    "trusted-compute": {
        "muse_sha": "38b23fad21dc8bb6d194239c7ac0d0a20b1c8566de4055442e3e927d8f97e59a",
        "mechanism": "Hoist M/N masks, compute one K offset per trip, and request a three-stage K loop to reduce predicate work and improve copy/dot overlap.",
        "resource_prediction": "The proposal preserved tile sizes and launcher settings; it gave no explicit register or shared-memory target.",
        "supported": "The independently written variant passed six correctness checks, and its TTIR recorded the explicit stage hint.",
        "contradicted": "The hoped-for distinct pipeline or instruction pattern did not appear: the saved PTX instruction multisets and resource counts match the baseline.",
        "uncertain": "The 1.00239× screening ratio did not establish a speedup against 3.84% A/A deviation. Runtime overlap and the exact generated source were not evaluated.",
    },
    "trusted-two-slab": {
        "muse_sha": "22ce595f5eb134442dd186de8a3db1b25c9413391daa230dd9e9f87781da43fc",
        "mechanism": "Use two ordered loads and FP32 dots per loop trip to halve K-loop trips and expose more load/dot scheduling opportunities.",
        "resource_prediction": "The proposal predicted more overlap but set no register or shared-memory ceiling.",
        "supported": "The trusted variant compiled to 12 two-dot trips instead of 24 one-dot trips; dynamic SASS instructions fell about 22.6% and branches 50%.",
        "contradicted": "The predicted speedup did not appear in screening: all eight pairs favored baseline, while registers rose 120→129 and shared memory 24,576→49,152 bytes.",
        "uncertain": "Greater overlap, occupancy effects, and the cause of the slowdown were not measured. A/A deviation was 4.85%, and the exact generated source was not evaluated.",
    },
}


def proposal_summary(record: dict) -> dict:
    known = KNOWN_SUMMARIES.get(record.get("run_id"))
    if (known and record.get("kind") == "kernel"
            and record.get("source_hash_valid") is True
            and record.get("source_sha256") == known[0]):
        return {"title": known[1], "summary": known[2]}
    hypothesis = " ".join(str(record.get("hypothesis") or "").split())
    return {"title": "Generated Triton proposal",
            "summary": textwrap.shorten(hypothesis, width=170, placeholder="…")
                or "No hypothesis was saved."}


def _valid_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _timing(record: dict) -> dict | None:
    value = record.get("timing") if record.get("kind") == "structural" else record.get("screening")
    return value if isinstance(value, dict) else None


def _warnings(record: dict, timing: dict | None) -> list[str]:
    if timing is None:
        return []
    warnings = []
    samples = timing.get("samples_ms", {})
    if isinstance(samples, dict):
        for side in ("baseline", "candidate"):
            raw = samples.get(side)
            if isinstance(raw, list):
                values = [value for value in raw if _valid_number(value) and value > 0]
                if values and (max(values) > 4 * statistics.median(values)
                               or max(values) > 4 * min(values)):
                    warnings.append(
                        f"Large raw {side} timing spread: maximum {max(values) * 1000:.1f} µs "
                        f"versus minimum {min(values) * 1000:.1f} µs and "
                        f"median {statistics.median(values) * 1000:.1f} µs. "
                        "All samples remain in the plot and artifact.")
    aa = record.get("aa")
    deviation = ((aa or {}).get("aa_variability") or {}).get(
        "median_absolute_ratio_deviation") if isinstance(aa, dict) else None
    if _valid_number(deviation) and deviation >= 0.03:
        warnings.append(
            f"A/A median absolute ratio deviation was {deviation * 100:.2f}%; "
            "interpret small screening shifts cautiously.")
    return warnings


def status_for(record: dict) -> dict:
    kind = record.get("kind")
    if kind == "kernel":
        review = {"label": "Awaiting review", "tone": "neutral"}
        static = {"label": ("Static validation passed for review"
                            if record.get("static_status") == "accepted_for_review_only"
                            else "Static review unavailable"), "tone": "neutral"}
        correctness = {"label": "Correctness not evaluated", "tone": "neutral"}
        performance = {"label": "Not measured", "tone": "neutral"}
    else:
        review = {"label": ("Human-reviewed implementation" if str(kind).startswith("trusted-")
                            else "Reviewed strategy"), "tone": "neutral"}
        static = None
        correctness_value = record.get("correctness")
        correctness = {"label": ("Correctness passed" if correctness_value == "PASS"
                                 else "Correctness failed" if correctness_value == "FAIL"
                                 else "Correctness unavailable"),
                       "tone": "pass" if correctness_value == "PASS" else "neutral"}
        timing = _timing(record)
        ratio = timing.get("median_paired_speedup") if timing else None
        decision = record.get("confirmation_decision")
        if (correctness_value == "PASS"
                and record.get("confirmation_artifact_verified") is True
                and isinstance(decision, dict) and decision.get("confirmed") is True):
            performance = {"label": "Confirmed", "tone": "confirmed"}
        elif record.get("id") in KNOWN_INCONCLUSIVE:
            performance = {"label": "Inconclusive", "tone": "neutral"}
        elif correctness_value != "PASS" or not _valid_number(ratio):
            performance = {"label": "Inconclusive", "tone": "neutral"}
        elif ratio > 1:
            performance = {"label": "Screening gain — unconfirmed", "tone": "screening"}
        elif ratio < 1:
            performance = {"label": "Screening loss", "tone": "loss"}
        else:
            performance = {"label": "Inconclusive", "tone": "neutral"}
    timing = _timing(record)
    return {"review": review, "static": static, "correctness": correctness,
            "performance": performance, "warnings": _warnings(record, timing),
            "interpretation": KNOWN_INCONCLUSIVE.get(record.get("id")),
            "recorded_outcome": record.get("outcome") or record.get("status")}


def hypothesis_evidence(trusted: dict, proposal: dict | None) -> dict | None:
    """Compare only an exact saved Muse→trusted hash relationship."""
    if (not proposal or proposal.get("kind") != "kernel"
            or proposal.get("source_hash_valid") is not True
            or trusted.get("exact_generated_source_evaluated") is not False
            or proposal.get("source_sha256") != trusted.get("generated_sha256")):
        return None
    known = STUDIES.get(trusted.get("kind"))
    if not known or known["muse_sha"] != trusted.get("generated_sha256"):
        return None
    feedback = trusted.get("feedback") or {}
    compiler = feedback.get("compiler_review") or {}
    compiler_lines = [f"{name.upper()}: {value}" for name, value in compiler.items()
                      if name in {"ttir", "ttgir", "ptx", "sass"} and isinstance(value, str)]
    resources = trusted.get("compiler_resources") or {}
    baseline = resources.get("baseline") or {}
    candidate = resources.get("trusted") or {}
    resource_text = "Unavailable"
    if all(_valid_number(side.get(key)) for side in (baseline, candidate)
           for key in ("registers_per_thread", "shared_memory_bytes")):
        resource_text = (f"Registers/thread {baseline['registers_per_thread']} → "
                         f"{candidate['registers_per_thread']}; shared memory "
                         f"{baseline['shared_memory_bytes']:,} → "
                         f"{candidate['shared_memory_bytes']:,} bytes.")
    timing = _timing(trusted)
    ratio = timing.get("median_paired_speedup") if timing else None
    aa = trusted.get("aa") or {}
    deviation = (aa.get("aa_variability") or {}).get("median_absolute_ratio_deviation")
    screening = "Unavailable"
    if _valid_number(ratio):
        screening = f"Median paired screening {ratio:.5f}×"
        if _valid_number(deviation):
            screening += f"; A/A median absolute ratio deviation {deviation * 100:.2f}%"
        screening += "; no confirmation artifact."
    passed = trusted.get("correctness_passed")
    total = trusted.get("correctness_total")
    correctness = (f"{passed}/{total} checks passed on the independently written kernel"
                   if isinstance(passed, int) and isinstance(total, int) and total
                   else "Unavailable")
    dynamic = trusted.get("dynamic_profile") or {}
    dynamic_text = "Unavailable"
    if isinstance(dynamic, dict):
        baseline_dynamic = dynamic.get("baseline") or {}
        candidate_dynamic = dynamic.get("trusted") or {}
        instruction = "sm__sass_inst_executed.sum"
        branch = "sm__sass_inst_executed_op_branch.sum"
        tensor = "sm__inst_executed_pipe_tensor_v2.sum"
        if all(_valid_number(side.get(key)) for side in
               (baseline_dynamic, candidate_dynamic)
               for key in (instruction, branch, tensor)):
            dynamic_text = (
                f"Dynamic SASS instructions {baseline_dynamic[instruction]:,} → "
                f"{candidate_dynamic[instruction]:,}; branches "
                f"{baseline_dynamic[branch]:,} → {candidate_dynamic[branch]:,}; "
                f"tensor-pipe instructions {baseline_dynamic[tensor]:,} → "
                f"{candidate_dynamic[tensor]:,}. Separate profiler launch, not paired timing.")
    interpretation = {key: known[key] for key in ("supported", "contradicted", "uncertain")}
    if not compiler_lines:
        interpretation["contradicted"] = "Compiler evidence unavailable; no compiler conclusion."
        interpretation["supported"] = (f"{correctness}." if correctness != "Unavailable"
                                       else "Compiler and correctness evidence unavailable.")
    if screening == "Unavailable":
        if compiler_lines:
            interpretation["contradicted"] = "Screening evidence unavailable; no speed conclusion."
        interpretation["uncertain"] = "Performance remains unmeasured for this trusted implementation."
    return {
        "muse_sha256": proposal["source_sha256"],
        "trusted_sha256": trusted.get("trusted_sha256"),
        "predicted": {"mechanism": known["mechanism"],
                      "improvement": proposal.get("hypothesis") or "Unavailable",
                      "resources": known["resource_prediction"]},
        "observed": {"compiler": compiler_lines or ["Unavailable"],
                     "correctness": correctness, "resources": resource_text,
                     "screening": screening, "dynamic": dynamic_text},
        "interpretation": interpretation,
    }


def decorate_history(history: dict) -> dict:
    """Attach display labels to in-memory records without touching artifacts."""
    records = {item["id"]: item for item in history["proposals"] + history["trusted"]}
    for item in records.values():
        item["display"] = status_for(item)
        if item.get("kind") == "kernel":
            item["summary"] = proposal_summary(item)
    for item in history["trusted"]:
        item["comparison"] = hypothesis_evidence(
            item, records.get(item.get("related_proposal")))
    return history
