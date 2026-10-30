"""Bounded, advisory cross-experiment planning from validated compiler diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re

from dashboard_data import ROOT
from experiment_004_evidence import load as load_004, sass_counts, sass_instructions
from mask_hoisting_memory import load as load_003
from muse_debrief import (API_BASE_URL, MODEL, canonical, digest, inspect_response,
                          require_complete_text, _get, NUMERIC_PERFORMANCE)

SECTION = Path("evaluations/muse-research-planning/v1")
PROMPT_VERSION = "cross_experiment_v1"
REQUEST_OPTIONS = {"max_output_tokens": 8000, "reasoning": {"effort": "low"}}
FIELDS = ("evidence_synthesis", "rejected_hypotheses", "remaining_uncertainty",
          "next_hypothesis", "proposed_experiment_005", "decision_criteria",
          "stop_condition", "research_rationale")
PLAN_FIELDS = ("workload", "controlled_variables", "expected_observations", "confounders")
DECISION_FIELDS = ("support", "weaken", "reject")


def evidence_package(root=ROOT):
    root = Path(root).resolve()
    third, warnings3 = load_003(root)
    fourth, warnings4 = load_004(root)
    if third is None or warnings3 or len(fourth) != 1 or warnings4:
        raise ValueError("Both completed experiments must validate: " + "; ".join(warnings3 + warnings4))
    fourth = fourth[0]
    a, b = third["diagnostic_evidence"], fourth["diagnostic_evidence"]
    if (third["outcome"] != "compiler_equivalent_screening_skipped"
            or fourth["outcome"] != "compiler_equivalent_screening_skipped"
            or third["correctness"] != fourth["correctness"] != "PASS"
            or a["latency_measured"] is not False or b["latency_measured"] is not False
            or third["screening"] is not None or fourth["screening"] is not None
            or b["source_identity"]["experiment_003_id"] != third["id"]
            or b["source_identity"]["experiment_003_result_sha256"] != a["artifact_hashes"]["result.json"]
            or any(third[k] != fourth[k] for k in ("baseline_sha256", "launcher_sha256",
                                                "trusted_implementation_sha256", "original_muse_source_sha256"))):
        raise ValueError("Experiment provenance or completion mismatch")
    folder3 = root / Path(third["artifact_path"]).parent
    counts3 = {}
    for side in ("baseline", "trusted"):
        path = folder3 / f"{side}.sass"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Experiment 003 SASS unavailable")
        counts3[side] = sass_counts(sass_instructions(path.read_text()))
    folder4 = root / Path(fourth["artifact_path"]).parent
    result4 = json.loads((folder4 / "result.json").read_bytes())  # hashed by the validator
    comparisons = result4["comparisons"]
    if counts3 != comparisons["control"]["sass_counts"]:
        raise ValueError("Experiment 003 and 004 control SASS counts disagree")
    def observation(entry, comparison, *, shape, counts, correctness_cases):
        return {"shape_mnk": shape, "dtype": entry["dtype"], "gpu": entry["gpu"],
                "configuration": entry["configuration"], "correctness": entry["correctness"],
                "correctness_cases": correctness_cases,
                "executable_identical_within_workload": comparison["executable_identical"],
                "metadata_identical_within_workload": comparison["metadata_identical"],
                "sass_counts": counts, "resources": comparison["resources"],
                "runtime_measured": False}
    c, p = comparisons["control"], comparisons["partial_m"]
    if any(not x["executable_identical"] or not x["metadata_identical"] or x["meaningful_compiler_difference"] for x in (c,p)):
        raise ValueError("Unexpected compiler comparison")
    return {"schema": "muse_cross_experiment_evidence_v1", "experiments": {
        "003": {"evidence_id": third["id"], "result_sha256": a["artifact_hashes"]["result.json"],
                "artifact_path": third["artifact_path"], "observations": {
                    "source_identity": {"baseline_sha256": third["baseline_sha256"],
                        "reviewed_implementation_sha256": third["trusted_implementation_sha256"],
                        "original_muse_source_sha256": third["original_muse_source_sha256"],
                        "exact_generated_source_evaluated": False},
                    "control_workload": observation(third, {**c, "resources": third["compiler"]["resources"]},
                                                    shape=third["shape_mnk"], counts=counts3,
                                                    correctness_cases=third["correctness_cases"]),
                    "limitations": a["limitations"]}},
        "004": {"evidence_id": fourth["id"], "result_sha256": b["artifact_hashes"]["result.json"],
                "artifact_path": fourth["artifact_path"], "observations": {
                    "source_identity": b["source_identity"],
                    "control_workload": observation(fourth, c, shape=[128, 3072, 768], counts=c["sass_counts"], correctness_cases=[4, 4]),
                    "partial_m_workload": observation(fourth, p, shape=[130, 3072, 768], counts=p["sass_counts"], correctness_cases=[4, 4]),
                    "cross_workload_caution": "Differences between M=128 and M=130 describe different workloads, not an optimization benefit.",
                    "limitations": b["limitations"]}}}}


def baseline(package):
    ids = [package["experiments"][n]["evidence_id"] for n in ("003", "004")]
    return {"name": "deterministic_stage_count_control", "hypothesis":
        "At fixed M=130, changing only the pipeline stage count from three to two changes compiled resource use or executable structure; any runtime effect remains unknown.",
        "experiment": {"workload": "FP16 GEMM M=130, N=3072, K=768 on RTX 4060 Ti",
            "controlled_variables": "Use the baseline source, same inputs, tile sizes, warps, compiler, GPU and correctness policy; vary only num_stages 3 versus 2.",
            "expected_observations": "Compare correctness, executable SASS, registers, shared memory and spills. Time only after human approval and a meaningful compiled difference.",
            "confounders": "Compiler version, clocks, caches and changed launch conditions if later timing is approved."},
        "decision_criteria": {"support": "Correctness passes and stage count changes compiled resources or executable structure.",
            "weaken": "Compilation differs but resource and executable differences do not match the proposed mechanism.",
            "reject": "The changed stage count compiles to the same relevant executable and resources, or correctness fails."},
        "negative_result_value": "A matching executable would rule out stage-count code-generation leverage for this fixed workload and environment.",
        "stop_condition": "Stop after one validated comparison if executable and resources match, or if correctness fails.",
        "citations": [{"evidence_id": ids[0], "observation_key": "control_workload"},
                      {"evidence_id": ids[1], "observation_key": "partial_m_workload"}]}


def prompt_for(package):
    example = {key: {"text": "Specific scientific statement for human review.", "citations": [
        {"evidence_id": package["experiments"]["003"]["evidence_id"], "observation_key": "control_workload"}]}
        for key in FIELDS if key not in ("proposed_experiment_005", "decision_criteria")}
    example["proposed_experiment_005"] = {k: "Specific test plan, hypothetical observations only." for k in PLAN_FIELDS}
    example["proposed_experiment_005"]["citations"] = [{"evidence_id": package["experiments"]["004"]["evidence_id"], "observation_key": "partial_m_workload"}]
    example["decision_criteria"] = {k: "Falsifiable observation." for k in DECISION_FIELDS}
    example["decision_criteria"]["citations"] = [{"evidence_id": package["experiments"]["004"]["evidence_id"], "observation_key": "partial_m_workload"}]
    return ("You are proposing ONE advisory Experiment 005. Return only JSON with exactly SCHEMA fields. "
        "Cite supplied evidence IDs and observation keys for every historical claim. Distinguish scope-specific "
        "compiler equivalence from universal equivalence. No runtime measurements exist: do not invent timings, "
        "speedups or predicted improvement. Outcomes must be hypothetical. Reject or weaken repeated mask hoisting "
        "unless a materially different controlled test is explained. Specify fixed workload, one changed variable, "
        "correctness and compiler observations, confounders, support/weaken/reject criteria and a stop condition. "
        "No code or execution request. Human review is required before implementation or execution. "
        "Numeric claims about past experiments must be grounded in cited observations.\nSCHEMA\n"
        + json.dumps(example, sort_keys=True) + "\nEVIDENCE\n" + json.dumps(package, sort_keys=True))


def parse_response(raw, package):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 30000:
        raise ValueError("missing or oversized response")
    try: answer = json.loads(raw)
    except ValueError as exc: raise ValueError("response is not JSON") from exc
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        raise ValueError("incorrect proposal fields")
    ids = {item["evidence_id"]: set(item["observations"]) for item in package["experiments"].values()}
    all_text = []
    for field in FIELDS:
        item = answer[field]
        keys = (set(PLAN_FIELDS) | {"citations"} if field == "proposed_experiment_005" else
                set(DECISION_FIELDS) | {"citations"} if field == "decision_criteria" else {"text", "citations"})
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError(f"invalid {field} structure")
        for key in keys - {"citations"}:
            value = item[key]
            if not isinstance(value, str) or not value.strip() or len(value) > 1400 or any(t in value for t in ("```", "@triton.jit", "\x00")):
                raise ValueError(f"invalid {field}.{key}")
            all_text.append(value)
        refs = item["citations"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= 5 or any(
                not isinstance(r, dict) or set(r) != {"evidence_id", "observation_key"}
                or r["evidence_id"] not in ids or r["observation_key"] not in ids[r["evidence_id"]] for r in refs):
            raise ValueError(f"invalid {field} citation")
    prose = " ".join(all_text)
    performance_pattern = re.compile(
        r"\b(?:achieved|measured|observed|confirmed|proved)\b.{0,45}\b(?:speedup|faster|slower|latency|runtime gain|performance gain)\b"
        r"|\b(?:speedup|faster|slower|latency|runtime gain|performance gain)\b.{0,45}\b(?:achieved|measured|observed|confirmed|proved)\b", re.I)
    if NUMERIC_PERFORMANCE.search(prose) or any(
            performance_pattern.search(sentence) and not re.search(r"\b(?:no|not|never|without|unmeasured)\b", sentence, re.I)
            for sentence in re.split(r"[.!?;]+", prose)):
        raise ValueError("unsupported runtime performance claim")
    if re.search(r"\b(?:was|were|did|has|have)\s+(?:\w+\s+){0,2}(?:faster|slower|improved|regressed)\b", prose, re.I):
        raise ValueError("unsupported observed performance outcome")
    if re.search(r"\b(?:universally|all shapes|all GPUs|always equivalent)\b", prose, re.I):
        raise ValueError("universal compiler equivalence unsupported")
    if re.search(r"\b(?:execute|run|implement)\s+(?:it|now|immediately|experiment 005)\b", prose, re.I):
        raise ValueError("proposal requests immediate execution")
    experiment = answer["proposed_experiment_005"]
    if not re.search(r"\b(?:hold|fixed|same|control|vary only|change only)\b", experiment["controlled_variables"], re.I):
        raise ValueError("uncontrolled experiment")
    if re.search(r"mask.hoist", answer["next_hypothesis"]["text"] + " " + experiment["controlled_variables"], re.I) and not re.search(r"\b(?:different|distinct|new mechanism)\b", answer["research_rationale"]["text"], re.I):
        raise ValueError("repeated mask-hoisting hypothesis")
    return answer


def comparison(answer, package):
    base = baseline(package)
    checks = {"evidence_grounding": "Citation keys valid; human checks claim-to-citation fit.",
        "falsifiability": "Support, weaken, reject, and stop fields are present; human checks discrimination.",
        "variable_control": "Control text is present; human checks whether it isolates one mechanism.",
        "negative_result_value": "Human review: does a negative result eliminate a plausible mechanism?",
        "rtx_4060_ti_feasibility": "Human review: compile/resource feasibility on the recorded GPU.",
        "avoids_repetition": "Mask-hoisting repetition check passed; human checks material difference."}
    return {"rubric_version": "v1", "muse": checks,
            "baseline": {"proposal": base, "review": checks},
            "judgment": "No automatic winner. Human scientific review required."}


def _base(root):
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / "evaluations", root / "evaluations/muse-research-planning", base)):
        raise ValueError("linked audit path")
    return base


def load_accepted(root=ROOT, package=None):
    package = package or evidence_package(root)
    path = _base(root) / "accepted.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 150000:
        return None
    try:
        saved = json.loads(path.read_bytes())
        if (saved["status"] != "accepted" or saved["model"] != MODEL or saved["response_model"] != MODEL
                or saved["prompt_version"] != PROMPT_VERSION or saved["evidence_sha256"] != digest(canonical(package))
                or saved["evidence_package"] != package or saved["request_options"] != REQUEST_OPTIONS
                or saved["prompt"] != prompt_for(package) or saved["proposal"] != parse_response(saved["raw_response"], package)
                or saved["comparison"] != comparison(saved["proposal"], package)):
            return None
        require_complete_text(saved["raw_response"], saved["response_metadata"])
        return saved
    except (OSError, ValueError, KeyError, TypeError): return None


def request_once(client, root=ROOT):
    package = evidence_package(root)
    base = _base(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "request.lock").open("a+b") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return "request_in_progress"
        if load_accepted(root, package): return "already_accepted"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = base / stamp
        folder.mkdir()
        prompt = prompt_for(package)
        audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "prompt_version": PROMPT_VERSION,
                 "model": MODEL, "prompt": prompt, "request_options": REQUEST_OPTIONS,
                 "evidence_package": package, "evidence_sha256": digest(canonical(package)),
                 "raw_response": None, "response_model": None}
        completed = False
        try:
            response = client.responses.create(model=MODEL, input=prompt, **REQUEST_OPTIONS)
            completed = True
            audit["raw_response"], audit["response_metadata"] = inspect_response(response)
            audit["response_model"] = _get(response, "model")
            audit["response_id"] = _get(response, "id")
            require_complete_text(audit["raw_response"], audit["response_metadata"])
            if audit["response_model"] != MODEL: raise ValueError("response model mismatch")
            audit["proposal"] = parse_response(audit["raw_response"], package)
            if evidence_package(root) != package: raise ValueError("source evidence changed during request")
            audit["comparison"] = comparison(audit["proposal"], package)
            audit["status"] = "accepted"
            (folder / "audit.json").write_bytes(canonical(audit))
            (base / "accepted.json").write_bytes(canonical(audit))
            return "accepted"
        except Exception as exc:
            audit["status"] = "rejected"
            audit["rejection_kind"] = getattr(exc, "kind", "invalid_proposal" if completed else "request_error")
            audit["rejection"] = f"{type(exc).__name__}: {str(exc)[:250]}" if completed and isinstance(exc, ValueError) else type(exc).__name__
            (folder / "audit.json").write_bytes(canonical(audit))
            return "rejected"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-experiment advisory Muse research planning")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--request-from-muse", action="store_true")
    args = parser.parse_args(argv)
    package = evidence_package(ROOT)
    if args.prepare_only:
        print(json.dumps({"evidence": package, "baseline": baseline(package), "prompt": prompt_for(package)}, indent=2))
        return 0
    if load_accepted(ROOT, package):
        print("already_accepted"); return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key: parser.error("MODEL_API_KEY is required")
    status = request_once(OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0), ROOT)
    print(status)
    return 0 if status in ("accepted", "already_accepted") else 1


if __name__ == "__main__": raise SystemExit(main())
