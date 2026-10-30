"""Experiment 006: advisory reflection on validated Experiment 005 evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re

from dashboard_data import ROOT
from experiment_005_evidence import (SECTION as SOURCE_SECTION, latest, manifest,
                                     read_manifest, sha, validate_timing, validate_verify)
from muse_debrief import (API_BASE_URL, MODEL, NUMBER, _get, _numbers, canonical,
                          digest, inspect_response, require_complete_text)

SECTION = Path("evaluations/muse-research-reflection/v1")
PROMPT_VERSION = "experiment_006_reflection_v1"
REQUEST_OPTIONS = {"max_output_tokens": 8000, "reasoning": {"effort": "low"}}
DECISIONS = {"abandon", "revise", "investigate_further"}
FIELDS = {"decision", "evidence_assessment", "mechanism_assessment",
          "remaining_uncertainties", "next_research_direction", "proposed_validation"}
TEXT_FIELDS = FIELDS - {"decision"}


def evidence_package(root=ROOT):
    """Use the Experiment 005 manifest and comparison validators; never run a GPU stage."""
    root = Path(root).resolve()
    record = latest(root)
    if record is None:
        raise ValueError("validated Experiment 005 verification is unavailable")
    run_id = record["run_id"]
    if record["screening"] is None:
        raise ValueError("validated Experiment 005 screening is unavailable")
    verify_dir, verified = validate_verify(root, run_id)
    source = root / SOURCE_SECTION / run_id
    proposal = json.loads((verify_dir / "wild_lab_accepted.json").read_bytes())
    idea = proposal["ideas"]["ideas"][1]
    observations = {
        "original_hypothesis": idea,
        "verification": {"correctness_passes": len(verified["correctness"]),
                         "shapes": verified["shapes"], "configuration": verified["config"],
                         "environment": verified["environment"],
                         "limitations": verified["limitations"]},
        "screening_scope": {"timing_scope": record["screening"]["timing_scope"],
                            "stage_decision": record["screening"]["decision"]},
        "confirmation_status": "validated" if record["confirmation"] else "missing",
        "profiling_status": "not validated by Experiment 005 evidence pipeline",
    }
    for row in record["screening"]["comparisons"]:
        observations[f"screening:{row['shape']}:{row['opponent']}"] = {
            "shape": row["shape"], "opponent": row["opponent"],
            "median_paired_ratio": row["summary"]["median_paired_ratio"],
            "paired_ratio_mad": row["summary"]["paired_ratio_mad"],
            "bootstrap_median_ratio_95pct": row["summary"]["bootstrap_median_ratio_95pct"],
            "baseline_median_ms": row["summary"]["baseline_median_ms"],
            "candidate_median_ms": row["summary"]["candidate_median_ms"],
            "rounds": row["summary"]["rounds"],
            "one_time_map_setup_ms": row["one_time_map_setup_ms"],
        }
    if record["confirmation"]:
        for row in record["confirmation"]["comparisons"]:
            observations[f"confirmation:{row['shape']}:{row['opponent']}"] = row["summary"]
    def validated_stage_dir(stage):
        return next(path for path in sorted((source / stage).iterdir(), reverse=True)
                    if path.is_dir() and not path.is_symlink()
                    and validate_timing(root, run_id, stage, path.name)[1] == record[stage])
    screening_dir = validated_stage_dir("screening")
    hashes = {"verify_manifest_sha256": sha((verify_dir / "manifest.json").read_bytes()),
              "verify_result_sha256": sha((verify_dir / "result.json").read_bytes()),
              "proposal_sha256": sha((verify_dir / "wild_lab_accepted.json").read_bytes()),
              "screening_manifest_sha256": sha((screening_dir / "manifest.json").read_bytes()),
              "screening_result_sha256": sha((screening_dir / "result.json").read_bytes())}
    if record["confirmation"]:
        confirmation_dir = validated_stage_dir("confirmation")
        hashes["confirmation_manifest_sha256"] = sha((confirmation_dir / "manifest.json").read_bytes())
        hashes["confirmation_result_sha256"] = sha((confirmation_dir / "result.json").read_bytes())
    return {"schema": "experiment_006_evidence_v1", "evidence_id": f"experiment_005:{run_id}",
            "source_hashes": hashes, "observations": observations,
            "evaluation_criteria": {
                "latency_support": "For one shape, screening and independent confirmation bootstrap lower bounds must both exceed 1 against row_major and grouped_m4.",
                "mechanism_support": "Cache locality requires profiling evidence; mapping or compiler differences alone do not establish cause.",
                "scope": "Intervals are descriptive; screening alone is unconfirmed. Human review is required for broader conclusions."}}


def prompt_for(package):
    evidence_id = package["evidence_id"]
    def field(text, key):
        return {"text": text, "citations": [{"evidence_id": evidence_id,
                                              "observation_key": key}]}
    example = {"decision": "<abandon|revise|investigate_further>",
               "evidence_assessment": field("Describe observed measurements and their uncertainty.", "screening_scope"),
               "mechanism_assessment": field("Label the causal explanation as an inference.", "original_hypothesis"),
               "remaining_uncertainties": field("Acknowledge confirmation and profiling status.", "confirmation_status"),
               "next_research_direction": field("Give one research direction for review.", "original_hypothesis"),
               "proposed_validation": field("Describe a discriminating validation step, without executing it.", "verification")}
    return ("Reflect on the original accepted Hilbert Tile Traversal hypothesis using only EVIDENCE. "
            "Choose decision independently: abandon, revise, or investigate_further. The SCHEMA example "
            "shows structure only and does not suggest a decision. Each prose field needs 1-8 local evidence "
            "citations. Separate observed measurements from inferred explanations explicitly. Cite every "
            "numerical measurement locally. Do not invent experiments, measurements, confirmation, profiling, "
            "or performance claims. Apply the predefined evaluation criteria without treating a descriptive "
            "interval as causal proof. Acknowledge missing confirmation and profiling evidence when applicable. "
            "Keep proposals advisory; do not request execution or provide code. Return JSON only.\nSCHEMA\n"
            + json.dumps(example, sort_keys=True) + "\nEVIDENCE\n"
            + json.dumps(package, sort_keys=True, allow_nan=False))


def parse_response(raw, package):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 30000:
        raise ValueError("missing or oversized reflection")
    try:
        answer = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("reflection is not JSON") from exc
    if not isinstance(answer, dict) or set(answer) != FIELDS or answer["decision"] not in DECISIONS:
        raise ValueError("invalid reflection fields or decision")
    observations = package["observations"]
    for name in TEXT_FIELDS:
        item = answer[name]
        if not isinstance(item, dict) or set(item) != {"text", "citations"}:
            raise ValueError(f"invalid {name} structure")
        prose, refs = item["text"], item["citations"]
        if (not isinstance(prose, str) or not prose.strip() or len(prose) > 1800
                or "\x00" in prose or "```" in prose or not isinstance(refs, list)
                or not 1 <= len(refs) <= 8):
            raise ValueError(f"invalid {name} text or citations")
        keys = set()
        for ref in refs:
            if (not isinstance(ref, dict) or set(ref) != {"evidence_id", "observation_key"}
                    or ref["evidence_id"] != package["evidence_id"]
                    or ref["observation_key"] not in observations
                    or ref["observation_key"] in keys):
                raise ValueError("invalid or duplicate evidence reference")
            keys.add(ref["observation_key"])
        allowed = set().union(*(_numbers(observations[key]) for key in keys))
        experiment_mentions = re.findall(r"\b(?:experiment|exp\.)\s*0*(\d+)\b", prose, re.I)
        if any(int(number) != 5 for number in experiment_mentions):
            raise ValueError("reference to nonexistent experiment")
        measurement_prose = re.sub(r"\b(?:experiment|exp\.)\s*0*5\b", "Experiment", prose, flags=re.I)
        if any(number not in allowed for number in NUMBER.findall(measurement_prose)):
            raise ValueError("unsupported numerical measurement")
        if re.search(r"\b(?:confirmed|proven|established)\s+(?:a\s+)?(?:speedup|latency|cache|locality|performance|mechanism)\b", prose, re.I):
            raise ValueError("unsupported performance or mechanism claim")
        if re.search(r"\b(?:improved|increased|better)\s+(?:L2|cache|DRAM|locality|coalescing)\b", prose, re.I):
            raise ValueError("unprofiled mechanism claim")
        if re.search(r"\b(?:achieved|delivered|proved|confirmed)\s+(?:a\s+)?(?:speedup|performance gain|faster runtime)\b|\b(?:is|was)\s+(?:faster|slower)\b", prose, re.I):
            raise ValueError("unsupported categorical performance claim")
        if experiment_mentions and not any(k.startswith("screening:") or k in {"verification", "original_hypothesis"} for k in keys):
            raise ValueError("experiment reference lacks evidence citation")
        if re.search(r"\b(?:measured|observed|achieved)\b.{0,50}\b(?:speedup|latency|faster|slower|runtime)\b", prose, re.I) and not any(k.startswith(("screening:", "confirmation:")) for k in keys):
            raise ValueError("performance claim lacks timing evidence")
        if re.search(r"\b(?:faster|slower|outperformed|speedup|slowdown)\b", prose, re.I) and not any(k.startswith(("screening:", "confirmation:")) for k in keys):
            raise ValueError("performance claim lacks timing evidence")
    uncertainty = answer["remaining_uncertainties"]["text"].lower()
    if observations["confirmation_status"] == "missing" and ("confirmation" not in uncertainty or not re.search(r"\b(?:missing|absent|unavailable|unconfirmed|not yet)\b", uncertainty)):
        raise ValueError("missing confirmation must be acknowledged")
    if "profiling" not in uncertainty or not re.search(r"\b(?:missing|absent|unavailable|unmeasured|not validated|no)\b", uncertainty):
        raise ValueError("profiling evidence limitation must be acknowledged")
    uncertainty_refs = {r["observation_key"] for r in answer["remaining_uncertainties"]["citations"]}
    if "confirmation_status" not in uncertainty_refs or "profiling_status" not in uncertainty_refs:
        raise ValueError("missing-stage citation required")
    return answer


def _base(root):
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / "evaluations", root / "evaluations/muse-research-reflection", base)):
        raise ValueError("linked reflection audit path")
    return base


def validate_audit(root, stamp):
    """Verify artifact bytes and their binding to the current validated evidence."""
    if not re.fullmatch(r"\d{8}T\d{12}Z", stamp):
        raise ValueError("invalid reflection audit ID")
    folder = _base(root) / stamp
    read_manifest(folder, schema="experiment_006_audit_v1")
    audit = json.loads((folder / "audit.json").read_bytes())
    package = evidence_package(root)
    if (audit.get("status") != "accepted" or audit.get("prompt_version") != PROMPT_VERSION
            or audit.get("model") != MODEL or audit.get("response_model") != MODEL
            or audit.get("request_options") != REQUEST_OPTIONS
            or audit.get("evidence") != package
            or audit.get("evidence_sha256") != digest(canonical(package))
            or audit.get("source_evidence_hashes") != package["source_hashes"]
            or audit.get("prompt") != prompt_for(package)
            or audit.get("parsed_reflection") != parse_response(audit.get("raw_response"), package)):
        raise ValueError("reflection audit contents mismatch")
    require_complete_text(audit["raw_response"], audit["response_metadata"])
    return audit


def request_once(client, root=ROOT):
    package = evidence_package(root)
    base = _base(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "request.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "request_in_progress"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = base / stamp
        folder.mkdir()
        audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
                 "prompt_version": PROMPT_VERSION, "model": MODEL,
                 "request_options": REQUEST_OPTIONS, "evidence": package,
                 "evidence_sha256": digest(canonical(package)), "source_evidence_hashes": package["source_hashes"],
                 "prompt": prompt_for(package), "raw_response": None,
                 "parsed_reflection": None, "response_model": None, "response_metadata": None}
        completed = False
        try:
            response = client.responses.create(model=MODEL, input=audit["prompt"], **REQUEST_OPTIONS)
            completed = True
            audit["raw_response"], audit["response_metadata"] = inspect_response(response)
            audit["response_model"] = _get(response, "model")
            audit["response_id"] = _get(response, "id")
            require_complete_text(audit["raw_response"], audit["response_metadata"])
            if audit["response_model"] != MODEL:
                raise ValueError("response model mismatch")
            audit["parsed_reflection"] = parse_response(audit["raw_response"], package)
            if evidence_package(root) != package:
                raise ValueError("source evidence changed during request")
            audit["status"] = "accepted"
        except Exception as exc:
            audit["status"] = "rejected"
            audit["rejection_kind"] = getattr(exc, "kind", "invalid_reflection" if completed else "request_error")
            audit["rejection"] = (f"{type(exc).__name__}: {str(exc)[:250]}"
                                  if completed and isinstance(exc, ValueError) else type(exc).__name__)
        (folder / "audit.json").write_bytes(canonical(audit))
        (folder / "manifest.json").write_bytes(canonical(manifest(folder, schema="experiment_006_audit_v1")))
        return audit["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experiment 006 Muse research reflection")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--request-from-muse", action="store_true")
    args = parser.parse_args(argv)
    package = evidence_package(ROOT)
    if args.prepare_only:
        print(json.dumps({"evidence": package, "prompt": prompt_for(package)}, indent=2))
        return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        parser.error("MODEL_API_KEY is required for a live request")
    status = request_once(OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0), ROOT)
    print(status)
    return 0 if status == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
