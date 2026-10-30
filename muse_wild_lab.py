"""Advisory GPU research ideation; no generated code or experiment execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re

from dashboard_data import ROOT
from muse_debrief import (API_BASE_URL, MODEL, canonical, digest, inspect_response,
                          require_complete_text, _get, NUMERIC_PERFORMANCE)
from muse_research_plan import evidence_package as validated_evidence

SECTION = Path("evaluations/muse-wild-research-lab/v1")
PROMPT_VERSION = "wild_ideation_v1"
REQUEST_OPTIONS = {"max_output_tokens": 8000, "reasoning": {"effort": "low"}}
CATEGORIES = ("plausible_overlooked", "cross_disciplinary", "high_risk_transformative")
IDEA_FIELDS = {"category", "name", "core_mechanism", "why_it_might_work",
               "what_makes_it_unconventional", "necessary_assumptions", "biggest_obstacle",
               "disproof_observation", "smallest_proof_of_concept", "numerical_equivalence",
               "evidence_references"}
REFERENCE_FIELDS = {"evidence_id", "observation_key"}
NUMERIC_EQUIVALENCE = {"exact", "approximate"}
PERFORMANCE = re.compile(
    r"\b(?:measured|observed|achieved|demonstrated|confirmed|proved|delivered)\b.{0,55}"
    r"\b(?:speedup|faster|slower|latency|runtime|throughput|performance gain)\b|"
    r"\b(?:speedup|faster|slower|latency|runtime|throughput|performance gain)\b.{0,55}"
    r"\b(?:measured|observed|achieved|demonstrated|confirmed|proved|delivered)\b", re.I)
CODE = re.compile(r"```|@triton\.jit|\b(?:import\s+triton|def\s+\w+\s*\(|tl\.load\s*\(|tl\.store\s*\()", re.I)


def context_package(root=ROOT):
    """Project validated records into a bounded, factual brainstorming context."""
    root = Path(root).resolve()
    verified = validated_evidence(root)
    three, four = (verified["experiments"][key] for key in ("003", "004"))
    path = root / four["artifact_path"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 200_000:
        raise ValueError("validated Experiment 004 environment unavailable")
    raw = path.read_bytes()
    if digest(raw) != four["result_sha256"]:
        raise ValueError("Experiment 004 environment hash mismatch")
    environment = json.loads(raw)["environment"]
    gpu = environment["gpu"]
    control = three["observations"]["control_workload"]
    partial = four["observations"]["partial_m_workload"]
    return {"schema": "wild_lab_context_v1", "validated_evidence_sha256": digest(canonical(verified)),
            "environment": {"gpu": gpu["name"], "compute_capability": [gpu["major"], gpu["minor"]],
                            "memory_bytes": gpu["total_memory"], "triton": environment["triton"],
                            "torch": environment["torch"], "cuda": environment["torch_cuda"]},
            "experiments": {
                "003": {"evidence_id": three["evidence_id"], "result_sha256": three["result_sha256"],
                    "observations": {"workload": control["shape_mnk"], "configuration": control["configuration"],
                        "correctness": control["correctness"], "compiler_equivalent_for_compared_sources":
                        control["executable_identical_within_workload"],
                        "runtime_measured": False,
                        "original_muse_source_evaluated": False}},
                "004": {"evidence_id": four["evidence_id"], "result_sha256": four["result_sha256"],
                    "observations": {"control_workload": [128, 3072, 768],
                        "partial_m_workload": partial["shape_mnk"],
                        "configuration": partial["configuration"], "correctness": partial["correctness"],
                        "compiler_equivalent_for_compared_sources_on_both_workloads": True,
                        "runtime_measured": False,
                        "cross_workload_caution": four["observations"]["cross_workload_caution"]}}},
            "interpretation_limit": "The unchanged reviewed mask-hoisting variant and baseline compiled equivalently within the recorded configurations. This does not establish universal equivalence or runtime performance. The original Muse-generated source was not evaluated."}


def prompt_for(context):
    example = {"ideas": [{"category": category, "name": "Memorable name",
        "core_mechanism": "Concise speculative mechanism.",
        "why_it_might_work": "Speculative causal rationale, conditional on assumptions.",
        "what_makes_it_unconventional": "What departs from ordinary tuning.",
        "necessary_assumptions": "Hardware, compiler, numerical or workload assumptions.",
        "biggest_obstacle": "One central technical obstacle.",
        "disproof_observation": "An observation that would refute the idea.",
        "smallest_proof_of_concept": "Advisory description only; no code or execution request.",
        "numerical_equivalence": "exact",
        "evidence_references": [{"evidence_id": context["experiments"]["003"]["evidence_id"],
                                 "observation_key": "compiler_equivalent_for_compared_sources"}]}
        for category in CATEGORIES]}
    return ("Wild Research Lab: brainstorm exactly three genuinely distinct GPU kernel optimization concepts "
            "for National Tensor. Return one JSON object matching SCHEMA exactly and in the shown category order. "
            "Categories: plausible but overlooked; unconventional cross-disciplinary; high-risk potentially "
            "transformative. Explore unusual memory layouts, warp cooperation, scheduling, numerical representations, "
            "mathematical restructuring, compiler behavior, or other promising directions. Be creative: ideas need not "
            "change one variable or resemble previous experiments. Separate verified historical facts from speculative "
            "mechanisms. Cite context evidence IDs and observation keys for historical claims. Prior mask-hoisting "
            "compiler equivalence is information, not a creativity constraint; do not simply repeat that hypothesis. "
            "Do not assert measured improvement, invent timing, or predict a precise speedup. Equivalence must be "
            "labeled exact or approximate; state necessary assumptions. Every idea is advisory and requires human "
            "review before implementation. Do not provide executable code, request execution, or approve experiments. "
            "No novelty scores. Keep each field concise.\nSCHEMA\n"
            + json.dumps(example, sort_keys=True) + "\nCONTEXT\n" + json.dumps(context, sort_keys=True))


def parse_response(raw, context, *, max_evidence_references=4):
    if not isinstance(max_evidence_references, int) or not 1 <= max_evidence_references <= 16:
        raise ValueError("invalid evidence reference limit")
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 30000:
        raise ValueError("missing or oversized response")
    try:
        result = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("response is not JSON") from exc
    if not isinstance(result, dict) or set(result) != {"ideas"} or not isinstance(result["ideas"], list) or len(result["ideas"]) != 3:
        raise ValueError("response requires exactly three ideas")
    available = {row["evidence_id"]: set(row["observations"])
                 for row in context["experiments"].values()}
    names, mechanisms = set(), set()
    for index, idea in enumerate(result["ideas"]):
        if not isinstance(idea, dict) or set(idea) != IDEA_FIELDS or idea["category"] != CATEGORIES[index]:
            raise ValueError("idea structure or category order mismatch")
        if not isinstance(idea["numerical_equivalence"], str) or idea["numerical_equivalence"] not in NUMERIC_EQUIVALENCE:
            raise ValueError("invalid numerical equivalence label")
        for key in IDEA_FIELDS - {"category", "numerical_equivalence", "evidence_references"}:
            value = idea[key]
            if (not isinstance(value, str) or not value.strip() or len(value) > 900
                    or "\x00" in value or CODE.search(value)):
                raise ValueError(f"invalid {key} text or executable code")
            for sentence in re.split(r"[.!?;]+", value):
                if (NUMERIC_PERFORMANCE.search(sentence)
                        or PERFORMANCE.search(sentence) and not re.search(r"\b(?:no|not|never|without|unmeasured)\b", sentence, re.I)
                        or re.search(r"\b(?:was|were|has|have)\s+(?:\w+\s+){0,2}(?:faster|slower|improved|regressed)\b", sentence, re.I)):
                    raise ValueError("unsupported observed performance claim")
            if re.search(r"\b(?:run|execute|implement)\s+(?:this|it|now|immediately)\b", value, re.I):
                raise ValueError("idea requests immediate execution")
        refs = idea["evidence_references"]
        if not isinstance(refs, list):
            raise ValueError("malformed evidence references")
        if not 1 <= len(refs) <= max_evidence_references:
            raise ValueError(f"evidence reference count must be 1 to {max_evidence_references}")
        if any(
                not isinstance(ref, dict) or set(ref) != REFERENCE_FIELDS
                or not isinstance(ref["evidence_id"], str)
                or not isinstance(ref["observation_key"], str)
                or ref["evidence_id"] not in available
                or ref["observation_key"] not in available[ref["evidence_id"]] for ref in refs):
            raise ValueError("invalid evidence reference")
        name = re.sub(r"\W+", "", idea["name"].casefold())
        mechanism = re.sub(r"\W+", "", idea["core_mechanism"].casefold())
        if name in names or mechanism in mechanisms:
            raise ValueError("duplicate idea name or mechanism")
        names.add(name); mechanisms.add(mechanism)
        if re.search(r"mask[ -]?hoist", idea["core_mechanism"] + " " + idea["smallest_proof_of_concept"], re.I):
            raise ValueError("repeated mask-hoisting direction")
    return result


def _base(root):
    root = Path(root).resolve()
    base = root / SECTION
    if any(path.is_symlink() for path in (root / "evaluations", root / "evaluations/muse-wild-research-lab", base)):
        raise ValueError("linked ideation audit path")
    return base


def load_accepted(root=ROOT, context=None):
    context = context or context_package(root)
    path = _base(root) / "accepted.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 120_000:
        return None
    try:
        raw = path.read_bytes()
        saved = json.loads(raw)
        if (saved["status"] != "accepted" or saved["prompt_version"] != PROMPT_VERSION
                or saved["model"] != MODEL or saved["response_model"] != MODEL
                or saved["context_sha256"] != digest(canonical(context)) or saved["context"] != context
                or saved["prompt"] != prompt_for(context) or saved["request_options"] != REQUEST_OPTIONS
                or saved["ideas"] != parse_response(saved["raw_response"], context)):
            return None
        require_complete_text(saved["raw_response"], saved["response_metadata"])
        round_id = saved["round_id"]
        if not isinstance(round_id, str) or not re.fullmatch(r"\d{8}T\d{12}Z", round_id):
            return None
        round_dir = _base(root) / round_id
        attempt = round_dir / "audit.json"
        if (round_dir.is_symlink() or not attempt.is_file() or attempt.is_symlink() or attempt.stat().st_size > 120_000
                or attempt.read_bytes() != raw):
            return None
        return saved
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _matching_attempt_exists(base, context):
    """A normal request is idempotent after an accepted or rejected attempt."""
    if not base.is_dir():
        return False
    expected = digest(canonical(context))
    for folder in base.iterdir():
        if (not folder.is_dir() or folder.is_symlink()
                or not re.fullmatch(r"\d{8}T\d{12}Z", folder.name)):
            continue
        path = folder / "audit.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 120_000:
            continue
        try:
            audit = json.loads(path.read_bytes())
            if (audit.get("context_sha256") == expected and audit.get("context") == context
                    and audit.get("prompt_version") == PROMPT_VERSION
                    and audit.get("model") == MODEL and audit.get("prompt") == prompt_for(context)
                    and audit.get("request_options") == REQUEST_OPTIONS
                    and audit.get("status") in {"accepted", "rejected"}):
                return True
        except (OSError, ValueError, TypeError):
            continue
    return False


def request_once(client, root=ROOT, *, new_round=False):
    context = context_package(root)
    base = _base(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "request.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "request_in_progress"
        prior = load_accepted(root, context)
        if prior and not new_round:
            return "already_accepted"
        if not new_round and _matching_attempt_exists(base, context):
            return "already_attempted"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = base / stamp
        folder.mkdir()
        audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
                 "prompt_version": PROMPT_VERSION, "round_id": stamp,
                 "explicit_new_round": bool(new_round), "model": MODEL,
                 "context": context, "context_sha256": digest(canonical(context)),
                 "prompt": prompt_for(context), "request_options": REQUEST_OPTIONS,
                 "previous_accepted_sha256": digest((base / "accepted.json").read_bytes()) if prior else None,
                 "raw_response": None, "response_model": None}
        request_completed = False
        try:
            response = client.responses.create(model=MODEL, input=audit["prompt"], **REQUEST_OPTIONS)
            request_completed = True
            audit["raw_response"], audit["response_metadata"] = inspect_response(response)
            audit["response_model"] = _get(response, "model")
            audit["response_id"] = _get(response, "id")
            require_complete_text(audit["raw_response"], audit["response_metadata"])
            if audit["response_model"] != MODEL:
                raise ValueError("response model mismatch")
            audit["ideas"] = parse_response(audit["raw_response"], context)
            if context_package(root) != context:
                raise ValueError("source evidence changed during request")
            audit["status"] = "accepted"
            (folder / "audit.json").write_bytes(canonical(audit))
            temporary = base / f".{stamp}.accepted.tmp"
            temporary.write_bytes(canonical(audit))
            os.replace(temporary, base / "accepted.json")
            return "accepted"
        except Exception as exc:
            audit["status"] = "rejected"
            audit["rejection_kind"] = getattr(exc, "kind", "invalid_ideas" if request_completed else "request_error")
            audit["rejection"] = (f"{type(exc).__name__}: {str(exc)[:250]}"
                                  if request_completed and isinstance(exc, ValueError) else type(exc).__name__)
            (folder / "audit.json").write_bytes(canonical(audit))
            return "rejected"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Advisory Muse Wild Research Lab")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true", help="show context and prompt; no API call")
    modes.add_argument("--request-from-muse", action="store_true", help="one live ideation request")
    parser.add_argument("--new-round", action="store_true", help="explicitly request a fresh round after acceptance")
    args = parser.parse_args(argv)
    if args.new_round and not args.request_from_muse:
        parser.error("--new-round requires --request-from-muse")
    context = context_package(ROOT)
    if args.prepare_only:
        print(json.dumps({"context": context, "prompt": prompt_for(context)}, indent=2))
        return 0
    if not args.new_round and (load_accepted(ROOT, context) or _matching_attempt_exists(_base(ROOT), context)):
        print("already_accepted" if load_accepted(ROOT, context) else "already_attempted")
        return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        parser.error("MODEL_API_KEY is required for a live request")
    status = request_once(OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0),
                          ROOT, new_round=args.new_round)
    print(status)
    return 0 if status in {"accepted", "already_accepted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
