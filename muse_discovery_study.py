"""Experiment 007: paired advisory hypothesis requests and manual blind review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import re

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import API_BASE_URL, MODEL, _get, canonical, digest, inspect_response, require_complete_text
from muse_research_reflection import SECTION as REFLECTION_SECTION, validate_audit
from muse_wild_lab import (CATEGORIES, CODE, IDEA_FIELDS, NUMERIC_PERFORMANCE, REQUEST_OPTIONS, context_package,
                           parse_response as parse_wild_response)

SECTION = Path("evaluations/muse-discovery-study/v1")
VERSION = "experiment_007_v1"
STAMP = re.compile(r"\d{8}T\d{12}Z\Z")
CONDITIONS = ("A", "B")
CRITERIA = ("novelty", "mechanism_specificity", "falsifiability",
            "evidence_alignment", "implementation_cost")
PROPOSAL_FIELDS = IDEA_FIELDS | {"testable_prediction"}
INSTRUCTIONS = (
    "Propose exactly three distinct advisory GPU optimization hypotheses for National Tensor. "
    "Return one JSON object with exactly the SCHEMA fields and categories in the shown order. "
    "For each idea, give a specific testable prediction with an observable comparison and a disproof observation. "
    "Explain mechanism, assumptions, obstacle, smallest proof of concept, and exact or approximate numerical equivalence. "
    "Cite supplied evidence IDs and observation keys for historical claims. Keep proposal prose self-contained: "
    "do not mention prior experiment numbers, prior results, or a prior reflection in the prose; put provenance in citations. "
    "Treat mechanisms as hypotheses. Do not invent measured performance, predict a precise speedup, provide code, "
    "request execution, or approve experiments. No confidence or quality scores. Return JSON only."
)


def _base(root):
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / "evaluations", root / "evaluations/muse-discovery-study", base)):
        raise ValueError("linked study path")
    return base


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _folder(root, study_id):
    if not STAMP.fullmatch(study_id):
        raise ValueError("invalid study ID")
    return _base(root) / study_id


def _write_artifact(folder, name, payload, schema):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / name).write_bytes(canonical(payload))
    (folder / "manifest.json").write_bytes(canonical(manifest(folder, schema=schema)))


def _read_artifact(folder, name, schema):
    read_manifest(folder, schema=schema)
    return json.loads((folder / name).read_bytes())


def sources(root, reflection_id):
    """No fallback: the specified accepted reflection and all upstream evidence must validate."""
    base_context = context_package(root)
    reflection = validate_audit(root, reflection_id)
    source_path = Path(root).resolve() / REFLECTION_SECTION / reflection_id / "audit.json"
    evidence = reflection["evidence"]
    return {"base_context": base_context, "experiment_005": evidence,
            "reflection": reflection["parsed_reflection"],
            "source_hashes": {"base_context_sha256": digest(canonical(base_context)),
                              "reflection_audit_sha256": sha(source_path.read_bytes()),
                              **evidence["source_hashes"]},
            "reflection_id": reflection_id}


def context_for(source, condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    context = json.loads(json.dumps(source["base_context"]))
    if condition == "B":
        evidence = source["experiment_005"]
        context["experiments"]["005"] = {"evidence_id": evidence["evidence_id"],
                                          "observations": evidence["observations"]}
        context["feedback"] = {"experiment_005_evaluation_criteria": evidence["evaluation_criteria"],
                               "accepted_reflection": source["reflection"]}
    return context


def prompt_for(context):
    reference = context["experiments"]["003"]
    schema = {"ideas": [{**{key: "Specific advisory statement." for key in IDEA_FIELDS
                             if key not in {"category", "numerical_equivalence", "evidence_references"}},
                         "category": category, "numerical_equivalence": "exact",
                         "testable_prediction": "If the mechanism matters, median latency differs from a fixed control.",
                         "disproof_observation": "Median latency is unchanged against the fixed control.",
                         "evidence_references": [{"evidence_id": reference["evidence_id"],
                                                  "observation_key": "control_workload"}]}
                        for category in CATEGORIES]}
    return INSTRUCTIONS + "\nSCHEMA\n" + json.dumps(schema, sort_keys=True) + "\nCONTEXT\n" + json.dumps(context, sort_keys=True, allow_nan=False)


def parse_proposals(raw, context, *, max_evidence_references=4):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 35000:
        raise ValueError("missing or oversized proposals")
    try:
        answer = json.loads(raw)
    except ValueError as exc:
        raise ValueError("proposal response is not JSON") from exc
    if not isinstance(answer, dict) or set(answer) != {"ideas"} or not isinstance(answer["ideas"], list) or len(answer["ideas"]) != 3:
        raise ValueError("expected exactly three ideas")
    legacy = {"ideas": []}
    for idea in answer["ideas"]:
        if not isinstance(idea, dict) or set(idea) != PROPOSAL_FIELDS:
            raise ValueError("proposal schema mismatch")
        prediction = idea["testable_prediction"]
        prose_fields = PROPOSAL_FIELDS - {"category", "numerical_equivalence", "evidence_references"}
        if any(not isinstance(idea[key], str) for key in prose_fields):
            raise ValueError("proposal prose must be text")
        if (not isinstance(prediction, str) or not 25 <= len(prediction) <= 900
                or not re.search(r"\b(?:if|when)\b", prediction, re.I)
                or not re.search(r"\b(?:latency|runtime|cache|traffic|register|spill|instruction|throughput|correctness|resource|memory|occupancy)\b", prediction, re.I)
                or not re.search(r"\b(?:than|versus|against|control|baseline|compared)\b", prediction, re.I)
                or prediction == idea["disproof_observation"]
                or NUMERIC_PERFORMANCE.search(prediction) or CODE.search(prediction)):
            raise ValueError("prediction must identify a testable observable")
        if (len(idea["disproof_observation"]) < 25 or not re.search(
                r"\b(?:latency|runtime|cache|traffic|register|spill|instruction|throughput|correctness|resource|memory|occupancy)\b",
                idea["disproof_observation"], re.I)):
            raise ValueError("disproof observation must identify an observable")
        prose = " ".join(idea[key] for key in prose_fields)
        if re.search(r"\b(?:experiment\s*0*\d+|prior\s+(?:result|reflection|experiment)|previous\s+(?:result|reflection|experiment)|condition\s+[AB]|evidence.informed\s+researcher|baseline\s+researcher)\b", prose, re.I):
            raise ValueError("proposal prose reveals prior research condition")
        legacy["ideas"].append({key: value for key, value in idea.items() if key != "testable_prediction"})
    parse_wild_response(json.dumps(legacy), context,
                        max_evidence_references=max_evidence_references)
    return answer


def prepare_study(root, reflection_id):
    source = sources(root, reflection_id)
    contexts = {condition: context_for(source, condition) for condition in CONDITIONS}
    prompts = {condition: prompt_for(contexts[condition]) for condition in CONDITIONS}
    baseline = prompts["A"]
    if (source["experiment_005"]["evidence_id"] in baseline
            or source["reflection_id"] in baseline
            or "accepted_reflection" in baseline
            or "Hilbert Tile Traversal" in baseline):
        raise ValueError("baseline prompt leaked experimental feedback")
    study_id = _stamp()
    record = {"schema": VERSION, "study_id": study_id, "created_utc": datetime.now(timezone.utc).isoformat(),
              "model": MODEL, "request_options": REQUEST_OPTIONS,
              "source": source, "contexts": contexts, "prompts": prompts,
              "instructions_sha256": digest(INSTRUCTIONS.encode()),
              "proposal_schema_fields": sorted(PROPOSAL_FIELDS), "proposal_count": 3}
    _write_artifact(_folder(root, study_id) / "setup", "study.json", record, "experiment_007_setup_v1")
    return study_id


def load_study(root, study_id):
    record = _read_artifact(_folder(root, study_id) / "setup", "study.json", "experiment_007_setup_v1")
    source = sources(root, record["source"]["reflection_id"])
    contexts = {c: context_for(source, c) for c in CONDITIONS}
    prompts = {c: prompt_for(contexts[c]) for c in CONDITIONS}
    if (record.get("schema") != VERSION or record.get("study_id") != study_id
            or record.get("model") != MODEL or record.get("request_options") != REQUEST_OPTIONS
            or record.get("source") != source or record.get("contexts") != contexts
            or record.get("prompts") != prompts
            or record.get("instructions_sha256") != digest(INSTRUCTIONS.encode())
            or record.get("proposal_schema_fields") != sorted(PROPOSAL_FIELDS)
            or record.get("proposal_count") != 3):
        raise ValueError("study setup or source evidence mismatch")
    return record


def load_request(root, study_id, condition, *, study=None):
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    study = study or load_study(root, study_id)
    folder = _folder(root, study_id) / "requests" / condition
    audit = _read_artifact(folder, "audit.json", "experiment_007_request_v1")
    if (audit.get("status") != "accepted" or audit.get("study_sha256") != digest(canonical(study))
            or audit.get("condition") != condition or audit.get("prompt") != study["prompts"][condition]
            or audit.get("model") != MODEL or audit.get("response_model") != MODEL
            or audit.get("request_options") != REQUEST_OPTIONS
            or audit.get("parsed_proposals") != parse_proposals(audit.get("raw_response"), study["contexts"][condition])):
        raise ValueError("request audit mismatch or incomplete")
    require_complete_text(audit["raw_response"], audit["response_metadata"])
    return audit


def request_condition(client, root, study_id, condition):
    study = load_study(root, study_id)
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    folder = _folder(root, study_id) / "requests" / condition
    if folder.exists():
        return "already_attempted"
    lock_path = _folder(root, study_id) / "request.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if folder.exists():
            return "already_attempted"
        audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "condition": condition,
                 "study_sha256": digest(canonical(study)), "model": MODEL,
                 "request_options": REQUEST_OPTIONS, "prompt": study["prompts"][condition],
                 "source_evidence_hashes": study["source"]["source_hashes"],
                 "raw_response": None, "parsed_proposals": None, "response_model": None,
                 "response_metadata": None}
        called = False
        try:
            response = client.responses.create(model=MODEL, input=audit["prompt"], **REQUEST_OPTIONS)
            called = True
            audit["raw_response"], audit["response_metadata"] = inspect_response(response)
            audit["response_model"] = _get(response, "model")
            audit["response_id"] = _get(response, "id")
            require_complete_text(audit["raw_response"], audit["response_metadata"])
            if audit["response_model"] != MODEL:
                raise ValueError("response model mismatch")
            audit["parsed_proposals"] = parse_proposals(audit["raw_response"], study["contexts"][condition])
            if load_study(root, study_id) != study:
                raise ValueError("source evidence changed during request")
            audit["status"] = "accepted"
        except Exception as exc:
            audit["status"] = "rejected"
            audit["rejection_kind"] = getattr(exc, "kind", "invalid_proposals" if called else "request_error")
            audit["rejection"] = (f"{type(exc).__name__}: {str(exc)[:250]}"
                                  if called and isinstance(exc, ValueError) else type(exc).__name__)
        _write_artifact(folder, "audit.json", audit, "experiment_007_request_v1")
        return audit["status"]


def prepare_review(root, study_id, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    study = load_study(root, study_id)
    audits = {c: load_request(root, study_id, c, study=study) for c in CONDITIONS}
    entries = [(c, index, idea) for c in CONDITIONS for index, idea in enumerate(audits[c]["parsed_proposals"]["ideas"])]
    random.Random(seed).shuffle(entries)
    review_id = _stamp()
    cards = []
    for number, (condition, index, idea) in enumerate(entries, 1):
        opaque_id = f"P{number:02d}"
        cards.append({"evaluation_id": opaque_id,
                      "proposal": {key: value for key, value in idea.items() if key != "evidence_references"}})
    folder = _folder(root, study_id) / "reviews" / review_id
    blind = {"schema": "experiment_007_blind_review_v1", "study_id": study_id,
             "review_id": review_id, "created_utc": datetime.now(timezone.utc).isoformat(),
             "criteria": list(CRITERIA), "score_scale": "1-5; for implementation_cost, 5 means lower estimated cost",
             "judgment_label": "subjective manual assessment, not objective proof of research superiority",
             "cards": cards}
    key = {"schema": "experiment_007_assignment_v1", "seed": seed,
           "blind_sha256": digest(canonical(blind)),
           "request_hashes": {c: digest(canonical(audits[c])) for c in CONDITIONS}}
    _write_artifact(folder / "blind", "review.json", blind, "experiment_007_blind_v1")
    _write_artifact(folder / "operator", "assignment.json", key, "experiment_007_assignment_v1")
    return review_id


def load_blind(root, study_id, review_id):
    if not STAMP.fullmatch(review_id):
        raise ValueError("invalid review ID")
    blind = _read_artifact(_folder(root, study_id) / "reviews" / review_id / "blind",
                           "review.json", "experiment_007_blind_v1")
    if (blind.get("study_id") != study_id or blind.get("review_id") != review_id
            or blind.get("criteria") != list(CRITERIA)
            or len(blind.get("cards", [])) != 6
            or [c["evaluation_id"] for c in blind["cards"]] != [f"P{n:02d}" for n in range(1, 7)]
            or any(set(c) != {"evaluation_id", "proposal"} or "evidence_references" in c["proposal"]
                   for c in blind["cards"])):
        raise ValueError("invalid blind review packet")
    return blind


def score_review(root, study_id, review_id, scores):
    blind = load_blind(root, study_id, review_id)
    if not isinstance(scores, dict) or set(scores) != {c["evaluation_id"] for c in blind["cards"]}:
        raise ValueError("scores must cover every blind evaluation ID")
    for item in scores.values():
        if not isinstance(item, dict) or set(item) != set(CRITERIA):
            raise ValueError("scores need every criterion")
        for score in item.values():
            if (not isinstance(score, dict) or set(score) != {"score", "justification"}
                    or type(score["score"]) is not int or not 1 <= score["score"] <= 5
                    or not isinstance(score["justification"], str)
                    or not 10 <= len(score["justification"].strip()) <= 1200):
                raise ValueError("each criterion needs a score and written justification")
    record = {"schema": "experiment_007_manual_scores_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "blind_sha256": digest(canonical(blind)), "judgment_label": blind["judgment_label"],
              "scores": scores}
    _write_artifact(_folder(root, study_id) / "reviews" / review_id / "scores",
                    "scores.json", record, "experiment_007_scores_v1")
    return record


def reveal_review(root, study_id, review_id):
    blind = load_blind(root, study_id, review_id)
    folder = _folder(root, study_id) / "reviews" / review_id
    scored = _read_artifact(folder / "scores", "scores.json", "experiment_007_scores_v1")
    if scored.get("blind_sha256") != digest(canonical(blind)):
        raise ValueError("scores do not match blind packet")
    # Recheck complete scoring before reading the private assignment.
    if set(scored.get("scores", {})) != {c["evaluation_id"] for c in blind["cards"]}:
        raise ValueError("incomplete blind scoring")
    for item in scored["scores"].values():
        if set(item) != set(CRITERIA) or any(
                type(v.get("score")) is not int or not 1 <= v["score"] <= 5
                or not isinstance(v.get("justification"), str) or len(v["justification"].strip()) < 10
                for v in item.values()):
            raise ValueError("invalid blind scores")
    key = _read_artifact(folder / "operator", "assignment.json", "experiment_007_assignment_v1")
    study = load_study(root, study_id)
    audits = {c: load_request(root, study_id, c, study=study) for c in CONDITIONS}
    original = [(c, i, idea) for c in CONDITIONS for i, idea in enumerate(audits[c]["parsed_proposals"]["ideas"])]
    random.Random(key["seed"]).shuffle(original)
    expected = [{"evaluation_id": f"P{n:02d}", "condition": c, "proposal_index": i,
                 "proposal_sha256": digest(canonical(idea))}
                for n, (c, i, idea) in enumerate(original, 1)]
    if (key.get("blind_sha256") != digest(canonical(blind))
            or key.get("request_hashes") != {c: digest(canonical(audits[c])) for c in CONDITIONS}
            or any(card["proposal"] != {k: v for k, v in idea.items() if k != "evidence_references"}
                   for card, (_, _, idea) in zip(blind["cards"], original))):
        raise ValueError("blind assignment or source request mismatch")
    summary = {condition: {criterion: sum(
        scored["scores"][entry["evaluation_id"]][criterion]["score"]
        for entry in expected if entry["condition"] == condition) / 3
        for criterion in CRITERIA} for condition in CONDITIONS}
    result = {"schema": "experiment_007_reveal_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "study_id": study_id, "review_id": review_id, "seed": key["seed"],
              "judgment_label": blind["judgment_label"], "scores_sha256": digest(canonical(scored)),
              "assignments": expected,
              "proposals": [{"evaluation_id": entry["evaluation_id"], "proposal": idea}
                            for entry, (_, _, idea) in zip(expected, original)],
              "scores": scored["scores"], "mean_subjective_scores_by_condition": summary}
    _write_artifact(folder / "reveal", "result.json", result, "experiment_007_reveal_v1")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experiment 007 hypothesis discovery study")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-study", action="store_true")
    mode.add_argument("--request-condition", choices=CONDITIONS)
    mode.add_argument("--prepare-review", action="store_true")
    mode.add_argument("--score-review", action="store_true")
    mode.add_argument("--reveal-review", action="store_true")
    parser.add_argument("--reflection-id")
    parser.add_argument("--study-id")
    parser.add_argument("--review-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scores-file", type=Path)
    args = parser.parse_args(argv)
    if args.prepare_study:
        if not args.reflection_id:
            parser.error("--reflection-id is required")
        print(prepare_study(ROOT, args.reflection_id)); return 0
    if not args.study_id:
        parser.error("--study-id is required")
    if args.request_condition:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(dotenv_path=ROOT / ".env")
        key = os.environ.get("MODEL_API_KEY")
        if not key:
            parser.error("MODEL_API_KEY is required for a live request")
        status = request_condition(OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0),
                                   ROOT, args.study_id, args.request_condition)
        print(status); return 0 if status == "accepted" else 1
    if args.prepare_review:
        if args.seed is None:
            parser.error("--seed is required")
        print(prepare_review(ROOT, args.study_id, args.seed)); return 0
    if not args.review_id:
        parser.error("--review-id is required")
    if args.score_review:
        if not args.scores_file:
            parser.error("--scores-file is required")
        print(json.dumps(score_review(ROOT, args.study_id, args.review_id,
                                      json.loads(args.scores_file.read_text())), indent=2)); return 0
    print(json.dumps(reveal_review(ROOT, args.study_id, args.review_id), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
