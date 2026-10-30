"""Experiment 008: explicit, auditable critique and revision of one Experiment 007 idea."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import re
from fractions import Fraction

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import (API_BASE_URL, MODEL, NUMBER, _get, _numbers, canonical, digest,
                          inspect_response, require_complete_text)
from muse_discovery_study import (CONDITIONS, CRITERIA, REQUEST_OPTIONS, SECTION as STUDY_SECTION,
                                  _read_artifact, context_for, load_blind, load_request,
                                  load_study, parse_proposals)

SECTION = Path("evaluations/muse-research-critic/v1")
VERSION = "experiment_008_v1"
CRITIC_VALIDATOR_VERSION = "critic_validator_v2"
REVALIDATION_SCHEMA = "experiment_008_revalidation_v1"
REVISION_PROMPT_VERSION = "revision_provenance_v3"
MAX_REVISED_PROPOSAL_REFERENCES = 16
STAMP = re.compile(r"\d{8}T\d{12}Z\Z")
RECOMMENDATIONS = {"proceed", "revise", "deprioritize"}
ACTIONS = {"revise", "defend", "withdraw"}
ASSESSMENTS = ("prior_research_overlap", "evidence_consistency", "hardware_mechanism",
               "falsifiability", "numerical_correctness", "research_value")
CRITIQUE_FIELDS = {"proposal_id", "proposal_sha256", "closest_related_prior_research",
                   "evidence_citations", "assessments", "observed_evidence", "inferred_explanations",
                   "unresolved_questions", "potential_technical_flaws", "unsupported_assumptions",
                   "proposed_discriminating_tests", "research_value_assessment", "recommendation",
                   "recommendation_explanation"}
REVISION_FIELDS = {"proposal_id", "proposal_sha256", "criticism_sha256", "action", "what_changed",
                   "why", "response_to_criticism", "evidence_citations", "revised_proposal"}
TEXT_FIELDS = ("observed_evidence", "inferred_explanations", "unresolved_questions",
               "potential_technical_flaws", "unsupported_assumptions", "proposed_discriminating_tests",
               "research_value_assessment", "recommendation_explanation")


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _base(root):
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / "evaluations", root / "evaluations/muse-research-critic", base)):
        raise ValueError("linked critic audit path")
    return base


def _folder(root, case_id):
    if not STAMP.fullmatch(case_id):
        raise ValueError("invalid critic case ID")
    return _base(root) / case_id


def _write(folder, name, content, schema):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / name).write_bytes(canonical(content))
    (folder / "manifest.json").write_bytes(canonical(manifest(folder, schema=schema)))


def _read(folder, name, schema):
    read_manifest(folder, schema=schema)
    return json.loads((folder / name).read_bytes())


def source_package(root, study_id, review_id, proposal_id):
    """Validate the full 007 chain before selecting one immutable proposal."""
    root = Path(root).resolve()
    study = load_study(root, study_id)
    audits = {c: load_request(root, study_id, c, study=study) for c in CONDITIONS}
    blind = load_blind(root, study_id, review_id)
    review_dir = root / STUDY_SECTION / study_id / "reviews" / review_id
    scores = _read_artifact(review_dir / "scores", "scores.json", "experiment_007_scores_v1")
    key = _read_artifact(review_dir / "operator", "assignment.json", "experiment_007_assignment_v1")
    reveal = _read_artifact(review_dir / "reveal", "result.json", "experiment_007_reveal_v1")
    if not isinstance(proposal_id, str) or not re.fullmatch(r"P0[1-6]", proposal_id):
        raise ValueError("invalid evaluation ID")
    if (blind.get("schema") != "experiment_007_blind_review_v1"
            or scores.get("schema") != "experiment_007_manual_scores_v1"
            or key.get("schema") != "experiment_007_assignment_v1"
            or reveal.get("schema") != "experiment_007_reveal_v1"
            or blind.get("study_id") != study_id or blind.get("review_id") != review_id
            or blind.get("criteria") != list(CRITERIA)
            or type(key.get("seed")) is not int or key["seed"] < 0
            or scores.get("blind_sha256") != digest(canonical(blind))
            or key.get("blind_sha256") != digest(canonical(blind))
            or key.get("request_hashes") != {c: digest(canonical(audits[c])) for c in CONDITIONS}
            or reveal.get("scores_sha256") != digest(canonical(scores))
            or reveal.get("seed") != key.get("seed")
            or reveal.get("study_id") != study_id or reveal.get("review_id") != review_id
            or reveal.get("scores") != scores.get("scores")):
        raise ValueError("Experiment 007 review provenance mismatch")
    entries = [(c, i, idea) for c in CONDITIONS
               for i, idea in enumerate(audits[c]["parsed_proposals"]["ideas"])]
    random.Random(key["seed"]).shuffle(entries)
    expected_assignments = [{"evaluation_id": f"P{n:02d}", "condition": c,
                             "proposal_index": i, "proposal_sha256": digest(canonical(idea))}
                            for n, (c, i, idea) in enumerate(entries, 1)]
    expected_proposals = [{"evaluation_id": item["evaluation_id"], "proposal": idea}
                          for item, (_, _, idea) in zip(expected_assignments, entries)]
    if (reveal.get("assignments") != expected_assignments
            or reveal.get("proposals") != expected_proposals
            or len(blind.get("cards", [])) != 6
            or any(card != {"evaluation_id": item["evaluation_id"],
                            "proposal": {k: v for k, v in idea.items() if k != "evidence_references"}}
                   for card, item, (_, _, idea) in zip(blind["cards"], expected_assignments, entries))
            or set(scores.get("scores", {})) != {x["evaluation_id"] for x in expected_assignments}):
        raise ValueError("Experiment 007 reveal or blind proposal mismatch")
    for scored in scores["scores"].values():
        if (set(scored) != set(blind["criteria"]) or any(
                not isinstance(v, dict) or set(v) != {"score", "justification"}
                or type(v["score"]) is not int or not 1 <= v["score"] <= 5
                or not isinstance(v["justification"], str) or len(v["justification"].strip()) < 10
                for v in scored.values())):
            raise ValueError("Experiment 007 scores incomplete")
    expected_means = {condition: {criterion: sum(
        scores["scores"][entry["evaluation_id"]][criterion]["score"]
        for entry in expected_assignments if entry["condition"] == condition) / 3
        for criterion in CRITERIA} for condition in CONDITIONS}
    if reveal.get("mean_subjective_scores_by_condition") != expected_means:
        raise ValueError("Experiment 007 score summary mismatch")
    position = int(proposal_id[1:]) - 1
    condition, index, proposal = entries[position]
    evidence = {**study["source"]["base_context"]["experiments"],
                "005": {"evidence_id": study["source"]["experiment_005"]["evidence_id"],
                        "observations": study["source"]["experiment_005"]["observations"]}}
    files = {"study_setup_sha256": sha((root / STUDY_SECTION / study_id / "setup/study.json").read_bytes()),
             "request_audit_sha256": sha((root / STUDY_SECTION / study_id / "requests" / condition / "audit.json").read_bytes()),
             "blind_review_sha256": sha((review_dir / "blind/review.json").read_bytes()),
             "scores_sha256": sha((review_dir / "scores/scores.json").read_bytes()),
             "reveal_sha256": sha((review_dir / "reveal/result.json").read_bytes()),
             "source_raw_response_sha256": digest(audits[condition]["raw_response"].encode()),
             **study["source"]["source_hashes"]}
    return {"schema": "experiment_008_source_v1", "study_id": study_id, "review_id": review_id,
            "proposal_id": proposal_id, "condition": condition, "proposal_index": index,
            "proposal": proposal, "proposal_sha256": digest(canonical(proposal)),
            "validated_evidence": evidence, "experiment_005_criteria":
                study["source"]["experiment_005"]["evaluation_criteria"],
            "prior_reflection": study["source"]["reflection"],
            "source_hashes": files}


def _references(refs, source, *, required=True, max_references=12):
    available = {item["evidence_id"]: item["observations"]
                 for item in source["validated_evidence"].values()}
    if not isinstance(refs, list):
        raise ValueError("malformed evidence citations")
    if len(refs) > max_references or (required and not refs):
        raise ValueError(f"evidence citation count must be 1 to {max_references}")
    seen = set()
    for ref in refs:
        if (not isinstance(ref, dict) or set(ref) != {"evidence_id", "observation_key"}
                or not isinstance(ref["evidence_id"], str) or not isinstance(ref["observation_key"], str)
                or ref["evidence_id"] not in available
                or ref["observation_key"] not in available[ref["evidence_id"]]
                or (ref["evidence_id"], ref["observation_key"]) in seen):
            raise ValueError("invalid or duplicate evidence citation")
        seen.add((ref["evidence_id"], ref["observation_key"]))
    return [available[ref["evidence_id"]][ref["observation_key"]] for ref in refs]


def _check_text(text, source, cited_values, refs, *, field, local_structural=False):
    def numeric_error(reason, match, content):
        fragment = content[max(0, match.start() - 55):min(len(content), match.end() + 55)]
        raise ValueError(f"{reason} in {field}: {match.group()!r} near {fragment!r}")

    def negated_in_clause(match):
        boundary = max(text.rfind(mark, 0, match.start()) for mark in ".!?;,")
        return bool(re.search(r"\b(?:no|not|without|never|unmeasured)\b",
                              text[boundary + 1:match.start()], re.I))

    if not isinstance(text, str) or not text.strip() or len(text) > 1800 or any(x in text for x in ("\x00", "```", "@triton.jit")):
        raise ValueError("invalid research assessment text")
    experiment_mentions = re.findall(r"\b(?:experiment|exp\.)\s*0*(\d+)\b", text, re.I)
    if any(int(x) not in (3, 4, 5) for x in experiment_mentions):
        raise ValueError("nonexistent experiment reference")
    cited_pairs = {(ref["evidence_id"], ref["observation_key"]) for ref in refs}
    cited_ids = {evidence_id for evidence_id, _ in cited_pairs}
    if any(not any(evidence_id == source["validated_evidence"][f"{int(number):03d}"]["evidence_id"]
                   for evidence_id, _ in cited_pairs) for number in experiment_mentions):
        raise ValueError("historical experiment claim lacks citation")
    bare_ids = re.findall(r"(?<![\w.])(00[3-5])(?!\d)", text)
    if any(source["validated_evidence"][number]["evidence_id"] not in cited_ids for number in bare_ids):
        raise ValueError("historical experiment identifier lacks citation")
    numeric_text = re.sub(r"\b(?:experiment|exp\.)\s*0*(?:3|4|5)\b", "Experiment", text, flags=re.I)
    numeric_text = re.sub(r"(?<![\w.])00[3-5](?!\d)", "Experiment", numeric_text)
    # Bracketed workload coordinates are separate dimensions, not thousands-formatted numbers.
    numeric_text = re.sub(r"\[\d+(?:\s*,\s*\d+)+\]",
                          lambda match: match.group().replace(",", " "), numeric_text)
    allowed = _numbers(source["proposal"]) | set().union(*(_numbers(value) for value in cited_values))
    if not local_structural:
        for experiment in source["validated_evidence"].values():
            if experiment["evidence_id"] not in cited_ids:
                continue
            observations = experiment["observations"]
            for key in ("configuration", "workload", "control_workload", "partial_m_workload"):
                if key in observations:
                    allowed |= _numbers(observations[key])
            if "verification" in observations:
                allowed |= _numbers(observations["verification"]["configuration"])
                allowed |= _numbers(observations["verification"]["shapes"])
    for match in re.finditer(r"(?<![\w.])(\d+)\s*([+*/-])\s*(\d+)\s*=\s*(\d+)(?![\w.])", numeric_text):
        left, operator, right, reported = match.groups()
        if left not in allowed or right not in allowed or (operator == "/" and int(right) == 0):
            numeric_error("derived arithmetic lacks validated operands", match, numeric_text)
        a, b = Fraction(int(left)), Fraction(int(right))
        actual = {"+": lambda: a + b, "-": lambda: a - b,
                  "*": lambda: a * b, "/": lambda: a / b}[operator]()
        if actual != int(reported):
            numeric_error("incorrect derived arithmetic", match, numeric_text)
        allowed.add(reported)
    for match in re.finditer(
            r"\bblock_m\s*=\s*(\d+)\s*,\s*M\s*=\s*(\d+)\s+is\s+exactly\s+divisible\s*\(\s*(\d+)\s+full tiles?\s*\)",
            numeric_text, re.I):
        block_m, m_rows, tiles = map(int, match.groups())
        if (str(block_m) not in allowed or str(m_rows) not in allowed or block_m == 0
                or m_rows % block_m or m_rows // block_m != tiles):
            numeric_error("incorrect derived tile count", match, numeric_text)
        allowed.add(str(tiles))
    # Ordered prose markers are formatting, not observations. Require a sequence from one.
    markers = list(re.finditer(r"(?<!\w)(\d{1,2})\)\s+", numeric_text))
    if [int(match.group(1)) for match in markers] == list(range(1, len(markers) + 1)):
        for match in reversed(markers):
            start, end = match.span(1)
            numeric_text = numeric_text[:start] + " " * (end - start) + numeric_text[end:]
    for match in NUMBER.finditer(numeric_text):
        if match.group() not in allowed:
            if local_structural:
                for experiment in source["validated_evidence"].values():
                    observations = experiment["observations"]
                    for key in ("configuration", "workload", "control_workload", "partial_m_workload"):
                        if key in observations and match.group() in _numbers(observations[key]):
                            numeric_error(f"validated numerical fact lacks citation to {experiment['evidence_id']}:{key}",
                                          match, numeric_text)
                    if "verification" in observations and any(
                            match.group() in _numbers(observations["verification"][key])
                            for key in ("configuration", "shapes")):
                        numeric_error(f"validated numerical fact lacks citation to {experiment['evidence_id']}:verification",
                                      match, numeric_text)
            numeric_error("unsupported numerical measurement", match, numeric_text)
    timing_fields = ("median_paired_ratio", "paired_ratio_mad", "bootstrap_median_ratio_95pct",
                     "baseline_median_ms", "candidate_median_ms", "one_time_map_setup_ms")
    timing_numbers = set().union(*(
        _numbers({name: value for name, value in
                  source["validated_evidence"]["005"]["observations"][key].items()
                  if name in timing_fields})
        for evidence_id, key in cited_pairs
        if evidence_id == source["validated_evidence"]["005"]["evidence_id"]
        and key.startswith(("screening:", "confirmation:"))))
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:ms|µs|ns|%|x|×)\b", text, re.I):
        if match.group(1) not in timing_numbers:
            numeric_error("numeric performance claim lacks a cited measurement", match, text)
    performance_claim = re.search(
        r"\b(?:measured|observed|achieved|delivered|confirmed)\b.{0,45}\b(?:speedup|slowdown|faster|slower)\b",
        text, re.I)
    if (performance_claim
            and not negated_in_clause(performance_claim)
            and not any(evidence_id == source["validated_evidence"]["005"]["evidence_id"]
                        and key.startswith(("screening:", "confirmation:"))
                        for evidence_id, key in cited_pairs)):
        numeric_error("measured performance claim lacks timing citation", performance_claim, text)
    stage_claim = re.search(r"\b(?:confirmed|completed|validated)\s+(?:independent\s+)?(?:confirmation|profiling|profile)\b|\b(?:confirmation|profiling)\s+(?:was\s+)?(?:completed|validated|confirmed)\b", text, re.I)
    if stage_claim and not re.search(r"\b(?:no|not|without|missing|absent|unconfirmed)\b", text[max(0, stage_claim.start() - 25):stage_claim.start()], re.I):
        if source["validated_evidence"]["005"]["observations"]["confirmation_status"] == "missing" or "profiling" in stage_claim.group().lower():
            raise ValueError("unsupported confirmation or profiling claim")
    categorical_claim = re.search(r"\b(?:proved|confirmed|established)\s+(?:a\s+)?(?:speedup|performance gain|cache|locality|mechanism)\b|\b(?:achieved|delivered)\s+(?:a\s+)?speedup\b", text, re.I)
    if categorical_claim and not negated_in_clause(categorical_claim):
        raise ValueError("unsupported categorical performance claim")


def critique_prompt(source):
    example = {"proposal_id": source["proposal_id"], "proposal_sha256": source["proposal_sha256"],
        "closest_related_prior_research": {"experiment_id": "<003|004|005|none>",
                                           "what_was_tested": "Describe the actual tested scope.",
                                           "relation": "Explain overlap or why a revisit is informative."},
        "evidence_citations": [{"evidence_id": source["validated_evidence"]["005"]["evidence_id"],
                                "observation_key": "confirmation_status"}],
        "assessments": {name: "Specific assessment, including strengths when supported." for name in ASSESSMENTS},
        "observed_evidence": "State only validated observations.",
        "inferred_explanations": "Label causal explanations as inference.",
        "unresolved_questions": "State what remains unverified.",
        "potential_technical_flaws": "Describe plausible flaws or state that none is identified.",
        "unsupported_assumptions": "Identify assumptions still requiring evidence.",
        "proposed_discriminating_tests": "Describe a controlled test that separates mechanisms.",
        "research_value_assessment": "State specific information a result would add.",
        "recommendation": "<proceed|revise|deprioritize>",
        "recommendation_explanation": "Explain the advisory recommendation."}
    return ("Independently critique this one advisory GPU hypothesis. Return JSON matching SCHEMA exactly. "
            "Evaluate prior research overlap, evidence consistency, hardware mechanism, falsifiability, numerical "
            "correctness (including reduction-order changes), and research value. Identify what each related "
            "experiment actually tested; revisiting a mechanism can be justified. Recognize strengths as well as "
            "flaws. Distinguish validated observations, inferred explanations, and unresolved questions. "
            "Cite validated evidence IDs and observation keys for historical claims. The prior reflection is model "
            "interpretation, not a measured result. Do not invent measurements or claim completed profiling or "
            "confirmation when absent. Recommendation is advisory and never authorizes GPU execution. No code.\n"
            "SCHEMA\n" + json.dumps(example, sort_keys=True) + "\nSOURCE\n"
            + json.dumps(source, sort_keys=True, allow_nan=False))


def parse_critique(raw, source):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 32000:
        raise ValueError("missing or oversized criticism")
    try:
        result = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("criticism is not JSON") from exc
    if (not isinstance(result, dict) or set(result) != CRITIQUE_FIELDS
            or result["proposal_id"] != source["proposal_id"]
            or result["proposal_sha256"] != source["proposal_sha256"]
            or result["recommendation"] not in RECOMMENDATIONS):
        raise ValueError("criticism identity, schema, or recommendation mismatch")
    prior = result["closest_related_prior_research"]
    if (not isinstance(prior, dict) or set(prior) != {"experiment_id", "what_was_tested", "relation"}
            or prior["experiment_id"] not in {"003", "004", "005", "none"}):
        raise ValueError("invalid prior research assessment")
    cited = _references(result["evidence_citations"], source)
    if prior["experiment_id"] != "none" and not any(
            ref["evidence_id"] == source["validated_evidence"][prior["experiment_id"]]["evidence_id"]
            for ref in result["evidence_citations"]):
        raise ValueError("closest experiment lacks citation")
    assessments = result["assessments"]
    if not isinstance(assessments, dict) or set(assessments) != set(ASSESSMENTS):
        raise ValueError("missing critic responsibilities")
    fields = [("closest_related_prior_research.what_was_tested", prior["what_was_tested"]),
              ("closest_related_prior_research.relation", prior["relation"])]
    fields += [(f"assessments.{name}", assessments[name]) for name in ASSESSMENTS]
    fields += [(name, result[name]) for name in TEXT_FIELDS]
    for field, value in fields:
        _check_text(value, source, cited, result["evidence_citations"], field=field)
    return result


def revision_prompt(source, criticism, *, version=REVISION_PROMPT_VERSION):
    if version not in ("legacy_v1", "revision_provenance_v2", REVISION_PROMPT_VERSION):
        raise ValueError("unknown revision prompt version")
    example = {"proposal_id": source["proposal_id"], "proposal_sha256": source["proposal_sha256"],
        "criticism_sha256": digest(canonical(criticism)), "action": "<revise|defend|withdraw>",
        "what_changed": "For revision, identify concrete proposal changes; otherwise say unchanged or withdrawn.",
        "why": "Explain why this action follows from the criticism and evidence.",
        "response_to_criticism": "Address the critic's strongest point directly.",
        "evidence_citations": [{"evidence_id": source["validated_evidence"]["005"]["evidence_id"],
                                "observation_key": "confirmation_status"}],
        "revised_proposal": "Full original proposal schema plus testable_prediction for revise; null otherwise."}
    return ("Respond to this advisory criticism in a separate step. Choose revise, defend, or withdraw. "
            "For revise, return one complete revised proposal in the original schema, with a specific testable "
            "prediction and disproof observation; explain what changed and why. For defend, retain the original "
            "proposal and cite evidence for the defense. For withdraw, explain why and return null proposal. "
            + ("Cite every historical claim and numerical workload reference in revised_proposal using the exact "
               "evidence_id and observation_key that support it, including workload dimensions and configuration "
               "values. Put those citations in revised_proposal.evidence_references. The response-level "
               "evidence_citations apply only to what_changed, why, and response_to_criticism; they do not cover "
               "revised_proposal. Revised-proposal citations do not cover the response-level explanations. "
               "Cite each scope directly, even when the same evidence appears in the criticism or source. "
               if version != "legacy_v1" else "")
            + ("Use between 1 and 16 distinct citations in revised_proposal.evidence_references; "
               "response-level evidence_citations have a separate limit of 12. "
               if version == REVISION_PROMPT_VERSION else "")
            + "Distinguish observed evidence from inferred mechanism. Do not invent measurements, completed profiling "
            "or confirmation. No code or execution approval. Return JSON matching SCHEMA exactly.\nSCHEMA\n"
            + json.dumps(example, sort_keys=True) + "\nSOURCE\n"
            + json.dumps({"proposal": source["proposal"], "proposal_id": source["proposal_id"],
                          "proposal_sha256": source["proposal_sha256"],
                          "validated_evidence": source["validated_evidence"],
                          "criticism": criticism}, sort_keys=True, allow_nan=False))


def parse_revision(raw, source, criticism, context, other_proposals):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 35000:
        raise ValueError("missing or oversized revision")
    try:
        result = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("revision is not JSON") from exc
    if (not isinstance(result, dict) or set(result) != REVISION_FIELDS
            or result["proposal_id"] != source["proposal_id"]
            or result["proposal_sha256"] != source["proposal_sha256"]
            or result["criticism_sha256"] != digest(canonical(criticism))
            or result["action"] not in ACTIONS):
        raise ValueError("revision identity, schema, or action mismatch")
    cited = _references(result["evidence_citations"], source)
    for field in ("what_changed", "why", "response_to_criticism"):
        _check_text(result[field], source, cited, result["evidence_citations"],
                    field=field, local_structural=True)
    revised = result["revised_proposal"]
    if result["action"] == "revise":
        if not isinstance(revised, dict) or revised == source["proposal"]:
            raise ValueError("revision must change the proposal")
        if "evidence_references" not in revised:
            raise ValueError("proposal schema mismatch")
        ideas = list(other_proposals)
        ideas[source["proposal_index"]] = revised
        revised_values = _references(revised["evidence_references"], source,
                                     max_references=MAX_REVISED_PROPOSAL_REFERENCES)
        parse_proposals(json.dumps({"ideas": ideas}), context,
                        max_evidence_references=MAX_REVISED_PROPOSAL_REFERENCES)
        for field, value in revised.items():
            if field not in {"category", "numerical_equivalence", "evidence_references"}:
                _check_text(value, source, revised_values, revised["evidence_references"],
                            field=f"revised_proposal.{field}", local_structural=True)
        if not result["what_changed"].strip() or result["what_changed"].lower() == "unchanged":
            raise ValueError("revision must identify what changed")
    elif revised is not None:
        raise ValueError("defense or withdrawal must preserve original proposal")
    if result["action"] == "defend" and not cited:
        raise ValueError("defense requires evidence")
    return result


def revision_context(study, source):
    """A revised A idea may cite validated feedback introduced after its original request."""
    context = json.loads(json.dumps(study["contexts"][source["condition"]]))
    context["experiments"]["005"] = source["validated_evidence"]["005"]
    return context


def prepare(root, study_id, review_id, proposal_id):
    source = source_package(root, study_id, review_id, proposal_id)
    case_id = _stamp()
    setup = {"schema": VERSION, "case_id": case_id, "created_utc": datetime.now(timezone.utc).isoformat(),
             "source": source, "source_sha256": digest(canonical(source)),
             "model": MODEL, "request_options": REQUEST_OPTIONS, "critique_prompt": critique_prompt(source)}
    _write(_folder(root, case_id) / "setup", "case.json", setup, "experiment_008_setup_v1")
    return case_id


def load_case(root, case_id):
    setup = _read(_folder(root, case_id) / "setup", "case.json", "experiment_008_setup_v1")
    source = setup["source"]
    actual = source_package(root, source["study_id"], source["review_id"], source["proposal_id"])
    if (setup.get("schema") != VERSION or setup.get("case_id") != case_id
            or source != actual or setup.get("source_sha256") != digest(canonical(actual))
            or setup.get("model") != MODEL or setup.get("request_options") != REQUEST_OPTIONS
            or setup.get("critique_prompt") != critique_prompt(actual)):
        raise ValueError("critic case source or prompt mismatch")
    return setup


def _request(client, root, case_id, stage):
    setup = load_case(root, case_id)
    folder = _folder(root, case_id) / stage
    if folder.exists():
        return "already_attempted"
    if stage == "criticism":
        prompt = setup["critique_prompt"]
        revision_setup = None
    else:
        revision_setup = load_revision_setup(root, case_id)
        prompt = revision_setup["revision_prompt"]
    with (_folder(root, case_id) / "request.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if folder.exists():
            return "already_attempted"
        audit = {"schema": f"experiment_008_{stage}_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                 "case_id": case_id, "source_sha256": setup["source_sha256"],
                 "model": MODEL, "request_options": REQUEST_OPTIONS, "prompt": prompt,
                 "raw_response": None, "parsed_response": None, "response_model": None,
                 "response_metadata": None}
        if revision_setup:
            audit["criticism_sha256"] = revision_setup["criticism_sha256"]
            if "criticism_source" in revision_setup:
                audit["criticism_source"] = revision_setup["criticism_source"]
            if "prompt_version" in revision_setup:
                audit["prompt_version"] = revision_setup["prompt_version"]
        called = False
        try:
            response = client.responses.create(model=MODEL, input=prompt, **REQUEST_OPTIONS)
            called = True
            audit["raw_response"], audit["response_metadata"] = inspect_response(response)
            audit["response_model"] = _get(response, "model")
            audit["response_id"] = _get(response, "id")
            require_complete_text(audit["raw_response"], audit["response_metadata"])
            if audit["response_model"] != MODEL:
                raise ValueError("response model mismatch")
            if stage == "criticism":
                audit["parsed_response"] = parse_critique(audit["raw_response"], setup["source"])
            else:
                study = load_study(root, setup["source"]["study_id"])
                original = load_request(root, setup["source"]["study_id"], setup["source"]["condition"], study=study)
                audit["parsed_response"] = parse_revision(
                    audit["raw_response"], setup["source"], revision_setup["criticism"],
                    revision_context(study, setup["source"]), original["parsed_proposals"]["ideas"])
            if load_case(root, case_id) != setup:
                raise ValueError("source changed during request")
            audit["status"] = "accepted"
        except Exception as exc:
            audit["status"] = "rejected"
            audit["rejection_kind"] = getattr(exc, "kind", "invalid_response" if called else "request_error")
            audit["rejection"] = (f"{type(exc).__name__}: {str(exc)[:250]}"
                                  if called and isinstance(exc, ValueError) else type(exc).__name__)
        _write(folder, "audit.json", audit, f"experiment_008_{stage}_v1")
        return audit["status"]


def request_critique(client, root, case_id):
    return _request(client, root, case_id, "criticism")


def load_criticism(root, case_id):
    setup = load_case(root, case_id)
    audit = _read(_folder(root, case_id) / "criticism", "audit.json", "experiment_008_criticism_v1")
    if (audit.get("status") != "accepted" or audit.get("case_id") != case_id
            or audit.get("source_sha256") != setup["source_sha256"]
            or audit.get("model") != MODEL or audit.get("response_model") != MODEL
            or audit.get("request_options") != REQUEST_OPTIONS
            or audit.get("prompt") != setup["critique_prompt"]
            or audit.get("parsed_response") != parse_critique(audit.get("raw_response"), setup["source"])):
        raise ValueError("criticism audit mismatch")
    require_complete_text(audit["raw_response"], audit["response_metadata"])
    return audit


def _historical_criticism(root, case_id):
    """Read a rejected, manifested response with exact case and request bindings."""
    setup = load_case(root, case_id)
    folder = _folder(root, case_id) / "criticism"
    audit = _read(folder, "audit.json", "experiment_008_criticism_v1")
    if (audit.get("schema") != "experiment_008_criticism_v1"
            or audit.get("status") != "rejected" or audit.get("parsed_response") is not None
            or audit.get("case_id") != case_id
            or audit.get("source_sha256") != setup["source_sha256"]
            or audit.get("prompt") != setup["critique_prompt"]
            or audit.get("model") != MODEL or audit.get("response_model") != MODEL
            or audit.get("request_options") != REQUEST_OPTIONS
            or not isinstance(audit.get("raw_response"), str)
            or not isinstance(audit.get("response_metadata"), dict)
            or audit["response_metadata"].get("text_characters") != len(audit["raw_response"])
            or not isinstance(audit.get("response_id"), str)
            or not isinstance(audit.get("rejection_kind"), str)):
        raise ValueError("historical criticism audit identity mismatch")
    return setup, audit, sha((folder / "audit.json").read_bytes()), sha((folder / "manifest.json").read_bytes())


def _revalidation_result(audit, source):
    try:
        require_complete_text(audit["raw_response"], audit["response_metadata"])
        return "valid", None, parse_critique(audit["raw_response"], source)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return "invalid", f"{type(exc).__name__}: {str(exc)[:300]}", None


def revalidate_criticism(root, case_id):
    """Write a new offline decision; never alter the rejected request or its manifest."""
    setup, audit, audit_hash, manifest_hash = _historical_criticism(root, case_id)
    status, reason, parsed = _revalidation_result(audit, setup["source"])
    if load_case(root, case_id) != setup:
        raise ValueError("source changed during offline revalidation")
    stamp = _stamp()
    record = {"schema": REVALIDATION_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "revalidation_id": stamp, "case_id": case_id,
              "proposal_sha256": setup["source"]["proposal_sha256"],
              "source_case_sha256": setup["source_sha256"],
              "original_audit_path": str(SECTION / case_id / "criticism/audit.json"),
              "original_audit_sha256": audit_hash,
              "original_manifest_sha256": manifest_hash,
              "original_response_sha256": digest(audit["raw_response"].encode()),
              "validator_version": CRITIC_VALIDATOR_VERSION,
              "validation_status": status, "rejection_reason": reason,
              "parsed_criticism": parsed}
    _write(_folder(root, case_id) / "revalidations" / "v1" / stamp,
           "result.json", record, REVALIDATION_SCHEMA)
    return record


def load_revalidation(root, case_id, revalidation_id):
    if not STAMP.fullmatch(revalidation_id):
        raise ValueError("invalid revalidation ID")
    setup, audit, audit_hash, manifest_hash = _historical_criticism(root, case_id)
    folder = _folder(root, case_id) / "revalidations" / "v1" / revalidation_id
    record = _read(folder, "result.json", REVALIDATION_SCHEMA)
    status, reason, parsed = _revalidation_result(audit, setup["source"])
    if (record.get("schema") != REVALIDATION_SCHEMA
            or record.get("revalidation_id") != revalidation_id
            or record.get("case_id") != case_id
            or record.get("proposal_sha256") != setup["source"]["proposal_sha256"]
            or record.get("source_case_sha256") != setup["source_sha256"]
            or record.get("original_audit_path") != str(SECTION / case_id / "criticism/audit.json")
            or record.get("original_audit_sha256") != audit_hash
            or record.get("original_manifest_sha256") != manifest_hash
            or record.get("original_response_sha256") != digest(audit["raw_response"].encode())
            or record.get("validator_version") != CRITIC_VALIDATOR_VERSION
            or record.get("validation_status") != status
            or record.get("rejection_reason") != reason
            or record.get("parsed_criticism") != parsed
            or not isinstance(record.get("created_utc"), str)):
        raise ValueError("revalidation content, source, or validator mismatch")
    return record


def _criticism_for_revision(root, case_id, revalidation_id=None):
    if revalidation_id is None:
        audit = load_criticism(root, case_id)
        return audit["parsed_response"], digest(canonical(audit)), None
    record = load_revalidation(root, case_id, revalidation_id)
    if record["validation_status"] != "valid":
        raise ValueError("invalid revalidation cannot authorize revision")
    binding = {"kind": "offline_revalidation", "revalidation_id": revalidation_id,
               "revalidation_sha256": digest(canonical(record)),
               "original_audit_sha256": record["original_audit_sha256"],
               "original_response_sha256": record["original_response_sha256"]}
    return record["parsed_criticism"], digest(canonical(record)), binding


def prepare_revision(root, case_id, *, revalidation_id=None):
    setup = load_case(root, case_id)
    criticism, criticism_hash, binding = _criticism_for_revision(root, case_id, revalidation_id)
    record = {"schema": "experiment_008_revision_setup_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "case_id": case_id, "source_sha256": setup["source_sha256"],
              "prompt_version": REVISION_PROMPT_VERSION,
              "criticism_sha256": criticism_hash,
              "criticism": criticism,
              "revision_prompt": revision_prompt(setup["source"], criticism)}
    if binding is not None:
        record["criticism_source"] = binding
    _write(_folder(root, case_id) / "revision_setup", "setup.json", record, "experiment_008_revision_setup_v1")
    return record


def load_revision_setup(root, case_id):
    setup = load_case(root, case_id)
    record = _read(_folder(root, case_id) / "revision_setup", "setup.json", "experiment_008_revision_setup_v1")
    binding = record.get("criticism_source")
    prompt_version = record.get("prompt_version")
    if prompt_version not in (None, "revision_provenance_v2", REVISION_PROMPT_VERSION):
        raise ValueError("unknown revision prompt version")
    if binding is None:
        criticism, criticism_hash, expected_binding = _criticism_for_revision(root, case_id)
    else:
        if not isinstance(binding, dict) or set(binding) != {"kind", "revalidation_id", "revalidation_sha256",
                                                         "original_audit_sha256", "original_response_sha256"}:
            raise ValueError("invalid revision revalidation binding")
        criticism, criticism_hash, expected_binding = _criticism_for_revision(
            root, case_id, binding["revalidation_id"])
    if (record.get("case_id") != case_id or record.get("source_sha256") != setup["source_sha256"]
            or record.get("criticism_sha256") != criticism_hash
            or record.get("criticism") != criticism
            or binding != expected_binding
            or record.get("revision_prompt") != revision_prompt(
                setup["source"], criticism, version=prompt_version or "legacy_v1")):
        raise ValueError("revision preparation mismatch")
    return record


def request_revision(client, root, case_id):
    return _request(client, root, case_id, "revision")


def validate_case(root, case_id):
    setup = load_case(root, case_id)
    result = {"case_id": case_id, "proposal_id": setup["source"]["proposal_id"],
              "source_sha256": setup["source_sha256"], "criticism": "not_requested", "revision": "not_requested"}
    if (_folder(root, case_id) / "criticism").exists():
        record = _read(_folder(root, case_id) / "criticism", "audit.json", "experiment_008_criticism_v1")
        if (record.get("status") == "rejected" and record.get("case_id") == case_id
                and record.get("source_sha256") == setup["source_sha256"]
                and record.get("prompt") == setup["critique_prompt"]
                and record.get("model") == MODEL and record.get("request_options") == REQUEST_OPTIONS
                and record.get("parsed_response") is None and record.get("rejection_kind")):
            result["criticism"] = "rejected"
        else:
            critique = load_criticism(root, case_id)
            result["criticism"] = critique["status"]
    if (_folder(root, case_id) / "revision_setup").exists():
        load_revision_setup(root, case_id)
    if (_folder(root, case_id) / "revision").exists():
        preparation = load_revision_setup(root, case_id)
        audit = _read(_folder(root, case_id) / "revision", "audit.json", "experiment_008_revision_v1")
        if (audit.get("status") == "rejected" and audit.get("case_id") == case_id
                and audit.get("source_sha256") == setup["source_sha256"]
                and audit.get("criticism_sha256") == preparation["criticism_sha256"]
                and audit.get("criticism_source") == preparation.get("criticism_source")
                and audit.get("prompt_version") == preparation.get("prompt_version")
                and audit.get("prompt") == preparation["revision_prompt"]
                and audit.get("model") == MODEL and audit.get("request_options") == REQUEST_OPTIONS
                and audit.get("parsed_response") is None and audit.get("rejection_kind")):
            result["revision"] = "rejected"
            return result
        study = load_study(root, setup["source"]["study_id"])
        original = load_request(root, setup["source"]["study_id"], setup["source"]["condition"], study=study)
        if (audit.get("status") != "accepted" or audit.get("case_id") != case_id
                or audit.get("source_sha256") != setup["source_sha256"]
                or audit.get("criticism_sha256") != preparation["criticism_sha256"]
                or audit.get("criticism_source") != preparation.get("criticism_source")
                or audit.get("prompt_version") != preparation.get("prompt_version")
                or audit.get("model") != MODEL or audit.get("response_model") != MODEL
                or audit.get("request_options") != REQUEST_OPTIONS
                or audit.get("prompt") != preparation["revision_prompt"]
                or audit.get("parsed_response") != parse_revision(
                    audit.get("raw_response"), setup["source"], preparation["criticism"],
                    revision_context(study, setup["source"]), original["parsed_proposals"]["ideas"])):
            raise ValueError("revision audit mismatch")
        require_complete_text(audit["raw_response"], audit["response_metadata"])
        result["revision"] = audit["status"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experiment 008 advisory research critic")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--criticize", action="store_true")
    modes.add_argument("--prepare-revision", action="store_true")
    modes.add_argument("--revalidate", action="store_true")
    modes.add_argument("--validate-revalidation", action="store_true")
    modes.add_argument("--revise", action="store_true")
    modes.add_argument("--validate", action="store_true")
    parser.add_argument("--study-id"); parser.add_argument("--review-id")
    parser.add_argument("--proposal-id"); parser.add_argument("--case-id")
    parser.add_argument("--revalidation-id")
    args = parser.parse_args(argv)
    if args.prepare:
        if not all((args.study_id, args.review_id, args.proposal_id)):
            parser.error("--study-id, --review-id, and --proposal-id are required")
        print(prepare(ROOT, args.study_id, args.review_id, args.proposal_id)); return 0
    if not args.case_id:
        parser.error("--case-id is required")
    if args.revalidate:
        record = revalidate_criticism(ROOT, args.case_id)
        print(json.dumps(record, indent=2)); return 0 if record["validation_status"] == "valid" else 1
    if args.validate_revalidation:
        if not args.revalidation_id:
            parser.error("--revalidation-id is required")
        record = load_revalidation(ROOT, args.case_id, args.revalidation_id)
        print(json.dumps(record, indent=2)); return 0 if record["validation_status"] == "valid" else 1
    if args.prepare_revision:
        print(json.dumps(prepare_revision(ROOT, args.case_id,
                                          revalidation_id=args.revalidation_id), indent=2)); return 0
    if args.validate:
        print(json.dumps(validate_case(ROOT, args.case_id), indent=2)); return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        parser.error("MODEL_API_KEY is required for a live request")
    client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0)
    status = request_critique(client, ROOT, args.case_id) if args.criticize else request_revision(client, ROOT, args.case_id)
    print(status)
    return 0 if status == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
