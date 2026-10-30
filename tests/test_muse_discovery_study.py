"""Deterministic CPU-only tests for Experiment 007."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_discovery_study as study

ROOT = Path(__file__).resolve().parents[1]
REFLECTION_ID = "20261030T141047613123Z"


@pytest.fixture(scope="module")
def source():
    return study.sources(ROOT, REFLECTION_ID)


def proposals(context, suffix):
    ids = context["experiments"]
    reference = ids["005"] if "005" in ids else ids["004"]
    key = "verification" if "005" in ids else "partial_m_workload"
    concepts = ("Tile Relay", "Sparse Map Echo", "Mixed Radix Tensor")
    return {"ideas": [{
        "category": category, "name": f"{name} {suffix}",
        "core_mechanism": f"Use {name.lower()} to change data movement across output tiles.",
        "why_it_might_work": "The mechanism might reduce redundant memory movement if the layout is favorable.",
        "what_makes_it_unconventional": "It changes data organization beyond parameter tuning.",
        "necessary_assumptions": "Compiler lowering preserves the intended mapping and resources fit.",
        "biggest_obstacle": "Shared memory pressure may erase the benefit.",
        "disproof_observation": "Latency is unchanged against the fixed control after correctness validation.",
        "smallest_proof_of_concept": "Review one isolated correctness and compiler diagnostic after approval.",
        "numerical_equivalence": "exact", "testable_prediction":
            "If the layout reduces movement, expect lower latency against the fixed control.",
        "evidence_references": [{"evidence_id": reference["evidence_id"],
                                 "observation_key": key}]}
        for category, name in zip(study.CATEGORIES, concepts)]}


class Client:
    def __init__(self, raw):
        self.raw = raw
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="fake", model=study.MODEL, status="completed",
                               output_text=self.raw, usage={"input_tokens": 10, "output_tokens": 20})


def complete_requests(tmp_path, monkeypatch, source):
    monkeypatch.setattr(study, "sources", lambda root, reflection_id: source)
    study_id = study.prepare_study(tmp_path, REFLECTION_ID)
    clients = {}
    for condition in study.CONDITIONS:
        context = study.context_for(source, condition)
        client = Client(json.dumps(proposals(context, condition)))
        assert study.request_condition(client, tmp_path, study_id, condition) == "accepted"
        clients[condition] = client
    return study_id, clients


def test_source_validation_and_condition_isolation(source):
    assert source["reflection"]["decision"] in {"abandon", "revise", "investigate_further"}
    a = study.context_for(source, "A")
    b = study.context_for(source, "B")
    assert "005" not in a["experiments"] and "feedback" not in a
    assert b["experiments"]["005"]["evidence_id"] == source["experiment_005"]["evidence_id"]
    assert b["feedback"]["accepted_reflection"] == source["reflection"]
    a_prompt, b_prompt = study.prompt_for(a), study.prompt_for(b)
    assert a_prompt.split("\nSCHEMA\n")[0] == b_prompt.split("\nSCHEMA\n")[0]
    assert a_prompt.split("\nCONTEXT\n")[0] == b_prompt.split("\nCONTEXT\n")[0]
    assert source["experiment_005"]["evidence_id"] not in a_prompt
    assert source["reflection_id"] not in a_prompt
    assert "Hilbert Tile Traversal" not in a_prompt
    assert "accepted_reflection" not in a_prompt
    assert "accepted_reflection" in b_prompt
    assert study.MODEL == "muse-spark-1.3"
    assert study.REQUEST_OPTIONS["reasoning"] == {"effort": "low"}


def test_missing_or_invalid_reflection_fails_closed(monkeypatch):
    with pytest.raises(ValueError):
        study.sources(ROOT, "not-a-stamp")
    with pytest.raises(ValueError):
        study.sources(ROOT, "20261030T000000000000Z")
    monkeypatch.setattr(study, "validate_audit", lambda *args: (_ for _ in ()).throw(ValueError("corrupt")))
    with pytest.raises(ValueError, match="corrupt"):
        study.sources(ROOT, REFLECTION_ID)


def test_matching_requests_and_audit_integrity(tmp_path, monkeypatch, source):
    study_id, clients = complete_requests(tmp_path, monkeypatch, source)
    saved = study.load_study(tmp_path, study_id)
    assert saved["proposal_count"] == 3
    assert saved["proposal_schema_fields"] == sorted(study.PROPOSAL_FIELDS)
    for condition in study.CONDITIONS:
        audit = study.load_request(tmp_path, study_id, condition)
        assert audit["parsed_proposals"]["ideas"] == proposals(saved["contexts"][condition], condition)["ideas"]
        assert audit["source_evidence_hashes"] == source["source_hashes"]
        assert audit["response_metadata"]["usage"]["input_tokens"] == 10
        assert clients[condition].calls == [{"model": study.MODEL,
            "input": saved["prompts"][condition], **study.REQUEST_OPTIONS}]
        assert study.request_condition(clients[condition], tmp_path, study_id, condition) == "already_attempted"
        assert len(clients[condition].calls) == 1
    setup = tmp_path / study.SECTION / study_id / "setup" / "study.json"
    original_setup = setup.read_bytes()
    setup.write_bytes(original_setup + b" ")
    with pytest.raises(ValueError, match="hashes"):
        study.load_study(tmp_path, study_id)
    setup.write_bytes(original_setup)
    folder = tmp_path / study.SECTION / study_id / "requests" / "A"
    (folder / "audit.json").write_bytes((folder / "audit.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes"):
        study.load_request(tmp_path, study_id, "A")


def test_proposal_schema_and_unsupported_claims(source):
    context = study.context_for(source, "B")
    valid = proposals(context, "B")
    assert study.parse_proposals(json.dumps(valid), context) == valid
    bad = copy.deepcopy(valid)
    bad["ideas"][0]["testable_prediction"] = "Maybe something changes."
    with pytest.raises(ValueError, match="testable"):
        study.parse_proposals(json.dumps(bad), context)
    bad = copy.deepcopy(valid)
    bad["ideas"][0]["why_it_might_work"] = "Prior reflection showed a speedup."
    with pytest.raises(ValueError):
        study.parse_proposals(json.dumps(bad), context)
    bad = copy.deepcopy(valid)
    bad["ideas"][0]["why_it_might_work"] = "Measured 2x speedup."
    with pytest.raises(ValueError):
        study.parse_proposals(json.dumps(bad), context)


def test_incomplete_request_is_rejected_and_retained(tmp_path, monkeypatch, source):
    monkeypatch.setattr(study, "sources", lambda root, reflection_id: source)
    study_id = study.prepare_study(tmp_path, REFLECTION_ID)
    context = study.context_for(source, "A")
    client = Client(json.dumps(proposals(context, "A")))
    def incomplete(**kwargs):
        client.calls.append(kwargs)
        return SimpleNamespace(id="incomplete", model=study.MODEL, status="incomplete",
                               incomplete_details={"reason": "max_output_tokens"}, output_text=client.raw)
    client.create = incomplete
    assert study.request_condition(client, tmp_path, study_id, "A") == "rejected"
    audit = json.loads((tmp_path / study.SECTION / study_id / "requests" / "A" / "audit.json").read_text())
    assert audit["rejection_kind"] == "incomplete_output"
    assert audit["raw_response"] == client.raw
    assert audit["parsed_proposals"] is None
    assert study.request_condition(client, tmp_path, study_id, "A") == "already_attempted"
    assert len(client.calls) == 1


def test_blind_assignment_scores_reveal_and_tampering(tmp_path, monkeypatch, source):
    study_id, _ = complete_requests(tmp_path, monkeypatch, source)
    review_id = study.prepare_review(tmp_path, study_id, seed=1729)
    blind = study.load_blind(tmp_path, study_id, review_id)
    second_review = study.prepare_review(tmp_path, study_id, seed=1729)
    assert study.load_blind(tmp_path, study_id, second_review)["cards"] == blind["cards"]
    assert len(blind["cards"]) == 6
    assert all("condition" not in json.dumps(card) for card in blind["cards"])
    assert all("evidence_references" not in card["proposal"] for card in blind["cards"])
    folder = tmp_path / study.SECTION / study_id / "reviews" / review_id
    key = json.loads((folder / "operator" / "assignment.json").read_text())
    assert key["seed"] == 1729
    assert "assignment" not in key
    scores = {card["evaluation_id"]: {criterion: {"score": 3, "justification":
        "This is a bounded subjective assessment."} for criterion in study.CRITERIA}
        for card in blind["cards"]}
    with pytest.raises(ValueError):
        study.reveal_review(tmp_path, study_id, review_id)
    with pytest.raises(ValueError):
        study.score_review(tmp_path, study_id, review_id, {"P01": scores["P01"]})
    study.score_review(tmp_path, study_id, review_id, scores)
    result = study.reveal_review(tmp_path, study_id, review_id)
    assert result["seed"] == 1729
    assert {x["condition"] for x in result["assignments"]} == {"A", "B"}
    assert "subjective" in result["judgment_label"]
    assert len(result["proposals"]) == 6
    assert all("evidence_references" in item["proposal"] for item in result["proposals"])
    assert result["mean_subjective_scores_by_condition"] == {
        condition: {criterion: 3.0 for criterion in study.CRITERIA}
        for condition in study.CONDITIONS}
    (folder / "scores" / "scores.json").write_text("{}")
    with pytest.raises(ValueError, match="hashes"):
        study.reveal_review(tmp_path, study_id, review_id)
