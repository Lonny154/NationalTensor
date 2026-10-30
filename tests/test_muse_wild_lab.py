import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_wild_lab as lab


def ideas(context):
    reference = {"evidence_id": context["experiments"]["004"]["evidence_id"],
                 "observation_key": "partial_m_workload"}
    concepts = (
        ("Tile Relay", "Have neighboring warps cooperatively relay a packed boundary tile through shared memory."),
        ("Sparse Map Echo", "Borrow sparse graph partitioning ideas to reorder output tiles for locality."),
        ("Mixed Radix Tensor", "Use a new numerical representation for intermediate products with a stated error budget."),
    )
    return {"ideas": [{"category": category, "name": name, "core_mechanism": mechanism,
        "why_it_might_work": "It might reduce redundant movement if locality is favorable.",
        "what_makes_it_unconventional": "It changes data organization beyond configuration tuning.",
        "necessary_assumptions": "The compiler preserves the intended layout and resource use fits the GPU.",
        "biggest_obstacle": "Shared memory pressure and compiler lowering may erase the benefit.",
        "disproof_observation": "A compiled proof of concept shows no useful structural change.",
        "smallest_proof_of_concept": "Review one isolated compiler and correctness diagnostic after approval.",
        "numerical_equivalence": "approximate" if index == 2 else "exact",
        "evidence_references": [reference]}
        for index, (category, (name, mechanism)) in enumerate(zip(lab.CATEGORIES, concepts))]}


@pytest.fixture
def context():
    return lab.context_package()


def response(value, *, status="completed", model=lab.MODEL):
    return SimpleNamespace(status=status, model=model, id="test-response",
        output=[{"type": "message", "status": status,
                 "content": [{"type": "output_text", "text": value}]}])


class Client:
    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = 0
        self.responses = self
    def create(self, **kwargs):
        self.calls += 1
        return self.queue.pop(0)


def test_context_is_verified_and_concise(context):
    assert context["environment"]["gpu"] == "NVIDIA GeForce RTX 4060 Ti"
    assert context["environment"]["triton"]
    assert context["experiments"]["003"]["observations"]["original_muse_source_evaluated"] is False
    assert context["experiments"]["004"]["observations"]["runtime_measured"] is False
    assert len(json.dumps(context)) < 4000
    assert "ideas need not change one variable" in lab.prompt_for(context)


def test_missing_evidence_fails_closed(tmp_path):
    with pytest.raises(ValueError):
        lab.context_package(tmp_path)


def test_three_distinct_structured_ideas(context):
    valid = ideas(context)
    assert lab.parse_response(json.dumps(valid), context) == valid
    for change in ("missing", "duplicate", "citation", "performance", "code", "repeat", "equivalence"):
        bad = copy.deepcopy(valid)
        if change == "missing": bad["ideas"].pop()
        if change == "duplicate": bad["ideas"][1]["core_mechanism"] = bad["ideas"][0]["core_mechanism"]
        if change == "citation": bad["ideas"][0]["evidence_references"][0]["observation_key"] = "fiction"
        if change == "performance": bad["ideas"][0]["why_it_might_work"] = "Measured 2x speedup."
        if change == "code": bad["ideas"][0]["smallest_proof_of_concept"] = "@triton.jit def kernel(x): pass"
        if change == "repeat": bad["ideas"][0]["core_mechanism"] = "Repeat mask hoisting on M=128."
        if change == "equivalence": bad["ideas"][0]["numerical_equivalence"] = "unknown"
        with pytest.raises(ValueError): lab.parse_response(json.dumps(bad), context)


def test_audit_rejection_idempotency_and_explicit_round(tmp_path, monkeypatch, context):
    monkeypatch.setattr(lab, "context_package", lambda root=lab.ROOT: context)
    raw = json.dumps(ideas(context))
    client = Client(response(raw, status="incomplete"), response(raw), response("{}"), response(raw))
    assert lab.request_once(client, tmp_path) == "rejected"
    assert lab.load_accepted(tmp_path, context) is None
    assert lab.request_once(client, tmp_path) == "already_attempted"
    assert client.calls == 1
    assert lab.request_once(client, tmp_path, new_round=True) == "accepted"
    first = lab.load_accepted(tmp_path, context)
    assert first["ideas"] == ideas(context)
    assert lab.request_once(client, tmp_path) == "already_accepted"
    assert client.calls == 2
    assert lab.request_once(client, tmp_path, new_round=True) == "rejected"
    assert lab.load_accepted(tmp_path, context) == first
    assert lab.request_once(client, tmp_path, new_round=True) == "accepted"
    second = lab.load_accepted(tmp_path, context)
    assert second["round_id"] != first["round_id"]
    assert second["previous_accepted_sha256"]
    assert len(list((tmp_path / lab.SECTION).glob("*/audit.json"))) == 4


def test_reject_wrong_model_and_corrupt_pointer(tmp_path, monkeypatch, context):
    monkeypatch.setattr(lab, "context_package", lambda root=lab.ROOT: context)
    wrong = Client(response(json.dumps(ideas(context)), model="wrong-model"))
    assert lab.request_once(wrong, tmp_path) == "rejected"
    assert lab.load_accepted(tmp_path, context) is None
    correct = Client(response(json.dumps(ideas(context))))
    assert lab.request_once(correct, tmp_path) == "already_attempted"
    assert lab.request_once(correct, tmp_path, new_round=True) == "accepted"
    path = tmp_path / lab.SECTION / "accepted.json"
    saved = json.loads(path.read_text())
    saved["context_sha256"] = "0" * 64
    path.write_text(json.dumps(saved))
    assert lab.load_accepted(tmp_path, context) is None


def test_dashboard_missing_and_accepted(monkeypatch, context):
    from dashboard_app import create_app
    monkeypatch.setattr(lab, "load_accepted", lambda root, context: None)
    client = create_app(Path.cwd()).test_client()
    page = client.get("/research")
    assert page.status_code == 200
    assert b"Wild Research Lab" in page.data
    assert b"No accepted ideation round yet" in page.data
    monkeypatch.setattr(lab, "load_accepted", lambda root, context: {
        "ideas": ideas(context), "response_model": lab.MODEL,
        "round_id": "test-round", "prompt_version": lab.PROMPT_VERSION})
    page = client.get("/research")
    assert page.status_code == 200
    assert page.data.count(b"SPECULATIVE") >= 3
    for name in (b"Tile Relay", b"Sparse Map Echo", b"Mixed Radix Tensor"):
        assert name in page.data
