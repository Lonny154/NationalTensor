"""CPU-only request/audit checks with a fake client; no Muse calls."""
import json
from types import SimpleNamespace

import pytest

import muse_critic_feasibility_010 as exp
from muse_debrief import canonical, digest
import muse_research_critic_v3 as v3


@pytest.fixture
def source():
    proposal = {"name": "Synthetic P06"}
    return {"proposal_id": "P06", "proposal_sha256": digest(canonical(proposal)),
            "proposal": proposal, "source_hashes": {"fixture_sha256": "a" * 64},
            "validated_evidence": {
                "003": {"evidence_id": "e:003", "observations": {"workload": [128, 3072, 768]}},
                "004": {"evidence_id": "e:004", "observations": {"control_workload": [128, 3072, 768]}},
                "005": {"evidence_id": "e:005", "observations": {"confirmation_status": "missing"}}}}


def response_text(source):
    return json.dumps({"protocol_version": v3.VERSION, "proposal_id": "P06",
                       "proposal_sha256": source["proposal_sha256"], "recommendation": "revise",
                       "claims": [{"claim_id": "C1", "type": "observation", "evidence_id": "e:005",
                                   "observation_key": "confirmation_status", "value_path": [],
                                   "recorded_value": "missing", "unit": "none"}]})


def fake_response(raw):
    return {"id": "resp-fixture", "model": exp.MODEL, "status": "completed", "max_output_tokens": 8000,
            "usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": raw}]}]}


def test_one_valid_attempt_and_read_only_manifest_validation(tmp_path, monkeypatch, source):
    monkeypatch.setattr(exp, "_source", lambda root: source)
    run_id = exp.prepare(tmp_path)
    request, _ = exp.load_request(tmp_path, run_id)
    assert request["proposal_id"] == "P06" and request["request_options"]["max_output_tokens"] == 8000
    assert request["temperature"] is None and request["prompt_sha256"]
    calls = []
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: (
        calls.append(kwargs), fake_response(response_text(source)))[1]))
    assert exp.request_once(client, tmp_path, run_id) == "valid"
    assert len(calls) == 1 and calls[0]["model"] == exp.MODEL
    result = exp.validate(tmp_path, run_id)
    assert result["structural_status"] == "valid"
    assert result["scientific_status"] == "pending_human_review"
    assert result["response_metadata"]["usage"]["output_tokens"] == 25
    with pytest.raises(ValueError, match="no retry"):
        exp.request_once(client, tmp_path, run_id)
    assert len(calls) == 1
    audit_path = tmp_path / exp.SECTION / run_id / "result/audit.json"
    audit_path.write_bytes(audit_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes"):
        exp.validate(tmp_path, run_id)


def test_invalid_output_is_preserved_without_retry(tmp_path, monkeypatch, source):
    monkeypatch.setattr(exp, "_source", lambda root: source)
    run_id = exp.prepare(tmp_path)
    raw = '{"bad":true}'
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: fake_response(raw)))
    assert exp.request_once(client, tmp_path, run_id) == "rejected"
    audit = json.loads((tmp_path / exp.SECTION / run_id / "result/audit.json").read_text())
    assert audit["raw_response"] == raw and audit["first_error"] and audit["parsed_response"] is None
    assert exp.validate(tmp_path, run_id)["structural_status"] == "rejected"


def test_transport_error_is_recorded_without_exception_text(tmp_path, monkeypatch, source):
    monkeypatch.setattr(exp, "_source", lambda root: source)
    run_id = exp.prepare(tmp_path)
    def fail(**kwargs):
        raise RuntimeError("secret-bearing transport detail")
    client = SimpleNamespace(responses=SimpleNamespace(create=fail))
    assert exp.request_once(client, tmp_path, run_id) == "rejected"
    audit = json.loads((tmp_path / exp.SECTION / run_id / "result/audit.json").read_text())
    assert audit["first_error"] == "transport_or_response_error"
    assert "secret-bearing" not in json.dumps(audit)
