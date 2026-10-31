"""CPU-only v3.1 prompt and Experiment 011 preparation checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_critic_feasibility_011 as exp
from muse_debrief import canonical, digest
from muse_research_critic import source_package
from muse_research_critic_v3 import critique_prompt_v3, parse_critique_v3
from muse_research_critic_v31 import PROMPT_VERSION, critique_prompt_v31

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RUN = "20261031T205719056741Z"


@pytest.fixture
def source():
    proposal = {"name": "Synthetic P06"}
    return {"proposal_id": "P06", "proposal_sha256": digest(canonical(proposal)),
            "proposal": proposal, "source_hashes": {"fixture_sha256": "a" * 64},
            "validated_evidence": {
                "003": {"evidence_id": "e:003", "observations": {"workload": [128, 3072, 768]}},
                "004": {"evidence_id": "e:004", "observations": {"control_workload": [128, 3072, 768]}},
                "005": {"evidence_id": "e:005", "observations": {
                    "confirmation_status": "missing",
                    "profiling_status": "not validated by Experiment 005 evidence pipeline",
                    "screening:512x3072x768:row_major": {"median_paired_ratio": 0.9971394589037794}}}}}


def response(source, claims):
    return {"protocol_version": "experiment_008_critic_v3", "proposal_id": "P06",
            "proposal_sha256": source["proposal_sha256"], "recommendation": "revise", "claims": claims}


def observation(ident, key, value, path, unit="none"):
    return {"claim_id": ident, "type": "observation", "evidence_id": "e:005",
            "observation_key": key, "value_path": path, "recorded_value": value, "unit": unit}


def test_scalar_paths_are_required_and_nested_paths_remain_exact(source):
    scalar = observation("C1", "confirmation_status", "missing", [])
    profiling = observation("C2", "profiling_status", "not validated by Experiment 005 evidence pipeline", [])
    assert parse_critique_v3(json.dumps(response(source, [scalar, profiling])), source)["structural_status"] == "valid"
    missing = copy.deepcopy(scalar); missing.pop("value_path")
    with pytest.raises(ValueError, match="observation schema mismatch"):
        parse_critique_v3(json.dumps(response(source, [missing])), source)
    wrong = copy.deepcopy(scalar); wrong["value_path"] = ["stage_decision"]
    with pytest.raises(ValueError, match="unknown observation value path"):
        parse_critique_v3(json.dumps(response(source, [wrong])), source)
    nested = observation("C1", "screening:512x3072x768:row_major", 0.9971394589037794,
                         ["median_paired_ratio"], "ratio")
    assert parse_critique_v3(json.dumps(response(source, [nested])), source)["structural_status"] == "valid"
    nested["value_path"] = []
    with pytest.raises(ValueError, match="recorded value does not match source"):
        parse_critique_v3(json.dumps(response(source, [nested])), source)


def test_prompt_has_complete_scalar_and_nested_examples_without_changing_v3(source):
    old = critique_prompt_v3(source)
    new = critique_prompt_v31(source)
    assert critique_prompt_v3(source) == old and old != new
    instructions, rest = new.split("\nSCHEMA\n", 1)
    schema_raw, rest = rest.split("\nEXAMPLES\n", 1)
    examples_raw, tail = rest.split("\nAGGREGATE_SETS\n", 1)
    examples = json.loads(examples_raw)
    scalar = examples["scalar_status_observation"]
    nested = examples["nested_screening_ratio_observation"]
    assert "Every observation MUST include value_path" in instructions
    assert "Use []" in instructions
    assert scalar["value_path"] == [] and nested["value_path"] == ["median_paired_ratio"]
    assert set(scalar) == set(nested) == {"claim_id", "type", "evidence_id", "observation_key",
                                         "value_path", "recorded_value", "unit"}
    assert json.loads(schema_raw)["protocol_version"] == "experiment_008_critic_v3"
    assert new.split("\nSOURCE\n", 1)[1] == old.split("\nSOURCE\n", 1)[1]
    assert tail.split("\nSOURCE\n", 1)[0] == old.split("\nAGGREGATE_SETS\n", 1)[1].split("\nSOURCE\n", 1)[0]


def test_original_experiment_010_response_remains_rejected_and_copy_is_only_hypothetical():
    audit_path = ROOT / "evaluations/muse-critic-feasibility/v1" / HISTORICAL_RUN / "result/audit.json"
    audit = json.loads(audit_path.read_bytes())
    source = source_package(ROOT, exp.STUDY_ID, exp.REVIEW_ID, "P06")
    original = audit["raw_response"]
    assert audit["structural_status"] == "rejected"
    with pytest.raises(ValueError, match="observation schema mismatch"):
        parse_critique_v3(original, source)
    candidate = json.loads(original)
    candidate["claims"][5]["value_path"] = []
    candidate["claims"][6]["value_path"] = []
    parsed = parse_critique_v3(json.dumps(candidate), source)
    assert parsed["structural_status"] == "valid"
    assert parsed["scientific_status"] == "pending_human_review"
    assert audit_path.read_bytes() == canonical(audit)


def test_separate_manifested_setup_and_one_attempt_mock(tmp_path, monkeypatch, source):
    monkeypatch.setattr(exp, "_source", lambda root: source)
    run_id = exp.prepare(tmp_path)
    request, _ = exp.load_request(tmp_path, run_id)
    assert exp.validate(tmp_path, run_id)["state"] == "prepared"
    assert request["prompt_version"] == PROMPT_VERSION
    assert request["response_schema"] == "experiment_008_critic_v3"
    assert request["model"] == exp.MODEL and request["request_options"]["max_output_tokens"] == 8000
    assert request["temperature"] is None and exp.SECTION != Path("evaluations/muse-critic-feasibility/v1")
    calls = []
    raw = json.dumps(response(source, [observation("C1", "confirmation_status", "missing", [])]))
    def create(**kwargs):
        calls.append(kwargs)
        return {"id": "fake", "model": exp.MODEL, "status": "completed", "max_output_tokens": 8000,
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": raw}]}]}
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    assert exp.request_once(client, tmp_path, run_id) == "valid"
    assert len(calls) == 1 and exp.validate(tmp_path, run_id)["structural_status"] == "valid"
    with pytest.raises(ValueError, match="no retry"):
        exp.request_once(client, tmp_path, run_id)
    assert len(calls) == 1
