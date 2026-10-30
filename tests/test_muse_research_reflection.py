"""CPU-only Experiment 006 evidence, response, and audit tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_research_reflection as reflection

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def package():
    return reflection.evidence_package(ROOT)


def answer(package):
    evidence_id = package["evidence_id"]
    timing_key = next(k for k in package["observations"] if k.startswith("screening:"))
    def field(text, *keys):
        return {"text": text, "citations": [
            {"evidence_id": evidence_id, "observation_key": key} for key in keys]}
    return {"decision": "revise",
            "evidence_assessment": field("Observed screening latency has descriptive uncertainty.", timing_key),
            "mechanism_assessment": field("The locality mechanism remains an inference.", "original_hypothesis"),
            "remaining_uncertainties": field("Confirmation is missing and profiling evidence is unavailable.",
                                             "confirmation_status", "profiling_status"),
            "next_research_direction": field("Review the tile traversal mechanism.", "original_hypothesis"),
            "proposed_validation": field("Consider an independently approved confirmation and targeted profiling.",
                                         "confirmation_status", "profiling_status")}


def test_loads_original_proposal_and_validated_screening(package):
    assert package == reflection.evidence_package(ROOT)
    assert package["observations"]["original_hypothesis"]["name"] == "Hilbert Tile Traversal"
    assert len([key for key in package["observations"] if key.startswith("screening:")]) == 8
    assert package["observations"]["confirmation_status"] == "missing"
    assert "screening_manifest_sha256" in package["source_hashes"]
    prompt = reflection.prompt_for(package)
    assert "bootstrap_median_ratio_95pct" in prompt
    assert "evaluation_criteria" in prompt
    assert "<abandon|revise|investigate_further>" in prompt


def test_missing_screening_and_verify_are_rejected(monkeypatch, package):
    monkeypatch.setattr(reflection, "latest", lambda root: None)
    with pytest.raises(ValueError, match="verification"):
        reflection.evidence_package(ROOT)
    monkeypatch.setattr(reflection, "latest", lambda root: {"run_id": "x", "screening": None})
    with pytest.raises(ValueError, match="screening"):
        reflection.evidence_package(ROOT)


def test_schema_and_mechanically_detectable_claims(package):
    valid = answer(package)
    assert reflection.parse_response(json.dumps(valid), package) == valid
    cases = []
    bad = copy.deepcopy(valid); bad["decision"] = "success"; cases.append(bad)
    bad = copy.deepcopy(valid); bad["evidence_assessment"]["citations"][0]["observation_key"] = "screening:imaginary"; cases.append(bad)
    bad = copy.deepcopy(valid); bad["evidence_assessment"]["text"] = "Observed 999 ms latency."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["evidence_assessment"]["text"] = "Experiment 999 measured latency."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["mechanism_assessment"]["text"] = "Hilbert improved cache locality."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["remaining_uncertainties"]["text"] = "The results need review."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["evidence_assessment"]["text"] = "Hilbert achieved a speedup."; cases.append(bad)
    for item in cases:
        with pytest.raises(ValueError):
            reflection.parse_response(json.dumps(item), package)


def test_audit_integrity_with_fake_client(tmp_path, monkeypatch, package):
    monkeypatch.setattr(reflection, "evidence_package", lambda root: package)
    raw = json.dumps(answer(package))
    response = SimpleNamespace(id="fake-response", model=reflection.MODEL,
                               status="completed", output_text=raw,
                               usage={"input_tokens": 20, "output_tokens": 30})
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    assert reflection.request_once(client, tmp_path) == "accepted"
    folder = next((tmp_path / reflection.SECTION).glob("*/audit.json")).parent
    audit_raw = (folder / "audit.json").read_bytes()
    audit = json.loads(audit_raw)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    assert manifest["artifact_hashes"]["audit.json"] == reflection.sha(audit_raw)
    assert audit["prompt"] == reflection.prompt_for(package)
    assert audit["raw_response"] == raw
    assert audit["parsed_reflection"] == answer(package)
    assert audit["response_metadata"]["usage"]["input_tokens"] == 20
    assert audit["source_evidence_hashes"] == package["source_hashes"]
    assert reflection.validate_audit(tmp_path, folder.name) == audit
    (folder / "audit.json").write_bytes(audit_raw + b" ")
    with pytest.raises(ValueError, match="hashes"):
        reflection.validate_audit(tmp_path, folder.name)


def test_incomplete_response_is_audited_as_rejected(tmp_path, monkeypatch, package):
    monkeypatch.setattr(reflection, "evidence_package", lambda root: package)
    response = SimpleNamespace(id="fake-incomplete", model=reflection.MODEL,
                               status="incomplete", incomplete_details={"reason": "max_output_tokens"},
                               output_text=json.dumps(answer(package)))
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    assert reflection.request_once(client, tmp_path) == "rejected"
    audit = json.loads(next((tmp_path / reflection.SECTION).glob("*/audit.json")).read_bytes())
    assert audit["rejection_kind"] == "incomplete_output"
    assert audit["parsed_reflection"] is None
