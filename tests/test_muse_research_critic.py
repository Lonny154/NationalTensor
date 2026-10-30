"""CPU-only checks for Experiment 008 source, criticism, and revision audits."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import muse_research_critic as critic
import muse_discovery_study as discovery

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "20261030T142150402688Z"
REVIEW_ID = "20261030T142419308682Z"
REJECTED_CASE_ID = "20261030T145230747865Z"
SECOND_REJECTED_CASE_ID = "20261030T150228098163Z"
FINAL_REJECTED_CASE_ID = "20261030T152949236420Z"
FINAL_REJECTED_REVISION_SHA256 = "609dfa48370920762e85a7fb01cf4cd769190f00c49630e29872f38eb8f63eac"
FINAL_REJECTED_MANIFEST_SHA256 = "740bc1af3fefc3600bfd2f9a45569dd9bdbae19b1fcbbb3733d4dc9eda44ab1f"
REJECTED_REVISION_SHA256 = "2a146295684b2458cc242cbf8304715bb6faf4055a5d6eafc4d5be5917345c63"
REJECTED_REVISION_MANIFEST_SHA256 = "7f9a3e027d175291c48592c1af66c565ac5aa829654e0ecd310697fdcc435330"


@pytest.fixture(scope="module")
def rejected_audit():
    folder = ROOT / critic.SECTION / REJECTED_CASE_ID
    audit_raw = (folder / "criticism" / "audit.json").read_bytes()
    assert critic.sha(audit_raw) == "3850c0acd753bc2ce6b5cb376e09c32adeeb3c3e88f03fa7f6b550fdc493a8e2"
    audit = json.loads(audit_raw)
    source = json.loads((folder / "setup" / "case.json").read_bytes())["source"]
    return audit, source


@pytest.fixture(scope="module")
def second_rejected_audit():
    folder = ROOT / critic.SECTION / SECOND_REJECTED_CASE_ID
    audit_raw = (folder / "criticism" / "audit.json").read_bytes()
    manifest_raw = (folder / "criticism" / "manifest.json").read_bytes()
    assert critic.sha(audit_raw) == "9bd3f0795b24762db2f71ef0045ccc77b0985007a4a9add8461997c4737160d6"
    assert critic.sha(manifest_raw) == "9499a0966b18e09e0f9da4daf464cb6fae1ece3e5c2487ba0e0208b090964a1b"
    audit = json.loads(audit_raw)
    source = json.loads((folder / "setup" / "case.json").read_bytes())["source"]
    return audit, source


@pytest.fixture(scope="module")
def rejected_revision():
    folder = ROOT / critic.SECTION / REJECTED_CASE_ID
    audit_raw = (folder / "revision" / "audit.json").read_bytes()
    manifest_raw = (folder / "revision" / "manifest.json").read_bytes()
    assert critic.sha(audit_raw) == REJECTED_REVISION_SHA256
    assert critic.sha(manifest_raw) == REJECTED_REVISION_MANIFEST_SHA256
    audit = json.loads(audit_raw)
    source = json.loads((folder / "setup" / "case.json").read_bytes())["source"]
    preparation = json.loads((folder / "revision_setup" / "setup.json").read_bytes())
    study = discovery.load_study(ROOT, source["study_id"])
    original = discovery.load_request(ROOT, source["study_id"], source["condition"], study=study)
    return audit, source, preparation, critic.revision_context(study, source), original["parsed_proposals"]["ideas"]


@pytest.fixture(scope="module")
def final_rejected_revision():
    folder = ROOT / critic.SECTION / FINAL_REJECTED_CASE_ID
    audit_raw = (folder / "revision" / "audit.json").read_bytes()
    manifest_raw = (folder / "revision" / "manifest.json").read_bytes()
    assert critic.sha(audit_raw) == FINAL_REJECTED_REVISION_SHA256
    assert critic.sha(manifest_raw) == FINAL_REJECTED_MANIFEST_SHA256
    audit = json.loads(audit_raw)
    source = json.loads((folder / "setup" / "case.json").read_bytes())["source"]
    preparation = json.loads((folder / "revision_setup" / "setup.json").read_bytes())
    study = discovery.load_study(ROOT, source["study_id"])
    original = discovery.load_request(ROOT, source["study_id"], source["condition"], study=study)
    return audit, source, preparation, critic.revision_context(study, source), original["parsed_proposals"]["ideas"]


@pytest.fixture(scope="module")
def source():
    return critic.source_package(ROOT, STUDY_ID, REVIEW_ID, "P04")


def reference(source, key="confirmation_status"):
    return {"evidence_id": source["validated_evidence"]["005"]["evidence_id"],
            "observation_key": key}


def criticism(source, recommendation="proceed"):
    return {"proposal_id": source["proposal_id"], "proposal_sha256": source["proposal_sha256"],
        "closest_related_prior_research": {"experiment_id": "005",
            "what_was_tested": "Experiment 005 tested Hilbert traversal against two schedules in screening.",
            "relation": "The proposed work redistribution addresses a different scheduling mechanism."},
        "evidence_citations": [reference(source, "screening_scope"), reference(source)],
        "assessments": {
            "prior_research_overlap": "The scheduling theme overlaps, while the tile ownership mechanism differs.",
            "evidence_consistency": "The screening result does not establish a repeatable speed benefit.",
            "hardware_mechanism": "Load balancing is plausible if work distribution is uneven.",
            "falsifiability": "A fixed control can distinguish timing from compilation changes.",
            "numerical_correctness": "Changing reduction order may alter rounding; exact equivalence needs checking.",
            "research_value": "The result would test whether redistribution helps this workload."},
        "observed_evidence": "Screening exists; independent confirmation is missing.",
        "inferred_explanations": "A load balance explanation remains an inference.",
        "unresolved_questions": "The scheduler and rounding behavior remain unverified.",
        "potential_technical_flaws": "Scheduling overhead might offset a load balance gain.",
        "unsupported_assumptions": "The GPU must have enough parallel work to benefit.",
        "proposed_discriminating_tests": "Hold the kernel body fixed and compare scheduler behavior and latency.",
        "research_value_assessment": "A controlled result would isolate the work redistribution question.",
        "recommendation": recommendation,
        "recommendation_explanation": "Proceed with a design review because the mechanism is distinct and testable."}


def revision(source, critique, *, action="revise"):
    revised = copy.deepcopy(source["proposal"]) if action == "revise" else None
    if revised:
        revised["name"] += " Refined"
        revised["core_mechanism"] += " Constrain the work partition to a fixed control."
    return {"proposal_id": source["proposal_id"], "proposal_sha256": source["proposal_sha256"],
            "criticism_sha256": critic.digest(critic.canonical(critique)), "action": action,
            "what_changed": "The partition is constrained to a fixed control." if revised else "Unchanged.",
            "why": "The criticism identified an unresolved scheduling assumption.",
            "response_to_criticism": "The controlled comparison addresses the strongest criticism.",
            "evidence_citations": [reference(source)], "revised_proposal": revised}


class Client:
    def __init__(self, raw, *, status="completed"):
        self.raw, self.status = raw, status
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="fake-response", model=critic.MODEL, status=self.status,
                               incomplete_details={"reason": "max_output_tokens"} if self.status != "completed" else None,
                               output_text=self.raw, usage={"input_tokens": 20, "output_tokens": 40})


def setup_case(tmp_path, monkeypatch, source):
    monkeypatch.setattr(critic, "source_package", lambda root, study_id, review_id, proposal_id: source)
    return critic.prepare(tmp_path, STUDY_ID, REVIEW_ID, source["proposal_id"])


def copy_rejected_case(tmp_path, monkeypatch, case_id):
    original = ROOT / critic.SECTION / case_id
    target = tmp_path / critic.SECTION / case_id
    shutil.copytree(original / "setup", target / "setup")
    shutil.copytree(original / "criticism", target / "criticism")
    source = json.loads((target / "setup" / "case.json").read_bytes())["source"]
    monkeypatch.setattr(critic, "source_package", lambda root, study_id, review_id, proposal_id: source)
    return target, source


@pytest.mark.parametrize("case_id", [REJECTED_CASE_ID, SECOND_REJECTED_CASE_ID])
def test_offline_revalidation_of_historical_rejections(tmp_path, monkeypatch, case_id):
    target, source = copy_rejected_case(tmp_path, monkeypatch, case_id)
    original_audit = (target / "criticism" / "audit.json").read_bytes()
    original_manifest = (target / "criticism" / "manifest.json").read_bytes()
    record = critic.revalidate_criticism(tmp_path, case_id)
    assert record["schema"] == critic.REVALIDATION_SCHEMA
    assert record["validator_version"] == critic.CRITIC_VALIDATOR_VERSION
    assert record["validation_status"] == "valid"
    assert record["rejection_reason"] is None
    assert record["proposal_sha256"] == source["proposal_sha256"]
    assert record["original_audit_sha256"] == critic.sha(original_audit)
    assert record["original_manifest_sha256"] == critic.sha(original_manifest)
    assert record["parsed_criticism"]["proposal_id"] == "P01"
    assert critic.load_revalidation(tmp_path, case_id, record["revalidation_id"]) == record
    assert (target / "criticism" / "audit.json").read_bytes() == original_audit
    assert (target / "criticism" / "manifest.json").read_bytes() == original_manifest
    assert critic.validate_case(tmp_path, case_id)["criticism"] == "rejected"


def test_revision_uses_revalidation_and_preserves_both_bindings(tmp_path, monkeypatch):
    target, source = copy_rejected_case(tmp_path, monkeypatch, REJECTED_CASE_ID)
    original_audit = (target / "criticism" / "audit.json").read_bytes()
    record = critic.revalidate_criticism(tmp_path, REJECTED_CASE_ID)
    setup = critic.prepare_revision(tmp_path, REJECTED_CASE_ID,
                                    revalidation_id=record["revalidation_id"])
    assert setup["criticism_source"]["kind"] == "offline_revalidation"
    assert setup["criticism_source"]["original_audit_sha256"] == critic.sha(original_audit)
    assert setup["criticism_source"]["original_response_sha256"] == record["original_response_sha256"]
    assert setup["criticism_source"]["revalidation_sha256"] == critic.digest(critic.canonical(record))
    assert critic.load_revision_setup(tmp_path, REJECTED_CASE_ID) == setup
    study = discovery.load_study(ROOT, STUDY_ID)
    request = discovery.load_request(ROOT, STUDY_ID, source["condition"], study=study)
    monkeypatch.setattr(critic, "load_study", lambda root, study_id: study)
    monkeypatch.setattr(critic, "load_request", lambda root, study_id, condition, *, study=None: request)
    reply = revision(source, record["parsed_criticism"], action="withdraw")
    assert critic.request_revision(Client(json.dumps(reply)), tmp_path, REJECTED_CASE_ID) == "accepted"
    revision_audit = json.loads((target / "revision" / "audit.json").read_bytes())
    assert revision_audit["criticism_source"] == setup["criticism_source"]
    assert revision_audit["criticism_sha256"] == critic.digest(critic.canonical(record))
    assert critic.validate_case(tmp_path, REJECTED_CASE_ID)["revision"] == "accepted"
    assert json.loads((target / "criticism" / "audit.json").read_bytes())["status"] == "rejected"


def test_invalid_revalidation_cannot_authorize_revision(tmp_path, monkeypatch):
    target, source = copy_rejected_case(tmp_path, monkeypatch, SECOND_REJECTED_CASE_ID)
    audit_path = target / "criticism" / "audit.json"
    audit = json.loads(audit_path.read_bytes())
    bad = json.loads(audit["raw_response"])
    bad["observed_evidence"] = "Observed latency was 999 ms."
    audit["raw_response"] = json.dumps(bad)
    audit["response_metadata"]["text_characters"] = len(audit["raw_response"])
    audit_path.write_bytes(critic.canonical(audit))
    (target / "criticism" / "manifest.json").write_bytes(critic.canonical(
        critic.manifest(target / "criticism", schema="experiment_008_criticism_v1")))
    record = critic.revalidate_criticism(tmp_path, SECOND_REJECTED_CASE_ID)
    assert record["validation_status"] == "invalid"
    assert "unsupported numerical measurement" in record["rejection_reason"]
    assert record["parsed_criticism"] is None
    with pytest.raises(ValueError, match="cannot authorize"):
        critic.prepare_revision(tmp_path, SECOND_REJECTED_CASE_ID,
                                revalidation_id=record["revalidation_id"])
    assert not (target / "revision_setup").exists()


def test_revalidation_tampering_and_source_changes_block_revision(tmp_path, monkeypatch):
    target, source = copy_rejected_case(tmp_path, monkeypatch, REJECTED_CASE_ID)
    record = critic.revalidate_criticism(tmp_path, REJECTED_CASE_ID)
    folder = target / "revalidations" / "v1" / record["revalidation_id"]
    saved_result = (folder / "result.json").read_bytes()
    (folder / "result.json").write_bytes(saved_result + b" ")
    with pytest.raises(ValueError, match="hashes"):
        critic.prepare_revision(tmp_path, REJECTED_CASE_ID,
                                revalidation_id=record["revalidation_id"])
    (folder / "result.json").write_bytes(saved_result)
    changed = json.loads(saved_result)
    changed["original_audit_sha256"] = "0" * 64
    (folder / "result.json").write_bytes(critic.canonical(changed))
    (folder / "manifest.json").write_bytes(critic.canonical(
        critic.manifest(folder, schema=critic.REVALIDATION_SCHEMA)))
    with pytest.raises(ValueError, match="source, or validator mismatch"):
        critic.prepare_revision(tmp_path, REJECTED_CASE_ID,
                                revalidation_id=record["revalidation_id"])
    (folder / "result.json").write_bytes(saved_result)
    (folder / "manifest.json").write_bytes(critic.canonical(
        critic.manifest(folder, schema=critic.REVALIDATION_SCHEMA)))
    audit_path = target / "criticism" / "audit.json"
    audit_path.write_bytes(audit_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes"):
        critic.prepare_revision(tmp_path, REJECTED_CASE_ID,
                                revalidation_id=record["revalidation_id"])


def test_validates_real_007_chain_and_missing_sources(source):
    assert source["proposal"] == discovery.load_request(ROOT, STUDY_ID, source["condition"])["parsed_proposals"]["ideas"][source["proposal_index"]]
    assert source["proposal_sha256"] == critic.digest(critic.canonical(source["proposal"]))
    assert len(source["validated_evidence"]) == 3
    assert source["validated_evidence"]["005"]["observations"]["confirmation_status"] == "missing"
    with pytest.raises(ValueError):
        critic.source_package(ROOT, STUDY_ID, "20261030T000000000000Z", "P04")
    with pytest.raises(ValueError, match="evaluation ID"):
        critic.source_package(ROOT, STUDY_ID, REVIEW_ID, "P99")


def test_criticism_schema_citations_and_claims(source):
    valid = criticism(source)
    assert critic.parse_critique(json.dumps(valid), source) == valid
    for mutation in ("bad_hash", "bad_citation", "missing_assessment", "fabricated_number", "shape_as_latency",
                     "nonexistent_experiment", "false_confirmation", "false_profiling"):
        bad = copy.deepcopy(valid)
        if mutation == "bad_hash": bad["proposal_sha256"] = "0" * 64
        if mutation == "bad_citation": bad["evidence_citations"][0]["observation_key"] = "imaginary"
        if mutation == "missing_assessment": bad["assessments"].pop("research_value")
        if mutation == "fabricated_number": bad["observed_evidence"] = "Measured latency was 999 ms."
        if mutation == "shape_as_latency": bad["observed_evidence"] = "Observed latency was 128 ms."
        if mutation == "nonexistent_experiment": bad["observed_evidence"] = "Experiment 099 established the result."
        if mutation == "false_confirmation": bad["observed_evidence"] = "Independent confirmation was completed."
        if mutation == "false_profiling": bad["observed_evidence"] = "Profiling was validated."
        with pytest.raises(ValueError):
            critic.parse_critique(json.dumps(bad), source)
    assert critic.parse_critique(json.dumps(criticism(source, "proceed")), source)["recommendation"] == "proceed"
    valid["unresolved_questions"] = "No confirmed profiling evidence exists for the proposed mechanism."
    assert critic.parse_critique(json.dumps(valid), source) == valid


def test_historical_rejected_response_is_valid_under_corrected_rules(rejected_audit):
    audit, source = rejected_audit
    assert audit["status"] == "rejected"
    assert audit["rejection"] == "ValueError: unsupported numerical measurement"
    parsed = critic.parse_critique(audit["raw_response"], source)
    assert parsed["proposal_id"] == "P01"
    assert parsed["recommendation"] == "deprioritize"
    assert critic.validate_case(ROOT, REJECTED_CASE_ID)["criticism"] == "rejected"
    with pytest.raises(ValueError):
        critic.load_criticism(ROOT, REJECTED_CASE_ID)


def test_historical_fixture_still_rejects_fabrications(rejected_audit):
    audit, source = rejected_audit
    original = json.loads(audit["raw_response"])
    timing_ref = reference(source, "screening:128x3072x768:row_major")
    cases = []
    bad = copy.deepcopy(original); bad["evidence_citations"].append(timing_ref)
    bad["observed_evidence"] = "Observed latency was 999 ms."; cases.append(bad)
    bad = copy.deepcopy(original); bad["evidence_citations"].append(timing_ref)
    bad["observed_evidence"] = "Observed latency was 128 ms."; cases.append(bad)
    bad = copy.deepcopy(original); bad["observed_evidence"] = "The variant achieved a speedup."; cases.append(bad)
    bad = copy.deepcopy(original); bad["observed_evidence"] = "Independent confirmation was completed."; cases.append(bad)
    bad = copy.deepcopy(original); bad["observed_evidence"] = "Profiling was validated."; cases.append(bad)
    bad = copy.deepcopy(original); bad["assessments"]["falsifiability"] = "The workload has 128/64=3 exact tiles."; cases.append(bad)
    bad = copy.deepcopy(original); bad["evidence_citations"][0]["observation_key"] = "imaginary"; cases.append(bad)
    for candidate in cases:
        with pytest.raises(ValueError):
            critic.parse_critique(json.dumps(candidate), source)


def test_second_historical_response_accepts_verified_tile_count_and_numbered_steps(second_rejected_audit):
    audit, source = second_rejected_audit
    assert audit["status"] == "rejected"
    assert audit["rejection"] == "ValueError: unsupported numerical measurement"
    parsed = critic.parse_critique(audit["raw_response"], source)
    assert parsed["recommendation"] == "deprioritize"
    assert "(2 full tiles)" in parsed["assessments"]["hardware_mechanism"]
    assert "1)" in parsed["potential_technical_flaws"]
    assert critic.validate_case(ROOT, SECOND_REJECTED_CASE_ID)["criticism"] == "rejected"
    with pytest.raises(ValueError):
        critic.load_criticism(ROOT, SECOND_REJECTED_CASE_ID)


def test_second_fixture_numerical_diagnostics_and_safeguards(second_rejected_audit):
    audit, source = second_rejected_audit
    original = json.loads(audit["raw_response"])
    cases = [
        ("assessments", "hardware_mechanism", "(2 full tiles)", "(3 full tiles)", "incorrect derived tile count", "3 full tiles"),
        ("potential_technical_flaws", None, "1) No edge-tile exposure", "9) No edge-tile exposure", "unsupported numerical measurement", "9) No edge-tile"),
        ("observed_evidence", None, original["observed_evidence"], "Observed latency was 999 ms.", "unsupported numerical measurement", "999 ms"),
        ("observed_evidence", None, original["observed_evidence"], "Observed latency was 128 ms.", "numeric performance claim lacks a cited measurement", "128 ms"),
    ]
    for field, subfield, before, after, reason, fragment in cases:
        bad = copy.deepcopy(original)
        if field == "assessments":
            bad[field][subfield] = bad[field][subfield].replace(before, after)
            expected_field = f"{field}.{subfield}"
        else:
            bad[field] = bad[field].replace(before, after)
            expected_field = field
        if "latency was" in after:
            bad["evidence_citations"].append(reference(source, "screening:128x3072x768:row_major"))
        with pytest.raises(ValueError) as error:
            critic.parse_critique(json.dumps(bad), source)
        assert reason in str(error.value)
        assert expected_field in str(error.value)
        assert fragment in str(error.value)
    for field, text in (("observed_evidence", "The variant achieved a speedup."),
                        ("observed_evidence", "Profiling was validated.")):
        bad = copy.deepcopy(original); bad[field] = text
        with pytest.raises(ValueError):
            critic.parse_critique(json.dumps(bad), source)
    bad = copy.deepcopy(original)
    bad["evidence_citations"][0]["observation_key"] = "imaginary"
    with pytest.raises(ValueError, match="citation"):
        critic.parse_critique(json.dumps(bad), source)


def test_historical_revision_is_grounded_but_uncited(rejected_revision):
    audit, source, preparation, context, originals = rejected_revision
    assert audit["status"] == "rejected"
    assert audit["rejection"].startswith("ValueError: unsupported numerical measurement in what_changed: '130'")
    with pytest.raises(ValueError) as error:
        critic.parse_revision(audit["raw_response"], source, preparation["criticism"], context, originals)
    assert "validated numerical fact lacks citation" in str(error.value)
    assert "partial_m_workload" in str(error.value)
    assert "what_changed: '130'" in str(error.value)
    assert critic.validate_case(ROOT, REJECTED_CASE_ID)["revision"] == "rejected"


def test_revision_prompt_versions_and_citation_scopes(source):
    critique = criticism(source)
    prompt = critic.revision_prompt(source, critique)
    instructions = prompt.split("\nSCHEMA\n", 1)[0]
    assert "every historical claim and numerical workload reference" in instructions
    assert "exact evidence_id and observation_key" in instructions
    assert "revised_proposal.evidence_references" in instructions
    assert "response-level evidence_citations apply only to what_changed, why, and response_to_criticism" in instructions
    assert "they do not cover revised_proposal" in instructions
    assert "Revised-proposal citations do not cover the response-level explanations" in instructions
    assert "even when the same evidence appears in the criticism or source" in instructions
    assert "between 1 and 16 distinct citations in revised_proposal.evidence_references" in instructions
    assert "response-level evidence_citations have a separate limit of 12" in instructions
    assert "Choose revise, defend, or withdraw" in instructions
    assert "130x3072x768" not in instructions
    legacy = critic.load_revision_setup(ROOT, REJECTED_CASE_ID)
    historical_source = critic.load_case(ROOT, REJECTED_CASE_ID)["source"]
    assert "prompt_version" not in legacy
    assert legacy["revision_prompt"] == critic.revision_prompt(
        historical_source, legacy["criticism"], version="legacy_v1")
    assert "revised_proposal.evidence_references" not in legacy["revision_prompt"].split("\nSCHEMA\n", 1)[0]


def test_final_historical_revision_citations_validate_without_changing_audit(final_rejected_revision):
    audit, source, preparation, context, originals = final_rejected_revision
    assert audit["status"] == "rejected"
    assert audit["rejection"] == "ValueError: invalid evidence reference"
    response = json.loads(audit["raw_response"])
    assert len(response["revised_proposal"]["evidence_references"]) == 11
    assert critic.parse_revision(audit["raw_response"], source, preparation["criticism"], context, originals) == response
    assert critic.validate_case(ROOT, FINAL_REJECTED_CASE_ID)["revision"] == "rejected"


def test_revised_citation_limit_and_reference_integrity(final_rejected_revision):
    audit, source, preparation, context, originals = final_rejected_revision
    response = json.loads(audit["raw_response"])
    refs = response["revised_proposal"]["evidence_references"]
    seen = {(ref["evidence_id"], ref["observation_key"]) for ref in refs}
    extras = []
    for experiment in source["validated_evidence"].values():
        for key in experiment["observations"]:
            pair = (experiment["evidence_id"], key)
            if pair not in seen:
                extras.append({"evidence_id": pair[0], "observation_key": pair[1]})
    within_limit = copy.deepcopy(response)
    within_limit["revised_proposal"]["evidence_references"].extend(extras[:5])
    assert len(within_limit["revised_proposal"]["evidence_references"]) == 16
    assert critic.parse_revision(json.dumps(within_limit), source, preparation["criticism"], context, originals) == within_limit

    too_many = copy.deepcopy(within_limit)
    too_many["revised_proposal"]["evidence_references"].append(extras[5])
    with pytest.raises(ValueError, match="evidence citation count must be 1 to 16"):
        critic.parse_revision(json.dumps(too_many), source, preparation["criticism"], context, originals)

    for bad_ref in (refs[0], {"evidence_id": "nonexistent", "observation_key": "workload"},
                    {"evidence_id": refs[0]["evidence_id"], "observation_key": "nonexistent"},
                    {"evidence_id": refs[0]["evidence_id"]}):
        invalid = copy.deepcopy(response)
        invalid["revised_proposal"]["evidence_references"][0] = bad_ref
        if bad_ref is refs[0]:
            invalid["revised_proposal"]["evidence_references"].append(bad_ref)
        with pytest.raises(ValueError, match="invalid or duplicate evidence citation"):
            critic.parse_revision(json.dumps(invalid), source, preparation["criticism"], context, originals)

    malformed = copy.deepcopy(response)
    malformed["revised_proposal"]["evidence_references"] = "not a list"
    with pytest.raises(ValueError, match="malformed evidence citations"):
        critic.parse_revision(json.dumps(malformed), source, preparation["criticism"], context, originals)

    unsupported = copy.deepcopy(response)
    unsupported["revised_proposal"]["evidence_references"] = [
        ref for ref in refs if ref["observation_key"] not in {"partial_m_workload", "cross_workload_caution"}
    ]
    with pytest.raises(ValueError, match="validated numerical fact lacks citation"):
        critic.parse_revision(json.dumps(unsupported), source, preparation["criticism"], context, originals)

    outer = copy.deepcopy(response)
    outer["evidence_citations"].extend(extras[:6])
    with pytest.raises(ValueError, match="evidence citation count must be 1 to 12"):
        critic.parse_revision(json.dumps(outer), source, preparation["criticism"], context, originals)


def test_revision_requires_local_citations_in_explanations_and_revised_fields(rejected_revision):
    audit, source, preparation, context, originals = rejected_revision
    original_response = json.loads(audit["raw_response"])
    experiment_004 = source["validated_evidence"]["004"]["evidence_id"]
    # A synthetic cited response demonstrates the rule without editing the historical response.
    cited = copy.deepcopy(original_response)
    for key in ("partial_m_workload", "configuration"):
        ref = {"evidence_id": experiment_004, "observation_key": key}
        cited["evidence_citations"].append(ref)
        cited["revised_proposal"]["evidence_references"].append(ref)
    assert critic.parse_revision(json.dumps(cited), source, preparation["criticism"], context, originals) == cited
    outer_missing = copy.deepcopy(cited)
    outer_missing["evidence_citations"] = original_response["evidence_citations"]
    with pytest.raises(ValueError, match="what_changed.*130"):
        critic.parse_revision(json.dumps(outer_missing), source, preparation["criticism"], context, originals)
    inner_missing = copy.deepcopy(cited)
    inner_missing["revised_proposal"]["evidence_references"] = original_response["revised_proposal"]["evidence_references"]
    with pytest.raises(ValueError, match="revised_proposal.testable_prediction.*130"):
        critic.parse_revision(json.dumps(inner_missing), source, preparation["criticism"], context, originals)


def test_revision_rejects_invented_workload_timing_and_unsupported_claims(rejected_revision):
    audit, source, preparation, context, originals = rejected_revision
    valid = json.loads(audit["raw_response"])
    experiment_004 = source["validated_evidence"]["004"]["evidence_id"]
    for key in ("partial_m_workload", "configuration"):
        ref = {"evidence_id": experiment_004, "observation_key": key}
        valid["evidence_citations"].append(ref)
        valid["revised_proposal"]["evidence_references"].append(ref)
    cases = []
    bad = copy.deepcopy(valid); bad["what_changed"] = bad["what_changed"].replace("130x3072x768", "999x3072x768"); cases.append(bad)
    bad = copy.deepcopy(valid); bad["revised_proposal"]["testable_prediction"] = bad["revised_proposal"]["testable_prediction"].replace("130,3072,768", "999,3072,768"); cases.append(bad)
    bad = copy.deepcopy(valid); bad["why"] = "Observed latency was 999 ms."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["why"] = "Observed latency was 128 ms."
    bad["evidence_citations"].append(reference(source, "screening:128x3072x768:row_major")); cases.append(bad)
    bad = copy.deepcopy(valid); bad["response_to_criticism"] = "The exact calculation is 128/64=3 tiles."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["why"] = "The variant achieved a speedup."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["why"] = "Profiling was validated."; cases.append(bad)
    bad = copy.deepcopy(valid); bad["evidence_citations"][0]["observation_key"] = "imaginary"; cases.append(bad)
    for candidate in cases:
        with pytest.raises(ValueError):
            critic.parse_revision(json.dumps(candidate), source, preparation["criticism"], context, originals)


def test_explicit_critique_revision_and_manifest_integrity(tmp_path, monkeypatch, source):
    case_id = setup_case(tmp_path, monkeypatch, source)
    case = critic.load_case(tmp_path, case_id)
    assert case["source"]["proposal"] == source["proposal"]
    assert critic.validate_case(tmp_path, case_id)["criticism"] == "not_requested"
    critique = criticism(source)
    client = Client(json.dumps(critique))
    assert critic.request_critique(client, tmp_path, case_id) == "accepted"
    assert critic.load_criticism(tmp_path, case_id)["parsed_response"] == critique
    assert critic.request_critique(client, tmp_path, case_id) == "already_attempted"
    assert len(client.calls) == 1
    prep = critic.prepare_revision(tmp_path, case_id)
    assert prep["criticism"] == critique
    assert prep["prompt_version"] == critic.REVISION_PROMPT_VERSION
    assert "revised_proposal.evidence_references" in prep["revision_prompt"]
    study = discovery.load_study(ROOT, STUDY_ID)
    request = discovery.load_request(ROOT, STUDY_ID, source["condition"], study=study)
    monkeypatch.setattr(critic, "load_study", lambda root, study_id: study)
    monkeypatch.setattr(critic, "load_request", lambda root, study_id, condition, *, study=None: request)
    revised = revision(source, critique)
    revised_client = Client(json.dumps(revised))
    assert critic.request_revision(revised_client, tmp_path, case_id) == "accepted"
    assert critic.validate_case(tmp_path, case_id)["revision"] == "accepted"
    assert critic.load_case(tmp_path, case_id)["source"]["proposal"] == source["proposal"]
    audit = tmp_path / critic.SECTION / case_id / "revision" / "audit.json"
    audit.write_bytes(audit.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes"):
        critic.validate_case(tmp_path, case_id)


def test_revision_provenance_and_actions(source):
    critique = criticism(source)
    study = discovery.load_study(ROOT, STUDY_ID)
    request = discovery.load_request(ROOT, STUDY_ID, source["condition"], study=study)
    context = critic.revision_context(study, source)
    originals = request["parsed_proposals"]["ideas"]
    for action in ("revise", "defend", "withdraw"):
        answer = revision(source, critique, action=action)
        assert critic.parse_revision(json.dumps(answer), source, critique, context, originals) == answer
    updated = revision(source, critique)
    updated["revised_proposal"]["evidence_references"] = [reference(source)]
    assert critic.parse_revision(json.dumps(updated), source, critique, context, originals) == updated
    bad = revision(source, critique)
    bad["criticism_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        critic.parse_revision(json.dumps(bad), source, critique, context, originals)
    bad = revision(source, critique)
    bad["revised_proposal"] = source["proposal"]
    with pytest.raises(ValueError, match="change"):
        critic.parse_revision(json.dumps(bad), source, critique, context, originals)


def test_rejected_incomplete_response_is_retained(tmp_path, monkeypatch, source):
    case_id = setup_case(tmp_path, monkeypatch, source)
    client = Client(json.dumps(criticism(source)), status="incomplete")
    assert critic.request_critique(client, tmp_path, case_id) == "rejected"
    audit = json.loads((tmp_path / critic.SECTION / case_id / "criticism" / "audit.json").read_text())
    assert audit["raw_response"] == client.raw
    assert audit["parsed_response"] is None
    assert audit["rejection_kind"] == "incomplete_output"
    assert critic.validate_case(tmp_path, case_id)["criticism"] == "rejected"
    assert critic.request_critique(client, tmp_path, case_id) == "already_attempted"
    with pytest.raises(ValueError):
        critic.prepare_revision(tmp_path, case_id)


def test_setup_tampering_is_detected(tmp_path, monkeypatch, source):
    case_id = setup_case(tmp_path, monkeypatch, source)
    path = tmp_path / critic.SECTION / case_id / "setup" / "case.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes"):
        critic.load_case(tmp_path, case_id)
