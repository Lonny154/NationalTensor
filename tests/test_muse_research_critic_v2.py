"""Synthetic CPU-only tests for the separately versioned critic ledger."""
import copy
import json

import pytest

import muse_research_critic as v1
import muse_research_critic_v2 as v2


E003 = "e:003"
E004 = "e:004"
E005 = "e:005"
KEYS = [f"screening:{shape}:{opponent}" for shape in ("512x3072x768", "2048x3072x768")
        for opponent in ("row_major", "grouped_m4")]
RATIOS = [0.9971394589037794, 0.9992195058576494, 0.9991797675787395, 1.0019068437051466]


@pytest.fixture
def source():
    return {"proposal_id": "P06", "proposal_sha256": "a" * 64,
            "proposal": {"name": "Untested grouping proposal"},
            "validated_evidence": {
                "003": {"evidence_id": E003, "observations": {
                    "workload": [128, 3072, 768], "compiler_equivalent_for_compared_sources": True}},
                "004": {"evidence_id": E004, "observations": {"control_workload": [128, 3072, 768]}},
                "005": {"evidence_id": E005, "observations": {
                    "confirmation_status": "missing", "screening_scope": {"stage_decision": "screening_unconfirmed"},
                    **{key: {"median_paired_ratio": ratio, "bootstrap_median_ratio_95pct": [0.95, 1.05]}
                       for key, ratio in zip(KEYS, RATIOS)}}}}}


def ref(evidence_id, key, value, path=None, unit="none"):
    return {"evidence_id": evidence_id, "observation_key": key, "value_path": path or [],
            "recorded_value": value, "unit": unit}


def response(source):
    result = {"protocol_version": v2.VERSION, "proposal_id": source["proposal_id"],
              "proposal_sha256": source["proposal_sha256"],
              "closest_related_prior_research": {"experiment_id": "none",
                  "what_was_tested": "No prior test of this proposal.",
                  "relation": "The prior screening is context only."},
              "evidence_citations": [{"evidence_id": E005, "observation_key": "confirmation_status"}],
              "assessments": {name: "Evidence is insufficient for this proposal." for name in v1.ASSESSMENTS},
              **{name: "Evidence is insufficient for this proposal." for name in v1.TEXT_FIELDS},
              "recommendation": "revise"}
    statement = "confirmation_status is missing; no independent confirmation was recorded."
    result["observed_evidence"] = statement
    result["claim_ledger"] = [{"claim_id": "C1", "claim_type": "observation",
        "field": "observed_evidence", "statement": statement,
        "references": [ref(E005, "confirmation_status", "missing")],
        "aggregate_scope": [], "arithmetic": None, "measurement_stage": "historical",
        "limitations": "This says nothing about the new proposal's performance."}]
    return result


def parse(result, source):
    return v2.parse_critique_v2(json.dumps(result), source)


def test_v2_prompt_is_explicit_and_v1_prompt_unchanged(source):
    before = v1.critique_prompt(source)
    prompt = v2.critique_prompt_v2(source)
    assert "claim_ledger" in prompt and "aggregate_scope" in prompt
    assert "Distinguish screening from independent confirmation" in prompt
    assert "Never attribute prior kernel measurements" in prompt
    assert v1.critique_prompt(source) == before
    assert parse(response(source), source)["protocol_version"] == v2.VERSION


def test_missing_screening_key_rejected(source):
    result = response(source)
    claim = result["claim_ledger"][0]
    claim["statement"] = result["observed_evidence"] = "screening:512x3072x768:row_major median_paired_ratio is 0.9971394589037794."
    claim["references"] = [ref(E005, KEYS[0], RATIOS[0], ["median_paired_ratio"], "ratio")]
    claim["measurement_stage"] = "screening"
    with pytest.raises(ValueError, match="missing from response citations"):
        parse(result, source)


def test_unrelated_historical_key_cannot_support_direct_claim(source):
    result = response(source)
    result["evidence_citations"].append({"evidence_id": E003, "observation_key": "workload"})
    claim = result["claim_ledger"][0]
    claim["statement"] = result["observed_evidence"] = "003 compiler_equivalent_for_compared_sources is true."
    claim["references"] = [ref(E003, "workload", [128, 3072, 768], unit="elements")]
    with pytest.raises(ValueError, match="exact key and value"):
        parse(result, source)


def test_direct_value_and_unit_must_match(source):
    result = response(source)
    for changed, message in [("recorded_value", "differs"), ("unit", "unit differs")]:
        bad = copy.deepcopy(result)
        bad["claim_ledger"][0]["references"][0][changed] = "confirmed" if changed == "recorded_value" else "ms"
        with pytest.raises(ValueError, match=message):
            parse(bad, source)


def rounded_range(source):
    result = response(source)
    statement = "E005 screening ratios range from 0.997 to 1.002 after rounding to three decimals."
    result["assessments"]["evidence_consistency"] = statement
    claim = {"claim_id": "C2", "claim_type": "derivation", "field": "assessments.evidence_consistency",
             "statement": statement,
             "references": [ref(E005, key, ratio, ["median_paired_ratio"], "ratio")
                            for key, ratio in zip(KEYS, RATIOS)],
             "aggregate_scope": [{"evidence_id": E005, "observation_key": key} for key in KEYS],
             "arithmetic": {"operation": "rounded_range", "operand_indices": [0, 1, 2, 3],
                            "precision_decimal_places": 3, "result": ["0.997", "1.002"]},
             "measurement_stage": "screening",
             "limitations": "These are Hilbert screening comparisons, not this proposal's measured results."}
    result["claim_ledger"].append(claim)
    result["evidence_citations"].extend({"evidence_id": E005, "observation_key": key} for key in KEYS)
    return result


def test_p06_four_ratio_rounded_range_metadata(source):
    result = rounded_range(source)
    assert parse(result, source)["claim_ledger"][1]["arithmetic"]["result"] == ["0.997", "1.002"]
    for mutation, message in [("result", "incorrect derived"), ("precision", "incorrect derived"),
                              ("operand", "every cited operand")]:
        bad = copy.deepcopy(result)
        expression = bad["claim_ledger"][1]["arithmetic"]
        if mutation == "result": expression["result"] = ["0.996", "1.002"]
        if mutation == "precision": expression["precision_decimal_places"] = 2
        if mutation == "operand": expression["operand_indices"] = [0, 1, 2]
        with pytest.raises(ValueError, match=message):
            parse(bad, source)


def test_aggregate_declared_scope_and_citations(source):
    result = rounded_range(source)
    bad = copy.deepcopy(result)
    bad["claim_ledger"][1]["references"].pop()
    with pytest.raises(ValueError, match="omits a declared source"):
        parse(bad, source)
    bad = copy.deepcopy(result)
    bad["evidence_citations"].pop()
    with pytest.raises(ValueError, match="missing from response citations"):
        parse(bad, source)


def test_screening_cannot_be_declared_confirmation(source):
    result = rounded_range(source)
    result["claim_ledger"][1]["measurement_stage"] = "confirmation"
    with pytest.raises(ValueError, match="cannot support confirmed performance"):
        parse(result, source)


def test_inference_is_not_measured_observation(source):
    result = response(source)
    statement = "This new grouping may improve locality, but that is unmeasured."
    result["inferred_explanations"] = statement
    claim = result["claim_ledger"][0]
    claim.update(claim_id="C2", claim_type="inference", field="inferred_explanations",
                 statement=statement, measurement_stage="screening")
    with pytest.raises(ValueError, match="inference cannot be labeled measured evidence"):
        parse(result, source)
    claim["measurement_stage"] = "not_applicable"
    assert parse(result, source)["claim_ledger"][0]["claim_type"] == "inference"
