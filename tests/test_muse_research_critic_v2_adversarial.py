"""Adversarial characterization only; production v2 behavior is deliberately unchanged."""
import json

import pytest

import muse_research_critic as v1
import muse_research_critic_v2 as v2


E003, E005 = "e:003", "e:005"
KEYS = [f"screening:{shape}:{opponent}" for shape in ("512x3072x768", "2048x3072x768")
        for opponent in ("row_major", "grouped_m4")]
RATIOS = [0.9971394589037794, 0.9992195058576494, 0.9991797675787395, 1.0019068437051466]


@pytest.fixture
def source():
    return {"proposal_id": "P06", "proposal_sha256": "a" * 64,
            "proposal": {"name": "Untested tile grouping"},
            "validated_evidence": {
                "003": {"evidence_id": E003, "observations": {
                    "workload": [128, 3072, 768], "compiler_equivalent_for_compared_sources": True}},
                "004": {"evidence_id": "e:004", "observations": {"control_workload": [128, 3072, 768]}},
                "005": {"evidence_id": E005, "observations": {
                    "confirmation_status": "missing", "screening_scope": {"stage_decision": "screening_unconfirmed"},
                    **{key: {"median_paired_ratio": ratio, "baseline_median_ms": 2.0,
                             "bootstrap_median_ratio_95pct": [0.95, 1.05]}
                       for key, ratio in zip(KEYS, RATIOS)}}}}}


def ref(evidence_id, key, value, path=None, unit="none"):
    return {"evidence_id": evidence_id, "observation_key": key,
            "value_path": [] if path is None else path, "recorded_value": value, "unit": unit}


def base(source):
    sentence = "confirmation_status is missing."
    return {"protocol_version": v2.VERSION, "proposal_id": source["proposal_id"],
            "proposal_sha256": source["proposal_sha256"],
            "closest_related_prior_research": {"experiment_id": "none",
                "what_was_tested": "No direct prior test of this proposal.",
                "relation": "Prior evidence is contextual."},
            "evidence_citations": [{"evidence_id": E005, "observation_key": "confirmation_status"}],
            "assessments": {name: "No direct test of this proposal." for name in v1.ASSESSMENTS},
            **{name: "No direct test of this proposal." for name in v1.TEXT_FIELDS},
            "observed_evidence": sentence, "recommendation": "revise",
            "claim_ledger": [{"claim_id": "C1", "claim_type": "observation",
                "field": "observed_evidence", "statement": sentence,
                "references": [ref(E005, "confirmation_status", "missing")],
                "aggregate_scope": [], "arithmetic": None, "measurement_stage": "historical",
                "limitations": "No conclusion about the new proposal follows."}]}


def parse(value, source):
    return v2.parse_critique_v2(json.dumps(value), source)


def screening_claim(value, *, key=KEYS[0], ratio=RATIOS[0]):
    sentence = f"{key} median_paired_ratio is {ratio}."
    value["observed_evidence"] = sentence
    claim = value["claim_ledger"][0]
    claim.update(statement=sentence, references=[ref(E005, key, ratio, ["median_paired_ratio"], "ratio")],
                 measurement_stage="screening")
    value["evidence_citations"] = [{"evidence_id": E005, "observation_key": key}]
    return value


def range_claim(value, keys=KEYS, ratios=RATIOS):
    sentence = "All four screening comparisons range from 0.997 to 1.002 after three-decimal rounding."
    value["assessments"]["evidence_consistency"] = sentence
    value["claim_ledger"].append({"claim_id": "C2", "claim_type": "derivation",
        "field": "assessments.evidence_consistency", "statement": sentence,
        "references": [ref(E005, key, ratio, ["median_paired_ratio"], "ratio")
                       for key, ratio in zip(keys, ratios)],
        "aggregate_scope": [{"evidence_id": E005, "observation_key": key} for key in keys],
        "arithmetic": {"operation": "rounded_range", "operand_indices": list(range(len(keys))),
                       "precision_decimal_places": 3, "result": ["0.997", "1.002"]},
        "measurement_stage": "screening",
        "limitations": "These are prior Hilbert screening ratios, not P06 measurements."})
    value["evidence_citations"].extend({"evidence_id": E005, "observation_key": key} for key in keys)
    return value


def test_negative_controls_exact_screening_and_complete_range(source):
    assert parse(screening_claim(base(source)), source)["claim_ledger"][0]["measurement_stage"] == "screening"
    assert parse(range_claim(base(source)), source)["claim_ledger"][1]["arithmetic"]["result"] == ["0.997", "1.002"]


def test_omitted_claim_in_unledgered_prose_passes(source):
    value = base(source)
    value["assessments"]["evidence_consistency"] = "P06 achieved a confirmed 30% speedup."
    assert parse(value, source) == value


def test_citation_laundering_in_observation_statement_passes(source):
    value = screening_claim(base(source))
    sentence = value["observed_evidence"] + " Therefore P06 is faster."
    value["observed_evidence"] = value["claim_ledger"][0]["statement"] = sentence
    assert parse(value, source) == value


def test_numerical_misattribution_in_mixed_statement_passes(source):
    value = screening_claim(base(source))
    sentence = value["observed_evidence"] + f" {KEYS[1]} also has median_paired_ratio {RATIOS[0]}."
    value["observed_evidence"] = value["claim_ledger"][0]["statement"] = sentence
    assert RATIOS[1] != RATIOS[0]
    assert parse(value, source) == value


def test_wrong_recorded_value_is_rejected(source):
    value = screening_claim(base(source))
    value["claim_ledger"][0]["references"][0]["recorded_value"] = RATIOS[1]
    with pytest.raises(ValueError, match="recorded value differs"):
        parse(value, source)


def test_arithmetic_on_incompatible_units_passes(source):
    value = base(source)
    sentence = "The sum 0.9971394589037794 + 2.0 is 2.997 at three decimals."
    value["assessments"]["evidence_consistency"] = sentence
    value["claim_ledger"].append({"claim_id": "C2", "claim_type": "derivation",
        "field": "assessments.evidence_consistency", "statement": sentence,
        "references": [ref(E005, KEYS[0], RATIOS[0], ["median_paired_ratio"], "ratio"),
                       ref(E005, KEYS[0], 2.0, ["baseline_median_ms"], "ms")],
        "aggregate_scope": [],
        "arithmetic": {"operation": "add", "operand_indices": [0, 1],
                       "precision_decimal_places": 3, "result": "2.997"},
        "measurement_stage": "screening", "limitations": "Mixed units have no interpretable sum."})
    value["evidence_citations"].append({"evidence_id": E005, "observation_key": KEYS[0]})
    assert parse(value, source) == value


def test_aggregate_can_omit_relevant_middle_observation(source):
    value = range_claim(base(source), [KEYS[0], KEYS[2], KEYS[3]],
                        [RATIOS[0], RATIOS[2], RATIOS[3]])
    assert "All four" in value["assessments"]["evidence_consistency"]
    assert parse(value, source) == value


def test_declared_aggregate_omission_is_rejected(source):
    value = range_claim(base(source))
    value["claim_ledger"][1]["references"].pop(1)
    with pytest.raises(ValueError, match="omits a declared source"):
        parse(value, source)


def test_screening_can_be_described_as_confirmed_in_prose(source):
    value = screening_claim(base(source))
    sentence = value["observed_evidence"] + " This independently confirmed P06's performance gain."
    value["observed_evidence"] = value["claim_ledger"][0]["statement"] = sentence
    assert value["claim_ledger"][0]["measurement_stage"] == "screening"
    assert parse(value, source) == value


def test_confirmation_metadata_mismatch_is_rejected(source):
    value = screening_claim(base(source))
    value["claim_ledger"][0]["measurement_stage"] = "confirmation"
    with pytest.raises(ValueError, match="cannot support confirmed performance"):
        parse(value, source)


def test_inference_can_claim_established_mechanism(source):
    value = base(source)
    sentence = "P06's cache locality mechanism was experimentally established."
    value["inferred_explanations"] = sentence
    value["claim_ledger"].append({"claim_id": "C2", "claim_type": "inference",
        "field": "inferred_explanations", "statement": sentence,
        "references": [ref(E005, "confirmation_status", "missing")],
        "aggregate_scope": [], "arithmetic": None, "measurement_stage": "not_applicable",
        "limitations": "No locality profiling was recorded."})
    assert parse(value, source) == value


def test_valid_plain_language_observation_is_rejected(source):
    value = screening_claim(base(source))
    sentence = f"The recorded screening median paired ratio was {RATIOS[0]}."
    value["observed_evidence"] = value["claim_ledger"][0]["statement"] = sentence
    with pytest.raises(ValueError, match="exact key and value"):
        parse(value, source)
