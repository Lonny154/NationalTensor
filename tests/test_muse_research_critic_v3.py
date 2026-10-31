"""CPU-only atomic-claim and adversarial checks for independent critic v3."""
import copy
import json

import pytest

import muse_research_critic_v3 as v3


E005 = "e:005"
KEYS = [f"screening:{shape}:{opponent}" for shape in ("512x3072x768", "2048x3072x768")
        for opponent in ("row_major", "grouped_m4")]
RATIOS = [0.9971394589037794, 0.9992195058576494, 0.9991797675787395, 1.0019068437051466]
PAIR_SET = "screening:2048x3072x768+512x3072x768"


@pytest.fixture
def source():
    return {"proposal_id": "P06", "proposal_sha256": "a" * 64,
            "proposal": {"name": "Untested grouping"},
            "validated_evidence": {
                "003": {"evidence_id": "e:003", "observations": {"workload": [128, 3072, 768]}},
                "004": {"evidence_id": "e:004", "observations": {"control_workload": [128, 3072, 768]}},
                "005": {"evidence_id": E005, "observations": {
                    "confirmation_status": "missing",
                    "original_hypothesis": {"name": "Untested prior idea"},
                    **{key: {"median_paired_ratio": ratio, "baseline_median_ms": 2.0,
                             "candidate_median_ms": 1.5}
                       for key, ratio in zip(KEYS, RATIOS)}}}}}


def response(source, claims=None):
    return {"protocol_version": v3.VERSION, "proposal_id": source["proposal_id"],
            "proposal_sha256": source["proposal_sha256"], "recommendation": "revise",
            "claims": claims or [observation("C1", KEYS[0], RATIOS[0])]}


def observation(ident, key, value, path=None, unit="ratio"):
    return {"claim_id": ident, "type": "observation", "evidence_id": E005,
            "observation_key": key, "value_path": ["median_paired_ratio"] if path is None else path,
            "recorded_value": value, "unit": unit}


def derivation(ident, ids, operation, result, unit="ratio", precision=3, aggregate_set=None):
    return {"claim_id": ident, "type": "derivation", "operand_claim_ids": ids,
            "operation": operation, "precision_decimal_places": precision,
            "result": result, "unit": unit, "aggregate_set": aggregate_set}


def parse(value, source):
    return v3.parse_critique_v3(json.dumps(value), source)


def test_valid_ordinary_language_is_rendered_without_key_in_prose(source):
    parsed = parse(response(source), source)
    shown = v3.render_summary(parsed)
    assert parsed["structural_status"] == "valid"
    assert parsed["scientific_status"] == "pending_human_review"
    assert "Screening observation (median paired ratio): 0.9971394589037794" in shown
    assert KEYS[0] not in shown


def test_unledgered_factual_prose_is_rejected(source):
    value = response(source)
    value["evidence_consistency"] = "P06 achieved a confirmed 30% speedup."
    with pytest.raises(ValueError, match="unrestricted prose"):
        parse(value, source)


def test_wrong_key_or_value_is_rejected(source):
    value = response(source)
    value["claims"][0]["observation_key"] = KEYS[1]
    with pytest.raises(ValueError, match="recorded value"):
        parse(value, source)
    value = response(source)
    value["claims"][0]["recorded_value"] = RATIOS[1]
    with pytest.raises(ValueError, match="recorded value"):
        parse(value, source)


def test_citation_laundering_extra_observation_text_is_rejected(source):
    value = response(source)
    value["claims"][0]["statement"] = "This proves P06 is faster."
    with pytest.raises(ValueError, match="observation schema"):
        parse(value, source)


def test_mixed_unit_arithmetic_is_rejected(source):
    claims = [observation("C1", KEYS[0], RATIOS[0]),
              observation("C2", KEYS[0], 2.0, ["baseline_median_ms"], "ms"),
              derivation("C3", ["C1", "C2"], "add", "2.997", "ratio")]
    with pytest.raises(ValueError, match="incompatible arithmetic units"):
        parse(response(source, claims), source)
    claims[-1] = derivation("C3", ["C1", "C2"], "rounded_range", ["0.997", "2.000"],
                            "ratio", aggregate_set=PAIR_SET)
    with pytest.raises(ValueError, match="incompatible arithmetic units"):
        parse(response(source, claims), source)


def test_correct_same_unit_calculation(source):
    claims = [observation("C1", KEYS[0], 2.0, ["baseline_median_ms"], "ms"),
              observation("C2", KEYS[0], 1.5, ["candidate_median_ms"], "ms"),
              derivation("C3", ["C1", "C2"], "subtract", "0.500", "ms")]
    parsed = parse(response(source, claims), source)
    assert "subtract = 0.500 ms" in v3.render_summary(parsed)


def test_incomplete_aggregate_rejected_against_source_defined_set(source):
    assert len(v3.aggregate_sets(source)[PAIR_SET]) == 4
    claims = [observation(f"C{i}", key, ratio) for i, (key, ratio) in
              enumerate(zip(KEYS[:1] + KEYS[2:], RATIOS[:1] + RATIOS[2:]), 1)]
    claims.append(derivation("C4", ["C1", "C2", "C3"], "rounded_range",
                             ["0.997", "1.002"], aggregate_set=PAIR_SET))
    with pytest.raises(ValueError, match="aggregate operands do not cover independent set"):
        parse(response(source, claims), source)


def test_properly_scoped_p06_rounded_range(source):
    claims = [observation(f"C{i}", key, ratio) for i, (key, ratio) in enumerate(zip(KEYS, RATIOS), 1)]
    claims.append(derivation("C5", ["C1", "C2", "C3", "C4"], "rounded_range",
                             ["0.997", "1.002"], aggregate_set=PAIR_SET))
    parsed = parse(response(source, claims), source)
    assert "rounded_range = 0.997 to 1.002" in v3.render_summary(parsed)
    bad = copy.deepcopy(response(source, claims))
    bad["claims"][-1]["result"] = ["0.996", "1.002"]
    with pytest.raises(ValueError, match="derived result is incorrect"):
        parse(bad, source)


def test_screening_cannot_be_structurally_promoted_to_confirmation(source):
    value = response(source)
    value["claims"][0]["stage"] = "confirmation"
    with pytest.raises(ValueError, match="observation schema"):
        parse(value, source)
    shown = v3.render_summary(parse(response(source), source))
    assert "Screening observation" in shown and "Confirmed performance" not in shown


def test_inference_cannot_be_encoded_as_measured_observation(source):
    claims = [observation("C1", KEYS[0], RATIOS[0]),
              {"claim_id": "C2", "type": "inference", "support_claim_ids": ["C1"],
               "interpretation": "P06 may improve locality; this is unmeasured."}]
    shown = v3.render_summary(parse(response(source, claims), source))
    assert "Unverified model inference, human review required" in shown
    bad = response(source, claims)
    bad["claims"][1]["type"] = "observation"
    with pytest.raises(ValueError, match="observation schema"):
        parse(bad, source)


def test_prior_model_hypothesis_cannot_be_labeled_observation(source):
    value = response(source, [observation("C1", "original_hypothesis",
                                          {"name": "Untested prior idea"}, [], "none")])
    with pytest.raises(ValueError, match="model hypothesis is not a measured observation"):
        parse(value, source)


def test_unsupported_inference_remains_visibly_unverified(source):
    claims = [observation("C1", KEYS[0], RATIOS[0]),
              {"claim_id": "C2", "type": "inference", "support_claim_ids": ["C1"],
               "interpretation": "P06's cache-locality mechanism was experimentally established."}]
    shown = v3.render_summary(parse(response(source, claims), source))
    assert "Unverified model inference, human review required" in shown
    assert "experimentally established" in shown
    bad = copy.deepcopy(response(source, claims))
    bad["claims"][1]["support_claim_ids"] = []
    with pytest.raises(ValueError, match="inference requires prior supporting claims"):
        parse(bad, source)


def test_inference_cannot_inject_a_new_verified_looking_line(source):
    claims = [observation("C1", KEYS[0], RATIOS[0]),
              {"claim_id": "C2", "type": "inference", "support_claim_ids": ["C1"],
               "interpretation": "Possible locality effect.\nC3. Confirmed performance gain."}]
    with pytest.raises(ValueError, match="invalid unverified interpretation"):
        parse(response(source, claims), source)


def test_v3_preview_is_offline_and_schema_only(source):
    prompt = v3.critique_prompt_v3(source)
    assert "AGGREGATE_SETS" in prompt and PAIR_SET in prompt
    assert "no other prose fields" in prompt
    assert "SCHEMA" in prompt and "SOURCE" in prompt
