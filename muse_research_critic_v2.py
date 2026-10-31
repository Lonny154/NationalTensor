"""Offline, separately versioned Experiment 008 critic protocol preview and validator.

No live request or Experiment 009 binding path is provided here.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
import re

from dashboard_data import ROOT
from muse_research_critic import (
    ASSESSMENTS, CRITIQUE_FIELDS, RECOMMENDATIONS, TEXT_FIELDS, source_package,
)

VERSION = "experiment_008_critic_v2"
CLAIM_TYPES = {"observation", "derivation", "inference"}
STAGES = {"historical", "screening", "confirmation", "not_applicable"}
FIELDS = ("closest_related_prior_research.what_was_tested",
          "closest_related_prior_research.relation",
          *(f"assessments.{name}" for name in ASSESSMENTS), *TEXT_FIELDS)
REF_FIELDS = {"evidence_id", "observation_key", "value_path", "recorded_value", "unit"}
CLAIM_FIELDS = {"claim_id", "claim_type", "field", "statement", "references",
                "aggregate_scope", "arithmetic", "measurement_stage", "limitations"}


def unit_for(path):
    leaf = path[-1] if path else ""
    if leaf.endswith("_ms"):
        return "ms"
    if "ratio" in leaf:
        return "ratio"
    if leaf in {"shared_memory_per_multiprocessor", "total_memory"}:
        return "bytes"
    if leaf in {"workload", "control_workload", "partial_m_workload", "shapes", "shape"}:
        return "elements"
    return "none"


def _value(source, ref):
    if not isinstance(ref, dict) or set(ref) != REF_FIELDS:
        raise ValueError("invalid claim reference schema")
    available = {v["evidence_id"]: v["observations"] for v in source["validated_evidence"].values()}
    evidence_id, key = ref["evidence_id"], ref["observation_key"]
    if not isinstance(evidence_id, str) or not isinstance(key, str):
        raise ValueError("invalid claim evidence ID or observation key")
    if evidence_id not in available or key not in available[evidence_id]:
        raise ValueError("invalid claim evidence ID or observation key")
    path = ref["value_path"]
    if not isinstance(path, list) or len(path) > 8 or any(not isinstance(p, (str, int)) or isinstance(p, bool) for p in path):
        raise ValueError("invalid claim value path")
    value = available[evidence_id][key]
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value:
            value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError("invalid claim value path")
    if type(value) is not type(ref["recorded_value"]) or value != ref["recorded_value"]:
        raise ValueError("claim recorded value differs from evidence")
    expected_unit = unit_for([key, *path])
    if ref["unit"] != expected_unit:
        raise ValueError("claim unit differs from evidence")
    return value


def _arithmetic(claim, values):
    expression = claim["arithmetic"]
    if claim["claim_type"] != "derivation":
        if expression is not None:
            raise ValueError("only derivations declare arithmetic")
        return
    if not isinstance(expression, dict) or set(expression) != {"operation", "operand_indices", "precision_decimal_places", "result"}:
        raise ValueError("derivation requires structured arithmetic")
    operation, indices = expression["operation"], expression["operand_indices"]
    precision = expression["precision_decimal_places"]
    if operation not in {"add", "subtract", "multiply", "divide", "rounded_range"}:
        raise ValueError("unsupported arithmetic operation")
    if not isinstance(indices, list) or len(indices) < 2 or len(indices) > 12 or any(type(i) is not int or i < 0 or i >= len(values) for i in indices):
        raise ValueError("invalid arithmetic operands")
    if sorted(indices) != list(range(len(values))):
        raise ValueError("derivation must use every cited operand exactly once")
    if type(precision) is not int or not 0 <= precision <= 12:
        raise ValueError("invalid rounding precision")
    if any(type(v) not in (int, float) for v in values):
        raise ValueError("arithmetic operands must be recorded numbers")
    nums = [Decimal(str(values[i])) for i in indices]
    if operation == "rounded_range":
        exact = [min(nums), max(nums)]
    elif operation == "add":
        exact = [sum(nums)]
    elif operation == "subtract" and len(nums) == 2:
        exact = [nums[0] - nums[1]]
    elif operation == "multiply":
        product = Decimal(1)
        for number in nums:
            product *= number
        exact = [product]
    elif operation == "divide" and len(nums) == 2 and nums[1] != 0:
        exact = [nums[0] / nums[1]]
    else:
        raise ValueError("invalid arithmetic operands for operation")
    quantum = Decimal(1).scaleb(-precision)
    expected = [format(number.quantize(quantum, rounding=ROUND_HALF_UP), f".{precision}f") for number in exact]
    result = expression["result"]
    if result != (expected if operation == "rounded_range" else expected[0]):
        raise ValueError("incorrect derived or rounded result")


def _fields(response):
    prior = response["closest_related_prior_research"]
    return {"closest_related_prior_research.what_was_tested": prior["what_was_tested"],
            "closest_related_prior_research.relation": prior["relation"],
            **{f"assessments.{name}": response["assessments"][name] for name in ASSESSMENTS},
            **{name: response[name] for name in TEXT_FIELDS}}


def validate_ledger(response, source):
    ledger = response["claim_ledger"]
    if not isinstance(ledger, list) or not 1 <= len(ledger) <= 16:
        raise ValueError("claim ledger must contain 1 to 16 entries")
    fields = _fields(response)
    seen = set()
    used_refs = set()
    for claim in ledger:
        if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
            raise ValueError("invalid claim ledger entry")
        claim_id = claim["claim_id"]
        if not isinstance(claim_id, str) or not re.fullmatch(r"C[1-9]\d{0,2}", claim_id) or claim_id in seen:
            raise ValueError("invalid or duplicate claim ID")
        seen.add(claim_id)
        if (not isinstance(claim["claim_type"], str) or claim["claim_type"] not in CLAIM_TYPES
                or not isinstance(claim["measurement_stage"], str) or claim["measurement_stage"] not in STAGES):
            raise ValueError("invalid claim type or measurement stage")
        field, statement = claim["field"], claim["statement"]
        if field not in fields or not isinstance(statement, str) or len(statement) < 4 or statement not in fields[field]:
            raise ValueError("claim statement is not present in named field")
        refs = claim["references"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= 12:
            raise ValueError("claim requires 1 to 12 references")
        if any(not isinstance(r, dict) or not isinstance(r.get("evidence_id"), str)
               or not isinstance(r.get("observation_key"), str) for r in refs):
            raise ValueError("duplicate or malformed claim references")
        identities = [(r["evidence_id"], r["observation_key"], json.dumps(r.get("value_path"))) for r in refs]
        if len(set(identities)) != len(refs):
            raise ValueError("duplicate or malformed claim references")
        values = [_value(source, ref) for ref in refs]
        scope = claim["aggregate_scope"]
        if not isinstance(scope, list) or any(not isinstance(item, dict) or set(item) != {"evidence_id", "observation_key"}
                                               or not isinstance(item["evidence_id"], str)
                                               or not isinstance(item["observation_key"], str)
                                               for item in scope):
            raise ValueError("invalid aggregate scope")
        scoped_pairs = [(item["evidence_id"], item["observation_key"]) for item in scope]
        ref_pairs = [(r["evidence_id"], r["observation_key"]) for r in refs]
        if scope and (len(set(scoped_pairs)) != len(scoped_pairs) or set(scoped_pairs) != set(ref_pairs)):
            raise ValueError("aggregate claim omits a declared source observation")
        if claim["claim_type"] == "observation":
            for ref, value in zip(refs, values):
                literal = json.dumps(value, ensure_ascii=False).strip('"')
                if ref["observation_key"] not in statement or literal not in statement:
                    raise ValueError("direct observation must state its exact key and value")
        if claim["claim_type"] == "derivation":
            arithmetic = claim["arithmetic"]
            if isinstance(arithmetic, dict) and arithmetic.get("operation") == "rounded_range" and not scope:
                raise ValueError("rounded aggregate needs declared scope")
        used_refs.update((r["evidence_id"], r["observation_key"]) for r in refs)
        stages = {"screening" if r["observation_key"].startswith("screening:") else
                  "confirmation" if r["observation_key"].startswith("confirmation:") else "historical"
                  for r in refs}
        if claim["measurement_stage"] == "confirmation" and (
                "screening" in stages or "confirmation" not in stages or
                source["validated_evidence"]["005"]["observations"]["confirmation_status"] == "missing"):
            raise ValueError("screening evidence cannot support confirmed performance")
        if claim["claim_type"] != "inference" and "screening" in stages and claim["measurement_stage"] != "screening":
            raise ValueError("screening observation must be labeled screening")
        if claim["claim_type"] == "inference" and claim["measurement_stage"] != "not_applicable":
            raise ValueError("inference cannot be labeled measured evidence")
        if not isinstance(claim["limitations"], str) or not claim["limitations"].strip():
            raise ValueError("claim limitations must be explicit")
        _arithmetic(claim, values)
        if claim["claim_type"] == "derivation":
            results = claim["arithmetic"]["result"]
            for result in results if isinstance(results, list) else [results]:
                if not isinstance(result, str) or result not in statement:
                    raise ValueError("derived result missing from statement")
    listed = {(r["evidence_id"], r["observation_key"]) for r in response["evidence_citations"]}
    if not used_refs <= listed:
        raise ValueError("claim reference missing from response citations")


def parse_critique_v2(raw, source):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 48000:
        raise ValueError("missing or oversized v2 criticism")
    try:
        response = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("v2 criticism is not JSON") from exc
    if not isinstance(response, dict) or set(response) != CRITIQUE_FIELDS | {"protocol_version", "claim_ledger"}:
        raise ValueError("v2 criticism schema mismatch")
    if (response["protocol_version"] != VERSION or response["proposal_id"] != source["proposal_id"]
            or response["proposal_sha256"] != source["proposal_sha256"]
            or not isinstance(response["recommendation"], str) or response["recommendation"] not in RECOMMENDATIONS):
        raise ValueError("v2 criticism identity mismatch")
    prior = response["closest_related_prior_research"]
    if (not isinstance(prior, dict) or set(prior) != {"experiment_id", "what_was_tested", "relation"}
            or not isinstance(prior["experiment_id"], str)
            or prior["experiment_id"] not in {"003", "004", "005", "none"}):
        raise ValueError("invalid prior research assessment")
    if not isinstance(response["assessments"], dict) or set(response["assessments"]) != set(ASSESSMENTS):
        raise ValueError("missing critic responsibilities")
    for value in _fields(response).values():
        if not isinstance(value, str) or not value.strip() or len(value) > 1800:
            raise ValueError("invalid v2 assessment text")
    refs = response["evidence_citations"]
    available = {v["evidence_id"]: v["observations"] for v in source["validated_evidence"].values()}
    if not isinstance(refs, list) or not 1 <= len(refs) <= 40:
        raise ValueError("v2 evidence citation count must be 1 to 40")
    pairs = []
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"evidence_id", "observation_key"}:
            raise ValueError("invalid v2 evidence citation")
        pair = (ref["evidence_id"], ref["observation_key"])
        if not all(isinstance(item, str) for item in pair):
            raise ValueError("invalid v2 evidence citation")
        if pair[0] not in available or pair[1] not in available[pair[0]] or pair in pairs:
            raise ValueError("invalid or duplicate v2 evidence citation")
        pairs.append(pair)
    if prior["experiment_id"] != "none" and source["validated_evidence"][prior["experiment_id"]]["evidence_id"] not in {x[0] for x in pairs}:
        raise ValueError("closest experiment lacks citation")
    validate_ledger(response, source)
    return response  # Structurally validated, pending human scientific review.


def critique_prompt_v2(source):
    example = {"protocol_version": VERSION, "proposal_id": source["proposal_id"],
               "proposal_sha256": source["proposal_sha256"],
               "closest_related_prior_research": {"experiment_id": "<003|004|005|none>",
                   "what_was_tested": "Actual tested scope.", "relation": "Overlap or justified revisit."},
               "evidence_citations": [{"evidence_id": source["validated_evidence"]["005"]["evidence_id"],
                                       "observation_key": "confirmation_status"}],
               "assessments": {name: "Specific assessment." for name in ASSESSMENTS},
               **{name: "Specific statement." for name in TEXT_FIELDS},
               "recommendation": "<proceed|revise|deprioritize>",
               "claim_ledger": [{"claim_id": "C1", "claim_type": "<observation|derivation|inference>",
                   "field": "observed_evidence", "statement": "Exact substring of that response field.",
                   "references": [{"evidence_id": "exact ID", "observation_key": "exact key",
                                   "value_path": ["nested field, if any"], "recorded_value": "exact source value",
                                   "unit": "ms|ratio|bytes|elements|none"}],
                   "aggregate_scope": [],
                   "arithmetic": None, "measurement_stage": "<historical|screening|confirmation|not_applicable>",
                   "limitations": "What these observations cannot establish."}]}
    instructions = (
        "Independently critique one advisory GPU hypothesis. Return JSON matching SCHEMA exactly. "
        "For every historical or quantitative claim, add a claim_ledger entry naming its response field, "
        "exact statement substring, and exact evidence IDs/observation keys. Include every source observation "
        "used by an aggregate claim in both references and aggregate_scope. Direct observation statements "
        "must include each exact observation key and value. Use value_path for nested values; copy recorded_value exactly and give "
        "unit (ms, ratio, bytes, elements, or none). Separate direct observations, arithmetic derivations, "
        "and scientific inferences. Prefer exact values. For rounding, give original operands in references "
        "and arithmetic {operation, operand_indices, precision_decimal_places, result}; use rounded_range "
        "for a min/max interval. Distinguish screening from independent confirmation. Never attribute prior "
        "kernel measurements to this untested proposal. State insufficient evidence rather than inventing "
        "support. Keep prose and ledger concise (at most 16 claims). Inference references are context, not proof. Recommendation is advisory; no code or GPU "
        "execution is authorized.\nSCHEMA\n")
    return instructions + json.dumps(example, sort_keys=True) + "\nSOURCE\n" + json.dumps(source, sort_keys=True, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Experiment 008 critic v2 preview")
    parser.add_argument("--preview", action="store_true", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--proposal-id", required=True)
    args = parser.parse_args(argv)
    source = source_package(ROOT, args.study_id, args.review_id, args.proposal_id)
    print(critique_prompt_v2(source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
