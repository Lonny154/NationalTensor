"""Offline atomic-claim critic protocol. No Muse request or Experiment 009 binding."""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import re

from dashboard_data import ROOT
from muse_research_critic import source_package

VERSION = "experiment_008_critic_v3"
TYPES = {"observation", "derivation", "inference"}
OPERATIONS = {"add", "subtract", "multiply", "divide", "rounded_range"}
RECOMMENDATIONS = {"proceed", "revise", "deprioritize"}
ROOT_FIELDS = {"protocol_version", "proposal_id", "proposal_sha256", "claims", "recommendation"}
OBS_FIELDS = {"claim_id", "type", "evidence_id", "observation_key", "value_path", "recorded_value", "unit"}
DER_FIELDS = {"claim_id", "type", "operand_claim_ids", "operation", "precision_decimal_places",
              "result", "unit", "aggregate_set"}
INF_FIELDS = {"claim_id", "type", "support_claim_ids", "interpretation"}


def _unit(key, path):
    leaf = path[-1] if path else key
    if not isinstance(leaf, str):
        leaf = key
    if leaf.endswith("_ms"):
        return "ms"
    if "ratio" in leaf:
        return "ratio"
    if leaf in {"shared_memory_per_multiprocessor", "total_memory"}:
        return "bytes"
    if leaf in {"workload", "control_workload", "partial_m_workload", "shapes", "shape"}:
        return "elements"
    return "none"


def aggregate_sets(source):
    """Comparison membership is computed from source evidence, never from model output."""
    evidence = source["validated_evidence"].get("005", {})
    evidence_id = evidence.get("evidence_id")
    observations = evidence.get("observations", {})
    by_shape = {}
    for key in sorted(observations):
        if key.startswith("screening:") and isinstance(observations[key], dict) and "median_paired_ratio" in observations[key]:
            shape = key.split(":")[1]
            by_shape.setdefault(shape, []).append((evidence_id, key, ("median_paired_ratio",)))
    sets = {}
    shapes = sorted(by_shape)
    if shapes:
        sets["screening:all"] = tuple(item for shape in shapes for item in by_shape[shape])
    for shape in shapes:
        sets[f"screening:{shape}"] = tuple(by_shape[shape])
    for i, left in enumerate(shapes):
        for right in shapes[i + 1:]:
            sets[f"screening:{left}+{right}"] = tuple(by_shape[left] + by_shape[right])
    return sets


def _observation(claim, source):
    if set(claim) != OBS_FIELDS:
        raise ValueError("observation schema mismatch")
    evidence_id, key, path = claim["evidence_id"], claim["observation_key"], claim["value_path"]
    if not isinstance(evidence_id, str) or not isinstance(key, str) or not isinstance(path, list) or len(path) > 8:
        raise ValueError("invalid observation identity")
    if any(type(part) not in (str, int) for part in path):
        raise ValueError("invalid observation value path")
    evidence = {v["evidence_id"]: v["observations"] for v in source["validated_evidence"].values()}
    if evidence_id not in evidence or key not in evidence[evidence_id]:
        raise ValueError("unknown observation ID or key")
    if key == "original_hypothesis":
        raise ValueError("model hypothesis is not a measured observation")
    value = evidence[evidence_id][key]
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value:
            value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError("unknown observation value path")
    if type(value) is not type(claim["recorded_value"]) or value != claim["recorded_value"]:
        raise ValueError("recorded value does not match source")
    unit = _unit(key, path)
    if claim["unit"] != unit:
        raise ValueError("observation unit does not match source")
    stage = "screening" if key.startswith("screening:") else "confirmation" if key.startswith("confirmation:") else "historical"
    if stage == "confirmation" and source["validated_evidence"]["005"]["observations"].get("confirmation_status") == "missing":
        raise ValueError("confirmation observation conflicts with missing confirmation")
    return {"value": value, "unit": unit, "stage": stage,
            "provenance": (evidence_id, key, tuple(path))}


def _result(operation, numbers, precision):
    if operation == "rounded_range":
        exact = [min(numbers), max(numbers)]
    elif operation == "add":
        exact = [sum(numbers)]
    elif operation == "subtract" and len(numbers) == 2:
        exact = [numbers[0] - numbers[1]]
    elif operation == "multiply" and len(numbers) == 2:
        exact = [numbers[0] * numbers[1]]
    elif operation == "divide" and len(numbers) == 2 and numbers[1] != 0:
        exact = [numbers[0] / numbers[1]]
    else:
        raise ValueError("invalid arithmetic operands")
    quantum = Decimal(1).scaleb(-precision)
    try:
        values = [format(number.quantize(quantum, rounding=ROUND_HALF_UP), f".{precision}f") for number in exact]
    except InvalidOperation as exc:
        raise ValueError("unrepresentable arithmetic result") from exc
    return values if operation == "rounded_range" else values[0]


def _derivation(claim, known, sets):
    if set(claim) != DER_FIELDS:
        raise ValueError("derivation schema mismatch")
    ids = claim["operand_claim_ids"]
    if not isinstance(ids, list) or not 2 <= len(ids) <= 12 or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid derivation operands")
    if any(x not in known or known[x]["type"] == "inference" for x in ids):
        raise ValueError("derivation requires prior measured or derived claims")
    operation, precision = claim["operation"], claim["precision_decimal_places"]
    if not isinstance(operation, str) or operation not in OPERATIONS or type(precision) is not int or not 0 <= precision <= 12:
        raise ValueError("invalid arithmetic operation or precision")
    operands = [known[x] for x in ids]
    if any(type(item["value"]) not in (int, float, str) or item["unit"] == "none" for item in operands):
        raise ValueError("non-numeric or unitless-unknown arithmetic operand")
    try:
        numbers = [Decimal(str(item["value"])) for item in operands]
    except InvalidOperation as exc:
        raise ValueError("non-numeric arithmetic operand") from exc
    units = [item["unit"] for item in operands]
    if operation in {"add", "subtract", "rounded_range"}:
        if len(set(units)) != 1:
            raise ValueError("incompatible arithmetic units")
        unit = units[0]
    elif operation == "multiply":
        if len(units) != 2 or (units[0] != "ratio" and units[1] != "ratio"):
            raise ValueError("unsupported multiplication units")
        unit = units[1] if units[0] == "ratio" else units[0]
    else:
        if len(units) != 2 or (units[0] != units[1] and units[1] != "ratio"):
            raise ValueError("unsupported division units")
        unit = "ratio" if units[0] == units[1] else units[0]
    if claim["unit"] != unit:
        raise ValueError("derived unit mismatch")
    stages = {item["stage"] for item in operands}
    stage = "screening" if "screening" in stages else "confirmation" if "confirmation" in stages else "historical"
    aggregate_set = claim["aggregate_set"]
    if operation == "rounded_range":
        if not isinstance(aggregate_set, str) or aggregate_set not in sets:
            raise ValueError("range requires an independent aggregate set")
        actual = [item["provenance"] for item in operands]
        if any(item["type"] != "observation" for item in operands) or len(actual) != len(sets[aggregate_set]) or set(actual) != set(sets[aggregate_set]):
            raise ValueError("aggregate operands do not cover independent set")
    elif aggregate_set is not None:
        raise ValueError("aggregate set is only supported for rounded ranges")
    result = _result(operation, numbers, precision)
    if claim["result"] != result:
        raise ValueError("derived result is incorrect")
    return {"value": result, "unit": unit, "stage": stage, "provenance": None}


def parse_critique_v3(raw, source):
    """Return structural status separately from required human scientific review."""
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 48000:
        raise ValueError("missing or oversized v3 response")
    try:
        response = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("v3 response is not JSON") from exc
    if not isinstance(response, dict) or set(response) != ROOT_FIELDS:
        raise ValueError("v3 response schema mismatch; unrestricted prose is not accepted")
    if (response["protocol_version"] != VERSION or response["proposal_id"] != source["proposal_id"]
            or response["proposal_sha256"] != source["proposal_sha256"]
            or not isinstance(response["recommendation"], str) or response["recommendation"] not in RECOMMENDATIONS):
        raise ValueError("v3 response identity or recommendation mismatch")
    claims = response["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= 16:
        raise ValueError("v3 requires 1 to 16 atomic claims")
    known, sets = {}, aggregate_sets(source)
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str) or not re.fullmatch(r"C[1-9]\d{0,2}", claim["claim_id"]) or claim["claim_id"] in known:
            raise ValueError("invalid or duplicate claim ID")
        kind = claim.get("type")
        if not isinstance(kind, str) or kind not in TYPES:
            raise ValueError("invalid claim type")
        if kind == "observation":
            result = _observation(claim, source)
        elif kind == "derivation":
            result = _derivation(claim, known, sets)
        else:
            if set(claim) != INF_FIELDS:
                raise ValueError("inference schema mismatch")
            support = claim["support_claim_ids"]
            if not isinstance(support, list) or not 1 <= len(support) <= 12 or any(not isinstance(x, str) or x not in known for x in support) or len(set(support)) != len(support):
                raise ValueError("inference requires prior supporting claims")
            interpretation = claim["interpretation"]
            if (not isinstance(interpretation, str) or not interpretation.strip()
                    or len(interpretation) > 500 or any(ord(char) < 32 for char in interpretation)):
                raise ValueError("invalid unverified interpretation")
            result = {"value": None, "unit": "none", "stage": "unmeasured", "provenance": None}
        known[claim["claim_id"]] = {"type": kind, **result}
    return {"structural_status": "valid", "scientific_status": "pending_human_review", "response": response}


def render_summary(validated):
    if validated.get("structural_status") != "valid" or validated.get("scientific_status") != "pending_human_review":
        raise ValueError("rendering requires a validated, unreviewed v3 response")
    lines = ["Structurally validated evidence; scientific review pending."]
    stages = {}
    for claim in validated["response"]["claims"]:
        ident = claim["claim_id"]
        if claim["type"] == "observation":
            stage = "Screening" if claim["observation_key"].startswith("screening:") else "Confirmation" if claim["observation_key"].startswith("confirmation:") else "Historical"
            name = claim["value_path"][-1] if claim["value_path"] else claim["observation_key"]
            name = str(name).replace("_", " ")
            value = json.dumps(claim["recorded_value"], ensure_ascii=False)
            suffix = "" if claim["unit"] in {"none", "ratio"} else f" {claim['unit']}"
            lines.append(f"{ident}. {stage} observation ({name}): {value}{suffix}.")
            stages[ident] = stage.lower()
        elif claim["type"] == "derivation":
            unit = "" if claim["unit"] == "ratio" else f" {claim['unit']}"
            result = claim["result"]
            shown = " to ".join(result) if isinstance(result, list) else result
            inherited = [stages[x] for x in claim["operand_claim_ids"]]
            stage = "screening" if "screening" in inherited else "confirmation" if "confirmation" in inherited else "historical"
            scope = f" over {claim['aggregate_set']}" if claim["aggregate_set"] else ""
            lines.append(f"{ident}. {stage.capitalize()} calculation{scope} from {', '.join(claim['operand_claim_ids'])}: {claim['operation']} = {shown}{unit}.")
            stages[ident] = stage
        else:
            lines.append(f"{ident}. Unverified model inference, human review required (support: {', '.join(claim['support_claim_ids'])}): {claim['interpretation']}")
    lines.append(f"Advisory recommendation: {validated['response']['recommendation']} (human review required).")
    return "\n".join(lines)


def critique_prompt_v3(source):
    example = {"protocol_version": VERSION, "proposal_id": source["proposal_id"],
               "proposal_sha256": source["proposal_sha256"], "recommendation": "<proceed|revise|deprioritize>",
               "claims": [
                   {"claim_id": "C1", "type": "observation", "evidence_id": "exact source ID",
                    "observation_key": "exact key", "value_path": ["nested key if needed"],
                    "recorded_value": "exact source value", "unit": "ms|ratio|bytes|elements|none"},
                   {"claim_id": "C2", "type": "derivation", "operand_claim_ids": ["C1", "another earlier ID"],
                    "operation": "add|subtract|multiply|divide|rounded_range", "precision_decimal_places": 3,
                    "result": "decimal string or two strings for range", "unit": "result unit",
                    "aggregate_set": "source-derived set name or null"},
                   {"claim_id": "C3", "type": "inference", "support_claim_ids": ["C1"],
                    "interpretation": "Unverified interpretation requiring human review."}]}
    instructions = (
        "Critique this untested GPU proposal using only atomic claims. Return JSON matching SCHEMA; "
        "no other prose fields. Observations copy exact evidence ID, key, nested value, and unit. "
        "Derivations cite earlier claim IDs, declare operation, decimal precision, result and unit. "
        "For rounded_range choose a provided aggregate set and include every member as an observation. "
        "Use inference only for interpretations; it remains unverified even with support claims. "
        "Do not promote screening to confirmation or prior-kernel results to this proposal. "
        "Say insufficient evidence with an inference instead of inventing a measurement. "
        "At most 16 claims. Recommendation is advisory; no code or execution.\nSCHEMA\n")
    available = {name: [{"evidence_id": eid, "observation_key": key, "value_path": list(path)}
                        for eid, key, path in members]
                 for name, members in aggregate_sets(source).items()}
    return instructions + json.dumps(example, sort_keys=True) + "\nAGGREGATE_SETS\n" + json.dumps(available, sort_keys=True) + "\nSOURCE\n" + json.dumps(source, sort_keys=True, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Experiment 008 atomic critic v3 preview")
    parser.add_argument("--preview", action="store_true", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--proposal-id", required=True)
    args = parser.parse_args(argv)
    print(critique_prompt_v3(source_package(ROOT, args.study_id, args.review_id, args.proposal_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
