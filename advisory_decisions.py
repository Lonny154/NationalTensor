"""Completed advisory state, bound to original plans and verified experiment memory."""

import hashlib
import json
from pathlib import Path
import re

from grouped_confirmation_memory import _read


RECORD = "docs/research_reviews/grouped_m4_fc1_deferred.json"


def _evidence_snapshot(root, entries, shape):
    return sorted([
        {"id": row["id"], "artifact_path": row["artifact_path"],
         "sha256": hashlib.sha256(_read(root, row["artifact_path"])).hexdigest()}
        for row in entries if "grouped_scheduling" in row["families"]
        and row["shape_mnk"] == shape and row["kind"] != "generated_proposal"
    ], key=lambda row: row["id"])


def load(root, entries):
    """A closure is advisory state, never an authorization to execute kernels."""
    root = Path(root).resolve()
    if not (root / RECORD).exists():
        return [], []
    try:
        from experiment_planner import load_saved_plans
        record = json.loads(_read(root, RECORD))
        plans, _ = load_saved_plans(root)
        saved = next(row for row in plans if row["run_id"] == record["plan_run_id"])
        if (saved["plan_sha256"] != record["plan_sha256"]
                or saved["plan"]["experiment_kind"] != "do_not_repeat"
                or saved["plan"]["target_shape_mnk"] != record["scope"]["shape_mnk"]
                or record["status"] != "completed_advisory_decision"
                or record["execution_approved"] is not False
                or record["family_wide_rejection"] is not False
                or record["future_controlled_replication_allowed"] is not True):
            raise ValueError("completed advisory identity or scope mismatch")
        indexed = {row["id"]: row for row in entries}
        snapshot = {row["id"]: row for row in record["evidence_snapshot"]}
        for ref in saved["plan"]["supporting_evidence"]:
            row = indexed[ref["id"]]
            spec = snapshot[ref["id"]]
            if row["artifact_path"] != ref["artifact_path"] or spec["artifact_path"] != ref["artifact_path"]:
                raise ValueError("completed advisory evidence path mismatch")
            _read(root, spec["artifact_path"], spec["sha256"])
        current = _evidence_snapshot(root, entries, record["scope"]["shape_mnk"])
        record.update({"artifact_path": RECORD, "plan_path": saved["artifact_path"],
                       "supporting_evidence": saved["plan"]["supporting_evidence"],
                       "new_verified_evidence": current != sorted(record["evidence_snapshot"], key=lambda row: row["id"])})
        return [record], []
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        return [], [f"Completed advisory decision rejected: {exc}"]


def repeated_decision(plan, context):
    """Detect a paraphrased pause of this scoped direction, without banning its family."""
    text = " ".join(plan[key] for key in
                    ("research_question", "optimization_mechanism", "technical_hypothesis"))
    named_m4 = re.search(r"grouped[ _-]*m4|group[ _-]*m\s*=\s*4", text, re.I)
    if plan["experiment_kind"] != "do_not_repeat" or not named_m4:
        return None
    for record in context.get("completed_advisory_decisions", []):
        if (record["scope"]["schedule"] == "grouped_m4"
                and record["scope"]["shape_mnk"] == plan["target_shape_mnk"]
                and not record["new_verified_evidence"]):
            return record
    return None
