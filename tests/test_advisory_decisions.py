"""Completed advisory lifecycle and distinct-next-request checks; mocked Muse only."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import advisory_decisions as decisions
import experiment_memory
import experiment_planner as planner


ROOT = Path(__file__).resolve().parents[1]


class AdvisoryDecisionTests(unittest.TestCase):
    def test_completion_survives_reload_and_reaches_next_prompt(self):
        first = experiment_memory.reconstruct(ROOT)
        self.assertEqual(first, experiment_memory.reconstruct(ROOT))
        self.assertEqual(first["warnings"], [])
        record, = first["completed_advisory_decisions"]
        self.assertEqual(record["status"], "completed_advisory_decision")
        self.assertEqual(record["direction_status"], "reviewed_and_deferred")
        self.assertFalse(record["new_verified_evidence"])
        self.assertFalse(record["execution_approved"])
        self.assertTrue(record["future_controlled_replication_allowed"])
        self.assertFalse(record["family_wide_rejection"])
        self.assertEqual(len(record["supporting_evidence"]), 2)
        context = planner.build_context(ROOT)
        self.assertEqual(context["completed_advisory_decisions"], [record])
        prompt = planner.build_prompt(context)
        self.assertIn("Select a distinct unresolved optimization or profiling question", prompt)
        self.assertIn("choose the technical direction yourself", prompt)
        self.assertIn("requires separate human approval", prompt)
        self.assertIn(record["plan_sha256"], prompt)

    def test_paraphrased_deferral_is_audited_without_another_accepted_plan(self):
        context = planner.build_context(ROOT)
        record, = context["completed_advisory_decisions"]
        plan = json.loads((ROOT / record["plan_path"]).read_text())
        plan["optimization_mechanism"] = "Defer grouped-M4 FC1 again because prior trials were inconclusive"
        response = SimpleNamespace(id="mock", output_text=json.dumps(plan))
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
        with tempfile.TemporaryDirectory() as temp:
            folder, metadata = planner.propose_to_directory(
                client, planner.build_prompt(context), context, Path(temp))
            self.assertEqual(metadata["status"], "duplicate_direction")
            self.assertEqual(metadata["duplicate_of"], record["plan_path"])
            self.assertFalse((folder / "plan.json").exists())
            audit = json.loads((folder / "proposal_audit.jsonl").read_text())
            self.assertIn("already completed", audit["rejection"])

    def test_other_directions_and_new_evidence_remain_eligible(self):
        context = planner.build_context(ROOT)
        record, = context["completed_advisory_decisions"]
        original = json.loads((ROOT / record["plan_path"]).read_text())
        self.assertIsNotNone(decisions.repeated_decision(original, context))
        other_shape = deepcopy(original)
        other_shape["target_shape_mnk"] = [128, 3072, 768]
        self.assertIsNone(decisions.repeated_decision(other_shape, context))
        replication = deepcopy(original)
        replication["experiment_kind"] = "profiling"
        self.assertIsNone(decisions.repeated_decision(replication, context))
        different = deepcopy(original)
        for key in ("research_question", "optimization_mechanism", "technical_hypothesis"):
            different[key] = "An unrelated unresolved research direction"
        self.assertIsNone(decisions.repeated_decision(different, context))
        new = deepcopy(context)
        new["completed_advisory_decisions"][0]["new_verified_evidence"] = True
        self.assertIsNone(decisions.repeated_decision(original, new))
        # New verified memory can flag reconsideration without erasing the old decision.
        entries = experiment_memory.reconstruct(ROOT)["entries"]
        extra = deepcopy(next(row for row in entries if row["id"] == "grouped_m4:confirmation_history"))
        extra["id"] = "grouped_m4:later_verified_replication"
        loaded, warnings = decisions.load(ROOT, entries + [extra])
        self.assertEqual(warnings, [])
        self.assertTrue(loaded[0]["new_verified_evidence"])
        self.assertEqual(loaded[0]["status"], "completed_advisory_decision")

    def test_tampered_completion_identity_is_not_exposed_as_verified(self):
        entries = experiment_memory.reconstruct(ROOT)["entries"]
        read = decisions._read

        def tampered(root, relative, digest=None):
            raw = read(root, relative, digest)
            if relative == decisions.RECORD:
                record = json.loads(raw)
                record["plan_sha256"] = "0" * 64
                return json.dumps(record).encode()
            return raw

        with patch.object(decisions, "_read", side_effect=tampered):
            loaded, warnings = decisions.load(ROOT, entries)
        self.assertEqual(loaded, [])
        self.assertTrue(any("identity" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
