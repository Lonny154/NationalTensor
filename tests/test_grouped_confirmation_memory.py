"""Historical confirmation/review ingestion and advisory citation checks; no GPU work."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import experiment_memory
import experiment_planner
import grouped_confirmation_memory as grouped


ROOT = Path(__file__).resolve().parents[1]


class GroupedConfirmationMemoryTests(unittest.TestCase):
    def test_all_three_attempts_survive_reconstruction_with_limitations(self):
        first = experiment_memory.reconstruct(ROOT)
        second = experiment_memory.reconstruct(ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first["warnings"], [])
        entry = next(row for row in first["entries"] if row["id"] == "grouped_m4:confirmation_history")
        attempts = entry["confirmation_evidence"]["attempts"]
        self.assertEqual(len(attempts), 3)
        self.assertEqual([row["pairs_favoring_candidate"] for row in attempts], [5, 5, 4])
        for attempt, median in zip(attempts, (1.0176651569959712, 1.0103786564323913,
                                             1.0024931750662138)):
            self.assertAlmostEqual(attempt["median_verification_pair_speedup"], median)
            self.assertEqual(attempt["status"], "unconfirmed")
            self.assertEqual(len(attempt["verification_pair_speedups"]), 7)
            self.assertEqual(attempt["method"]["iterations_per_side_per_round"], 200)
            self.assertIn("not recorded", attempt["aa_control"])
            self.assertEqual(len(attempt["artifact_sha256"]), 64)
        self.assertIn("legacy", attempts[0]["reference_precision_policy"])
        self.assertEqual(attempts[1]["reference_precision_policy"],
                         "fp16_torch_mm_without_reduced_precision_reduction")
        self.assertIsNone(entry["confirmation_evidence"]["pooled_latency"])
        self.assertEqual(entry["outcome"], "inconclusive_grouped_m4")

    def test_planner_can_cite_complete_history_and_review_without_new_execution(self):
        context = experiment_planner.build_context(ROOT)
        entry = next(row for row in context["relevant_experiment_memory"]
                     if row["id"] == "grouped_m4:confirmation_history")
        review = entry["review_status"]
        self.assertEqual(review["status"], "reviewed_not_approved_for_execution")
        self.assertTrue(review["future_controlled_replication_allowed"])
        self.assertIn("limited expected value", review["reason"])
        self.assertIn("no failure classification", review["scope"])
        self.assertIn("GROUP_M=4", review["schedule_distinction"]["grouped_m4"])
        self.assertIn("does not define", review["schedule_distinction"]["hypothetical_grouped_n"])
        plan = json.loads((ROOT / "examples/experiment_planner_plan.json").read_text())
        plan["supporting_evidence"] = [{
            "id": entry["id"], "artifact_path": entry["artifact_path"],
            "observation_keys": ["confirmation_evidence", "review_status", "outcome"],
            "why_relevant": "Complete prior evidence and review constrain a later research hypothesis.",
        }]
        experiment_planner.parse_plan(json.dumps(plan), context["relevant_experiment_memory"])
        prompt = experiment_planner.build_prompt(context)
        self.assertIn("Hypothetical grouped-N needs an explicit separately reviewed mapping", prompt)
        self.assertIn("justified future controlled replication remains eligible", prompt)
        for attempt in entry["confirmation_evidence"]["attempts"]:
            self.assertIn(attempt["artifact_path"], prompt)
        # The immutable advisory remains an accepted historical model response.
        saved, warnings = experiment_planner.load_saved_plans(ROOT)
        self.assertEqual(warnings, [])
        original = next(row for row in saved if row["run_id"] == "20261028T142437794739Z")
        self.assertEqual(original["plan_sha256"], review["plan_sha256"])

    def test_tampered_artifacts_sources_and_plan_are_rejected(self):
        review = json.loads((ROOT / grouped.REVIEW).read_text())
        paths = [grouped.REVIEW, review["plan_path"], *review["source_files"],
                 *(spec["path"] for spec in review["confirmation_artifacts"])]
        targets = [spec["path"] for spec in review["confirmation_artifacts"]]
        targets += ["grouped_matmul.py", review["plan_path"]]
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for relative in paths:
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / relative, destination)
                self.assertEqual(grouped.load(root)[1], [])
                path = root / target
                path.write_bytes(path.read_bytes() + b"\n")
                entry, warnings = grouped.load(root)
                self.assertIsNone(entry)
                self.assertTrue(any("hash mismatch" in warning for warning in warnings))

    def test_contradictory_decision_is_rejected_even_with_updated_artifact_hash(self):
        review = json.loads((ROOT / grouped.REVIEW).read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [grouped.REVIEW, review["plan_path"], *review["source_files"],
                     *(spec["path"] for spec in review["confirmation_artifacts"])]
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, path)
            spec = review["confirmation_artifacts"][0]
            path = root / spec["path"]
            saved = json.loads(path.read_text())
            saved["shapes"][0]["confirmation"]["decision"]["pairs_favoring_candidate"] = 7
            path.write_text(json.dumps(saved))
            spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / grouped.REVIEW).write_text(json.dumps(review))
            entry, warnings = grouped.load(root)
            self.assertIsNone(entry)
            self.assertTrue(any("contradicts raw rounds" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
