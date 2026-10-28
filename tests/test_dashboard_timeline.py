"""Unified activity presentation tests; fixture execution is mocked and CPU-only."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from dashboard_app import create_app
from dashboard_data import load_history
from dashboard_timeline import FILTERS, build_activity, filter_events
import experiment_memory
import test_research_workflow as fixtures


ROOT = Path(__file__).resolve().parents[1]


class UnifiedTimelineTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ControlledWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        fixture.approve()
        fixture.manager.start_job(fixtures.PLAN_RUN, background=False)
        self.job = fixture.manager.jobs()[0]
        for section in ("kernel-evolution", "structural-loop", "trusted-compute", "trusted-two-slab",
                        "planned-profiling", "stage-reduction", "experiment-003-mask-hoisting-m128",
                        "experiment-planner", "grouped-m4"):
            shutil.copytree(ROOT / "evaluations" / section, self.root / "evaluations" / section, dirs_exist_ok=True)
        shutil.copytree(ROOT / "runs/20261027T010046401977Z", self.root / "runs/20261027T010046401977Z")
        shutil.copytree(ROOT / "docs/research_reviews", self.root / "docs/research_reviews")
        shutil.copy2(ROOT / "docs/day2_checkpoint.md", self.root / "docs/day2_checkpoint.md")
        for filename in ("structural_loop.py", "planned_profile.py"):
            shutil.copy2(ROOT / filename, self.root / filename)

    def test_all_activity_and_filters_classify_science_decisions_and_replays(self):
        activity = build_activity(self.root)
        self.assertEqual(activity["warnings"], [])
        indexed = {row["id"]: row for row in activity["events"]}
        replay = indexed[self.job["memory_id"]]
        self.assertEqual(replay["outcome"], "compiler_equivalent_screening_skipped")
        self.assertEqual(replay["filters"], ["workflow", "compiler"])
        self.assertIsNone(replay["measurement"])
        self.assertEqual(replay["measurement_label"], "Not measured")
        scientific = {row["id"] for row in filter_events(activity["events"], "scientific")}
        compiler = {row["id"] for row in filter_events(activity["events"], "compiler")}
        self.assertNotIn(replay["id"], scientific)
        self.assertIn(replay["id"], compiler)
        self.assertIn("diagnostic:mask_hoisting_m128:20261028T145501296300Z", scientific & compiler)
        self.assertIn("stage_reduction:20261028T140056186673Z", scientific)
        self.assertEqual(indexed["plan:20261028T144337265266Z"]["outcome"], "reviewed_and_deferred")
        self.assertIn("reviewed_not_approved_for_execution", {row["outcome"] for row in activity["events"]})
        confirmations = [row for row in activity["events"] if row["experiment_type"] == "performance confirmation"]
        self.assertEqual(len(confirmations), 3)
        self.assertTrue(all(row["outcome"] == "unconfirmed" for row in confirmations))
        self.assertIn("Inconclusive", {row["interpretation"] for row in activity["events"]})
        self.assertIn("Screening loss", {row["interpretation"] for row in activity["events"]})
        self.assertEqual(filter_events(activity["events"], "all"), activity["events"])
        for value in FILTERS:
            self.assertTrue(all(value == "all" or value in row["filters"] for row in filter_events(activity["events"], value)))

    def test_performance_uses_saved_values_only_and_never_pools_or_mutates_science(self):
        history, memory = load_history(self.root), experiment_memory.reconstruct(self.root)
        original_history, original_memory = deepcopy(history), deepcopy(memory)
        original_files = {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in self.root.rglob("*") if path.is_file()}
        activity = build_activity(self.root, history=history, memory=memory)
        raw = {row["id"]: row for row in history["timeline"]}
        remembered = {row["id"]: row for row in memory["entries"]}
        for row in activity["performance"]:
            self.assertFalse(row["workflow_validation"])
            if row["id"] in raw:
                record = raw[row["id"]]
                timing = record.get("timing") if record["kind"] == "structural" else record["screening"]
                self.assertEqual(row["measurement"]["baseline_ms"], timing["baseline_ms"])
                self.assertEqual(row["measurement"]["candidate_ms"], timing["candidate_ms"])
                self.assertEqual(row["measurement"]["median_paired_speedup"], timing["median_paired_speedup"])
            else:
                timing = remembered[row["id"]]["screening"]
                self.assertEqual(row["measurement"]["baseline_ms"], timing["baseline_median_ms"])
                self.assertEqual(row["measurement"]["candidate_ms"], timing["candidate_median_ms"])
        self.assertNotIn(self.job["memory_id"], {row["id"] for row in activity["performance"]})
        self.assertNotIn("diagnostic:mask_hoisting_m128:20261028T145501296300Z", {row["id"] for row in activity["performance"]})
        self.assertEqual(history, original_history)
        self.assertEqual(memory, original_memory)
        self.assertEqual(original_files, {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in self.root.rglob("*") if path.is_file()})

    def test_tampered_workflow_is_excluded_instead_of_shown_as_completed(self):
        path = self.root / "evaluations/research-workflow/jobs" / self.job["id"] / "artifacts/correctness.json"
        path.write_text("{}")
        activity = build_activity(self.root)
        self.assertNotIn(self.job["memory_id"], {row["id"] for row in activity["events"]})
        self.assertTrue(activity["warnings"])

    def test_tampered_compiler_artifact_and_inconsistent_legacy_timing_are_excluded(self):
        compiler = self.root / "evaluations/trusted-two-slab/20261027T161358957319Z/baseline.ptx"
        compiler.write_bytes(compiler.read_bytes() + b"\n")
        path = self.root / "evaluations/structural-loop/20261027T135435023171Z/experiments.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["timing"]["paired_rounds"][0]["baseline_ms"] *= 2
        path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        activity = build_activity(self.root)
        identifiers = {row["id"] for row in activity["events"]}
        self.assertNotIn("trusted:trusted-two-slab:20261027T161358957319Z", identifiers)
        self.assertNotIn("structural:20261027T135435023171Z:1", identifiers)
        self.assertNotIn("trusted:trusted-two-slab:20261027T161358957319Z", {row["id"] for row in activity["performance"]})
        self.assertTrue(any("Timeline record excluded" in warning for warning in activity["warnings"]))

    def test_rejected_human_review_appears_as_a_decision_without_runtime_values(self):
        fixture = fixtures.ControlledWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.manager.decide(fixtures.PLAN_RUN, "rejected")
        activity = build_activity(fixture.root)
        review = next(row for row in activity["events"] if row["id"] == "approval:" + fixtures.PLAN_RUN)
        self.assertEqual(review["outcome"], "rejected")
        self.assertEqual(review["filters"], ["decisions"])
        self.assertIsNone(review["measurement"])

    def test_dashboard_filter_api_and_evidence_navigation(self):
        with patch("research_workflow.execute_trusted", side_effect=AssertionError("Timeline launched execution")) as execute:
            client = create_app(self.root).test_client()
            page = client.get("/?view=timeline", base_url=fixtures.BASE_URL)
            self.assertEqual(page.status_code, 200)
            self.assertIn(self.job["memory_id"].encode(), page.data)
            self.assertIn(b"Workflow validation", page.data)
            self.assertIn(b"Not measured", page.data)
            for label in FILTERS.values():
                self.assertIn(label.encode(), page.data)
            for value in FILTERS:
                data = client.get("/api/research/timeline?filter=" + value, base_url=fixtures.BASE_URL).get_json()
                expected = filter_events(build_activity(self.root)["events"], value)
                self.assertEqual([row["id"] for row in data["events"]], [row["id"] for row in expected])
                self.assertTrue(all("evidence_url" in row for row in data["events"]))
                self.assertEqual(client.get("/?view=timeline&filter=" + value, base_url=fixtures.BASE_URL).status_code, 200)
            data = client.get("/api/research/timeline", base_url=fixtures.BASE_URL).get_json()
            examples = {}
            for row in data["events"]:
                examples.setdefault(row["evidence"]["kind"], row)
            examples["workflow"] = next(row for row in data["events"] if row["id"] == self.job["memory_id"])
            for row in examples.values():
                self.assertEqual(client.get(row["evidence_url"], base_url=fixtures.BASE_URL).status_code, 200)
            performance = client.get("/?view=performance", base_url=fixtures.BASE_URL)
            self.assertEqual(performance.status_code, 200)
            self.assertNotIn(self.job["memory_id"].encode(), performance.data)
            self.assertIn(b"Not measured", performance.data)
            self.assertEqual(client.get("/?view=timeline&filter=unknown", base_url=fixtures.BASE_URL).status_code, 400)
            self.assertEqual(client.get("/api/research/timeline?filter=unknown", base_url=fixtures.BASE_URL).status_code, 400)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
