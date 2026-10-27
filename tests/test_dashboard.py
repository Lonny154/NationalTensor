"""Read-only dashboard tests; no GPU, Muse client, or generated source import."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dashboard_app import _sample_chart, create_app
import dashboard_data
from dashboard_interpretation import (
    decorate_history, hypothesis_evidence, proposal_summary, status_for,
)


class DashboardTests(unittest.TestCase):
    def test_checkpoint_counts_status_and_source_relationships(self):
        data = dashboard_data.load_history()
        self.assertEqual(data["warnings"], [])
        self.assertEqual(data["overview"]["muse_proposals"], 10)
        self.assertEqual(data["overview"]["structural_proposals"], 6)
        self.assertEqual(data["overview"]["kernel_proposals"], 4)
        self.assertEqual(data["overview"]["rejected_responses"], 1)
        self.assertEqual(data["overview"]["trusted_implementations"], 2)
        self.assertEqual((data["overview"]["trusted_correctness_passed"],
                          data["overview"]["trusted_correctness_total"]), (12, 12))
        self.assertEqual(data["overview"]["confirmed_improvements"], 0)
        kernel = {item["run_id"]: item for item in data["proposals"]
                  if item["kind"] == "kernel"}
        latest = kernel["20261027T163630745455Z"]
        self.assertEqual(latest["outcome"], "unevaluated")
        self.assertEqual(latest["correctness"], "not evaluated")
        self.assertIsNone(latest["timing"])
        self.assertTrue(latest["source_hash_valid"])
        trusted = {item["kind"]: item for item in data["trusted"]}
        self.assertEqual(kernel["20261027T150316362990Z"]["related_trusted"],
                         [trusted["trusted-compute"]["id"]])
        self.assertEqual(kernel["20261027T160014125560Z"]["feedback_link"],
                         trusted["trusted-compute"]["id"])
        self.assertEqual(kernel["20261027T160014125560Z"]["related_trusted"],
                         [trusted["trusted-two-slab"]["id"]])
        self.assertEqual(latest["feedback_link"], trusted["trusted-two-slab"]["id"])
        self.assertEqual(latest["related_trusted"], [])
        self.assertTrue(all(item["exact_generated_source_evaluated"] is False
                            for item in data["trusted"]))
        self.assertTrue(all(not item["confirmed"] for item in data["timeline"]))

    def test_saved_raw_samples_and_methods_are_preserved(self):
        data = dashboard_data.load_history()
        for item in data["proposals"] + data["trusted"]:
            timing = item.get("timing") if item["kind"] == "structural" else item.get("screening")
            if not timing:
                continue
            chart = _sample_chart(timing)
            self.assertEqual(chart["count"], sum(len(values) for values in
                                                timing["samples_ms"].values()))
            self.assertEqual(item["measurement_method"], "python_launch_cuda_event")
            self.assertFalse(item["confirmed"])
        self.assertEqual(len(data["reliability"]["captures"]), 2)
        self.assertEqual({tuple(row["order"]) for row in data["reliability"]["captures"]},
                         {("python_launch", "cuda_graph"),
                          ("cuda_graph", "python_launch")})

    def test_large_compiler_comparison_is_loaded_only_on_request(self):
        paths = []
        real_reader = dashboard_data._read_json

        def record(root, section, relative, warnings, **kwargs):
            paths.append(relative)
            return real_reader(root, section, relative, warnings, **kwargs)

        with patch.object(dashboard_data, "_read_json", side_effect=record):
            data = dashboard_data.load_history()
            self.assertFalse(any("compare/comparison.json" in path for path in paths))
            two_slab = next(item for item in data["trusted"]
                            if item["kind"] == "trusted-two-slab")
            detail = dashboard_data.load_compiler_detail(dashboard_data.ROOT, two_slab)
        self.assertEqual(detail["comparisons"]["ptx"]["baseline_instructions"], 397)
        self.assertEqual(detail["comparisons"]["sass"]["trusted_instructions"], 496)
        ops = detail["comparisons"]["ptx"]["operations"]
        self.assertIn({"name": "cp.async (all forms)", "baseline": 26,
                       "trusted": 50}, ops)
        self.assertTrue(any("sass-compare/comparison.json" in path for path in paths))

    def test_missing_malformed_and_outside_artifacts_do_not_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "evaluations" / "kernel-evolution" / "20261027T000000000000Z"
            run.mkdir(parents=True)
            (run / "metadata.json").write_text("{invalid")
            data = dashboard_data.load_history(root)
            self.assertEqual(data["overview"]["muse_proposals"], 0)
            self.assertTrue(any("Malformed JSON" in message for message in data["warnings"]))
            outside = root / "secret.txt"
            outside.write_text("not dashboard data")
            warnings = []
            self.assertIsNone(dashboard_data._read_text(
                root, "kernel-evolution", "../../../secret.txt", warnings))
            self.assertTrue(any("Blocked path" in message for message in warnings))

    def test_untrusted_source_is_escaped_and_http_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_id = "20261027T000000000000Z"
            run = root / "evaluations" / "kernel-evolution" / run_id
            run.mkdir(parents=True)
            source = "@triton.jit\ndef matmul_kernel():\n    # <script>alert(1)</script>\n"
            digest = hashlib.sha256(source.encode()).hexdigest()
            (run / "candidate.py").write_text(source)
            (run / "metadata.json").write_text(json.dumps({
                "status": "awaiting_human_review", "created_utc": "2026-10-27T00:00:00Z",
                "candidate_sha256": digest, "shape_mnk": [512, 3072, 768],
                "hypothesis": "<script>alert(2)</script>",
                "static_findings": {"ast_check": "accepted_for_review_only"}}))
            response = json.dumps({"kernel_source": source})
            (run / "proposal_audit.jsonl").write_text(json.dumps({
                "prompt": "Request\n{}", "raw_response": response}) + "\n")
            client = create_app(root).test_client()
            page = client.get(f"/?view=timeline&experiment=kernel:{run_id}")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"&lt;script&gt;alert(1)&lt;/script&gt;", page.data)
            self.assertIn(b"&lt;script&gt;alert(2)&lt;/script&gt;", page.data)
            self.assertNotIn(b"<script>", page.data)
            self.assertEqual(client.post("/").status_code, 405)
            self.assertEqual(client.get("/?experiment=../../secret.txt").status_code, 404)
            self.assertIn("default-src 'none'", page.headers["Content-Security-Policy"])

    def test_outcome_labels_keep_review_correctness_and_screening_separate(self):
        data = decorate_history(dashboard_data.load_history())
        by_id = {item["id"]: item for item in data["timeline"]}
        small_m = by_id["structural:20261027T135435023171Z:2"]
        self.assertEqual(small_m["outcome"], "promising_unconfirmed")
        self.assertEqual(small_m["display"]["performance"]["label"], "Inconclusive")
        self.assertEqual(small_m["display"]["correctness"]["label"],
                         "Correctness passed")
        self.assertTrue(any("Large raw" in warning for warning in
                            small_m["display"]["warnings"]))
        self.assertEqual(by_id["trusted:trusted-compute:20261027T154233319712Z"]
                         ["display"]["performance"]["label"], "Inconclusive")
        self.assertEqual(by_id["trusted:trusted-two-slab:20261027T161358957319Z"]
                         ["display"]["performance"]["label"], "Screening loss")
        latest = by_id["kernel:20261027T163630745455Z"]
        self.assertEqual(latest["display"]["review"]["label"], "Awaiting review")
        self.assertEqual(latest["display"]["static"]["label"],
                         "Static validation passed for review")
        self.assertEqual(latest["display"]["correctness"]["label"],
                         "Correctness not evaluated")
        self.assertEqual(latest["display"]["performance"]["label"], "Not measured")
        generic = {"id": "new", "kind": "structural", "correctness": "PASS",
                   "timing": {"median_paired_speedup": 1.08}}
        self.assertEqual(status_for(generic)["performance"]["label"],
                         "Screening gain — unconfirmed")
        generic["confirmed"] = True
        self.assertEqual(status_for(generic)["performance"]["label"],
                         "Screening gain — unconfirmed")
        generic["confirmation_artifact_verified"] = True
        generic["confirmation_decision"] = {"confirmed": True}
        self.assertEqual(status_for(generic)["performance"]["label"], "Confirmed")
        generic["correctness"] = "FAIL"
        self.assertEqual(status_for(generic)["correctness"]["label"],
                         "Correctness failed")
        self.assertNotEqual(status_for(generic)["performance"]["label"], "Confirmed")

    def test_latest_summary_is_short_and_unknown_summary_is_safe(self):
        data = decorate_history(dashboard_data.load_history())
        latest = next(item for item in data["proposals"]
                      if item["id"] == "kernel:20261027T163630745455Z")
        self.assertEqual(latest["summary"]["title"], "K-bound-specialized loop")
        self.assertLess(len(latest["summary"]["summary"]), 230)
        self.assertNotEqual(latest["summary"]["summary"], latest["hypothesis"])
        fallback = proposal_summary({"run_id": "unknown", "kind": "kernel",
                                     "hypothesis": "<script>alert(1)</script> " * 20})
        self.assertLessEqual(len(fallback["summary"]), 170)
        self.assertIn("<script>", fallback["summary"])

    def test_hypothesis_evidence_uses_hashes_and_handles_missing_evidence(self):
        data = decorate_history(dashboard_data.load_history())
        by_id = {item["id"]: item for item in data["timeline"]}
        for trusted_id, proposal_id in (
            ("trusted:trusted-compute:20261027T154233319712Z",
             "kernel:20261027T150316362990Z"),
            ("trusted:trusted-two-slab:20261027T161358957319Z",
             "kernel:20261027T160014125560Z"),
        ):
            trusted, proposal = by_id[trusted_id], by_id[proposal_id]
            comparison = hypothesis_evidence(trusted, proposal)
            self.assertEqual(comparison["muse_sha256"], proposal["source_sha256"])
            self.assertEqual(comparison["trusted_sha256"], trusted["trusted_sha256"])
            self.assertIn("6/6 checks passed", comparison["observed"]["correctness"])
            self.assertIn("no confirmation artifact", comparison["observed"]["screening"])
            self.assertIsNone(hypothesis_evidence(trusted, {**proposal,
                                                            "source_sha256": "0" * 64}))
            self.assertIsNone(hypothesis_evidence(trusted, {**proposal,
                                                            "source_hash_valid": False}))
        two = deepcopy(by_id["trusted:trusted-two-slab:20261027T161358957319Z"])
        proposal = by_id["kernel:20261027T160014125560Z"]
        two["feedback"] = None
        two["screening"] = None
        two["aa"] = None
        two["dynamic_profile"] = None
        two["compiler_resources"] = {"baseline": {}, "trusted": {}}
        missing = hypothesis_evidence(two, proposal)
        self.assertEqual(missing["observed"]["compiler"], ["Unavailable"])
        self.assertEqual(missing["observed"]["resources"], "Unavailable")
        self.assertEqual(missing["observed"]["screening"], "Unavailable")
        self.assertIn("unavailable", missing["interpretation"]["contradicted"].lower())

    def test_comparison_escapes_model_text_and_all_views_render(self):
        data = dashboard_data.load_history()
        proposal = next(item for item in data["proposals"]
                        if item["id"] == "kernel:20261027T160014125560Z")
        proposal["hypothesis"] = "<img src=x onerror=alert(1)>"
        with patch("dashboard_app.load_history", return_value=data):
            client = create_app().test_client()
            for view in ("overview", "timeline", "performance", "reliability"):
                self.assertEqual(client.get(f"/?view={view}").status_code, 200)
            page = client.get("/?view=timeline&experiment="
                              "trusted:trusted-two-slab:20261027T161358957319Z")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"&lt;img src=x onerror=alert(1)&gt;", page.data)
            self.assertNotIn(b"<img src=x", page.data)
            self.assertIn(b"Muse predicted", page.data)
            self.assertIn(b"Observed evidence", page.data)


if __name__ == "__main__":
    unittest.main()
