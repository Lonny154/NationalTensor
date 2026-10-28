"""Mock-only advisory planner tests; no GPU or live Muse requests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dashboard_app import _planner_references, create_app
import dashboard_data
import experiment_planner as planner


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "experiment_planner_plan.json"
FAILED_RESPONSE = (ROOT / "tests" / "fixtures" /
                   "planner_invalid_response_20261027T231039939029Z.json")
FAILED_RESPONSE_SHA256 = "87cb3757c51907126d96014dae25d2d81e2c2c9ab512a139a8d2a3ea52dfc556"


def fixture():
    context = planner.build_context(ROOT)
    plan = json.loads(EXAMPLE.read_text())
    return context, plan


def client_for(plan):
    response = SimpleNamespace(id="mock-plan", output_text=json.dumps(plan))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    return SimpleNamespace(responses=SimpleNamespace(create=create)), calls


class PlannerTests(unittest.TestCase):
    def test_history_retrieval_keeps_outcomes_shapes_and_provenance(self):
        context, plan = fixture()
        rows = context["relevant_experiment_memory"]
        self.assertEqual(context["hardware"]["compute_capability"], [8, 9])
        self.assertEqual(context["current_numeric_config"]["block_k"], 32)
        self.assertEqual(len(context["baseline_source_sha256"]), 64)
        self.assertLessEqual(len(rows), planner.MAX_CONTEXT)
        self.assertIn([128, 3072, 768], context["available_workload_shapes_mnk"])
        self.assertIn([1024, 1024, 1024], context["available_workload_shapes_mnk"])
        self.assertTrue(any(row["shape_mnk"] == [128, 3072, 768] for row in rows))
        self.assertTrue(any(row["shape_mnk"] == [1024, 1024, 1024] for row in rows))
        two = next(row for row in rows if row["id"].startswith("trusted:trusted-two-slab:"))
        tuned = next(row for row in rows if row["id"].startswith("stage_reduction:"))
        self.assertEqual(two["performance_interpretation"], "Screening loss")
        self.assertEqual(two["correctness"], "PASS")
        self.assertEqual(two["trusted_implementation_sha256"],
                         "5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0")
        self.assertAlmostEqual(two["screening"]["aa_median_absolute_ratio_deviation"],
                               0.04851305314242971)
        self.assertEqual(tuned["outcome"], "two_stage_screening_loss")
        self.assertEqual(tuned["correctness"], "PASS")
        self.assertAlmostEqual(tuned["screening"]["median_paired_speedup"],
                               0.951883129051782)
        self.assertEqual(tuned["compiler_resources"]["stages_2"]
                         ["shared_memory_bytes"], 24576)
        self.assertEqual(tuned["profiling_evidence"]["metrics"]["stages_2"]
                         ["sm__maximum_warps_per_active_cycle_pct"], 33.33)
        self.assertTrue(any(row["kind"] == "generated_proposal"
                            and row["correctness"] == "not evaluated" for row in rows))
        self.assertEqual(planner.parse_plan(json.dumps(plan), rows), plan)
        defer = deepcopy(plan)
        defer["experiment_kind"] = "do_not_repeat"
        defer["optimization_mechanism"] = "Do not repeat two-slab K-loop unrolling without new evidence"
        self.assertEqual(planner.parse_plan(json.dumps(defer), rows)["experiment_kind"],
                         "do_not_repeat")
        prompt = planner.build_prompt(context)
        header = prompt.split("\n", 1)[0]
        prompt_example = json.loads(header.split("VALID_JSON_EXAMPLE: ", 1)[1])
        self.assertEqual(planner.parse_plan(json.dumps(prompt_example), rows), prompt_example)
        self.assertIsInstance(prompt_example["experiment_kind"], str)
        self.assertEqual(prompt_example["target_shape_mnk"], [512, 3072, 768])
        self.assertIn("experiment_kind MUST be one JSON string", header)
        self.assertIn("target_shape_mnk MUST be a JSON array", header)
        self.assertIn(two["original_muse_source_sha256"], prompt)
        self.assertIn("Screening is not confirmation", prompt)
        self.assertIn("does not rule out an entire optimization family", prompt)
        self.assertIn("do not prove identical runtime performance", prompt)
        self.assertIn("Keep resource improvement separate from performance improvement", prompt)
        self.assertIn("two_stage_screening_loss", prompt)
        self.assertIn("do_not_repeat", prompt)

    def test_actual_failed_response_requires_correct_json_types_and_valid_citations(self):
        context, _ = fixture()
        raw = FAILED_RESPONSE.read_text().rstrip("\n")
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), FAILED_RESPONSE_SHA256)
        original = json.loads(raw)
        self.assertEqual(original["experiment_kind"], ["profiling"])
        self.assertEqual(original["target_shape_mnk"], "[512, 3072, 768]")
        with self.assertRaisesRegex(ValueError, "unsupported experiment_kind"):
            planner.parse_plan(raw, context["relevant_experiment_memory"])
        corrected = deepcopy(original)
        corrected["experiment_kind"] = "profiling"
        with self.assertRaisesRegex(ValueError, "target_shape_mnk"):
            planner.parse_plan(json.dumps(corrected), context["relevant_experiment_memory"])
        corrected["target_shape_mnk"] = [512, 3072, 768]
        self.assertEqual(planner.parse_plan(json.dumps(corrected),
                                            context["relevant_experiment_memory"]), corrected)
        bad = deepcopy(corrected)
        bad["supporting_evidence"][0]["id"] = "trusted:fabricated"
        with self.assertRaisesRegex(ValueError, "evidence reference"):
            planner.parse_plan(json.dumps(bad), context["relevant_experiment_memory"])
        bad = deepcopy(corrected)
        bad["supporting_evidence"][0]["observation_keys"] = ["invented_metric"]
        with self.assertRaisesRegex(ValueError, "evidence observation key"):
            planner.parse_plan(json.dumps(bad), context["relevant_experiment_memory"])

    def test_structured_plan_rejects_fabricated_and_unavailable_evidence(self):
        context, plan = fixture()
        rows = context["relevant_experiment_memory"]
        variants = []
        bad = deepcopy(plan); bad["supporting_evidence"][0]["id"] = "trusted:invented"; variants.append(bad)
        bad = deepcopy(plan); bad["supporting_evidence"][0]["artifact_path"] = "/tmp/invented"; variants.append(bad)
        bad = deepcopy(plan); bad["supporting_evidence"][0]["observation_keys"] = ["unknown"]; variants.append(bad)
        bad = deepcopy(plan); bad["supporting_evidence"][0]["observation_keys"] = ["compiler_evidence", "screening"]
        generated = next(row for row in rows if row["kind"] == "generated_proposal")
        bad["supporting_evidence"][0]["id"] = generated["id"]
        bad["supporting_evidence"][0]["artifact_path"] = generated["artifact_path"]
        variants.append(bad)
        bad = deepcopy(plan); bad["kernel_source"] = "@triton.jit"; variants.append(bad)
        bad = deepcopy(plan); bad["technical_hypothesis"] = "```python\npass\n```"; variants.append(bad)
        bad = deepcopy(plan); bad["experiment_kind"] = "execute_kernel"; variants.append(bad)
        bad = deepcopy(plan); bad["target_shape_mnk"] = [True, 3, 4]; variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    planner.parse_plan(json.dumps(bad), rows)
        with self.assertRaises(ValueError):
            planner.parse_plan("{bad", rows)

    def test_one_mocked_advisory_plan_is_audited_and_duplicate_direction_rejected(self):
        context, plan = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = planner.build_prompt(context)
            client, calls = client_for(plan)
            folder, metadata = planner.propose_to_directory(client, prompt, context, root)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["model"], planner.MODEL)
            self.assertEqual(metadata["status"], "advisory_saved")
            self.assertEqual(metadata["plan_sha256"], hashlib.sha256(
                (folder / "plan.json").read_bytes()).hexdigest())
            self.assertEqual(metadata["evidence_provenance"][0]
                             ["trusted_implementation_sha256"],
                             "5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0")
            audit = json.loads((folder / "proposal_audit.jsonl").read_text())
            self.assertEqual(audit["prompt"], prompt)
            self.assertEqual(json.loads(audit["raw_response"]), plan)
            self.assertFalse((folder / "candidate.py").exists())
            saved, warnings = planner.load_saved_plans(root)
            self.assertEqual(warnings, [])
            self.assertEqual(saved[0]["plan"], plan)

            repeated = deepcopy(plan)
            repeated["optimization_mechanism"] = "  MEASURE  occupancy and memory-pipeline behavior before another K-loop rewrite  "
            repeated["research_question"] = "Rephrase the same direction?"
            client, _ = client_for(repeated)
            duplicate_folder, duplicate = planner.propose_to_directory(
                client, prompt, context, root)
            self.assertEqual(duplicate["status"], "duplicate_direction")
            self.assertFalse((duplicate_folder / "plan.json").exists())
            self.assertEqual(json.loads((duplicate_folder / "proposal_audit.jsonl").read_text())
                             ["duplicate_of"], saved[0]["artifact_path"])

            different = deepcopy(plan)
            different["optimization_mechanism"] = "Profile cache reuse on the smaller M workload"
            different["target_shape_mnk"] = [128, 3072, 768]
            client, _ = client_for(different)
            _, accepted = planner.propose_to_directory(client, prompt, context, root)
            self.assertEqual(accepted["status"], "advisory_saved")
            self.assertEqual(len(planner.load_saved_plans(root)[0]), 2)
            (folder / "plan.json").write_text("{tampered")
            loaded, warnings = planner.load_saved_plans(root)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(any("mismatched planner artifact" in warning
                                for warning in warnings))

    def test_invalid_response_and_api_failure_keep_sanitized_audits(self):
        context, plan = fixture()
        prompt = planner.build_prompt(context)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = deepcopy(plan)
            bad["supporting_evidence"][0]["id"] = "fabricated"
            client, _ = client_for(bad)
            folder, metadata = planner.propose_to_directory(client, prompt, context, root)
            self.assertEqual(metadata["status"], "invalid_response")
            self.assertFalse((folder / "plan.json").exists())
            self.assertIn("rejection", json.loads((folder / "proposal_audit.jsonl").read_text()))

            def fail(**kwargs):
                raise RuntimeError("secret bearer header")

            client = SimpleNamespace(responses=SimpleNamespace(create=fail))
            folder, metadata = planner.propose_to_directory(client, prompt, context, root)
            self.assertEqual(metadata["status"], "request_failed")
            self.assertNotIn("secret", (folder / "metadata.json").read_text())
            self.assertNotIn("secret", (folder / "proposal_audit.jsonl").read_text())
            self.assertEqual(planner.load_saved_plans(root)[1], [])

    def test_missing_malformed_and_outside_artifacts_are_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                planner.build_context(root)
            base = root / planner.PLAN_SECTION
            bad = base / "20261027T000000000000Z"
            bad.mkdir(parents=True)
            (bad / "metadata.json").write_text("{bad")
            self.assertTrue(planner.load_saved_plans(root)[1])
            (bad / "metadata.json").write_text(json.dumps({"status": "advisory_saved"}))
            self.assertTrue(planner.load_saved_plans(root)[1])
            (bad / "metadata.json").write_text(json.dumps({"status": "prepared_only"}))
            self.assertEqual(planner.load_saved_plans(root), ([], []))
            outside = root.parent / (root.name + "-outside")
            outside.mkdir()
            self.addCleanup(lambda: outside.rmdir())
            (base / "20261027T000000000001Z").symlink_to(outside, target_is_directory=True)
            self.assertEqual(planner.load_saved_plans(root), ([], []))

    def test_dashboard_escapes_saved_model_text_and_never_launches_plan(self):
        context, plan = fixture()
        plan["research_question"] = "<script>alert(1)</script> Why profile?"
        prompt = planner.build_prompt(context)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client, _ = client_for(plan)
            planner.propose_to_directory(client, prompt, context, root)
            history = dashboard_data.load_history(ROOT)
            records = {row["id"]: row for row in history["timeline"]}
            saved = planner.load_saved_plans(root)[0][0]
            self.assertTrue(_planner_references(saved, history, records)[0]["verified"])
            altered = deepcopy(history)
            next(row for row in altered["trusted"]
                 if row["kind"] == "trusted-two-slab")["trusted_sha256"] = "0" * 64
            self.assertFalse(_planner_references(saved, altered, records)[0]["verified"])
            with patch("dashboard_app.load_history", return_value=history), \
                 patch("dashboard_proposals.OpenAI", side_effect=AssertionError("Muse called")), \
                 patch("torch.cuda.is_available", side_effect=AssertionError("GPU called")):
                browser = create_app(root).test_client()
                for view in ("overview", "timeline", "performance", "reliability"):
                    self.assertEqual(browser.get(f"/?view={view}").status_code, 200)
                page = browser.get("/?view=overview")
            self.assertIn(b"RESEARCH PLANNER", page.data)
            self.assertIn(b"&lt;script&gt;alert(1)&lt;/script&gt;", page.data)
            self.assertNotIn(b"<script>alert(1)</script>", page.data)
            self.assertIn(b"ADVISORY", page.data)
            self.assertIn(b"original muse sha", page.data.lower())
            self.assertIn(b"Screening loss", page.data)
            self.assertIn(b"24576", page.data)
            self.assertIn(b"49152", page.data)

    def test_prepare_only_cli_never_imports_model_client(self):
        context, _ = fixture()
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(planner, "ROOT", Path(temp)), \
             patch.object(planner, "build_context", return_value=context):
            self.assertEqual(planner.main(["--prepare-only"]), 0)
            folders = list((Path(temp) / planner.PLAN_SECTION).iterdir())
            self.assertEqual(len(folders), 1)
            self.assertTrue((folders[0] / "prompt.txt").exists())
            self.assertFalse((folders[0] / "plan.json").exists())
            self.assertFalse((folders[0] / "proposal_audit.jsonl").exists())
        with self.assertRaises(SystemExit):
            planner.main([])


if __name__ == "__main__":
    unittest.main()
