"""CPU-only tests for the opt-in reviewed schedule search."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import optimize


class StructuralOptimizeTests(unittest.TestCase):
    def test_proposal_is_strict_json_with_fixed_numeric_config(self):
        tried = {"direct_2d"}
        config, hypothesis = optimize.parse_structural_proposal(
            json.dumps({"schedule": "grouped_m4", "hypothesis": "Reuse B tiles"}), tried)
        self.assertEqual(config, {**optimize.BASELINE, "schedule": "grouped_m4"})
        self.assertEqual(hypothesis, "Reuse B tiles")
        self.assertTrue(optimize.valid_structural_config(config))
        for payload in (
            {"schedule": "direct_2d", "hypothesis": "same mapping"},
            {"schedule": "arbitrary_code", "hypothesis": "run this"},
            {"schedule": "grouped_m4", "hypothesis": " "},
            {"schedule": "grouped_m4", "hypothesis": "reuse", "block_m": 128},
            {"schedule": "grouped_m4", "hypothesis": "reuse", "code": "exec('x')"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                optimize.parse_structural_proposal(json.dumps(payload), tried)
        with self.assertRaises(ValueError):
            optimize.parse_structural_proposal(
                json.dumps({"schedule": "grouped_m4", "hypothesis": "repeat"}),
                {"direct_2d", "grouped_m4"})
        with self.assertRaises(json.JSONDecodeError):
            optimize.parse_structural_proposal("```json\n{}\n```", tried)
        self.assertFalse(optimize.valid_structural_config(
            {**optimize.BASELINE, "block_m": 128, "schedule": "grouped_m4"}))

    def test_muse_retry_is_audited_and_never_changes_numeric_parameters(self):
        responses = iter((
            SimpleNamespace(output_text=json.dumps({
                "schedule": "direct_2d", "hypothesis": "duplicate"}), id="one", usage=None),
            SimpleNamespace(output_text=json.dumps({
                "schedule": "grouped_m4", "hypothesis": "B reuse"}), id="two", usage=None),
        ))
        inputs = []

        def create(**kwargs):
            inputs.append(kwargs)
            return next(responses)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        audit = []
        with patch.object(optimize.torch.cuda, "get_device_name", return_value="fake GPU"):
            config, hypothesis = optimize.propose_structural(
                client, [], {"direct_2d"}, audit=audit.append)
        self.assertEqual(config, {**optimize.BASELINE, "schedule": "grouped_m4"})
        self.assertEqual(hypothesis, "B reuse")
        self.assertEqual(len(inputs), 2)
        self.assertEqual([entry["response_id"] for entry in audit], ["one", "two"])
        self.assertIn("Previous response invalid", inputs[1]["input"])
        self.assertEqual(audit[0]["prompt"], inputs[0]["input"])
        self.assertEqual(audit[1]["response"], json.dumps({
            "schedule": "grouped_m4", "hypothesis": "B reuse"}))

    def test_structural_compare_dispatches_and_keeps_raw_paired_data(self):
        baseline_out, grouped_out = object(), object()
        launches = []

        def fake_launch(_a, _b, output, config):
            launches.append((output, config["schedule"]))

        def fake_timer(function, iterations):
            for _ in range(iterations):
                function()
            return 2.0 if launches[-1][0] is baseline_out else 1.0

        config = {**optimize.BASELINE, "schedule": "grouped_m4"}
        with patch.object(optimize.grouped_matmul, "launch_scheduled", fake_launch):
            timing = optimize.compare_structural(
                object(), object(), baseline_out, grouped_out,
                config, 4, 13, warmups=0, iterations=2,
                timer=fake_timer, synchronize=lambda: None)
        self.assertEqual(timing["baseline_schedule"], "direct_2d")
        self.assertEqual(timing["candidate_schedule"], "grouped_m4")
        self.assertEqual(len(timing["paired_rounds"]), 4)
        self.assertEqual(timing["median_paired_speedup"], 2.0)
        self.assertIn((baseline_out, "direct_2d"), launches)
        self.assertIn((grouped_out, "grouped_m4"), launches)
        self.assertEqual({tuple(row["order"]) for row in timing["paired_rounds"]},
                         {("baseline", "candidate"), ("candidate", "baseline")})

    def test_histories_do_not_reinterpret_missing_schedule(self):
        gpu = "fake GPU"
        numeric_identity = optimize.run_identity(gpu)
        structural_identity = optimize.structural_identity(gpu)
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root)

            def save(name, identity, config):
                folder = runs / name
                folder.mkdir()
                (folder / "metadata.json").write_text(json.dumps(identity))
                (folder / "experiments.jsonl").write_text(json.dumps({
                    "number": 1, "config": config, "status": "PASS",
                    "hypothesis": "test", "timing": {
                        "measurement_mode": "python_launch_cuda_event",
                        "median_paired_speedup": 1.01}}) + "\n")

            save("numeric", numeric_identity, {**optimize.BASELINE, "block_k": 64})
            save("schedule_with_numeric_metadata", numeric_identity,
                 {**optimize.BASELINE, "schedule": "grouped_m4"})
            save("structural", structural_identity,
                 {**optimize.BASELINE, "schedule": "grouped_m4"})
            save("missing_schedule", structural_identity, optimize.BASELINE)
            structural, _, structural_legacy = optimize.load_history(runs, structural_identity)
            numeric, _, _ = optimize.load_history(runs, numeric_identity)
        self.assertEqual([row["run"] for row in structural], ["structural"])
        self.assertEqual(structural[0]["config"]["schedule"], "grouped_m4")
        self.assertEqual(structural_legacy, 0)
        self.assertEqual([row["run"] for row in numeric], ["numeric"])

    def test_prior_grouped_schedule_prevents_another_proposal(self):
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root)
            folder = runs / "prior"
            folder.mkdir()
            (folder / "metadata.json").write_text(json.dumps(
                optimize.structural_identity("fake GPU")))
            (folder / "experiments.jsonl").write_text(json.dumps({
                "config": {**optimize.BASELINE, "schedule": "grouped_m4"},
                "status": "FAIL", "hypothesis": "previous attempt"}) + "\n")
            args = SimpleNamespace(show_history=False)
            with patch.object(optimize, "OpenAI", side_effect=AssertionError("API called")), \
                    patch.object(optimize.torch, "manual_seed",
                                 side_effect=AssertionError("GPU touched")):
                optimize.run_structural(args, "fake GPU", {}, runs)

    def test_structural_runner_saves_policy_schedule_and_raw_timing_without_gpu_or_api(self):
        response = SimpleNamespace(output_text=json.dumps({
            "schedule": "grouped_m4", "hypothesis": "B tile reuse"}),
            id="fake-response", usage=None)
        client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **_kwargs: response))
        policy = {"policy": "fp16_torch_mm_without_reduced_precision_reduction",
                  "settings_before": {"allow_fp16_reduced_precision_reduction": True},
                  "settings_during": {"allow_fp16_reduced_precision_reduction": False},
                  "settings_after": {"allow_fp16_reduced_precision_reduction": True}}
        raw_rows = [{"round": 1, "order": ["baseline", "candidate"],
                     "baseline_ms": 1.0, "candidate_ms": 1.01,
                     "baseline_over_candidate": 1 / 1.01}]
        timing = {"baseline_ms": 1.0, "candidate_ms": 1.01,
                  "speedup": 1 / 1.01, "median_paired_speedup": 1 / 1.01,
                  "baseline_schedule": "direct_2d", "candidate_schedule": "grouped_m4",
                  "paired_rounds": raw_rows}
        configs = []

        def fake_correct(_a, _b, _reference, config):
            configs.append(config)
            return object()

        args = SimpleNamespace(show_history=False, repeats=8,
                               confirmation_pairs=7, min_confirmation_speedup=1.02)
        method = optimize.benchmark_method(8, 7, 1.02)
        identity = optimize.structural_identity("fake GPU")
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(optimize.os.environ, {"MODEL_API_KEY": "test-only"}), \
                patch.object(optimize, "structural_identity", return_value=identity), \
                patch.object(optimize, "OpenAI", return_value=client), \
                patch.object(optimize.torch.cuda, "get_device_name", return_value="fake GPU"), \
                patch.object(optimize.torch, "manual_seed"), \
                patch.object(optimize.torch, "randn", return_value=object()), \
                patch.object(optimize.reference_precision, "fp16_mm_reference",
                             return_value=(object(), policy)), \
                patch.object(optimize, "correct_structural_output", side_effect=fake_correct), \
                patch.object(optimize, "compare_structural", return_value=timing):
            optimize.run_structural(args, "fake GPU", method, Path(root))
            folder = next(Path(root).iterdir())
            metadata = json.loads((folder / "metadata.json").read_text())
            record = json.loads((folder / "experiments.jsonl").read_text())
            summary = json.loads((folder / "summary.json").read_text())
            audit = json.loads((folder / "proposal_audit.jsonl").read_text())
        self.assertEqual([config["schedule"] for config in configs],
                         ["direct_2d", "grouped_m4"])
        self.assertEqual(metadata["reference_precision"], policy)
        self.assertEqual(record["reference_precision"], policy)
        self.assertEqual(record["config"], {**optimize.BASELINE, "schedule": "grouped_m4"})
        self.assertEqual(record["timing"]["paired_rounds"], raw_rows)
        self.assertEqual(record["source_sha256"], metadata["source_sha256"])
        self.assertEqual(summary["confirmation_status"], "not_run")
        self.assertEqual(audit["response_id"], "fake-response")


if __name__ == "__main__":
    unittest.main()
