"""CPU-only tests for validated proposals, feedback, and isolated artifacts."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import split_k_matmul
import structural_loop as loop


def response(strategy, parameters, reason="prior result was slow"):
    return SimpleNamespace(output_text=json.dumps({
        "strategy": strategy, "parameters": parameters,
        "hypothesis": "Test a reviewed schedule",
        "expected_mechanism": "Change CTA work distribution",
        "reasoning_from_history": reason,
    }), id="mock", usage=None)


class StructuralLoopTests(unittest.TestCase):
    def test_all_shapes_and_strict_configuration_validation(self):
        for shape in loop.SHAPES.values():
            identity = loop.run_identity(shape, "mock GPU", 42)
            self.assertEqual(identity["shape_mnk"], list(shape))
            self.assertEqual(identity["measurement_method"]["measurement_mode"],
                             "python_launch_cuda_event")
        with self.assertRaises(ValueError):
            loop.run_identity((16, 16, 16), "mock GPU", 42)
        self.assertTrue(loop.valid_config(loop.BASELINE))
        self.assertTrue(loop.valid_config({**loop.BASELINE, "schedule": "split_k",
                                           "split_factor": 2}))
        for bad in ({**loop.BASELINE, "schedule": "split_k"},
                    {**loop.BASELINE, "schedule": "split_k", "split_factor": True},
                    {**loop.BASELINE, "schedule": "split_k", "split_factor": 8},
                    {**loop.BASELINE, "block_m": 128},
                    {**loop.BASELINE, "schedule": "arbitrary"}):
            with self.subTest(config=bad):
                self.assertFalse(loop.valid_config(bad))

    def test_proposal_rejects_code_extra_fields_duplicate_and_bad_factor(self):
        tried = {("direct_2d", None)}
        config, explanation = loop.parse_proposal(
            response("split_k", {"split_factor": 4}).output_text, tried)
        self.assertEqual(config["split_factor"], 4)
        self.assertIn("hypothesis", explanation)
        for raw in (
            response("direct_2d", {}).output_text,
            response("split_k", {"split_factor": 8}).output_text,
            response("grouped_m4", {"code": "print(1)"}).output_text,
            response("unknown", {}).output_text,
            response("split_k", {"split_factor": True}).output_text,
            response("split_k", {"split_factor": 2}).output_text[:-1] + ', "code": "x"}',
            "```json\n{}\n```",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                loop.parse_proposal(raw, tried)
        with self.assertRaises(ValueError):
            loop.parse_proposal(response("split_k", {"split_factor": 4}).output_text,
                                tried | {("split_k", 4)})

    def test_feedback_from_result_and_failure_reaches_next_prompt(self):
        identity = loop.run_identity(loop.SHAPES["small_m"], "mock GPU", 42)
        calls = []
        replies = iter((response("grouped_m4", {}),
                        response("split_k", {"split_factor": 2},
                                 "grouped failed correctness")))

        def create(**kwargs):
            calls.append(kwargs)
            return next(replies)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        saved, audits = [], []

        def evaluate(config, number):
            if number == 1:
                return {"status": "FAIL", "correctness": "FAIL",
                        "failure_stage": "correctness", "error": "mock mismatch"}
            return {"status": "PASS", "correctness": "PASS", "timing": {
                "measurement_mode": "python_launch_cuda_event",
                "baseline_ms": 0.06, "candidate_ms": 0.058,
                "median_paired_speedup": 1.02,
                "statistics": {"paired_ratios": {"mad": 0.02, "cv": 0.03}},
                "paired_rounds": [{"order": ["baseline", "candidate"],
                                   "baseline_ms": 0.06, "candidate_ms": 0.058}],
            }}

        rows = loop.run_experiment_loop(
            client, identity, [], {"aa_variability": {
                "median_absolute_ratio_deviation": 0.04}}, 2,
            evaluate, saved.append, audits.append)
        self.assertEqual(rows, saved)
        self.assertEqual([row["schedule"] for row in rows],
                         ["grouped_m4", "split_k"])
        self.assertEqual(rows[1]["optimization_status"], "promising_unconfirmed")
        self.assertEqual(rows[1]["shape_mnk"], [128, 3072, 768])
        self.assertEqual(rows[1]["source_sha256"], identity["source_sha256"])
        self.assertEqual(rows[1]["measurement_method"], identity["measurement_method"])
        second_prompt = calls[1]["input"]
        self.assertIn("mock mismatch", second_prompt)
        self.assertIn('"failure_stage": "correctness"', second_prompt)
        self.assertIn('"aa_median_absolute_ratio_deviation": 0.04', second_prompt)
        self.assertEqual([item["response_id"] for item in audits], ["mock", "mock"])

    def test_measured_feedback_and_rejected_proposal_reach_next_request(self):
        identity = loop.run_identity(loop.SHAPES["fc1"], "mock GPU", 42)
        calls = []
        replies = iter((response("direct_2d", {}),
                        response("grouped_m4", {}),
                        response("split_k", {"split_factor": 2})))

        def create(**kwargs):
            calls.append(kwargs["input"])
            return next(replies)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        rows = loop.run_experiment_loop(
            client, identity, [], None, 2,
            lambda _config, _number: {"status": "PASS", "correctness": "PASS",
                "timing": {"measurement_mode": "python_launch_cuda_event",
                           "baseline_ms": 0.06, "candidate_ms": 0.059,
                           "median_paired_speedup": 1.017,
                           "statistics": {"paired_ratios": {"mad": 0.03,
                                                              "cv": 0.04}}}},
            lambda _record: None, lambda _entry: None)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(calls), 3)
        self.assertIn("Previous proposal rejected", calls[1])
        self.assertIn('"median_paired_speedup": 1.017', calls[2])
        self.assertIn('"paired_ratio_mad": 0.03', calls[2])
        self.assertIn('"optimization_status": "promising_unconfirmed"', calls[2])

    def test_compatible_history_and_serialization_do_not_mix_methods(self):
        identity = loop.run_identity(loop.SHAPES["fc1"], "mock GPU", 42)
        record = {"number": 1, "config": {**loop.BASELINE,
                                         "schedule": "split_k", "split_factor": 2},
                  "status": "PASS", "shape_mnk": identity["shape_mnk"],
                  "source_sha256": identity["source_sha256"],
                  "measurement_method": identity["measurement_method"],
                  "hypothesis": "test", "expected_mechanism": "parallel K"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, metadata, row in (
                ("compatible", identity, record),
                ("wrong_shape", {**identity, "shape_mnk": [1, 2, 3]}, record),
                ("wrong_method", {**identity, "measurement_method": {
                    "measurement_mode": "cuda_graph"}}, record),
                ("wrong_source", {**identity, "source_sha256": {}}, record),
            ):
                folder = root / name
                folder.mkdir()
                (folder / "metadata.json").write_text(json.dumps(metadata))
                loop.append_jsonl(folder / "experiments.jsonl", row)
            loaded = loop.load_compatible_history(root, identity)
            self.assertEqual([row["run"] for row in loaded], ["compatible"])
            self.assertEqual(loaded[0]["config"]["split_factor"], 2)
            self.assertEqual(loaded[0]["measurement_method"],
                             identity["measurement_method"])

    def test_full_mocked_run_writes_reproducible_feedback_artifacts(self):
        identity = loop.run_identity(loop.SHAPES["small_m"], "mock GPU", 42)
        prompts = []
        replies = iter((response("grouped_m4", {}),
                        response("split_k", {"split_factor": 2})))

        def create(**kwargs):
            prompts.append(kwargs["input"])
            return next(replies)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        timing = {"measurement_mode": "python_launch_cuda_event",
                  "baseline_ms": 0.06, "candidate_ms": 0.059,
                  "median_paired_speedup": 1.01,
                  "statistics": {"paired_ratios": {"mad": 0.02, "cv": 0.03}},
                  "paired_rounds": [{"order": ["baseline", "candidate"],
                                     "baseline_ms": 0.06, "candidate_ms": 0.059}]}
        policy = {"policy": loop.REFERENCE_POLICY}
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(loop, "run_identity", return_value=identity), \
                patch.dict(loop.os.environ, {"MODEL_API_KEY": "mock-key"}), \
                patch.object(loop.torch.cuda, "is_available", return_value=True), \
                patch.object(loop.torch.cuda, "get_device_name", return_value="mock GPU"), \
                patch.object(loop, "prepare_workload",
                             return_value=(object(), object(), object(), object(), policy)), \
                patch.object(loop, "measure_aa", return_value={"aa_variability": {
                    "median_absolute_ratio_deviation": 0.04}}), \
                patch.object(loop, "evaluate_gpu", return_value={
                    "status": "PASS", "correctness": "PASS", "timing": timing}), \
                patch.object(loop.optimize, "OpenAI", return_value=client):
            run_dir = loop.run_search("small_m", 2, 42, Path(temp))
            metadata = json.loads((run_dir / "metadata.json").read_text())
            rows = [json.loads(line) for line in
                    (run_dir / "experiments.jsonl").read_text().splitlines()]
            audit = [json.loads(line) for line in
                     (run_dir / "proposal_audit.jsonl").read_text().splitlines()]
            summary = json.loads((run_dir / "summary.json").read_text())
            aa = json.loads((run_dir / "aa.json").read_text())
        self.assertEqual(metadata["shape_mnk"], [128, 3072, 768])
        self.assertEqual(metadata["input_seed"], 42)
        self.assertEqual(metadata["reference_precision"], policy)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["schedule"] for row in rows],
                         ["grouped_m4", "split_k"])
        self.assertEqual(rows[1]["measurement_method"], loop.METHOD)
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[1]["prompt"], prompts[1])
        self.assertIn('"median_paired_speedup": 1.01', prompts[1])
        self.assertEqual(summary["confirmed_winner"], None)
        self.assertEqual(summary["best_screening_experiment"], 1)
        self.assertEqual(aa["aa_variability"]["median_absolute_ratio_deviation"], 0.04)

    def test_workspace_allocation_and_dispatch_require_preallocation(self):
        a = torch.empty((8, 16), dtype=torch.float16)
        b = torch.empty((16, 12), dtype=torch.float16)
        workspace = split_k_matmul.allocate_workspace(a, b, 2)
        self.assertEqual(workspace.shape, (2, 8, 12))
        self.assertEqual(workspace.dtype, torch.float32)
        with self.assertRaises(ValueError):
            split_k_matmul.allocate_workspace(a, b, 3)
        with self.assertRaises(ValueError):
            split_k_matmul.allocate_workspace(a, torch.empty((15, 12)), 2)
        config = {**loop.BASELINE, "schedule": "split_k", "split_factor": 2}
        out = object()
        with self.assertRaisesRegex(ValueError, "preallocated"):
            loop.launch_strategy(a, b, out, config, {})
        with patch.object(split_k_matmul, "launch_split_k") as launch:
            loop.launch_strategy(a, b, out, config, {id(out): workspace})
            launch.assert_called_once_with(a, b, out, config, workspace)

    def test_confirmation_is_separate_and_uses_existing_pair_rule(self):
        identity = loop.run_identity(loop.SHAPES["small_m"], "mock GPU", 42)
        config = {**loop.BASELINE, "schedule": "split_k", "split_factor": 2}
        rounds = [{"order": order, "baseline_ms": 1.04,
                   "candidate_ms": 1.0}
                  for _ in range(7)
                  for order in (["baseline", "candidate"],
                                ["candidate", "baseline"])]
        timing = {"paired_rounds": rounds, "measurement_mode":
                  "python_launch_cuda_event", "candidate_launches":
                  ["partial_gemm", "fp32_reduction"]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            search_run = root / "source"
            search_run.mkdir()
            (search_run / "metadata.json").write_text(json.dumps(identity))
            loop.append_jsonl(search_run / "experiments.jsonl", {
                "number": 1, "config": config, "status": "PASS",
                "timing": {"median_paired_speedup": 1.01},
                "source_sha256": identity["source_sha256"],
                "measurement_method": identity["measurement_method"]})
            with patch.object(loop, "run_identity", return_value=identity), \
                    patch.object(loop.torch.cuda, "get_device_name",
                              return_value="mock GPU"), \
                    patch.object(loop, "prepare_workload",
                                 return_value=(object(), object(), object(),
                                               object(), {"policy": loop.REFERENCE_POLICY})), \
                    patch.object(loop, "checked_output",
                                 return_value=(object(), {})), \
                    patch.object(loop, "measure_aa", return_value={"mode": "aa"}), \
                    patch.object(loop, "compare_config", return_value=timing):
                result = loop.confirm_search_run(search_run, root)
            confirmations = list((root / "confirmations").glob("*/confirmation.json"))
            self.assertEqual(len(confirmations), 1)
            saved = json.loads(confirmations[0].read_text())
        self.assertEqual(result["status"], "heuristically_confirmed")
        self.assertEqual(saved["decision"]["pairs_favoring_candidate"], 7)
        self.assertEqual(saved["timing"]["candidate_launches"],
                         ["partial_gemm", "fp32_reduction"])


if __name__ == "__main__":
    unittest.main()
