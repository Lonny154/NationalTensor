import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import optimize


class OptimizeIntegrationTests(unittest.TestCase):
    def test_compare_records_paired_evidence_with_old_scalar_fields(self):
        base_out, candidate_out = object(), object()
        calls = []
        samples = {
            "baseline": iter([2.0, 2.0, 2.0, 2.0]),
            "candidate": iter([1.0, 1.0, 1.0, 1.0]),
        }

        def fake_launch(_a, _b, output, _config):
            calls.append(output)

        def fake_timer(function, iterations):
            for _ in range(iterations):
                function()
            side = "baseline" if calls[-1] is base_out else "candidate"
            return next(samples[side])

        with patch.object(optimize, "launch", fake_launch):
            timing = optimize.compare(
                object(), object(), base_out, candidate_out,
                {**optimize.BASELINE, "block_k": 64}, 4, 13,
                warmups=1, iterations=3, timer=fake_timer,
                synchronize=lambda: None,
            )
        self.assertEqual(timing["benchmark_schema_version"], 2)
        self.assertEqual(timing["measurement_mode"], "python_launch_cuda_event")
        self.assertEqual(timing["speedup"], 2.0)
        self.assertEqual(timing["median_paired_speedup"], 2.0)
        self.assertEqual(timing["samples_ms"], {
            "baseline": [2.0] * 4, "candidate": [1.0] * 4})
        self.assertEqual(len(timing["paired_rounds"]), 4)
        self.assertEqual([row["order"] for row in timing["paired_rounds"]].count(
            ["baseline", "candidate"]), 2)
        self.assertEqual(timing["method"]["order_seed"], 13)
        self.assertEqual(timing["method"]["clock"], "injected_timer")
        self.assertIn(base_out, calls)
        self.assertIn(candidate_out, calls)
        json.dumps(timing)

    def test_confirmation_requires_every_balanced_pair_and_threshold(self):
        def rows(candidate_values):
            result = []
            for number, value in enumerate(candidate_values):
                result.extend([
                    {"order": ["baseline", "candidate"],
                     "baseline_ms": 1.0, "candidate_ms": value},
                    {"order": ["candidate", "baseline"],
                     "baseline_ms": 1.0, "candidate_ms": value},
                ])
            return {"paired_rounds": result}

        good = optimize.confirmation_decision(rows([0.97] * 7))
        self.assertTrue(good["confirmed"])
        self.assertEqual(good["pairs_favoring_candidate"], 7)
        self.assertIn("not a statistical significance test", good["interpretation"])
        self.assertFalse(optimize.confirmation_decision(
            rows([0.99] * 7))["confirmed"])
        lost_one = optimize.confirmation_decision(rows([0.97] * 6 + [1.01]))
        self.assertFalse(lost_one["confirmed"])
        self.assertEqual(lost_one["pairs_favoring_candidate"], 6)
        self.assertTrue(optimize.confirmation_decision(
            rows([0.99] * 7), min_speedup=1.005)["confirmed"])
        bad = rows([0.97] * 7)
        bad["paired_rounds"][1]["order"] = ["baseline", "candidate"]
        with self.assertRaisesRegex(ValueError, "order-balanced"):
            optimize.confirmation_decision(bad)

    def test_odd_screening_request_keeps_cli_semantics_and_balances_order(self):
        self.assertEqual(optimize.balanced_rounds(9), 10)
        method = optimize.benchmark_method(10, 7, 1.02)
        self.assertEqual(method["screening"]["rounds"], 10)
        self.assertEqual(method["confirmation"]["rounds"], 14)
        self.assertEqual(method["confirmation"]["iterations_per_block"], 200)

    def test_failed_confirmation_never_produces_confirmed_winner(self):
        winner = {"number": 1, "status": "PASS", "timing": {
            "median_paired_speedup": 1.05}}
        self.assertIs(optimize.select_screening_winner([winner]), winner)
        rows = []
        for value in [0.97] * 6 + [1.01]:
            rows.extend([
                {"order": ["baseline", "candidate"],
                 "baseline_ms": 1.0, "candidate_ms": value},
                {"order": ["candidate", "baseline"],
                 "baseline_ms": 1.0, "candidate_ms": value},
            ])
        summary = {"confirmed_winner": None}
        decision = optimize.apply_confirmation(
            summary, winner, {"paired_rounds": rows}, 7, 1.02)
        self.assertFalse(decision["confirmed"])
        self.assertEqual(summary["confirmation_status"], "unconfirmed")
        self.assertIsNone(summary["confirmed_winner"])
        self.assertEqual(summary["screening_winner"]["optimization_status"],
                         "unconfirmed")

    def test_history_reads_legacy_and_new_timings_without_reinterpreting(self):
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root)
            identity = {
                "gpu": "fake GPU", "shape": [512, 3072, 768],
                "dtype": "fp16", "baseline": optimize.BASELINE,
                "kernel_sha256": "known-hash",
            }
            config = {**optimize.BASELINE, "block_k": 64}
            old = runs / "old"
            old.mkdir()
            (old / "metadata.json").write_text(json.dumps(identity))
            (old / "experiments.jsonl").write_text(json.dumps({
                "number": 1, "config": config, "status": "PASS",
                "timing": {"baseline_ms": 1.0, "candidate_ms": 0.9,
                           "speedup": 1 / 0.9, "samples_ms": {}},
            }) + "\n")
            new = runs / "new"
            new.mkdir()
            (new / "metadata.json").write_text(json.dumps(identity))
            (new / "experiments.jsonl").write_text(json.dumps({
                "number": 1, "config": config, "status": "PASS",
                "timing": {"benchmark_schema_version": 2,
                           "measurement_mode": "python_launch_cuda_event",
                           "baseline_ms": 1.0, "candidate_ms": 0.97,
                           "speedup": 1 / 0.97,
                           "median_paired_speedup": 1 / 0.97,
                           "paired_rounds": []},
            }) + "\n")
            (new / "summary.json").write_text(json.dumps({
                "screening_winner": {"number": 1},
                "confirmation_status": "unconfirmed",
            }))
            history, skipped, legacy = optimize.load_history(runs, identity)

        self.assertEqual((skipped, legacy), (0, 0))
        self.assertEqual(len(history), 2)
        by_run = {item["run"]: item for item in history}
        self.assertEqual(by_run["old"]["measurement_mode"], "legacy_triton_do_bench")
        self.assertEqual(by_run["new"]["measurement_mode"], "python_launch_cuda_event")
        self.assertEqual(by_run["new"]["optimization_status"], "unconfirmed")
        self.assertEqual(by_run["old"]["timing"]["speedup"], 1 / 0.9)
        model_rows = optimize.model_history(history)
        self.assertEqual(next(item for item in model_rows if item["run"] == "new")
                         ["optimization_status"], "unconfirmed")


if __name__ == "__main__":
    unittest.main()
