from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_search
import optimize


class SearchComparisonTests(unittest.TestCase):
    def test_aa_core_result_is_preserved_at_start_and_end(self):
        plan = compare_search.build_plan(1, 1, 123)["trial_plans"][0]
        timing = {
            "method": {"iterations_per_side_per_round": 100},
            "rounds": [
                {"round": 1, "order": ["baseline", "candidate"],
                 "baseline_ms": 2.0, "candidate_ms": 3.0},
                {"round": 2, "order": ["candidate", "baseline"],
                 "baseline_ms": 4.0, "candidate_ms": 5.0},
            ],
        }
        events = []
        with patch("benchmark_core.run_aa_calibration", return_value=timing) as bench:
            trial = compare_search.run_trial(
                plan,
                lambda _history, _tried, _audit: (
                    compare_search.search_space()[0], "test"),
                lambda _config, _seed: {"status": "FAIL",
                                        "failure_stage": "correctness"},
                lambda _winner, _seed: self.fail("No winner to confirm"),
                lambda _stage, seed: compare_search.measure_aa(object(), object(), seed),
                events.append,
            )
        self.assertEqual(bench.call_count, 2)
        self.assertEqual(trial["status"], "complete")
        for stage in ("start", "end"):
            result = trial["aa"][stage]
            self.assertEqual(result["status"], "PASS")
            self.assertIs(result["timing"], timing)
            self.assertEqual(result["timed_cuda_event_interval_s"], 1.4)
        self.assertEqual([event["stage"] for event in events
                          if event["kind"] == "aa"], ["start", "end"])
        self.assertEqual(timing["rounds"][0]["order"], ["baseline", "candidate"])
        self.assertEqual(compare_search.timed_event_interval_s({
            "method": timing["method"], "paired_rounds": timing["rounds"]}), 1.4)

    def test_space_sampling_and_strategy_schedule_are_reproducible(self):
        first = compare_search.build_plan(4, 4, 20261026)
        self.assertEqual(first, compare_search.build_plan(4, 4, 20261026))
        self.assertNotEqual(first, compare_search.build_plan(4, 4, 20261027))
        self.assertEqual(first["available_nonbaseline_configs"], 107)
        self.assertEqual(len(compare_search.search_space()), 107)
        self.assertNotIn(optimize.BASELINE, compare_search.search_space())
        first_turns = []
        for trial in first["trial_plans"]:
            configurations = trial["random_configs"]
            self.assertEqual(len({optimize.key(config) for config in configurations}), 4)
            self.assertTrue(all(optimize.valid_config(config) for config in configurations))
            for slot in trial["slots"]:
                self.assertEqual(set(slot["order"]), set(compare_search.STRATEGIES))
                first_turns.append(slot["order"][0])
        self.assertEqual(first_turns.count("muse"), 8)
        self.assertEqual(first_turns.count("random"), 8)
        self.assertEqual([trial["confirmation_order"][0]
                          for trial in first["trial_plans"]],
                         ["muse", "random", "muse", "random"])

    def test_trial_histories_are_muse_only_and_reset_between_trials(self):
        plan = compare_search.build_plan(2, 2, 123)
        space = compare_search.search_space()
        seen = []
        events = []

        def proposer(history, tried, audit):
            seen.append((len(history), len(tried),
                         [row["strategy"] for row in history]))
            config = next(config for config in space if optimize.key(config) not in tried)
            audit({"attempt": 1, "model": compare_search.MODEL,
                   "prompt": "exact prompt", "response": "exact response",
                   "api_wall_s": 0.25})
            return config, "test hypothesis"

        def evaluate(_config, _seed):
            return {"status": "PASS", "timing": {"median_paired_speedup": 1.03},
                    "gpu_work_wall_s": 0.5, "screening_wall_s": 0.4}

        def confirm(_winner, _seed):
            return {"status": "confirmed", "best_confirmed_speedup": 1.025,
                    "gpu_work_wall_s": 1.0, "confirmation_wall_s": 0.9,
                    "timing": {"paired_rounds": []}}

        trials = [compare_search.run_trial(
            trial, proposer, evaluate, confirm,
            lambda _stage, _seed: {"status": "PASS", "gpu_work_wall_s": 0.1},
            events.append)
            for trial in plan["trial_plans"]]
        self.assertEqual(seen, [(0, 1, []), (1, 2, ["muse"])] * 2)
        self.assertTrue(all(trial["status"] == "complete" for trial in trials))
        self.assertTrue(all(trial["api_calls"] == 2 for trial in trials))
        self.assertTrue(all(trial["proposal_retries"] == 0 for trial in trials))
        self.assertTrue(all(
            [entry["strategy"] for entry in trial["actual_strategy_execution_order"]]
            == [arm for slot in planned["slots"] for arm in slot["order"]]
            for trial, planned in zip(trials, plan["trial_plans"])))
        self.assertTrue(all(trial["arms"]["muse"]["valid_configurations_evaluated"] == 2
                            and trial["arms"]["random"]["valid_configurations_evaluated"] == 2
                            for trial in trials))
        self.assertEqual(sum(event["kind"] == "api_attempt" for event in events), 4)
        self.assertTrue(all(event["prompt_sha256"] for event in events
                            if event["kind"] == "api_attempt"))

    def test_failures_count_toward_budget_and_retries_are_preserved(self):
        plan = compare_search.build_plan(1, 2, 55)["trial_plans"][0]
        space = compare_search.search_space()
        events = []
        evaluation_count = 0

        def proposer(_history, tried, audit):
            config = next(config for config in space if optimize.key(config) not in tried)
            if len(tried) == 1:
                audit({"attempt": 1, "prompt": "bad proposal", "response": "{}",
                       "api_wall_s": 0.1})
            audit({"attempt": 2, "prompt": "valid proposal", "response": "valid",
                   "api_wall_s": 0.2})
            return config, "hypothesis"

        def evaluate(_config, _seed):
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count == 1:
                return {"status": "FAIL", "failure_stage": "correctness",
                        "gpu_work_wall_s": 0.2}
            return {"status": "PASS", "timing": {"median_paired_speedup": 0.98},
                    "gpu_work_wall_s": 0.4, "screening_wall_s": 0.3}

        trial = compare_search.run_trial(
            plan, proposer, evaluate,
            lambda _winner, _seed: self.fail("No screening winner expected"),
            lambda _stage, _seed: {"status": "PASS"}, events.append)
        self.assertEqual(trial["status"], "complete")
        self.assertEqual(trial["api_calls"], 3)
        self.assertEqual(trial["proposal_retries"], 1)
        self.assertAlmostEqual(trial["api_wall_s"], 0.5)
        self.assertEqual(sum(trial["arms"][arm]["valid_configurations_evaluated"]
                             for arm in compare_search.STRATEGIES), 4)
        self.assertEqual(sum(trial["arms"][arm]["correctness_failures"]
                             for arm in compare_search.STRATEGIES), 1)
        self.assertEqual(trial["arms"]["muse"]["best_confirmed_speedup"], 1.0)
        self.assertEqual(len(trial["arms"]["muse"]["screening_curve"]), 2)
        self.assertEqual(trial["arms"]["muse"]["screening_curve"][0]
                         ["best_vs_baseline"], 1.0)

    def test_incomplete_trial_is_retained_and_excluded_from_paired_outcome(self):
        plan = compare_search.build_plan(1, 2, 66)["trial_plans"][0]
        space = compare_search.search_space()
        events = []

        def proposer(_history, tried, audit):
            audit({"attempt": 1, "prompt": "prompt", "response": "response",
                   "api_wall_s": 0.1})
            if len(tried) > 1:
                raise RuntimeError("proposal exhausted")
            return space[0], "hypothesis"

        trial = compare_search.run_trial(
            plan, proposer,
            lambda _config, _seed: {"status": "PASS",
                                    "timing": {"median_paired_speedup": 1.01}},
            lambda _winner, _seed: self.fail("Incomplete trial must not confirm"),
            lambda _stage, _seed: {"status": "PASS"}, events.append)
        self.assertEqual(trial["status"], "incomplete")
        self.assertEqual(trial["arms"]["muse"]["valid_configurations_evaluated"], 1)
        self.assertEqual(trial["arms"]["random"]["valid_configurations_evaluated"], 2)
        self.assertEqual(trial["api_calls"], 2)
        self.assertEqual(len(trial["actual_strategy_execution_order"]), 3)
        self.assertTrue(any(event["kind"] == "proposal_failure" for event in events))
        aggregate = compare_search.aggregate_trials([trial], 2)
        self.assertEqual(aggregate["completed_paired_trials"], 0)
        self.assertEqual(aggregate["incomplete_trials"][0]["trial"], 1)
        self.assertEqual(aggregate["valid_configurations_evaluated"],
                         {"muse": 1, "random": 2})

    def test_aggregation_uses_completed_paired_trials_and_baseline_fallback(self):
        plan = compare_search.build_plan(1, 1, 77)["trial_plans"][0]
        space = compare_search.search_space()
        trial = compare_search.run_trial(
            plan, lambda _history, _tried, _audit: (space[0], "hypothesis"),
            lambda config, _seed: {"status": "PASS", "timing": {
                "median_paired_speedup": 1.04 if config == space[0] else 0.98}},
            lambda _winner, _seed: {"status": "confirmed", "timing": {},
                                    "best_confirmed_speedup": 1.025},
            lambda _stage, _seed: {"status": "PASS"}, lambda _event: None)
        aggregate = compare_search.aggregate_trials([trial], 1)
        self.assertEqual(aggregate["completed_paired_trials"], 1)
        self.assertEqual(aggregate["paired_trial_outcomes"][0]["muse"], 1.025)
        self.assertEqual(aggregate["paired_trial_outcomes"][0]["random"], 1.0)
        self.assertAlmostEqual(aggregate["mean_paired_difference"], 0.025)
        json.dumps(aggregate)

    def test_proposal_audit_keeps_exact_prompt_response_and_retries(self):
        valid = {**optimize.BASELINE, "num_warps": 8}
        payloads = [" {\"invalid\": true} ", json.dumps({**valid,
                    "hypothesis": "test"})]

        class FakeResponses:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return type("Response", (), {"output_text": payloads[len(self.calls) - 1],
                                             "id": f"response-{len(self.calls)}",
                                             "usage": None})()

        client = type("Client", (), {"responses": FakeResponses()})()
        audit = []
        with patch.object(optimize.torch.cuda, "get_device_name", return_value="fake GPU"), \
             redirect_stdout(StringIO()):
            config, hypothesis = optimize.propose(
                client, [], {optimize.key(optimize.BASELINE)}, audit=audit.append)
        self.assertEqual((config, hypothesis), (valid, "test"))
        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual([entry["prompt"] for entry in audit],
                         [call["input"] for call in client.responses.calls])
        self.assertEqual([entry["response"] for entry in audit], payloads)
        self.assertTrue(all(entry["model"] == compare_search.MODEL for entry in audit))

    def test_api_error_audit_never_saves_exception_text(self):
        class FakeResponses:
            def create(self, **_kwargs):
                raise RuntimeError("secret-token-in-error")

        client = type("Client", (), {"responses": FakeResponses()})()
        audit = []
        with patch.object(optimize.torch.cuda, "get_device_name", return_value="fake GPU"), \
             self.assertRaises(RuntimeError):
            optimize.propose(client, [], {optimize.key(optimize.BASELINE)},
                             audit=audit.append)
        self.assertEqual(audit[0]["error_type"], "RuntimeError")
        self.assertNotIn("secret-token-in-error", json.dumps(audit))

    def test_dry_run_does_not_touch_cuda_api_or_files(self):
        with tempfile.TemporaryDirectory() as root:
            out_dir = Path(root) / "unused"
            output = StringIO()
            with patch.object(compare_search, "run_full", side_effect=AssertionError), \
                 patch.object(optimize.torch.cuda, "is_available", side_effect=AssertionError), \
                 redirect_stdout(output):
                result = compare_search.main(["--dry-run", "--out-dir", str(out_dir)])
            self.assertEqual(result, 0)
            self.assertFalse(out_dir.exists())
            self.assertIn("Trial 4", output.getvalue())
            self.assertIn("muse -> random", output.getvalue())


if __name__ == "__main__":
    unittest.main()
