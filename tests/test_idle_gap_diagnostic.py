import unittest
from unittest.mock import patch

from idle_gap_diagnostic import (
    analyze, analyze_settling, condition_schedule, package_measurement,
    run_settling_blocks, settling_schedule,
)
import optimize


class IdleGapDiagnosticTests(unittest.TestCase):
    def test_schedule_is_seeded_balanced_and_reversed(self):
        schedule = condition_schedule(42)
        self.assertEqual(schedule, condition_schedule(42))
        self.assertEqual(len(schedule), 4)
        self.assertEqual([row["condition"] for row in schedule[:2]],
                         list(reversed([row["condition"] for row in schedule[2:]])))
        for pair in (1, 2):
            self.assertEqual({row["condition"] for row in schedule
                              if row["pair"] == pair}, {"immediate", "idle"})

    def test_settling_schedule_balances_all_four_cells_and_preparation_order(self):
        schedule = settling_schedule(42)
        self.assertEqual(schedule, settling_schedule(42))
        self.assertEqual(len(schedule), 8)
        self.assertEqual([(row["condition"], row["settling"])
                          for row in schedule[:4]],
                         list(reversed([(row["condition"], row["settling"])
                                        for row in schedule[4:]])))
        for pair in (1, 2):
            entries = [row for row in schedule if row["pair"] == pair]
            self.assertEqual({(row["condition"], row["settling"])
                              for row in entries},
                             {("immediate", False), ("idle", False),
                              ("immediate", True), ("idle", True)})
            self.assertEqual(len({row["screen_order_seed"] for row in entries}), 1)
        self.assertEqual(schedule[0]["preparation_order"],
                         list(reversed(schedule[4]["preparation_order"])))

    def test_settling_runs_one_untimed_block_for_each_side(self):
        launches = []
        outputs = {"baseline": object(), "candidate": object()}

        class FakeTimer:
            def __init__(self):
                self.blocks = []

            def __call__(self, function, iterations):
                self_test.assertEqual(iterations, optimize.SCREEN_ITERATIONS)
                for _ in range(iterations):
                    function()
                self.blocks.append({"cuda_event_ms_per_launch": 0.75})

        self_test = self
        timer = FakeTimer()
        with patch.object(optimize, "launch", side_effect=lambda _a, _b, out, config:
                          launches.append((out, config))):
            records = run_settling_blocks(object(), object(), outputs,
                                          ["candidate", "baseline"], timer)
        self.assertEqual([row["side"] for row in records],
                         ["candidate", "baseline"])
        self.assertEqual(len(launches), 2 * optimize.SCREEN_ITERATIONS)
        self.assertTrue(all(out is outputs["candidate"] and config ==
                            {"block_m": 128, "block_n": 128, "block_k": 32,
                             "num_warps": 8, "num_stages": 3}
                            for out, config in launches[:optimize.SCREEN_ITERATIONS]))
        self.assertTrue(all(out is outputs["baseline"] and config == optimize.BASELINE
                            for out, config in launches[optimize.SCREEN_ITERATIONS:]))
        with self.assertRaisesRegex(ValueError, "each side once"):
            run_settling_blocks(None, None, outputs, ["baseline", "baseline"], timer)

    def test_result_preserves_order_raw_ratios_and_first_two_rounds(self):
        timing = {"method": {}, "median_paired_speedup": 1.01,
                  "paired_rounds": [
                      {"round": 1, "order": ["candidate", "baseline"],
                       "candidate_ms": 0.060, "baseline_ms": 0.120,
                       "baseline_over_candidate": 2.0},
                      {"round": 2, "order": ["baseline", "candidate"],
                       "baseline_ms": 0.061, "candidate_ms": 0.062,
                       "baseline_over_candidate": 0.061 / 0.062},
                  ]}
        blocks = [{"cuda_event_ms_per_launch": value,
                   "host_submission_ms_per_block": index + 1.0,
                   "host_wall_ms_per_block": index + 2.0}
                  for index, value in enumerate((0.060, 0.120, 0.061, 0.062))]
        packaged = package_measurement(timing, blocks)
        self.assertIs(packaged["benchmark"], timing)
        self.assertEqual(timing["median_paired_speedup"], 1.01)
        self.assertEqual(timing["paired_rounds"][0]["baseline_over_candidate"], 2.0)
        self.assertIs(timing["paired_rounds"][0]["blocks"]["candidate"], blocks[0])
        self.assertIs(timing["paired_rounds"][0]["blocks"]["baseline"], blocks[1])
        self.assertEqual(packaged["first_two_rounds"][0]["order"],
                         ["candidate", "baseline"])
        self.assertEqual(packaged["first_two_rounds"][0]["blocks"]["baseline"]
                         ["host_submission_ms_per_block"], 2.0)
        with self.assertRaisesRegex(ValueError, "count"):
            package_measurement(timing, blocks[:-1])

    def test_aa_core_result_shape_is_packaged(self):
        timing = {"method": {}, "rounds": [
            {"round": 1, "order": ["baseline", "candidate"],
             "baseline_ms": 0.06, "candidate_ms": 0.06},
        ]}
        blocks = [{"cuda_event_ms_per_launch": 0.06},
                  {"cuda_event_ms_per_launch": 0.06}]
        result = package_measurement(timing, blocks)
        self.assertEqual(result["first_two_rounds"][0]["order"],
                         ["baseline", "candidate"])
        self.assertIs(timing["rounds"][0]["blocks"]["baseline"], blocks[0])

    def test_analysis_counts_first_block_anomalies_without_filtering(self):
        def entry(pair, condition, baseline_us, ratio):
            return {"pair": pair, "condition": condition, "measurement": {
                "first_two_rounds": [{"baseline_ms": baseline_us / 1000,
                                      "blocks": {"baseline": {
                                          "host_submission_ms_per_block": 5.0}}}],
                "benchmark": {"median_paired_speedup": ratio,
                              "paired_rounds": [
                                  {"baseline_ms": baseline_us / 1000},
                                  {"baseline_ms": 0.060},
                                  {"baseline_ms": 0.060},
                                  {"baseline_ms": 0.061},
                              ]},
            }}

        comparisons = [entry(1, "immediate", 60, 0.99),
                       entry(1, "idle", 750, 1.01),
                       entry(2, "idle", 300, 1.03),
                       entry(2, "immediate", 61, 0.98)]
        aa = {stage: {"benchmark": {
            "paired_ratios": {"median": 1.0},
            "aa_variability": {"median_absolute_ratio_deviation": 0.002}}}
              for stage in ("before", "after")}
        result = analyze(comparisons, aa)
        self.assertEqual(result["conditions"]["idle"]
                         ["first_baseline_over_100us_count"], 2)
        self.assertEqual(result["conditions"]["immediate"]
                         ["first_baseline_over_100us_count"], 0)
        self.assertEqual(result["conditions"]["idle"]["first_baseline_us"],
                         [750, 300])
        self.assertEqual(result["conditions"]["idle"]
                         ["later_baseline_median_us"], [60.5, 60.5])
        self.assertAlmostEqual(result["conditions"]["idle"]
                               ["median_paired_speedup_across_comparisons"], 1.02)
        self.assertAlmostEqual(result["idle_minus_immediate_median_paired_speedup"],
                               0.035)
        self.assertEqual([round(row["idle_minus_immediate_paired_speedup"], 3)
                          for row in result["pairwise_differences"]], [0.02, 0.05])

    def test_settling_analysis_keeps_original_and_preparation_timings(self):
        comparisons = []
        for pair in (1, 2):
            for condition in ("immediate", "idle"):
                for prepared in (False, True):
                    first_us = (61 if condition == "immediate" or prepared else 750)
                    preparation_us = 750 if condition == "idle" else 61
                    comparisons.append({
                        "pair": pair, "condition": condition, "settling": prepared,
                        "measurement": {
                            "first_two_rounds": [{"order": ["baseline", "candidate"],
                                                  "baseline_ms": first_us / 1000,
                                                  "candidate_ms": 0.062}],
                            "preparation": {"blocks": ([{"side": "baseline",
                                "timing": {"cuda_event_ms_per_launch":
                                           preparation_us / 1000}},
                                {"side": "candidate", "timing": {
                                    "cuda_event_ms_per_launch": 0.061}}]
                                if prepared else [])},
                            "benchmark": {"paired_ratios": {
                                "median": 0.975 if prepared else 0.977}},
                        },
                    })
        aa = {stage: {"benchmark": {
            "paired_ratios": {"median": 1.0},
            "aa_variability": {"median_absolute_ratio_deviation": 0.001}}}
              for stage in ("before", "after")}
        result = analyze_settling(comparisons, aa)
        self.assertEqual(result["cells"]["idle_original"]
                         ["first_timed_baseline_us"], [750, 750])
        self.assertEqual(result["cells"]["idle_original"]
                         ["first_timed_block_over_100us_count"], 2)
        self.assertEqual(result["cells"]["idle_settled"]
                         ["first_preparation_block_us"], [750, 750])
        self.assertEqual(result["cells"]["idle_settled"]
                         ["preparation_us_by_side"]["baseline"], [750, 750])
        self.assertEqual(result["cells"]["idle_settled"]
                         ["first_timed_baseline_us"], [61, 61])
        self.assertTrue(result["idle_settling_moved_anomaly_in_this_capture"])
        self.assertEqual(result["cells"]["idle_settled"]
                         ["median_paired_speedups"], [0.975, 0.975])


if __name__ == "__main__":
    unittest.main()
