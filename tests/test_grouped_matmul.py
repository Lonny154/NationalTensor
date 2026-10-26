"""CPU-only checks for the reviewed grouped schedule and benchmark gates."""

import unittest
from unittest.mock import patch

from grouped_matmul import (
    DIRECT_2D, GROUPED_M4, launch_scheduled, tile_for_pid,
)
from benchmark_grouped_m4 import confirm_raw, screening_summary


class GroupedMappingTests(unittest.TestCase):
    def test_every_tile_exactly_once(self):
        # 8x24 is the primary grid; 6x24 has an incomplete final M group.
        # The remaining grids check rectangular and degenerate layouts.
        for num_m, num_n in ((8, 24), (6, 24), (5, 7), (7, 5), (1, 13), (13, 1)):
            with self.subTest(grid=(num_m, num_n)):
                seen = [tile_for_pid(pid, num_m, num_n)
                        for pid in range(num_m * num_n)]
                expected = {(m, n) for m in range(num_m) for n in range(num_n)}
                self.assertEqual(len(seen), len(set(seen)))
                self.assertEqual(set(seen), expected)

    def test_invalid_program_id_or_grid(self):
        for args in ((-1, 6, 24), (144, 6, 24), (0, 0, 24), (0, 6, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tile_for_pid(*args)

    def test_dispatch_requires_explicit_reviewed_schedule(self):
        with patch("grouped_matmul.launch_direct") as direct, \
                patch("grouped_matmul.launch_grouped") as grouped:
            launch_scheduled(None, None, None, {"schedule": DIRECT_2D})
            direct.assert_called_once()
            grouped.assert_not_called()
            launch_scheduled(None, None, None, {"schedule": GROUPED_M4})
            grouped.assert_called_once()
            for config in ({}, {"schedule": "model_supplied_code"}):
                with self.assertRaises(ValueError):
                    launch_scheduled(None, None, None, config)


class BenchmarkGateTests(unittest.TestCase):
    def test_screening_uses_median_paired_ratio(self):
        summary = screening_summary({
            "ratio_of_medians": 0.99,
            "paired_ratios": {"median": 1.01},
        })
        self.assertEqual(summary["ratio_of_medians"], 0.99)
        self.assertEqual(summary["median_paired_speedup"], 1.01)
        self.assertEqual(summary["selection_metric"], "median_paired_speedup")
        self.assertTrue(summary["selected_for_confirmation"])
        self.assertFalse(screening_summary({
            "ratio_of_medians": 1.01,
            "paired_ratios": {"median": 0.99},
        })["selected_for_confirmation"])

    def test_confirmation_uses_existing_seven_pair_rule(self):
        rows = []
        for pair in range(7):
            for order in (["baseline", "candidate"], ["candidate", "baseline"]):
                rows.append({"order": order, "baseline_ms": 1.03,
                             "candidate_ms": 1.0})
        decision = confirm_raw({"rounds": rows})
        self.assertTrue(decision["confirmed"])
        self.assertEqual(decision["pairs_favoring_candidate"], 7)
        rows[0]["baseline_ms"] = 0.90
        rows[1]["baseline_ms"] = 0.90
        self.assertFalse(confirm_raw({"rounds": rows})["confirmed"])


if __name__ == "__main__":
    unittest.main()
