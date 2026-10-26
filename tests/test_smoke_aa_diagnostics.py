from contextlib import contextmanager
import unittest

from smoke_aa import (
    DiagnosticTimer, GraphDiagnosticTimer, attach_diagnostics,
    capture_graphs, comparison_summary, diagnostic_summary,
)


class DiagnosticTimerTests(unittest.TestCase):
    def test_records_distinct_event_submission_and_enclosing_wall_intervals(self):
        calls = []
        timestamps = iter((0, 1_000_000, 4_000_000, 8_000_000))
        events = []

        class FakeEvent:
            def __init__(self):
                self.name = "start" if not events else "end"
                events.append(self)

            def record(self):
                calls.append(f"{self.name}.record")

            def synchronize(self):
                calls.append(f"{self.name}.synchronize")

            def elapsed_time(self, other):
                self_test.assertIs(other, events[1])
                calls.append("elapsed_time")
                return 12.0

        self_test = self
        def clock_ns():
            calls.append("clock")
            return next(timestamps)

        timer = DiagnosticTimer(event_factory=FakeEvent, clock_ns=clock_ns)
        elapsed = timer(lambda: calls.append("launch"), 3)

        self.assertEqual(elapsed, 4.0)
        self.assertEqual(timer.blocks, [{
            "cuda_event_ms_per_block": 12.0,
            "cuda_event_ms_per_launch": 4.0,
            "host_submission_ms_per_block": 3.0,
            "host_wall_ms_per_block": 8.0,
        }])
        self.assertEqual(calls, [
            "clock", "start.record", "clock", "launch", "launch",
            "launch", "clock", "end.record", "end.synchronize",
            "clock", "elapsed_time",
        ])

    def test_blocks_follow_recorded_execution_order_and_form_band_summary(self):
        result = {"method": {}, "rounds": [
            {"round": 1, "order": ["candidate", "baseline"],
             "candidate_ms": 0.059, "baseline_ms": 0.063},
            {"round": 2, "order": ["baseline", "candidate"],
             "baseline_ms": 0.068, "candidate_ms": 0.063},
        ]}
        blocks = [
            {"cuda_event_ms_per_launch": event,
             "cuda_event_ms_per_block": event * 100,
             "host_submission_ms_per_block": submission,
             "host_wall_ms_per_block": wall}
            for event, submission, wall in (
                (0.059, 1.0, 7.0), (0.063, 2.0, 8.0),
                (0.068, 3.0, 9.0), (0.063, 2.0, 8.0),
            )
        ]
        attach_diagnostics(result, blocks)
        self.assertIs(result["rounds"][0]["blocks"]["candidate"], blocks[0])
        self.assertIs(result["rounds"][0]["blocks"]["baseline"], blocks[1])
        self.assertIs(result["rounds"][1]["blocks"]["baseline"], blocks[2])
        self.assertEqual(result["method"]["clock"], "cuda_event")

        summary = diagnostic_summary(result)
        self.assertGreater(summary["submission_vs_event_all_blocks"], 0.99)
        bands = summary["latency_bands_from_prior_aa_trace"]
        self.assertEqual([band["count"] for band in bands], [1, 2, 1])
        self.assertEqual(bands[1]["submission_median_ms_per_block"], 2.0)
        self.assertEqual(bands[1]["wall_median_ms_per_block"], 8.0)

    def test_rejects_missing_or_misaligned_blocks(self):
        result = {"method": {}, "rounds": [
            {"order": ["baseline", "candidate"],
             "baseline_ms": 1.0, "candidate_ms": 2.0},
        ]}
        with self.assertRaisesRegex(ValueError, "count"):
            attach_diagnostics(result, [])
        with self.assertRaisesRegex(ValueError, "does not match"):
            attach_diagnostics(result, [
                {"cuda_event_ms_per_launch": 2.0},
                {"cuda_event_ms_per_launch": 1.0},
            ])

    def test_graph_timer_replays_once_and_retains_100_launch_denominator(self):
        calls = []
        timestamps = iter((0, 1_000_000, 2_000_000, 9_000_000))
        events = []

        class FakeEvent:
            def __init__(self):
                self.name = "start" if not events else "end"
                events.append(self)

            def record(self):
                calls.append(f"{self.name}.record")

            def synchronize(self):
                calls.append("end.synchronize")

            def elapsed_time(self, _other):
                return 8.0

        class FakeGraph:
            def replay(self):
                calls.append("graph.replay")

        def clock_ns():
            calls.append("clock")
            return next(timestamps)

        timer = GraphDiagnosticTimer({"baseline": FakeGraph()},
                                     event_factory=FakeEvent, clock_ns=clock_ns)
        value = timer("baseline", lambda: self.fail("eager launch inside graph timer"), 100)
        self.assertEqual(value, 0.08)
        self.assertEqual(timer.blocks[0]["cuda_event_ms_per_block"], 8.0)
        self.assertEqual(timer.blocks[0]["host_submission_ms_per_block"], 1.0)
        self.assertEqual(timer.blocks[0]["host_wall_ms_per_block"], 9.0)
        self.assertEqual(calls, [
            "clock", "start.record", "clock", "graph.replay", "clock",
            "end.record", "end.synchronize", "clock",
        ])

    def test_graph_capture_has_100_launches_per_distinct_output(self):
        launches = []
        graphs = []
        syncs = []
        outputs = {"baseline": object(), "candidate": object()}

        class FakeGraph:
            def replay(self):
                self.replays += 1

            def __init__(self):
                self.replays = 0
                graphs.append(self)

        @contextmanager
        def capture_context(graph):
            graph.capturing = True
            yield
            graph.capturing = False

        result = capture_graphs(
            lambda _a, _b, output, _config: launches.append(output),
            object(), object(), {"tile": 1}, outputs, 100,
            graph_factory=FakeGraph, capture_context=capture_context,
            synchronize=lambda: syncs.append(True),
        )
        self.assertEqual(len(launches), 200)
        self.assertTrue(all(output is outputs["baseline"] for output in launches[:100]))
        self.assertTrue(all(output is outputs["candidate"] for output in launches[100:]))
        self.assertEqual([graph.replays for graph in graphs], [1, 1])
        self.assertEqual(len(syncs), 2)
        self.assertIsNot(result["baseline"], result["candidate"])

    def test_comparison_keeps_modes_separate(self):
        def capture(ratio, deviation, cv, count):
            return {
                "benchmark": {
                    "paired_ratios": {"median": ratio},
                    "aa_variability": {"median_absolute_ratio_deviation": deviation},
                    "baseline_ms": {"cv": cv}, "candidate_ms": {"cv": cv},
                },
                "diagnostic_summary": {
                    "latency_bands_from_prior_aa_trace": [
                        {"label": "<61 us", "count": count}
                    ],
                    "relative_latency_band_counts": {"within_3_percent_of_mode_median": count},
                },
            }

        summary = comparison_summary({
            "python_launch": capture(1.01, 0.07, 0.05, 2),
            "cuda_graph": capture(1.00, 0.02, 0.01, 4),
        })
        self.assertAlmostEqual(
            summary["graph_minus_python_median_absolute_ratio_deviation"], -0.05
        )
        self.assertEqual(summary["observed_graph_aa_deviation"], "lower")
        self.assertEqual(summary["metrics"]["python_launch"]["baseline_cv"], 0.05)
        self.assertEqual(summary["metrics"]["cuda_graph"]["baseline_cv"], 0.01)


if __name__ == "__main__":
    unittest.main()
