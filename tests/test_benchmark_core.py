import math
import json
import unittest

from benchmark_core import run_aa_calibration, run_paired


class FakeBenchmark:
    def __init__(self, samples):
        self.samples = {name: iter(values) for name, values in samples.items()}
        self.outputs = []
        self.calls = []
        self.syncs = 0

    def output_factory(self, _a, _b):
        output = object()
        self.outputs.append(output)
        return output

    def launch(self, _a, _b, output, config):
        name = "baseline" if output is self.outputs[0] else "candidate"
        self.calls.append((name, config))

    def synchronize(self):
        self.syncs += 1

    def timer(self, function, iterations):
        for _ in range(iterations):
            function()
        name = self.calls[-1][0]
        return next(self.samples[name])

    def run(self, baseline=None, candidate=None, **kwargs):
        return run_paired(
            self.launch, object(), object(), baseline or {"tile": 1},
            candidate or {"tile": 2}, timer=self.timer,
            output_factory=self.output_factory, synchronize=self.synchronize,
            **kwargs,
        )


class PairedBenchmarkTests(unittest.TestCase):
    def test_balanced_orders_priming_warmups_and_statistics(self):
        fake = FakeBenchmark({
            "baseline": [4.0, 4.0, 2.0, 2.0],
            "candidate": [2.0, 2.0, 2.0, 2.0],
        })
        result = fake.run(rounds=4, warmups=2, iterations=3, seed=7)

        orders = [tuple(row["order"]) for row in result["rounds"]]
        self.assertEqual(orders.count(("baseline", "candidate")), 2)
        self.assertEqual(orders.count(("candidate", "baseline")), 2)
        self.assertEqual(result["samples_ms"]["baseline"], [4.0, 4.0, 2.0, 2.0])
        self.assertEqual(result["baseline_ms"]["median"], 3.0)
        self.assertEqual(result["baseline_ms"]["mad"], 1.0)
        self.assertEqual(result["paired_ratios"]["median"], 1.5)
        self.assertEqual(result["ratio_of_medians"], 1.5)
        self.assertEqual(result["paired_ratios"]["p25"], 1.0)
        self.assertEqual([row["baseline_over_candidate"] for row in result["rounds"]],
                         [2.0, 2.0, 1.0, 1.0])
        self.assertEqual(result["method"]["order_seed"], 7)
        self.assertEqual(result["method"]["clock"], "injected_timer")
        json.dumps(result)
        self.assertEqual(len(fake.outputs), 2)
        self.assertIsNot(fake.outputs[0], fake.outputs[1])
        # Each side is primed and warmed before any timed round.
        self.assertEqual(len(fake.calls), 2 * (1 + 2) + 4 * 2 * (2 + 3))
        self.assertEqual(fake.calls[:6], [
            ("baseline", {"tile": 1}), ("baseline", {"tile": 1}),
            ("baseline", {"tile": 1}), ("candidate", {"tile": 2}),
            ("candidate", {"tile": 2}), ("candidate", {"tile": 2}),
        ])
        self.assertEqual(result["method"]["initial_warmups_per_side"], 2)

    def test_same_seed_replays_schedule(self):
        schedules = []
        for _ in range(2):
            fake = FakeBenchmark({"baseline": [1.0] * 6, "candidate": [1.0] * 6})
            result = fake.run(rounds=6, warmups=0, iterations=1, seed=99)
            schedules.append([row["order"] for row in result["rounds"]])
        self.assertEqual(*schedules)

    def test_paired_median_can_differ_from_ratio_of_medians(self):
        fake = FakeBenchmark({
            "baseline": [10.0, 1.0, 10.0, 1.0],
            "candidate": [5.0, 1.0, 1.0, 5.0],
        })
        result = fake.run(rounds=4, warmups=0, iterations=1)
        self.assertEqual(result["paired_ratios"]["median"], 1.5)
        self.assertAlmostEqual(result["ratio_of_medians"], 5.5 / 3.0)

    def test_aa_calibration_uses_distinct_outputs_and_records_variability(self):
        fake = FakeBenchmark({"baseline": [1.0, 1.1], "candidate": [1.0, 1.0]})
        config = {"tile": 1}
        result = run_aa_calibration(
            fake.launch, object(), object(), config, rounds=2, warmups=0,
            iterations=1, timer=fake.timer, output_factory=fake.output_factory,
            synchronize=fake.synchronize,
        )
        self.assertEqual(result["mode"], "aa")
        self.assertIsNot(fake.outputs[0], fake.outputs[1])
        self.assertTrue(all(call[1] == config for call in fake.calls))
        self.assertAlmostEqual(result["aa_variability"]["max_absolute_ratio_deviation"], 0.1)
        self.assertAlmostEqual(result["aa_variability"]["median_absolute_ratio_deviation"], 0.05)

    def test_invalid_arguments_and_timer_values(self):
        fake = FakeBenchmark({"baseline": [1.0] * 4, "candidate": [1.0] * 4})
        for kwargs in ({"rounds": 3}, {"rounds": 0}, {"warmups": -1},
                       {"iterations": 0}, {"seed": -1}, {"rounds": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fake.run(**kwargs)
        with self.assertRaises(ValueError):
            fake.run(mode="aa")
        for bad in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                trial = FakeBenchmark({"baseline": [bad, 1.0], "candidate": [1.0, 1.0]})
                trial.run(rounds=2, warmups=0, iterations=1)
        overflow = FakeBenchmark({"baseline": [1e308, 1.0],
                                  "candidate": [1e-308, 1.0]})
        with self.assertRaisesRegex(ValueError, "ratio is nonfinite"):
            overflow.run(rounds=2, warmups=0, iterations=1)

    def test_rejects_shared_output_buffer(self):
        fake = FakeBenchmark({"baseline": [1.0, 1.0], "candidate": [1.0, 1.0]})
        shared = object()
        with self.assertRaisesRegex(ValueError, "distinct outputs"):
            run_paired(fake.launch, object(), object(), {}, {}, rounds=2,
                       timer=fake.timer, synchronize=fake.synchronize,
                       output_factory=lambda _a, _b: shared)

    def test_preparation_and_side_timer_leave_balanced_warmups_intact(self):
        fake = FakeBenchmark({"baseline": [1.0, 1.0], "candidate": [1.0, 1.0]})
        prepared = []
        measured = []

        def prepare(outputs):
            prepared.append((len(fake.calls), outputs))

        def timer_by_side(name, function, iterations):
            measured.append((name, iterations))
            function()
            return 1.0

        result = run_paired(
            fake.launch, object(), object(), {"tile": 1}, {"tile": 1},
            rounds=2, warmups=2, iterations=100, timer_by_side=timer_by_side,
            prepare=prepare, output_factory=fake.output_factory,
            synchronize=fake.synchronize, mode="aa",
        )
        self.assertEqual(prepared[0][0], 2 * (1 + 2))
        self.assertIsNot(prepared[0][1]["baseline"], prepared[0][1]["candidate"])
        self.assertEqual(len(measured), 4)
        self.assertTrue(all(iterations == 100 for _, iterations in measured))
        self.assertEqual(result["method"]["clock"], "injected_timer")
        self.assertEqual(result["paired_ratios"]["median"], 1.0)
        with self.assertRaisesRegex(ValueError, "timer or timer_by_side"):
            fake.run(timer_by_side=timer_by_side)


if __name__ == "__main__":
    unittest.main()
