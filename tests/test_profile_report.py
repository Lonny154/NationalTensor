"""CPU-only validation of saved Nsight Compute captures and memory retrieval."""

import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import experiment_memory
import experiment_planner
import planned_profile
import profile_report


class ProfileReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / profile_report.SECTION / profile_report.RUN_ID
        self.folder.parent.mkdir(parents=True)
        shutil.copytree(planned_profile.OUTPUT_BASE / profile_report.RUN_ID, self.folder)
        shutil.copy2(planned_profile.ROOT / "planned_profile.py", self.root / "planned_profile.py")
        for filename, _, _ in planned_profile.KERNELS.values():
            shutil.copy2(planned_profile.ROOT / filename, self.root / filename)
        plan_rel = Path("evaluations/experiment-planner") / planned_profile.PLAN_RUN
        (self.root / plan_rel).parent.mkdir(parents=True)
        shutil.copytree(planned_profile.ROOT / plan_rel, self.root / plan_rel)
        for section, run, artifact in (
            ("trusted-compute", "20261027T154233319712Z", "experiment.json"),
            ("trusted-two-slab", "20261027T161358957319Z", "check.json"),
        ):
            rel = Path("evaluations") / section / run / artifact
            (self.root / rel).parent.mkdir(parents=True)
            shutil.copy2(planned_profile.ROOT / rel, self.root / rel)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())

    def test_six_named_metric_records_and_provenance(self):
        report = profile_report.analyze(self.folder, root=self.root)
        self.assertEqual(report["status"], "profiling_complete_verified")
        self.assertEqual(len(report["captures"]), 6)
        self.assertEqual(report["summary"]["two_slab"][
            "sm__warps_active.avg.pct_of_peak_sustained_active"]["captures"], [15.88, 15.86])
        self.assertEqual(report["summary"]["baseline"][
            "sm__maximum_warps_per_active_cycle_pct"]["captures"], [33.33, 33.33])
        self.assertTrue(all(len(row["metrics"]) == 12 for row in report["captures"]))
        self.assertTrue(all(row["auxiliary_metrics"]["profiler__replayer_passes"]["value"] == 3
                            for row in report["captures"]))
        loaded, warnings = profile_report.load_verified_report(self.root)
        self.assertEqual(warnings, [])
        self.assertEqual(loaded, report)

    def test_missing_metric_or_wrong_unit_is_rejected(self):
        file = self.folder / "baseline-1.csv"
        original = file.read_text()
        file.write_text(original.replace(
            '"sm__warps_active.avg.pct_of_peak_sustained_active"', '"other_metric"', 1))
        with self.assertRaisesRegex(ValueError, "missing metric"):
            profile_report.validate_capture(self.folder, self.manifest,
                                           self.manifest["captures"][0], self.root)
        lines = original.splitlines()
        rows = list(csv.reader(lines[-3:]))
        rows[1][rows[0].index("dram__bytes.sum.per_second")] = "wrong_unit"
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(rows)
        file.write_text("\n".join(lines[:-3]) + "\n" + stream.getvalue())
        with self.assertRaisesRegex(ValueError, "mismatched unit"):
            profile_report.validate_capture(self.folder, self.manifest,
                                           self.manifest["captures"][0], self.root)

    def test_second_gemm_record_is_rejected(self):
        file = self.folder / "baseline-1.csv"
        text = file.read_text()
        file.write_text(text + text.splitlines()[-1] + "\n")
        with self.assertRaisesRegex(ValueError, "one Nsight Compute GEMM"):
            profile_report.validate_capture(self.folder, self.manifest,
                                           self.manifest["captures"][0], self.root)

    def test_replay_metadata_and_selection_mismatch_rejected(self):
        file = self.folder / "two_slab-2.log"
        original = file.read_text()
        file.write_text(original.replace("--profile-from-start off", "--profile-from-start on", 1))
        with self.assertRaisesRegex(ValueError, "capture command"):
            profile_report.validate_capture(self.folder, self.manifest,
                                           self.manifest["captures"][-1], self.root)
        file.write_text(original.replace('"correctness": "PASS"', '"correctness": "FAIL"'))
        with self.assertRaisesRegex(ValueError, "correctness"):
            profile_report.validate_capture(self.folder, self.manifest,
                                           self.manifest["captures"][-1], self.root)

    def test_source_identity_and_raw_hash_mismatch_rejected(self):
        source = self.root / "trusted_two_slab_kernel.py"
        source.write_text(source.read_text().replace("2 * BK", "3 * BK", 1))
        with self.assertRaisesRegex(ValueError, "source identity"):
            profile_report.analyze(self.folder, root=self.root)
        shutil.copy2(planned_profile.ROOT / "trusted_two_slab_kernel.py", source)
        csv_path = self.folder / "baseline-2.csv"
        csv_path.write_text(csv_path.read_text() + "\n")
        report, warnings = profile_report.load_verified_report(self.root)
        self.assertIsNone(report)
        self.assertTrue(warnings)

    def test_memory_and_planner_retrieve_distinct_profile_evidence(self):
        memory = experiment_memory.reconstruct(planned_profile.ROOT)
        entry = next(row for row in memory["entries"] if row["kind"] == "profiling_study")
        self.assertIsNone(entry["screening"])
        self.assertEqual(entry["confirmation"], "not a performance experiment")
        self.assertEqual(entry["profiling_evidence"]["shared_memory_block_limit"]["two_slab"],
                         [2.0, 2.0])
        context = experiment_planner.build_context()
        selected = context["relevant_experiment_memory"][0]
        self.assertEqual(selected["id"], entry["id"])
        self.assertIn("profiling_evidence", selected)
        self.assertEqual(selected["compatibility"]["measurement_method"], "different")
        self.assertFalse(selected["compatibility"]["timing_directly_comparable"])
        prompt = experiment_planner.build_prompt(context)
        self.assertIn("achieved_occupancy_pct", prompt)
        self.assertIn("Nsight Compute replay counters are distinct", prompt)

    def test_missing_report_or_malformed_metadata_is_nonfatal(self):
        (self.folder / "report_metadata.json").unlink()
        report, warnings = profile_report.load_verified_report(self.root)
        self.assertIsNone(report)
        self.assertTrue(warnings)
        (self.folder / "report_metadata.json").write_text("{bad")
        report, warnings = profile_report.load_verified_report(self.root)
        self.assertIsNone(report)
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
