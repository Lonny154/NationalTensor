"""CPU-only checks for the advisory profiling preparation boundary."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import planned_profile as profile


class PlannedProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename, _, _ in profile.KERNELS.values():
            shutil.copy2(profile.ROOT / filename, self.root / filename)
        plan_dir = Path("evaluations/experiment-planner") / profile.PLAN_RUN
        target = self.root / plan_dir
        target.parent.mkdir(parents=True)
        shutil.copytree(profile.ROOT / plan_dir, target)
        for section, run, artifact in (
            ("trusted-compute", "20261027T154233319712Z", "experiment.json"),
            ("trusted-two-slab", "20261027T161358957319Z", "check.json"),
        ):
            src = profile.ROOT / "evaluations" / section / run / artifact
            dst = self.root / "evaluations" / section / run / artifact
            dst.parent.mkdir(parents=True)
            shutil.copy2(src, dst)

    def test_valid_record_and_source_hashes(self):
        verified = profile.verify_inputs(self.root)
        self.assertEqual(set(verified["identities"]), set(profile.KERNELS))
        self.assertEqual(verified["identities"]["two_slab"]["kernel_sha256"],
                         profile.KERNELS["two_slab"][1])

    def test_plan_hash_rejected(self):
        path = self.root / "evaluations/experiment-planner" / profile.PLAN_RUN / "plan.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash or audit"):
            profile.verify_inputs(self.root)

    def test_plan_kind_shape_and_missing_evidence_rejected(self):
        record = profile.verify_inputs(self.root)["record"]
        from experiment_planner import load_saved_plans
        for mutation in (
            lambda x: x["plan"].update(experiment_kind="kernel_transformation"),
            lambda x: x["plan"].update(target_shape_mnk=[128, 3072, 768]),
            lambda x: x["plan"].update(supporting_evidence=[]),
        ):
            altered = copy.deepcopy(record)
            mutation(altered)
            with self.subTest(plan=altered["plan"]):
                with patch("experiment_planner.load_saved_plans", return_value=([altered], [])):
                    with self.assertRaises(ValueError):
                        profile.verify_inputs(self.root)

    def test_missing_or_changed_trusted_evidence_rejected(self):
        file = self.root / "evaluations/trusted-two-slab/20261027T161358957319Z/check.json"
        saved = file.read_text()
        file.unlink()
        with self.assertRaises(FileNotFoundError):
            profile.verify_inputs(self.root)
        file.write_text(saved.replace("correctness_and_compiler_captured", "wrong", 1))
        with self.assertRaisesRegex(ValueError, "correctness"):
            profile.verify_inputs(self.root)
        data = json.loads(saved)
        data["correctness"][0]["status"] = "FAIL"
        file.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "correctness"):
            profile.verify_inputs(self.root)

    def test_source_identity_rejected(self):
        source = self.root / "trusted_compute_kernel.py"
        source.write_text(source.read_text().replace("num_stages=3", "num_stages=2"))
        with self.assertRaisesRegex(ValueError, "source identity"):
            profile.verify_inputs(self.root)

    def test_profiler_capability_discovery_and_unsupported_metrics(self):
        class Result:
            def __init__(self, text):
                self.stdout, self.stderr, self.returncode = text, "", 0
        values = [Result("Version test\n"),
                  Result("sm__warps_active.avg.pct_of_peak_sustained_active Counter % active\n"),
                  Result("launch__registers_per_thread Ratio register/thread allocated\n")]
        with patch("planned_profile.subprocess.run", side_effect=values) as run:
            found = profile.discover_profiler(executable="/usr/bin/ncu")
        self.assertEqual(found["status"], "ready")
        self.assertEqual(len(found["metrics"]), 2)
        self.assertIn("dram__bytes.sum.per_second", found["unsupported_metrics"])
        self.assertEqual(found["metric_units"]["launch__registers_per_thread"], "register/thread")
        self.assertEqual(run.call_count, 3)
        with patch("planned_profile.shutil.which", return_value=None):
            self.assertEqual(profile.discover_profiler()["status"], "unavailable")

    def test_manifest_commands_unique_and_no_gpu(self):
        fake = {"status": "ready", "executable": "/usr/bin/ncu", "version": "test",
                "chip": "ad106", "metrics": [next(iter(profile.METRICS))],
                "metric_units": {next(iter(profile.METRICS)): "%"},
                "unsupported_metrics": []}
        with patch("planned_profile.subprocess.run") as subprocess_run:
            path, digest = profile.prepare(self.root, self.root / "evaluations/planned-profiling",
                                           fake, "20261028T010000000000Z")
            subprocess_run.assert_not_called()
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["status"], "prepared_only")
        self.assertEqual(len(manifest["captures"]), 6)
        self.assertTrue(all(row["status"] == "not_run" for row in manifest["captures"]))
        commands = (path.parent / "COMMANDS.txt").read_text()
        self.assertIn(digest, commands)
        self.assertIn("timeout 180s", commands)
        self.assertIn("--run-one", commands)
        self.assertNotIn("candidate.py", commands)
        with self.assertRaises(FileExistsError):
            profile.prepare(self.root, self.root / "evaluations/planned-profiling",
                            fake, "20261028T010000000000Z")

    def test_capture_requires_hash_and_rejects_outside_manifest(self):
        with self.assertRaisesRegex(ValueError, "manifest path"):
            profile.checked_manifest(self.root / "fake.json", "a" * 64)
        with self.assertRaisesRegex(ValueError, "manifest path"):
            profile.checked_manifest(self.root / "fake.json", "bad")


if __name__ == "__main__":
    unittest.main()
