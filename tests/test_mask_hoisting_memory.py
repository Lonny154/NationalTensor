"""Experiment 003 ingestion stays read-only and distinguishes code from latency evidence."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import experiment_memory
import experiment_planner as planner
import mask_hoisting_memory as diagnostic


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "diagnostic:mask_hoisting_m128:20261028T145501296300Z"


class MaskHoistingMemoryTests(unittest.TestCase):
    def copy_fixture(self, root):
        shutil.copytree(ROOT / diagnostic.FOLDER, root / diagnostic.FOLDER)
        saved = json.loads((ROOT / diagnostic.FOLDER / "diagnostic.json").read_text())
        identity = saved["source_identity"]
        for relative in (identity["plan_path"], identity["prior_path"], "trusted_compute_kernel.py",
                         "reference_precision.py", "benchmark_core.py", "benchmark_proposal.py"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, path)

    def test_reconstruction_preserves_skipped_status_and_compiler_evidence(self):
        first = experiment_memory.reconstruct(ROOT)
        self.assertEqual(first, experiment_memory.reconstruct(ROOT))
        self.assertEqual(first["warnings"], [])
        entry = next(row for row in first["entries"] if row["id"] == IDENTIFIER)
        self.assertEqual(entry["outcome"], "compiler_equivalent_screening_skipped")
        self.assertEqual(entry["correctness_cases"], [4, 4])
        self.assertIsNone(entry["screening"])
        evidence = entry["diagnostic_evidence"]
        self.assertFalse(evidence["latency_measured"])
        self.assertEqual(evidence["screening_status"], "skipped")
        self.assertIn("conditional timing phase", evidence["screening_skip_reason"])
        self.assertEqual(evidence["executable_section"]["size_bytes"], 6912)
        self.assertTrue(evidence["executable_section"]["identical"])
        self.assertTrue(evidence["constant_data_and_launch_metadata_identical"])
        self.assertEqual(evidence["tolerances"], {"rtol": 0.02, "atol": 0.02})
        self.assertIn("Not measured", entry["performance_interpretation"])
        self.assertEqual(entry["compiler"]["resources"]["baseline"],
                         entry["compiler"]["resources"]["trusted"])

    def test_planner_can_cite_diagnostic_but_cannot_cite_nonexistent_screening(self):
        context = planner.build_context(ROOT)
        entry = next(row for row in context["relevant_experiment_memory"] if row["id"] == IDENTIFIER)
        self.assertEqual(entry["compatibility"]["measurement_method"], "unmeasured")
        self.assertFalse(entry["compatibility"]["timing_directly_comparable"])
        self.assertFalse(entry["screening"])
        plan = json.loads((ROOT / "examples/experiment_planner_plan.json").read_text())
        plan["supporting_evidence"] = [{"id": IDENTIFIER, "artifact_path": entry["artifact_path"],
            "observation_keys": ["diagnostic_evidence", "compiler_resources", "correctness", "outcome"],
            "why_relevant": "Compiler equivalence on M=128 was observed without runtime screening."}]
        self.assertEqual(planner.parse_plan(json.dumps(plan), context["relevant_experiment_memory"]), plan)
        bad = deepcopy(plan)
        bad["supporting_evidence"][0]["observation_keys"] = ["screening"]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            planner.parse_plan(json.dumps(bad), context["relevant_experiment_memory"])
        self.assertIn("not a measured performance failure", planner.build_prompt(context))

    def test_corrupted_records_binary_and_sources_are_not_ingested(self):
        targets = [diagnostic.FOLDER + name for name in diagnostic.HASHES]
        targets += [diagnostic.FOLDER + name for name in
                    ("trusted.cubin", "baseline.sass", "mask_hoisting_m128_experiment.py", "verify_artifacts.py")]
        targets += ["trusted_compute_kernel.py", "reference_precision.py"]
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.copy_fixture(root)
                self.assertEqual(diagnostic.load(root)[1], [])
                path = root / target
                path.write_bytes(path.read_bytes() + b"\n")
                entry, warnings = diagnostic.load(root)
                self.assertIsNone(entry)
                self.assertTrue(warnings)

    def test_ingestion_does_not_import_gpu_runners_or_execute_saved_verifier(self):
        command = [sys.executable, "-c",
                   "import sys; from pathlib import Path; import mask_hoisting_memory as m; "
                   "entry, warnings = m.load(Path('.')); assert entry and not warnings; "
                   "assert 'torch' not in sys.modules; "
                   "assert 'mask_hoisting_m128_experiment' not in sys.modules"]
        subprocess.run(command, cwd=ROOT, check=True, timeout=10, capture_output=True)


if __name__ == "__main__":
    unittest.main()
