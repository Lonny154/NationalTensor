"""Experiment 004 preparation tests: CPU fixtures, no CUDA or Muse calls."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

import experiment_004_evidence as e004
import experiment_memory
from dashboard_app import create_app
from dashboard_timeline import build_activity
import mask_hoisting_memory as e003
import test_research_workflow as fixtures


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


class PartialBoundaryTests(unittest.TestCase):
    def test_workloads_grid_and_last_tile_correctness(self):
        self.assertEqual(e004.SHAPES["control"], [128, 3072, 768])
        self.assertEqual(e004.SHAPES["partial_m"], [130, 3072, 768])
        self.assertEqual(e004.grid(e004.SHAPES["control"]), [2, 24])
        self.assertEqual(e004.grid(e004.SHAPES["partial_m"]), [3, 24])
        self.assertEqual(e004.CONFIG, {"block_m": 64, "block_n": 128, "block_k": 32,
                                       "num_warps": 4, "num_stages": 3})
        reference = torch.zeros((130, 3072), dtype=torch.float16)
        output = reference.clone()
        report = e004.check_output(output, reference, e004.SHAPES["partial_m"])
        self.assertEqual(report["boundary_start_row"], 128)
        self.assertEqual(report["boundary_row_count"], 2)
        output[129, 7] = 1
        with self.assertRaisesRegex(ValueError, r"\(129, 7\).*bad_elements=1"):
            e004.check_output(output, reference, e004.SHAPES["partial_m"])
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            e004.check_output(output[:128], reference, e004.SHAPES["partial_m"])

    def test_complete_binary_metadata_and_predicate_comparison(self):
        folder = ROOT / e003.FOLDER
        diagnostic = json.loads((folder / "diagnostic.json").read_text())
        compiler = diagnostic["compiler"]
        same = e004.compare_compiler(folder, compiler)
        self.assertTrue(same["executable_identical"])
        self.assertTrue(same["metadata_identical"])
        self.assertFalse(same["meaningful_compiler_difference"])
        self.assertEqual(same["sass_counts"]["baseline"], same["sass_counts"]["trusted"])
        self.assertGreater(same["sass_counts"]["baseline"]["instructions"], 0)
        with tempfile.TemporaryDirectory() as temp:
            copy_folder = Path(temp)
            for name in ("baseline", "trusted"):
                for kind in ("cubin", "sass", "ptx"):
                    shutil.copy2(folder / f"{name}.{kind}", copy_folder / f"{name}.{kind}")
            original = (copy_folder / "trusted.cubin").read_bytes()
            section = e003._sections(original)[".text.matmul_kernel"]
            offset = original.find(section)
            self.assertGreater(offset, 0)
            altered = bytearray(original)
            altered[offset] ^= 1
            (copy_folder / "trusted.cubin").write_bytes(altered)
            changed = e004.compare_compiler(copy_folder, compiler)
            self.assertFalse(changed["executable_identical"])
            self.assertTrue(changed["meaningful_compiler_difference"])
            self.assertTrue(changed["metadata_identical"])

    def fixture(self, root):
        """Synthetic signed-by-manifest structure, never represented as a GPU run."""
        old = ROOT / e003.FOLDER
        shutil.copytree(old, root / e003.FOLDER)
        identity = json.loads((old / "diagnostic.json").read_text())["source_identity"]
        for relative in (identity["plan_path"], identity["prior_path"],
                         "benchmark_proposal.py", "trusted_compute_kernel.py", "reference_precision.py",
                         "benchmark_core.py", "experiment_004_partial_boundary.py"):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        historical, warnings = e003.load(root)
        self.assertEqual(warnings, [])
        stamp = "20261029T170000000000Z"
        folder = root / e004.SECTION / stamp
        folder.mkdir(parents=True)
        for name in ("benchmark_proposal.py", "trusted_compute_kernel.py", "reference_precision.py",
                     "experiment_004_partial_boundary.py"):
            shutil.copy2(root / name, folder / name)
        source = {**identity, "experiment_003_id": historical["id"],
                  "experiment_003_result_sha256": e004.sha((root / historical["artifact_path"]).read_bytes()),
                  "experiment_004_runner_sha256": e004.sha((root / "experiment_004_partial_boundary.py").read_bytes()),
                  "baseline_module_sha256": e004.sha((root / "benchmark_proposal.py").read_bytes()),
                  "variant_tested": "unchanged trusted_compute_kernel.py reviewed mask-hoisting kernel; not Muse-generated source"}
        compiler = json.loads((old / "diagnostic.json").read_text())["compiler"]
        comparisons = {}
        for key, shape in e004.SHAPES.items():
            sub = folder / key
            sub.mkdir()
            for side in ("baseline", "trusted"):
                for artifact in compiler[side]["artifacts"].values():
                    shutil.copy2(old / artifact["path"], sub / artifact["path"])
            comparison = e004.compare_compiler(sub, compiler)
            comparisons[key] = comparison
            write_json(sub / "diagnostic.json", {"shape_mnk": shape, "grid": e004.grid(shape),
                "partial_m_boundary": key == "partial_m", "compiler": compiler, "comparison": comparison})
            checks = [{"implementation": side, "seed": seed, "status": "PASS",
                       "output_shape": shape[:2], "boundary_rows_checked": True,
                       "boundary_start_row": ((shape[0] - 1) // 64) * 64,
                       "boundary_row_count": shape[0] - ((shape[0] - 1) // 64) * 64,
                       "reference_precision": {"policy": "fp16_torch_mm_without_reduced_precision_reduction"}}
                      for side in ("baseline", "trusted") for seed in e004.SEEDS]
            write_json(sub / "correctness.json", {"status": "PASS", "shape_mnk": shape,
                "tolerances": e004.TOLERANCES, "checks": checks})
        write_json(folder / "result.json", {"status": "complete", "created_utc": "2026-10-29T17:00:00+00:00",
            "source_identity": source, "shapes": e004.SHAPES, "config": e004.CONFIG,
            "dtype": "fp16", "input_seeds": list(e004.SEEDS),
            "environment": {"gpu": {"name": "NVIDIA GeForce RTX 4060 Ti"}},
            "runtime_screening": "not_run", "comparisons": comparisons,
            "outcome": "compiler_equivalent_screening_skipped",
            "screening_decision": "No timing was run.", "limitations": ["Synthetic fixture only."]})
        files = {str(path.relative_to(folder)): e004.sha(path.read_bytes())
                 for path in folder.rglob("*") if path.is_file()}
        write_json(folder / "manifest.json", {"schema": "experiment_004_v1", "status": "complete",
            "shapes": e004.SHAPES, "config": e004.CONFIG,
            "files": sorted(files), "artifact_hashes": files})
        return folder

    def test_manifest_ingestion_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.fixture(root)
            entries, warnings = e004.load(root)
            self.assertEqual(warnings, [])
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["shape_mnk"], [130, 3072, 768])
            self.assertEqual(entry["correctness_cases"], [8, 8])
            self.assertIsNone(entry["screening"])
            self.assertFalse(entry["diagnostic_evidence"]["latency_measured"])
            self.assertFalse(entry["exact_generated_source_evaluated"])
            path = folder / "partial_m" / "trusted.cubin"
            path.write_bytes(path.read_bytes() + b"tampered")
            self.assertEqual(e004.load(root)[0], [])
            self.assertIn("rejected", e004.load(root)[1][0])

    def test_manifest_rejects_fabricated_correctness_even_with_rehashed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.fixture(root)
            path = folder / "partial_m" / "correctness.json"
            checks = json.loads(path.read_text())
            checks["checks"][0]["output_shape"] = [128, 3072]
            write_json(path, checks)
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifact_hashes"]["partial_m/correctness.json"] = e004.sha(path.read_bytes())
            write_json(manifest_path, manifest)
            self.assertEqual(e004.load(root)[0], [])
            self.assertIn("correctness", e004.load(root)[1][0])

    def test_validated_entry_joins_existing_compiler_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.fixture(root)
            entry = e004.validate_directory(root, folder.relative_to(root))
            with patch.object(e004, "load", return_value=([entry], [])):
                memory = experiment_memory.reconstruct(ROOT)
                activity = build_activity(ROOT, memory=memory)
            event = next(row for row in activity["events"] if row["id"] == entry["id"])
            self.assertIn("compiler", event["filters"])
            self.assertIn("scientific", event["filters"])
            self.assertEqual(event["title"], "Partial M boundary tile — compiler diagnostic")
            self.assertEqual(event["measurement_label"], "Not measured")
            self.assertIsNone(event["measurement"])

    def test_dashboard_detail_accepts_complete_record_and_rejects_tampered_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = self.fixture(root)
            entry = e004.validate_directory(root, folder.relative_to(root))
            client = create_app(root).test_client()
            url = "/research/evidence/" + entry["id"]
            with patch("dashboard_app.experiment_memory.reconstruct",
                       return_value={"entries": [], "warnings": []}):
                response = client.get(url, base_url=fixtures.BASE_URL)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Control M=128", response.data)
                self.assertIn(b"Partial M=130", response.data)
                self.assertIn(b"Not measured", response.data)
                (folder / "partial_m" / "trusted.cubin").write_bytes(b"tampered")
                self.assertEqual(client.get(url, base_url=fixtures.BASE_URL).status_code, 404)


if __name__ == "__main__":
    unittest.main()
