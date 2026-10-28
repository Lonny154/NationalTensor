"""Artifact-derived memory tests; generated kernels are never imported."""

import json
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dashboard_app import create_app
from dashboard_data import load_history
from dashboard_proposals import _context as dashboard_request_context
import experiment_memory as memory
import kernel_evolution
import stage_reduction_memory


ROOT = Path(__file__).resolve().parents[1]
TWO_SLAB = "kernel:20261027T160014125560Z"
TRUSTED_TWO_SLAB = "trusted:trusted-two-slab:20261027T161358957319Z"
STAGE_REDUCTION = "stage_reduction:20261028T140056186673Z"
K_BOUND = "kernel:20261027T163630745455Z"


class ExperimentMemoryTests(unittest.TestCase):
    def test_reconstruction_survives_reload_and_keeps_provenance_separate(self):
        first = memory.reconstruct(ROOT)
        second = memory.reconstruct(ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first["warnings"], [])
        by_id = {row["id"]: row for row in first["entries"]}
        generated = by_id[TWO_SLAB]
        trusted = by_id[TRUSTED_TWO_SLAB]
        self.assertEqual(generated["correctness"], "not evaluated")
        self.assertIsNone(generated["screening"])
        self.assertEqual(generated["original_muse_source_sha256"],
                         "22ce595f5eb134442dd186de8a3db1b25c9413391daa230dd9e9f87781da43fc")
        self.assertEqual(trusted["original_muse_source_sha256"],
                         generated["original_muse_source_sha256"])
        self.assertEqual(trusted["trusted_implementation_sha256"],
                         "5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0")
        self.assertFalse(trusted["exact_generated_source_evaluated"])
        self.assertEqual(trusted["correctness"], "PASS")
        self.assertAlmostEqual(trusted["screening"]["median_paired_speedup"], 0.9473613339471774)
        self.assertAlmostEqual(trusted["screening"]["aa_median_absolute_ratio_deviation"],
                               0.04851305314242971)
        self.assertEqual(trusted["compiler"]["resources"]["trusted"]
                         ["shared_memory_bytes"], 49152)
        self.assertEqual(trusted["confirmation"], "not confirmed")
        self.assertEqual(trusted["performance_interpretation"], "Screening loss")
        self.assertEqual(trusted["provenance"],
                         "model_suggested_human_reviewed_implementation")
        tuned = by_id[STAGE_REDUCTION]
        self.assertEqual(tuned["correctness_cases"], [12, 12])
        self.assertEqual(tuned["outcome"], "two_stage_screening_loss")
        self.assertEqual(tuned["confirmation"], "not confirmed")
        self.assertAlmostEqual(tuned["screening"]["median_paired_speedup"],
                               0.951883129051782)
        self.assertLess(tuned["compiler"]["resources"]["stages_2"]
                        ["shared_memory_bytes"],
                        tuned["compiler"]["resources"]["stages_3"]
                        ["shared_memory_bytes"])
        profile = tuned["profiling_evidence"]
        self.assertGreater(profile["metrics"]["stages_2"]
                           ["sm__maximum_warps_per_active_cycle_pct"],
                           profile["metrics"]["stages_3"]
                           ["sm__maximum_warps_per_active_cycle_pct"])
        self.assertIn("not a causal", profile["interpretation"])

    def test_family_comparison_exact_related_and_structure_hint(self):
        remembered = memory.reconstruct(ROOT)
        older = next(entry for entry in remembered["entries"] if entry["id"] == TWO_SLAB)
        exact = memory.compare_candidate(
            remembered, source_sha256=older["original_muse_source_sha256"],
            hypothesis=older["hypothesis"], mechanism=older["mechanism"],
            source=older["source"], shape=older["shape_mnk"], gpu=older["gpu"])
        self.assertEqual(exact["exact_duplicate_of"], TWO_SLAB)
        self.assertIn("k_loop_unrolling", exact["families"])

        changed = older["source"].replace(
            "for p in range(num_pairs):", "for p in tl.range(0, num_pairs, num_stages=2):")
        self.assertNotEqual(changed, older["source"])
        related = memory.compare_candidate(
            remembered, source_sha256=kernel_evolution.source_sha256(changed),
            hypothesis=older["hypothesis"], mechanism=older["mechanism"],
            source=changed, shape=older["shape_mnk"], gpu=older["gpu"])
        self.assertIsNone(related["exact_duplicate_of"])
        self.assertIn(TRUSTED_TWO_SLAB, [row["id"] for row in related["related"]])
        self.assertIn(TWO_SLAB, [row["id"] for row in related["related"]])
        whitespace = older["source"] + "\n"
        hint = memory.compare_candidate(
            remembered, source_sha256=kernel_evolution.source_sha256(whitespace),
            hypothesis=older["hypothesis"], mechanism=older["mechanism"],
            source=whitespace, shape=older["shape_mnk"], gpu=older["gpu"])
        self.assertIsNone(hint["exact_duplicate_of"])
        self.assertTrue(next(row for row in hint["related"] if row["id"] == TWO_SLAB)
                        ["source_structure_match"])
        self.assertEqual(memory.classify_families("an unfamiliar idea"), [])
        unknown = {"entries": [{"id": "kernel:earlier", "kind": "generated_proposal",
                                "original_muse_source_sha256": "1" * 64,
                                "families": [], "source": "x = 1\n",
                                "shape_mnk": [512, 3072, 768], "gpu": "GPU",
                                "outcome": "awaiting review", "screening": None,
                                "performance_interpretation": "Not measured",
                                "artifact_path": "evaluations/kernel-evolution/example/candidate.py"}]}
        ambiguous = memory.compare_candidate(
            unknown, source_sha256=kernel_evolution.source_sha256("x=1\n"),
            hypothesis="an unfamiliar idea", mechanism="unknown mechanism",
            source="x=1\n", shape=(512, 3072, 768), gpu="GPU")
        self.assertIsNone(ambiguous["exact_duplicate_of"])
        self.assertTrue(ambiguous["related"][0]["source_structure_match"])

    def test_stage_reduction_memory_rejects_tampered_identity(self):
        original_read = stage_reduction_memory._read

        def altered(root, relative):
            raw = original_read(root, relative)
            if relative == stage_reduction_memory.ARTIFACT:
                value = json.loads(raw)
                value["plan_sha256"] = "0" * 64
                return json.dumps(value).encode()
            return raw

        with patch("stage_reduction_memory._read", side_effect=altered):
            entry, warnings = stage_reduction_memory.load(ROOT)
        self.assertIsNone(entry)
        self.assertTrue(any("identity mismatch" in warning for warning in warnings))

    def test_k_bound_is_primary_with_mask_hoisting_secondary(self):
        remembered = memory.reconstruct(ROOT)
        entry = next(row for row in remembered["entries"] if row["id"] == K_BOUND)
        self.assertEqual(entry["primary_family"], "k_bound_specialization")
        self.assertIn("mask_hoisting", entry["secondary_families"])
        self.assertEqual(entry["families"][0], "k_bound_specialization")
        relation = memory.compare_candidate(
            remembered, source_sha256=entry["original_muse_source_sha256"],
            hypothesis=entry["hypothesis"], mechanism=entry["mechanism"],
            source=entry["source"], shape=entry["shape_mnk"], gpu=entry["gpu"])
        self.assertEqual(relation["exact_duplicate_of"], K_BOUND)
        self.assertEqual(relation["primary_family"], "k_bound_specialization")
        self.assertIn("mask_hoisting", relation["secondary_families"])
        self.assertEqual(memory.classify_families("an unfamiliar idea"), [])
        unknown = memory.compare_candidate(
            remembered, source_sha256="f" * 64,
            hypothesis="an unfamiliar idea", mechanism="new mechanism",
            source="x = 7", shape=(512, 3072, 768), gpu=entry["gpu"])
        self.assertIsNone(unknown["primary_family"])
        self.assertEqual(unknown["secondary_families"], [])

    def test_prompt_retrieval_includes_actual_two_slab_evidence(self):
        history = load_history(ROOT)
        latest = max((item for item in history["proposals"] if item["kind"] == "kernel"),
                     key=lambda item: item["run_id"])
        review_dir = ROOT / "evaluations" / "kernel-evolution" / latest["run_id"]
        review = kernel_evolution.reviewed_unevaluated_proposal(review_dir)
        context = kernel_evolution.relevant_memory(ROOT, latest["gpu"], review)
        self.assertLessEqual(len(context), 6)
        two_slab = next(item for item in context if item["id"] == TRUSTED_TWO_SLAB)
        self.assertAlmostEqual(two_slab["screening"]["median_paired_speedup"],
                               0.9473613339471774)
        self.assertEqual(two_slab["compiler_resources"]["trusted"]
                         ["registers_per_thread"], 129)
        self.assertEqual(two_slab["compatibility"]["workload"], "same")
        self.assertEqual(two_slab["compatibility"]["measurement_method"], "same")
        self.assertNotIn("source", two_slab)
        prompt = kernel_evolution.build_prompt(
            latest["gpu"], {}, [], review=review, memory_context=context)
        payload = json.loads(prompt.split("\n", 1)[1])
        self.assertEqual(payload["relevant_experiment_memory"], context)
        self.assertIn("5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0",
                      prompt)
        self.assertEqual(two_slab["compiler_resources"]["trusted"]
                         ["shared_memory_bytes"], 49152)
        self.assertIn("trusted result belongs only", prompt)

    def test_cli_prepared_prompt_retrieves_memory_without_model_call(self):
        history = load_history(ROOT)
        latest = max((item for item in history["proposals"] if item["kind"] == "kernel"),
                     key=lambda item: item["run_id"])
        review_dir = ROOT / "evaluations" / "kernel-evolution" / latest["run_id"]
        with tempfile.TemporaryDirectory() as temp, \
             patch("kernel_evolution.OpenAI", side_effect=AssertionError("Muse called")):
            result = kernel_evolution.main([
                "propose", "--prepare-only", "--review-from", str(review_dir),
                "--out-dir", temp, "--structural-history-dir", str(Path(temp) / "none")])
            self.assertEqual(result, 0)
            folder = next(Path(temp).iterdir())
            prompt = (folder / "prompt.txt").read_text()
            payload = json.loads(prompt.split("\n", 1)[1])
            self.assertIn(TRUSTED_TWO_SLAB,
                          [item["id"] for item in payload["relevant_experiment_memory"]])
            self.assertTrue(all("source" not in item
                                for item in payload["relevant_experiment_memory"]))

    def test_dashboard_request_reconstructs_memory_after_restart(self):
        first = dashboard_request_context(ROOT)[3]
        second = dashboard_request_context(ROOT)[3]
        for prompt in (first, second):
            payload = json.loads(prompt.split("\n", 1)[1])
            self.assertIn(TRUSTED_TWO_SLAB,
                          [item["id"] for item in payload["relevant_experiment_memory"]])
        self.assertEqual(first, second)

    def test_incompatible_and_malformed_artifacts_are_not_equated(self):
        trusted = next(item for item in memory.reconstruct(ROOT)["entries"]
                       if item["id"] == TRUSTED_TWO_SLAB)
        incompatible = memory.compatibility(
            trusted, shape=(128, 3072, 768), dtype="fp16", gpu="other GPU",
            method="cuda_graph", baseline_sha256="0" * 64,
            launcher_sha256="0" * 64, numeric_config={"block_m": 16})
        self.assertFalse(incompatible["timing_directly_comparable"])
        self.assertEqual(incompatible["measurement_method"], "different")
        self.assertEqual(incompatible["source_version"], "different")
        altered = deepcopy(load_history(ROOT))
        altered_trusted = next(item for item in altered["trusted"]
                               if item["id"] == TRUSTED_TWO_SLAB)
        altered_trusted["generated_sha256"] = "0" * 64
        self.assertNotIn(TRUSTED_TWO_SLAB,
                         [entry["id"] for entry in memory.from_history(altered)["entries"]])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = root / "evaluations" / "kernel-evolution" / "20261027T000000000000Z"
            bad.mkdir(parents=True)
            (bad / "metadata.json").write_text("{invalid")
            result = memory.reconstruct(root)
            self.assertEqual(result["entries"], [])
            self.assertTrue(result["warnings"])
            source = ROOT / "evaluations" / "kernel-evolution" / "20261027T160014125560Z"
            for name in ("metadata.json", "candidate.py", "proposal_audit.jsonl"):
                shutil.copy2(source / name, bad / name)
            (bad / "candidate.py").write_text("tampered")
            result = memory.reconstruct(root)
            self.assertEqual(result["entries"], [])
            self.assertTrue(any("SHA-256 mismatch" in warning for warning in result["warnings"]))

    def test_dashboard_related_section_uses_saved_relationships(self):
        latest = max((item for item in load_history(ROOT)["proposals"]
                      if item["kind"] == "kernel"), key=lambda item: item["run_id"])
        related = memory.related_for(memory.reconstruct(ROOT), latest["id"])
        self.assertTrue(related)
        page = create_app(ROOT).test_client().get(f"/experiments/{latest['id']}")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"EXPERIMENT MEMORY", page.data)
        self.assertIn(related[0]["id"].encode(), page.data)
        if related[0]["screening"]:
            self.assertIn(f"{related[0]['screening']['median_paired_speedup']:.5f}".encode(),
                          page.data)
        self.assertIn(b"trusted implementation", page.data)
        bound = create_app(ROOT).test_client().get(f"/experiments/{K_BOUND}")
        self.assertIn(b"Primary mechanism: K-bound specialization", bound.data)
        self.assertIn(b"Secondary: Mask hoisting", bound.data)

    def test_distinct_family_revision_is_saved_for_review_with_relationship(self):
        older_run = "20261027T160014125560Z"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prior = root / "evaluations" / "kernel-evolution" / older_run
            prior.mkdir(parents=True)
            original = ROOT / "evaluations" / "kernel-evolution" / older_run
            for name in ("metadata.json", "candidate.py", "proposal_audit.jsonl"):
                shutil.copy2(original / name, prior / name)
            revision = (prior / "candidate.py").read_text().replace(
                "for p in range(num_pairs):",
                "for p in tl.range(0, num_pairs, num_stages=2):")
            self.assertNotEqual(kernel_evolution.source_sha256(revision),
                                json.loads((prior / "metadata.json").read_text())
                                ["candidate_sha256"])
            response = SimpleNamespace(id="mock", output_text=json.dumps({
                "kernel_source": revision,
                "hypothesis": "Two-slab K-loop unrolling with revised operand scheduling",
                "expected_mechanism": "Change K-loop load and dot organization",
                "reasoning_from_feedback": "Prior source was not evaluated",
            }))
            client = SimpleNamespace(responses=SimpleNamespace(
                create=lambda **_kwargs: response))
            folder, metadata = kernel_evolution.propose_to_directory(
                client, "test prompt\n{}", prior.parent, review_from=prior,
                gpu="NVIDIA GeForce RTX 4060 Ti", memory_root=root,
                max_attempts=1)
            self.assertEqual(metadata["status"], "awaiting_human_review")
            self.assertTrue((folder / "candidate.py").exists())
            self.assertIn(f"kernel:{older_run}",
                          [row["id"] for row in metadata["memory_relationships"]])
            self.assertIsNone(json.loads((folder / "proposal_audit.jsonl").read_text())
                              ["memory_relationships"]["exact_duplicate_of"])


if __name__ == "__main__":
    unittest.main()
