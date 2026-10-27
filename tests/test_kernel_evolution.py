"""Mock-only checks: generated code is never imported by these tests."""

import json
import io
import hashlib
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import kernel_evolution as proposals
import kernel_evolution_eval as worker


def modified_source(version=1):
    source = proposals.baseline_source()
    source = source.replace(
        "    a_ptrs = A + m[:, None] * K + k[None, :]\n"
        "    b_ptrs = B + k[:, None] * N + n[None, :]\n\n", "")
    source = source.replace(
        "    for offset in range(tl.cdiv(K, BK)):\n",
        "    for offset in range(tl.cdiv(K, BK)):\n"
        "        global_k = offset * BK + k\n"
        "        a_ptrs = A + m[:, None] * K + global_k[None, :]\n"
        "        b_ptrs = B + global_k[:, None] * N + n[None, :]\n")
    source = source.replace("\n        a_ptrs += BK\n        b_ptrs += BK * N\n", "")
    if version == 2:
        source = source.replace("range(tl.cdiv(K, BK))",
                                "tl.range(0, tl.cdiv(K, BK), num_stages=3)")
    return source


def response(source, reason="No prior evaluated candidate"):
    return SimpleNamespace(output_text=json.dumps({
        "kernel_source": source,
        "hypothesis": "Reorganize the K loop to improve scheduling",
        "expected_mechanism": "Change instruction organization without changing output math",
        "reasoning_from_feedback": reason,
    }), id="mock-response")


class KernelEvolutionTests(unittest.TestCase):
    def test_cli_requires_explicit_live_request_flag(self):
        with patch.object(proposals, "OpenAI",
                          side_effect=AssertionError("live Muse call")):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    proposals.main(["propose", "--gpu", "mock GPU"])
        self.assertEqual(caught.exception.code, 2)

    def test_modified_source_passes_review_filter_and_unsafe_python_fails(self):
        source = modified_source()
        findings = proposals.inspect_kernel_source(source)
        self.assertEqual(findings["source_sha256"], proposals.source_sha256(source))
        self.assertIn("tl.dot", findings["calls"])
        pipelined = proposals.baseline_source().replace(
            "range(tl.cdiv(K, BK))", "tl.range(0, tl.cdiv(K, BK), num_stages=3)")
        self.assertIn("tl.range", proposals.inspect_kernel_source(pipelined)["calls"])
        bad = (
            proposals.baseline_source(),
            "import os\n" + source,
            source.replace("@triton.jit", "@other.decorator"),
            source.replace("def matmul_kernel(", "def replacement_kernel("),
            source.replace("    pid_m =", "    open('/tmp/x', 'w')\n    pid_m ="),
            source.replace("    pid_m =", "    while True: pass\n    pid_m ="),
            source.replace("    pid_m =", "    tl.float32 += 1\n    pid_m ="),
            source.replace("tl.dot(a, b, acc)", "tl.inline_asm_elementwise(a, b)"),
            source + "\nimport subprocess\n",
        )
        for item in bad:
            with self.subTest(item=item[:60]), self.assertRaises((ValueError, SyntaxError)):
                proposals.inspect_kernel_source(item)

    def test_response_schema_and_prompt_contain_baseline_and_hardware(self):
        obj, findings = proposals.parse_response(response(modified_source()).output_text)
        self.assertEqual(obj["kernel_source"], modified_source())
        self.assertEqual(findings["ast_check"], "accepted_for_review_only")
        example = Path(__file__).resolve().parents[1] / "examples/kernel_evolution_proposal.json"
        example_obj, example_findings = proposals.parse_response(example.read_text())
        self.assertEqual(example_findings["source_sha256"], proposals.source_sha256(
            example_obj["kernel_source"]))
        prompt = proposals.build_prompt(
            "RTX 4060 Ti", {"compute_capability": [8, 9]},
            [{"schedule": "grouped_m4", "median_paired_speedup": 1.005}])
        self.assertIn("def matmul_kernel", prompt)
        self.assertIn("def launch", prompt)
        self.assertIn("RTX 4060 Ti", prompt)
        self.assertIn("grouped_m4", prompt)
        with self.assertRaises(ValueError):
            proposals.parse_response(json.dumps({"kernel_source": modified_source(),
                                                  "hypothesis": "x"}))

    def test_compute_focused_mode_rejects_schedule_only_candidate(self):
        schedule_only = proposals.baseline_source().replace(
            "m = pid_m * BM + tl.arange(0, BM)",
            "m = (pid_m + 0) * BM + tl.arange(0, BM)")
        proposals.inspect_kernel_source(schedule_only)
        with self.assertRaisesRegex(ValueError, "changes only tile scheduling"):
            proposals.parse_response(response(schedule_only).output_text,
                                     require_compute_change=True)
        obj, _ = proposals.parse_response(response(modified_source()).output_text,
                                          require_compute_change=True)
        self.assertEqual(obj["kernel_source"], modified_source())
        swizzle_and_compute = modified_source().replace(
            "pid_m = tl.program_id(0)",
            "pid_m0 = tl.program_id(0)\n    pid_m = pid_m0 + 0")
        with self.assertRaisesRegex(ValueError, "direct 2D program IDs"):
            proposals.parse_response(response(swizzle_and_compute).output_text,
                                     require_compute_change=True)

    def test_independent_request_preserves_review_source_and_never_calls_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prior = root / "prior"
            prior.mkdir()
            source = modified_source()
            digest = proposals.source_sha256(source)
            (prior / "candidate.py").write_text(source)
            (prior / "metadata.json").write_text(json.dumps({
                "status": "awaiting_human_review", "shape_mnk": list(proposals.SHAPE),
                "candidate_sha256": digest, "hypothesis": "prior hypothesis",
                "expected_mechanism": "prior mechanism", "gpu": "mock GPU",
                "hardware": {"compute_capability": [8, 9]}}))
            rejected = response(proposals.baseline_source().replace(
                "    pid_m = tl.program_id(0)",
                "    GROUP_M: tl.constexpr = 8\n    pid_m = tl.program_id(0)"))
            (prior / "proposal_audit.jsonl").write_text(json.dumps({
                "attempt": 1, "raw_response": rejected.output_text,
                "rejection": "ValueError: unsupported syntax: AnnAssign"}) + "\n")
            review = proposals.reviewed_unevaluated_proposal(prior)
            prompt = proposals.build_prompt("mock GPU", {"compute_capability": [8, 9]},
                                            [], review=review)
            self.assertIn("program-ID-only scheduling swizzle", prompt)
            self.assertIn("not compiled, checked for correctness, or timed", prompt)
            self.assertIn("AnnAssign", prompt)
            with patch.object(proposals, "OpenAI",
                              side_effect=AssertionError("live Muse call")):
                output = proposals.prepare_request(prompt, root / "requests", review=review,
                                                   gpu="mock GPU", hardware={})
            self.assertEqual((output / "prompt.txt").read_text(), prompt)
            saved = json.loads((output / "metadata.json").read_text())
            self.assertFalse(saved["api_call_made"])
            self.assertEqual(saved["reviewed_candidate_sha256"], digest)
            self.assertEqual(saved["prompt_sha256"], hashlib.sha256(
                prompt.encode()).hexdigest())
            self.assertEqual((prior / "candidate.py").read_text(), source)
            (prior / "candidate.py").write_text(source + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                proposals.reviewed_unevaluated_proposal(prior)

    def test_trusted_feedback_prompt_keeps_generated_and_reviewed_sources_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prior = root / "prior"
            prior.mkdir()
            source = modified_source()
            generated_sha = proposals.source_sha256(source)
            trusted_sha = "a" * 64
            (prior / "candidate.py").write_text(source)
            (prior / "metadata.json").write_text(json.dumps({
                "status": "awaiting_human_review", "shape_mnk": list(proposals.SHAPE),
                "candidate_sha256": generated_sha, "hypothesis": "prior mask idea",
                "expected_mechanism": "reduce mask work", "gpu": "mock GPU",
                "hardware": {"compute_capability": [8, 9]}}))
            (prior / "proposal_audit.jsonl").write_text("")
            experiment = {
                "provenance": "model_suggested_human_reviewed_implementation",
                "exact_generated_source_evaluated": False,
                "muse_candidate_sha256": generated_sha,
                "trusted_implementation_sha256": trusted_sha,
                "shape_mnk": list(proposals.SHAPE), "gpu": "mock GPU",
                "correctness": [{"status": "PASS", "input_seed": 42}],
                "screening": {
                    "measurement_mode": "python_launch_cuda_event",
                    "baseline_ms": 0.06134, "candidate_ms": 0.06119,
                    "median_paired_speedup": 1.00239,
                    "statistics": {"paired_ratios": {"mad": 0.04186}}},
                "aa": {"aa_variability": {
                    "median_absolute_ratio_deviation": 0.0384,
                    "max_absolute_ratio_deviation": 0.1056}},
                "compiler": {
                    "baseline": {"registers_per_thread": 120, "spills": 0,
                                 "shared_memory_bytes": 24576},
                    "trusted": {"registers_per_thread": 120, "spills": 0,
                                "shared_memory_bytes": 24576}},
            }
            feedback = {
                "report_type": "trusted_implementation_feedback",
                "provenance": experiment["provenance"],
                "exact_generated_source_evaluated": False,
                "original_muse_candidate_sha256": generated_sha,
                "trusted_implementation_sha256": trusted_sha,
                "experiment_artifact": "experiment.json",
                "workload_mnk": list(proposals.SHAPE), "gpu": "mock GPU",
                "correctness": "PASS on reviewed implementation only",
                "screening": {"baseline_median_ms": 0.06134,
                              "trusted_median_ms": 0.06119,
                              "median_paired_speedup": 1.00239,
                              "paired_ratio_mad": 0.04186},
                "aa_calibration": {"median_absolute_ratio_deviation": 0.0384,
                                   "max_absolute_ratio_deviation": 0.1056},
                "compiler_review": {"ptx": "same async-copy and matrix-operation counts",
                                    "resources": "120 registers, 24576 bytes shared"},
                "feedback_for_next_proposal": "No confirmed gain; change data movement.",
            }
            (root / "experiment.json").write_text(json.dumps(experiment))
            feedback_path = root / "feedback.json"
            feedback_path.write_text(json.dumps(feedback))
            review = proposals.reviewed_unevaluated_proposal(prior)
            trusted = proposals.reviewed_trusted_feedback(feedback_path, review)
            prompt = proposals.build_prompt("mock GPU", {}, [], review=review,
                                            trusted_feedback=trusted)
            context = json.loads(prompt.split("\n", 1)[1])
            self.assertIsNone(context["reviewed_generated_candidate_feedback"])
            self.assertEqual(context["prior_unevaluated_review"]["candidate_sha256"],
                             generated_sha)
            external = context["external_trusted_implementation_feedback"]
            self.assertEqual(external["original_muse_candidate_sha256"], generated_sha)
            self.assertEqual(external["trusted_implementation_sha256"], trusted_sha)
            self.assertFalse(external["exact_generated_source_evaluated"])
            self.assertEqual(external["compiler_resources"]["trusted"]
                             ["shared_memory_bytes"], 24576)
            self.assertIn("same async-copy", prompt)
            self.assertIn("0.0384", prompt)
            self.assertIn("data movement", prompt)
            self.assertIn("NOT the exact Muse-generated source", prompt)

            with patch.object(proposals, "OpenAI",
                              side_effect=AssertionError("live Muse call")):
                self.assertEqual(proposals.main([
                    "propose", "--prepare-only", "--out-dir", str(root / "requests"),
                    "--structural-history-dir", str(root / "no-history"),
                    "--review-from", str(prior), "--trusted-feedback",
                    str(feedback_path)]), 0)
            prepared = next((root / "requests").iterdir())
            prepared_meta = json.loads((prepared / "metadata.json").read_text())
            self.assertEqual(prepared_meta["trusted_feedback_sha256"],
                             proposals.source_sha256(feedback_path.read_text()))
            self.assertEqual(prepared_meta["trusted_experiment_sha256"],
                             proposals.source_sha256((root / "experiment.json").read_text()))
            self.assertEqual(prepared_meta["trusted_feedback_provenance"],
                             "model_suggested_human_reviewed_implementation")
            self.assertFalse(prepared_meta["api_call_made"])

            client = SimpleNamespace(responses=SimpleNamespace(
                create=lambda **_kw: response(modified_source(2))))
            next_folder, next_meta = proposals.propose_to_directory(
                client, prompt, root / "proposals", review_from=prior,
                trusted_feedback=trusted, gpu="mock GPU")
            audit = json.loads((next_folder / "proposal_audit.jsonl").read_text())
            self.assertEqual(audit["prompt"], prompt)
            self.assertEqual(next_meta["review_from"], str(prior))
            self.assertIsNone(next_meta["feedback_from"])
            self.assertEqual(next_meta["trusted_implementation_sha256"], trusted_sha)

            feedback["original_muse_candidate_sha256"] = "b" * 64
            feedback_path.write_text(json.dumps(feedback))
            with self.assertRaisesRegex(ValueError, "provenance"):
                proposals.reviewed_trusted_feedback(feedback_path, review)

    def test_two_slab_feedback_reaches_mocked_proposal_with_distinct_sources(self):
        evidence_dir = (Path(__file__).resolve().parents[1] / "evaluations" /
                        "trusted-two-slab" / "20261027T161358957319Z")
        prior_dir = (Path(__file__).resolve().parents[1] / "evaluations" /
                     "kernel-evolution" / "20261027T160014125560Z")
        review = proposals.reviewed_unevaluated_proposal(prior_dir)
        trusted = proposals.reviewed_trusted_feedback(evidence_dir / "feedback.json", review)
        self.assertFalse(trusted["exact_generated_source_evaluated"])
        self.assertEqual(trusted["original_muse_candidate_sha256"],
                         "22ce595f5eb134442dd186de8a3db1b25c9413391daa230dd9e9f87781da43fc")
        self.assertEqual(trusted["trusted_implementation_sha256"],
                         "5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0")
        self.assertEqual(trusted["matmul_inspector_static_instruction_counts"]["ptx"],
                         {"baseline": 397, "trusted": 498})
        prompt = proposals.build_prompt("mock GPU", {}, [],
                                        review=review, trusted_feedback=trusted)
        context = json.loads(prompt.split("\n", 1)[1])
        external = context["external_trusted_implementation_feedback"]
        self.assertEqual(external["dynamic_profile"]["trusted"]
                         ["sm__sass_inst_executed_op_branch.sum"], 9216)
        self.assertEqual(external["compiler_resources"]["trusted"]
                         ["shared_memory_bytes"], 49152)
        self.assertEqual(len(external["correctness_records"]), 6)
        for phrase in ("0.94736x", "4.85%", "129", "49152", "occupancy",
                       "distinguish measured observations from speculative"):
            self.assertIn(phrase, prompt)
        with tempfile.TemporaryDirectory() as temp:
            client = SimpleNamespace(responses=SimpleNamespace(
                create=lambda **_kw: response(modified_source(2))))
            folder, metadata = proposals.propose_to_directory(
                client, prompt, Path(temp), review_from=prior_dir,
                trusted_feedback=trusted, gpu="mock GPU")
            self.assertEqual(metadata["trusted_implementation_sha256"],
                             trusted["trusted_implementation_sha256"])
            self.assertEqual(json.loads((folder / "proposal_audit.jsonl").read_text())
                             ["prompt"], prompt)
            with patch.object(proposals, "OpenAI",
                              side_effect=AssertionError("live Muse call")):
                prepared = proposals.prepare_request(prompt, Path(temp), review=review,
                                                     gpu="mock GPU", hardware={},
                                                     trusted_feedback=trusted)
            self.assertFalse(json.loads((prepared / "metadata.json").read_text())
                             ["api_call_made"])

    def test_accepted_swizzle_cpu_mapping_covers_tiles_but_changes_order(self):
        def remap(pid_m0, pid_n0, num_pid_m, num_pid_n):
            pid = pid_m0 * num_pid_n + pid_n0
            group_width = 4 * num_pid_n
            first_m = (pid // group_width) * 4
            group_size = min(num_pid_m - first_m, 4)
            return (first_m + (pid % group_width) % group_size,
                    (pid % group_width) // group_size)

        for m, n, k in ((512, 3072, 768), (128, 3072, 768),
                        (1024, 1024, 1024)):
            with self.subTest(shape=(m, n, k)):
                gm, gn = (m + 63) // 64, (n + 127) // 128
                launched = [remap(pm, pn, gm, gn)
                            for pn in range(gn) for pm in range(gm)]
                grouped = [remap(pid // gn, pid % gn, gm, gn)
                           for pid in range(gm * gn)]
                self.assertEqual(len(launched), len(set(launched)))
                self.assertEqual(set(launched), {(pm, pn) for pm in range(gm)
                                                 for pn in range(gn)})
                self.assertNotEqual(launched[1], grouped[1])

    def test_two_mocked_proposals_preserve_versions_and_feedback(self):
        prompts = []
        replies = iter((response(modified_source(1)),
                        response(modified_source(2), "Prior compilation failed")))

        def create(**kwargs):
            prompts.append(kwargs["input"])
            return next(replies)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_prompt = proposals.build_prompt("mock GPU", {}, [])
            first, first_meta = proposals.propose_to_directory(
                client, first_prompt, root, gpu="mock GPU", hardware={})
            self.assertEqual(first_meta["status"], "awaiting_human_review")
            self.assertEqual((first / "candidate.py").read_text(), modified_source(1))
            evaluation_path = root / "screening.json"
            evaluation_path.write_text(json.dumps({
                "candidate_sha256": first_meta["candidate_sha256"],
                "status": "failed", "gpu": "mock GPU",
                "compilation": "FAIL", "correctness": [],
                "failure_stage": "compilation", "error": "mock compiler error"}))
            feedback = proposals.reviewed_feedback(first, evaluation_path)
            second_prompt = proposals.build_prompt("mock GPU", {}, [], feedback)
            second, second_meta = proposals.propose_to_directory(
                client, second_prompt, root, feedback_from=first,
                gpu="mock GPU", hardware={})
            audits = [json.loads(line) for line in
                      (second / "proposal_audit.jsonl").read_text().splitlines()]
        second_context = json.loads(prompts[1].split("\n", 1)[1])
        self.assertEqual(second_context["reviewed_generated_candidate_feedback"]
                         ["kernel_source"], modified_source(1))
        self.assertIn("mock compiler error", prompts[1])
        self.assertIn("Reorganize the K loop", prompts[1])
        self.assertEqual(second_meta["feedback_from"], str(first))
        self.assertNotEqual(first_meta["candidate_sha256"],
                            second_meta["candidate_sha256"])
        self.assertEqual(audits[0]["prompt"], second_prompt)
        self.assertEqual(audits[0]["raw_response"], response(
            modified_source(2), "Prior compilation failed").output_text)

    def test_rejected_attempt_is_audited_without_source_execution(self):
        calls = []
        replies = iter((response(proposals.baseline_source()),
                        response(modified_source())))
        client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **kw: (calls.append(kw["input"]), next(replies))[1]))
        with tempfile.TemporaryDirectory() as temp:
            folder, metadata = proposals.propose_to_directory(
                client, "prompt", Path(temp), gpu="mock GPU")
            audits = [json.loads(line) for line in
                      (folder / "proposal_audit.jsonl").read_text().splitlines()]
        self.assertEqual(metadata["status"], "awaiting_human_review")
        self.assertIn("does not modify", audits[0]["rejection"])
        self.assertIn("Previous proposal rejected", calls[1])

    def test_worker_requires_exact_human_approval_and_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = SimpleNamespace(responses=SimpleNamespace(
                create=lambda **_kw: response(modified_source())))
            candidate, meta = proposals.propose_to_directory(
                client, "prompt", root, gpu="mock GPU")
            approval_file = root / "approval.json"
            approval = {"scope": "kernel_evolution_v1",
                        "candidate_sha256": meta["candidate_sha256"],
                        "approved_by": "human reviewer",
                        "approved_at": "2026-10-27T15:00:00+00:00"}
            approval_file.write_text(json.dumps({**approval,
                                                  "candidate_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "explicit approval"):
                worker.verify_approval(candidate, approval_file)
            approval_file.write_text(json.dumps(approval))
            output = root / "evaluation.json"
            with patch.object(worker, "load_reviewed_kernel",
                              side_effect=AssertionError("candidate imported")), \
                    patch.object(worker, "verify_isolation",
                                 side_effect=RuntimeError("isolation unavailable")):
                with self.assertRaisesRegex(RuntimeError, "isolation unavailable"):
                    worker.evaluate(candidate, approval_file, output, mode="screen")
            self.assertFalse(output.exists())
            prior = root / "prior.json"
            prior.write_text(json.dumps({"candidate_sha256": "wrong",
                                         "status": "screening_promising"}))
            with patch.object(worker, "load_reviewed_kernel",
                              side_effect=AssertionError("candidate imported")), \
                    patch.object(worker, "verify_isolation",
                                 return_value={"verified": True}):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    worker.evaluate(candidate, approval_file, output,
                                    mode="confirm", prior_evaluation=prior)
            self.assertFalse(output.exists())

    def test_mocked_approved_evaluation_serializes_feedback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = SimpleNamespace(responses=SimpleNamespace(
                create=lambda **_kw: response(modified_source())))
            candidate, meta = proposals.propose_to_directory(
                client, "prompt", root, gpu="mock GPU")
            approval_file = root / "approval.json"
            approval_file.write_text(json.dumps({
                "scope": "kernel_evolution_v1",
                "candidate_sha256": meta["candidate_sha256"],
                "approved_by": "human reviewer",
                "approved_at": "2026-10-27T15:00:00+00:00"}))
            output = root / "screening.json"
            timing = {"measurement_mode": "python_launch_cuda_event",
                      "median_paired_speedup": 1.01, "baseline_ms": 0.06,
                      "candidate_ms": 0.059,
                      "statistics": {"paired_ratios": {"mad": 0.02, "cv": 0.03}},
                      "paired_rounds": [{"baseline_ms": 0.06,
                                         "candidate_ms": 0.059}]}
            temporary = SimpleNamespace(cleanup=lambda: None)
            with patch.object(worker, "load_reviewed_kernel",
                              return_value=(object(), temporary)), \
                    patch.object(worker, "check_correctness", return_value=(
                        [{"status": "PASS", "shape_mnk": list(proposals.SHAPE),
                          "input_seed": 42}], (object(), object(), object()))), \
                    patch.object(worker.benchmark_core, "run_aa_calibration",
                                 return_value={"mode": "aa"}), \
                    patch.object(worker, "measure_paired", return_value=timing), \
                    patch.object(worker, "verify_isolation",
                                 return_value={"verified": True}), \
                    patch.object(worker.torch.cuda, "get_device_name",
                                 return_value="mock GPU"):
                result = worker.evaluate(candidate, approval_file, output,
                                         mode="screen")
            saved = json.loads(output.read_text())
        self.assertEqual(result["status"], "screening_promising")
        self.assertEqual(saved["candidate_sha256"], meta["candidate_sha256"])
        self.assertEqual(saved["screening"]["paired_rounds"], timing["paired_rounds"])
        self.assertEqual(saved["aa"], {"mode": "aa"})


if __name__ == "__main__":
    unittest.main()
