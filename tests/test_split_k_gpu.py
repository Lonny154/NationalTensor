"""Limited GPU correctness checks for the reviewed Split-K implementation."""

import unittest

import torch

import reference_precision
import split_k_matmul
import structural_loop


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class SplitKCorrectnessTests(unittest.TestCase):
    def test_partial_workspace_is_fully_overwritten_and_output_matches_reference(self):
        shapes = structural_loop.SHAPES
        for shape_name in ("small_m", "fc1", "square"):
            for seed in (42, 43):
                for split_factor in ((2, 4) if shape_name == "small_m" else (2,)):
                    with self.subTest(shape=shape_name, seed=seed,
                                      split_factor=split_factor):
                        m, n, k = shapes[shape_name]
                        generator = torch.Generator(device="cuda")
                        generator.manual_seed(seed)
                        a = torch.randn((m, k), device="cuda", dtype=torch.float16,
                                        generator=generator)
                        b = torch.randn((k, n), device="cuda", dtype=torch.float16,
                                        generator=generator)
                        reference, _ = reference_precision.fp16_mm_reference(a, b)
                        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
                        workspace = split_k_matmul.allocate_workspace(
                            a, b, split_factor)
                        workspace.fill_(float("nan"))
                        config = {**structural_loop.BASELINE, "schedule": "split_k",
                                  "split_factor": split_factor}
                        split_k_matmul.launch_split_k(a, b, out, config, workspace)
                        torch.cuda.synchronize()
                        self.assertTrue(torch.isfinite(workspace).all().item())
                        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)

    def test_reduction_uses_all_fp32_partials_and_overwrites_output(self):
        workspace = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        ], device="cuda", dtype=torch.float32)
        out = torch.full((2, 3), float("nan"), device="cuda", dtype=torch.float16)
        split_k_matmul.reduction_kernel[(1,)](workspace, out, 2, 3, 2, 256,
                                               num_warps=4)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, workspace.sum(0).to(torch.float16),
                                   rtol=0, atol=0)

    def test_paired_adapter_times_complete_split_k_call(self):
        m, n, k = structural_loop.SHAPES["small_m"]
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        reference, _ = reference_precision.fp16_mm_reference(a, b)
        base_out, _ = structural_loop.checked_output(
            a, b, reference, structural_loop.BASELINE)
        config = {**structural_loop.BASELINE, "schedule": "split_k",
                  "split_factor": 2}
        out, workspaces = structural_loop.checked_output(a, b, reference, config)
        timing = structural_loop.compare_config(
            a, b, base_out, out, config, workspaces,
            rounds=2, iterations=2, order_seed=7)
        self.assertEqual(timing["candidate_launches"],
                         ["partial_gemm", "fp32_reduction"])
        self.assertEqual(timing["measurement_mode"], "python_launch_cuda_event")
        self.assertEqual(len(timing["paired_rounds"]), 2)
        self.assertEqual(len(timing["samples_ms"]["candidate"]), 2)


if __name__ == "__main__":
    unittest.main()
