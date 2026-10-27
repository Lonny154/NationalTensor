"""Bounded correctness, compiler, and optional screening for a trusted K-loop variant."""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

import torch
import triton

import benchmark_core
import benchmark_proposal
import optimize
import reference_precision
import trusted_two_slab_kernel


ORIGINAL_MUSE_SHA256 = "22ce595f5eb134442dd186de8a3db1b25c9413391daa230dd9e9f87781da43fc"
ORIGINAL_MUSE_PATH = Path("evaluations/kernel-evolution/20261027T160014125560Z/candidate.py")
SHAPES = ((512, 3072, 768), (509, 3071, 769), (65, 129, 33))
SEEDS = (42, 43)
CONFIG = dict(optimize.BASELINE)
ROOT = Path("evaluations/trusted-two-slab")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _identity():
    if _sha256(ORIGINAL_MUSE_PATH.read_bytes()) != ORIGINAL_MUSE_SHA256:
        raise ValueError("original Muse proposal changed")
    return {
        "provenance": "model_suggested_human_reviewed_implementation",
        "exact_generated_source_evaluated": False,
        "original_muse_candidate_sha256": ORIGINAL_MUSE_SHA256,
        "original_muse_candidate_path": str(ORIGINAL_MUSE_PATH),
        "trusted_implementation_sha256": _sha256(inspect.getsource(
            trusted_two_slab_kernel.matmul_kernel.fn).encode("utf-8")),
        "trusted_module_sha256": _sha256(Path(trusted_two_slab_kernel.__file__).read_bytes()),
        "baseline_kernel_sha256": _sha256(inspect.getsource(
            benchmark_proposal.matmul_kernel.fn).encode("utf-8")),
        "baseline_launcher_sha256": _sha256(inspect.getsource(
            benchmark_proposal.launch).encode("utf-8")),
        "gpu": torch.cuda.get_device_name(0),
        "numeric_config": CONFIG,
        "correctness_shapes_mnk": [list(shape) for shape in SHAPES],
        "correctness_seeds": list(SEEDS),
        "measurement_method": optimize.benchmark_method(8, 7, 1.02),
    }


def _inputs(shape, seed):
    m, n, k = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16,
                    generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16,
                    generator=generator)
    reference, policy = reference_precision.fp16_mm_reference(a, b)
    return a, b, reference, policy


def _capture(folder, name, handle):
    saved = {}
    for kind in ("ttir", "ttgir", "llir", "ptx", "sass", "cubin"):
        try:
            content = handle.asm.get(kind)
        except Exception:
            content = None
        if content is None:
            continue
        data = content if isinstance(content, bytes) else str(content).encode("utf-8")
        path = folder / f"{name}.{kind}"
        path.write_bytes(data)
        saved[kind] = {"path": path.name, "sha256": _sha256(data),
                       "size_bytes": len(data)}
    return {"artifacts": saved,
            "registers_per_thread": getattr(handle, "n_regs", None),
            "spills": getattr(handle, "n_spills", None),
            "shared_memory_bytes": getattr(handle.metadata, "shared", None)}


def check(root=ROOT):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    identity = _identity()
    folder = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    result = {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "incomplete", "correctness": []}
    try:
        primary = None
        for shape in SHAPES:
            for seed in SEEDS:
                a, b, reference, policy = _inputs(shape, seed)
                out = torch.empty((shape[0], shape[1]), device="cuda", dtype=torch.float16)
                row = {"shape_mnk": list(shape), "input_seed": seed,
                       "reference_precision": policy}
                try:
                    handle = trusted_two_slab_kernel.launch(a, b, out, CONFIG)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
                except Exception as exc:
                    row.update({"status": "FAIL", "error":
                                f"{type(exc).__name__}: {str(exc)[:2000]}"})
                    result["correctness"].append(row)
                    result["status"] = "correctness_failed"
                    return folder, result
                row["status"] = "PASS"
                result["correctness"].append(row)
                if shape == SHAPES[0] and seed == SEEDS[0]:
                    primary = (a, b, handle)

        a, b, trusted_handle = primary
        baseline_out = torch.empty((SHAPES[0][0], SHAPES[0][1]),
                                   device="cuda", dtype=torch.float16)
        grid = (triton.cdiv(SHAPES[0][0], CONFIG["block_m"]),
                triton.cdiv(SHAPES[0][1], CONFIG["block_n"]))
        baseline_handle = benchmark_proposal.matmul_kernel[grid](
            a, b, baseline_out, SHAPES[0][0], SHAPES[0][1], SHAPES[0][2],
            CONFIG["block_m"], CONFIG["block_n"], CONFIG["block_k"],
            num_warps=CONFIG["num_warps"], num_stages=CONFIG["num_stages"])
        torch.cuda.synchronize()
        result["compilation"] = {"baseline": "PASS", "trusted": "PASS"}
        result["compiler"] = {
            "baseline": _capture(folder, "baseline", baseline_handle),
            "trusted": _capture(folder, "trusted", trusted_handle),
        }
        result["status"] = "correctness_and_compiler_captured"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
    finally:
        (folder / "check.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return folder, result


def screen(folder):
    previous = json.loads((folder / "check.json").read_text())
    identity = _identity()
    if (previous.get("status") != "correctness_and_compiler_captured"
            or any(previous.get(key) != identity[key] for key in identity)
            or len(previous.get("correctness", [])) != len(SHAPES) * len(SEEDS)
            or any(row["status"] != "PASS" for row in previous["correctness"])):
        raise ValueError("prior trusted correctness and source identities are incompatible")
    output = folder / "screening.json"
    if output.exists():
        raise FileExistsError(output)
    result = {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
              "prior_check_sha256": _sha256((folder / "check.json").read_bytes()),
              "status": "incomplete"}
    try:
        a, b, reference, policy = _inputs(SHAPES[0], SEEDS[0])
        trusted_out = torch.empty_like(reference)
        trusted_two_slab_kernel.launch(a, b, trusted_out, CONFIG)
        torch.cuda.synchronize()
        torch.testing.assert_close(trusted_out, reference, rtol=0.02, atol=0.02)
        result["primary_reference_precision"] = policy
        baseline_out = torch.empty_like(reference)

        def launch(x, y, out, config):
            if out is baseline_out:
                benchmark_proposal.launch(x, y, out, config)
            elif out is trusted_out:
                trusted_two_slab_kernel.launch(x, y, out, config)
            else:
                raise ValueError("unexpected benchmark output")

        result["aa"] = benchmark_core.run_aa_calibration(
            benchmark_proposal.launch, a, b, CONFIG,
            rounds=20, warmups=10, iterations=100, seed=42)
        result["screening"] = optimize.compare(
            a, b, baseline_out, trusted_out, CONFIG, 8, 43,
            warmups=10, iterations=100, launcher=launch,
            baseline_config=CONFIG)
        result["screening"]["candidate_launches"] = ["trusted_two_slab_gemm"]
        result["status"] = "screening_complete_unconfirmed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--out-dir", type=Path, default=ROOT)
    screen_parser = sub.add_parser("screen")
    screen_parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "check":
        folder, result = check(args.out_dir)
        print(f"Saved {result['status']}: {folder / 'check.json'}")
        return 0 if result["status"] == "correctness_and_compiler_captured" else 1
    output, result = screen(args.artifact_dir)
    print(f"Saved {result['status']}: {output}")
    return 0 if result["status"] == "screening_complete_unconfirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
