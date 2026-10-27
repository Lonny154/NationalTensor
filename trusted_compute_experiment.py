"""One trusted, manually reviewed compute experiment for the FC1 GEMM."""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re

import torch
import triton

import benchmark_core
import benchmark_proposal
import optimize
import reference_precision
import trusted_compute_kernel


MUSE_CANDIDATE_SHA256 = "38b23fad21dc8bb6d194239c7ac0d0a20b1c8566de4055442e3e927d8f97e59a"
MUSE_CANDIDATE_PATH = Path("evaluations/kernel-evolution/20261027T150316362990Z/candidate.py")
CORRECTNESS_SHAPES = ((512, 3072, 768), (509, 3071, 769), (65, 129, 33))
CORRECTNESS_SEEDS = (42, 43)
ROOT = Path("evaluations/trusted-compute")
CONFIG = dict(optimize.BASELINE)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


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


def _correctness():
    checks = []
    primary = None
    for shape in CORRECTNESS_SHAPES:
        for seed in CORRECTNESS_SEEDS:
            a, b, reference, policy = _inputs(shape, seed)
            out = torch.empty((shape[0], shape[1]), device="cuda", dtype=torch.float16)
            row = {"shape_mnk": list(shape), "input_seed": seed,
                   "reference_precision": policy}
            try:
                trusted_compute_kernel.launch(a, b, out, CONFIG)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
            except Exception as exc:
                row.update({"status": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:2000]}"})
                checks.append(row)
                return checks, None
            row["status"] = "PASS"
            checks.append(row)
            if shape == CORRECTNESS_SHAPES[0] and seed == CORRECTNESS_SEEDS[0]:
                primary = (a, b, out)
    return checks, primary


def _compiled_artifacts(folder, name, handle):
    assembly = handle.asm
    saved = {}
    for kind in ("ttir", "ttgir", "llir", "ptx"):
        if kind in assembly:
            content = str(assembly[kind])
            filename = f"{name}.{kind}"
            (folder / filename).write_text(content)
            saved[kind] = {"path": filename,
                           "sha256": _sha256(content.encode("utf-8"))}
    ptx = str(assembly.get("ptx", ""))
    instructions = {opcode: len(re.findall(pattern, ptx)) for opcode, pattern in {
        "cp_async": r"\bcp\.async\b",
        "ld_global": r"\bld\.global\b",
        "st_global": r"\bst\.global\b",
        "mma_sync": r"\bmma\.sync\b",
        "ldmatrix": r"\bldmatrix\b",
    }.items()}
    return {"artifacts": saved,
            "registers_per_thread": getattr(handle, "n_regs", None),
            "spills": getattr(handle, "n_spills", None),
            "shared_memory_bytes": getattr(handle.metadata, "shared", None),
            "ptx_instruction_counts": instructions}


def _screen(primary):
    a, b, trusted_out = primary
    baseline_out = torch.empty_like(trusted_out)

    def launch(x, y, out, config):
        if out is baseline_out:
            benchmark_proposal.launch(x, y, out, config)
        elif out is trusted_out:
            trusted_compute_kernel.launch(x, y, out, config)
        else:
            raise ValueError("unexpected output buffer")

    timing = optimize.compare(
        a, b, baseline_out, trusted_out, CONFIG, 8, 43,
        warmups=10, iterations=100, launcher=launch, baseline_config=CONFIG)
    timing["candidate_launches"] = ["trusted_compute_gemm"]
    return timing


def run(root=ROOT):
    if _sha256(MUSE_CANDIDATE_PATH.read_bytes()) != MUSE_CANDIDATE_SHA256:
        raise ValueError("Muse source hash changed; provenance cannot be recorded")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    folder = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "model_suggested_human_reviewed_implementation",
        "exact_generated_source_evaluated": False,
        "muse_candidate_sha256": MUSE_CANDIDATE_SHA256,
        "muse_candidate_path": str(MUSE_CANDIDATE_PATH),
        "trusted_implementation_sha256": _sha256(
            inspect.getsource(trusted_compute_kernel.matmul_kernel.fn).encode("utf-8")),
        "trusted_module_sha256": _sha256(Path(trusted_compute_kernel.__file__).read_bytes()),
        "baseline_kernel_sha256": _sha256(
            inspect.getsource(benchmark_proposal.matmul_kernel.fn).encode("utf-8")),
        "baseline_launcher_sha256": _sha256(
            inspect.getsource(benchmark_proposal.launch).encode("utf-8")),
        "gpu": torch.cuda.get_device_name(0),
        "shape_mnk": list(CORRECTNESS_SHAPES[0]),
        "numeric_config": CONFIG,
        "input_seeds": list(CORRECTNESS_SEEDS),
        "measurement_method": optimize.benchmark_method(8, 7, 1.02),
        "correctness": [],
    }
    try:
        checks, primary = _correctness()
        result["correctness"] = checks
        if primary is None:
            result["status"] = "correctness_failed"
            return folder, result

        a, b, trusted_out = primary
        baseline_out = torch.empty_like(trusted_out)
        m, k = a.shape
        n = b.shape[1]
        grid = (triton.cdiv(m, CONFIG["block_m"]),
                triton.cdiv(n, CONFIG["block_n"]))
        baseline_handle = benchmark_proposal.matmul_kernel[grid](
            a, b, baseline_out, m, n, k,
            CONFIG["block_m"], CONFIG["block_n"], CONFIG["block_k"],
            num_warps=CONFIG["num_warps"], num_stages=CONFIG["num_stages"])
        trusted_handle = trusted_compute_kernel.launch(a, b, trusted_out, CONFIG)
        torch.cuda.synchronize()
        result["compilation"] = {"baseline": "PASS", "trusted": "PASS"}
        result["compiler"] = {
            "baseline": _compiled_artifacts(folder, "baseline", baseline_handle),
            "trusted": _compiled_artifacts(folder, "trusted", trusted_handle),
        }

        result["aa"] = benchmark_core.run_aa_calibration(
            benchmark_proposal.launch, a, b, CONFIG,
            rounds=20, warmups=10, iterations=100, seed=42)
        result["screening"] = _screen(primary)
        result["status"] = (
            "screening_promising" if result["screening"]["median_paired_speedup"] > 1.0
            else "screening_no_gain")
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:2000]}"
    finally:
        (folder / "experiment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return folder, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    folder, result = run(args.out_dir)
    print(f"Saved {result['status']}: {folder / 'experiment.json'}")
    return 0 if result["status"] in {"screening_promising", "screening_no_gain"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
