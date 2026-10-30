"""Experiment 004: compare reviewed mask hoisting at full and partial M tiles.

No timing, screening, Muse request, or generated-source execution is in this runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shutil

import torch

import benchmark_proposal
import experiment_004_evidence as evidence
import mask_hoisting_m128_experiment as experiment_003_runner
from mask_hoisting_memory import load as load_experiment_003
from stage_reduction_experiment import environment
import trusted_compute_experiment as trusted_inputs
import trusted_compute_kernel

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_identity():
    """Bind unchanged reviewed source to the hash-validated Experiment 003 control."""
    prior = experiment_003_runner.identity()
    historical, warnings = load_experiment_003(ROOT)
    if historical is None or warnings:
        raise ValueError("Experiment 003 control evidence unavailable")
    if (prior["trusted_implementation_sha256"] != historical["trusted_implementation_sha256"]
            or prior["baseline_kernel_sha256"] != historical["baseline_sha256"]
            or prior["original_muse_source_sha256"] != historical["original_muse_source_sha256"]):
        raise ValueError("reviewed source differs from Experiment 003")
    return {**prior, "experiment_003_id": historical["id"],
            "experiment_003_result_sha256": evidence.sha((ROOT / historical["artifact_path"]).read_bytes()),
            "experiment_004_runner_sha256": evidence.sha(Path(__file__).read_bytes()),
            "baseline_module_sha256": evidence.sha(Path(benchmark_proposal.__file__).read_bytes()),
            "variant_tested": "unchanged trusted_compute_kernel.py reviewed mask-hoisting kernel; not Muse-generated source"}


def verify_correctness(shape):
    checks = {"status": "incomplete", "shape_mnk": list(shape),
              "tolerances": evidence.TOLERANCES, "checks": []}
    for seed in evidence.SEEDS:
        a, b, reference, policy = trusted_inputs._inputs(shape, seed)
        for name, launch in (("baseline", benchmark_proposal.launch),
                             ("trusted", trusted_compute_kernel.launch)):
            row = {"implementation": name, "seed": seed, "shape_mnk": list(shape),
                   "reference_precision": policy, "status": "incomplete"}
            try:
                # NaN initialization exposes missed stores, especially in rows 128-129.
                out = torch.full_like(reference, float("nan"))
                launch(a, b, out, evidence.CONFIG)
                torch.cuda.synchronize()
                row.update(evidence.check_output(out, reference, shape, **evidence.TOLERANCES))
                row["status"] = "PASS"
            except Exception as exc:
                row.update(status="FAIL", error=f"{type(exc).__name__}: {str(exc)[:2000]}")
                checks["checks"].append(row)
                checks["status"] = "FAIL"
                return checks
            checks["checks"].append(row)
    checks["status"] = "PASS"
    return checks


def capture_shape(folder, shape):
    folder.mkdir()
    m, n, k = shape
    a = torch.empty((m, k), device="cuda", dtype=torch.float16)
    b = torch.empty((k, n), device="cuda", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    diagnostic = {"shape_mnk": list(shape), "grid": evidence.grid(shape),
                  "partial_m_boundary": m % evidence.CONFIG["block_m"] != 0,
                  "compiler": {}}
    for name, kernel in (("baseline", benchmark_proposal.matmul_kernel),
                         ("trusted", trusted_compute_kernel.matmul_kernel)):
        handle = kernel.warmup(a, b, out, m, n, k, 64, 128, 32,
                               num_warps=4, num_stages=3, grid=tuple(diagnostic["grid"]))
        diagnostic["compiler"][name] = experiment_003_runner.capture(folder, name, handle)
        if diagnostic["compiler"][name]["sass_status"] != "captured":
            raise RuntimeError(f"{name} complete decoded SASS unavailable")
    diagnostic["comparison"] = evidence.compare_compiler(folder, diagnostic["compiler"])
    for kind, lines in (("ptx", evidence.normalized_ptx), ("sass", evidence.sass_instructions)):
        baseline = lines((folder / f"baseline.{kind}").read_text())
        candidate = lines((folder / f"trusted.{kind}").read_text())
        (folder / f"{kind}.diff").write_text("\n".join(difflib.unified_diff(
            baseline, candidate, fromfile="baseline", tofile="trusted")) + "\n")
    write_json(folder / "diagnostic.json", diagnostic)
    checks = verify_correctness(shape)
    write_json(folder / "correctness.json", checks)
    if checks["status"] != "PASS":
        raise ValueError(f"correctness failed for M={m}; see {folder / 'correctness.json'}")
    return diagnostic["comparison"]


def run():
    identity = source_identity()
    if not torch.cuda.is_available() or "RTX 4060 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Experiment 004 requires the RTX 4060 Ti CUDA device")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / evidence.SECTION / stamp
    folder.mkdir(parents=True, exist_ok=False)
    result = {"experiment": "004_partial_boundary_tiles", "status": "incomplete",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_identity": identity, "shapes": evidence.SHAPES,
              "config": evidence.CONFIG, "dtype": "fp16", "input_seeds": list(evidence.SEEDS),
              "environment": environment(), "runtime_screening": "not_run",
              "comparisons": {}, "limitations": [
                  "Compiler comparison is scoped to the recorded RTX 4060 Ti environment and these two shapes.",
                  "A code difference does not establish a latency or performance difference.",
                  "No CUDA-event timing or performance confirmation was run."]}
    try:
        for name, shape in evidence.SHAPES.items():
            result["comparisons"][name] = capture_shape(folder / name, shape)
        control = result["comparisons"]["control"]
        boundary = result["comparisons"]["partial_m"]
        if control["meaningful_compiler_difference"]:
            result["outcome"] = "control_difference_screening_skipped"
            result["screening_decision"] = (
                "The M=128 control no longer reproduces Experiment 003 compiler equivalence. "
                "Investigate toolchain or environment drift before considering timing; none was run.")
        elif boundary["meaningful_compiler_difference"]:
            result["outcome"] = "compiler_difference_screening_skipped"
            result["screening_decision"] = (
                "A separate paired CUDA-event benchmark may be considered after human review; none was run.")
        else:
            result["outcome"] = "compiler_equivalent_screening_skipped"
            result["screening_decision"] = (
                "No compiler difference in the partial-M comparison warrants timing this unchanged pair; none was run.")
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:2000]}")
    finally:
        write_json(folder / "result.json", result)
        for file in ("benchmark_proposal.py", "trusted_compute_kernel.py",
                     "reference_precision.py", "experiment_004_partial_boundary.py"):
            shutil.copy2(ROOT / file, folder / file)
        (folder / "COMMANDS.txt").write_text(
            ".venv/bin/python experiment_004_partial_boundary.py --run\n"
            "# Each run creates a fresh UTC-stamped directory; no screening command is provided.\n")
    if result["status"] == "complete":
        files = {str(path.relative_to(folder)): evidence.sha(path.read_bytes())
                 for path in folder.rglob("*") if path.is_file()}
        write_json(folder / "manifest.json", {"schema": "experiment_004_v1", "status": "complete",
                   "shapes": evidence.SHAPES, "config": evidence.CONFIG,
                   "files": sorted(files), "artifact_hashes": files})
    return folder, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True,
                        help="execute one new GPU compiler/correctness diagnostic; never benchmark")
    args = parser.parse_args(argv)
    folder, result = run()
    print(f"{result['status']}: {folder}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
