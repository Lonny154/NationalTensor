"""Experiment 002: compare the reviewed two-slab GEMM at three and two stages."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import torch
import triton

import benchmark_core
import trusted_two_slab_experiment as trusted
import trusted_two_slab_kernel


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "evaluations/experiment-planner/20261028T010924452160Z/plan.json"
PLAN_SHA256 = "1586d7e9861a35a5fbde0e4d2aa9943cabdd83742d9b886a17e01bb005b211cc"
PRIOR = ROOT / "evaluations/trusted-two-slab/20261027T161358957319Z/check.json"
TRUSTED_SHA256 = "5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0"
TRUSTED_MODULE_SHA256 = "98188d25c726c2301cd81e36e6dfd22f02db5f3acbc3d05d385d63ce7cffd06d"
CONFIG_3 = dict(trusted.CONFIG)
CONFIG_2 = {**CONFIG_3, "num_stages": 2}
SHAPE = trusted.SHAPES[0]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_sources():
    if sha256(PLAN.read_bytes()) != PLAN_SHA256:
        raise ValueError("saved Muse plan hash changed")
    plan = json.loads(PLAN.read_text())
    if plan["experiment_kind"] != "numeric_tuning" or plan["target_shape_mnk"] != list(SHAPE):
        raise ValueError("saved Muse plan target changed")
    identity = trusted._identity()
    if (identity["trusted_implementation_sha256"] != TRUSTED_SHA256
            or identity["trusted_module_sha256"] != TRUSTED_MODULE_SHA256):
        raise ValueError("reviewed two-slab source changed")
    prior = json.loads(PRIOR.read_text())
    if (prior["status"] != "correctness_and_compiler_captured"
            or prior["trusted_implementation_sha256"] != TRUSTED_SHA256
            or prior["trusted_module_sha256"] != TRUSTED_MODULE_SHA256
            or prior["numeric_config"] != CONFIG_3
            or len(prior["correctness"]) != 6
            or any(row["status"] != "PASS" for row in prior["correctness"])):
        raise ValueError("saved trusted check is incompatible")
    if {key for key in CONFIG_3 if CONFIG_3[key] != CONFIG_2[key]} != {"num_stages"}:
        raise ValueError("configurations differ in more than num_stages")
    return identity


def environment():
    device = torch.cuda.get_device_properties(0)
    fields = ("name", "major", "minor", "multi_processor_count", "total_memory",
              "shared_memory_per_multiprocessor", "regs_per_multiprocessor",
              "max_threads_per_multi_processor", "warp_size")
    gpu = {name: getattr(device, name, None) for name in fields}
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        smi = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "triton": triton.__version__, "gpu": gpu, "nvidia_smi_uuid_driver": smi}


def resource_limits(capture, gpu):
    """Simple resource ceilings; NCU can supply exact allocation limits separately."""
    threads = CONFIG_3["num_warps"] * gpu["warp_size"]
    registers = capture["registers_per_thread"]
    shared = capture["shared_memory_bytes"]
    regs_per_sm = gpu["regs_per_multiprocessor"]
    shared_per_sm = gpu["shared_memory_per_multiprocessor"]
    threads_per_sm = gpu["max_threads_per_multi_processor"]
    if not all(isinstance(x, int) and x > 0 for x in
               (threads, registers, shared, regs_per_sm, shared_per_sm, threads_per_sm)):
        return {"status": "unavailable"}
    limits = {"registers": regs_per_sm // (registers * threads),
              "shared_memory": shared_per_sm // shared,
              "threads": threads_per_sm // threads}
    blocks = min(limits.values())
    return {"status": "arithmetic_upper_bound", "threads_per_block": threads,
            "blocks_per_sm_by_resource": limits, "upper_bound_blocks_per_sm": blocks,
            "upper_bound_occupancy_fraction": blocks * threads / threads_per_sm,
            "caveat": "Ignores allocation granularity and architecture block limits; compiler metadata is not achieved occupancy."}


def run(folder, rounds, aa_rounds, iterations, warmups):
    identity = verify_sources()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    folder.mkdir(parents=True, exist_ok=False)
    result = {"status": "incomplete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "experiment": "002_pipeline_stage_reduction", "plan_path": str(PLAN.relative_to(ROOT)),
              "plan_sha256": PLAN_SHA256, "prior_check_path": str(PRIOR.relative_to(ROOT)),
              "prior_check_sha256": sha256(PRIOR.read_bytes()),
              "runner_sha256": sha256(Path(__file__).read_bytes()), "source_identity": identity,
              "workload": {"shape_mnk": list(SHAPE), "dtype": "float16", "seed": 42},
              "configurations": {"stages_3": CONFIG_3, "stages_2": CONFIG_2},
              "environment": environment(), "correctness": [],
              "measurement_settings": {"rounds": rounds, "aa_rounds": aa_rounds,
                                       "iterations": iterations, "warmups": warmups}}
    path = folder / "result.json"
    try:
        primary = None
        handles = {}
        for shape in trusted.SHAPES:
            for seed in trusted.SEEDS:
                a, b, reference, policy = trusted._inputs(shape, seed)
                for name, config in (("stages_3", CONFIG_3), ("stages_2", CONFIG_2)):
                    out = torch.empty_like(reference)
                    row = {"configuration": name, "shape_mnk": list(shape),
                           "seed": seed, "reference_precision": policy}
                    try:
                        handle = trusted_two_slab_kernel.launch(a, b, out, config)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
                    except Exception as exc:
                        row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}"[:2000])
                        result["correctness"].append(row)
                        raise
                    row["status"] = "PASS"
                    result["correctness"].append(row)
                    if shape == SHAPE and seed == 42:
                        handles[name] = handle
                        primary = (a, b)
        result["compiler"] = {}
        for name, handle in handles.items():
            capture = trusted._capture(folder, name, handle)
            capture["occupancy_resource_estimate"] = resource_limits(
                capture, result["environment"]["gpu"])
            result["compiler"][name] = capture

        a, b = primary
        launch = trusted_two_slab_kernel.launch
        common = {"warmups": warmups, "iterations": iterations}
        result["aa_stages_3"] = benchmark_core.run_aa_calibration(
            launch, a, b, CONFIG_3, rounds=aa_rounds, seed=42, **common)
        result["comparison"] = benchmark_core.run_paired(
            launch, a, b, CONFIG_3, CONFIG_2, rounds=rounds, seed=43, **common)
        result["aa_stages_2"] = benchmark_core.run_aa_calibration(
            launch, a, b, CONFIG_2, rounds=aa_rounds, seed=44, **common)
        ratio = result["comparison"]["paired_ratios"]["median"]
        noise = max(result[key]["aa_variability"]["median_absolute_ratio_deviation"]
                    for key in ("aa_stages_3", "aa_stages_2"))
        pair_mad = result["comparison"]["paired_ratios"]["mad"]
        result["interpretation"] = {
            "median_paired_speedup_3_over_2": ratio,
            "absolute_ratio_effect": abs(ratio - 1),
            "largest_aa_median_absolute_ratio_deviation": noise,
            "paired_ratio_mad": pair_mad,
            "exceeds_observed_noise_heuristic": abs(ratio - 1) > max(noise, pair_mad),
            "noise_rule": "absolute paired median ratio deviation from 1 exceeds both A/A median absolute ratio deviations and comparison paired-ratio MAD; heuristic, not a significance test",
        }
        result["status"] = "screening_complete_unconfirmed"
    except Exception as exc:
        result["status"] = "correctness_failed" if any(
            row["status"] == "FAIL" for row in result["correctness"]) else "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-one", type=int, choices=(2, 3),
                        help="check FC1 seed 42, then expose one reviewed launch to NCU")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evaluations/stage-reduction" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--aa-rounds", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    if args.profile_one is not None:
        verify_sources()
        a, b, reference, _ = trusted._inputs(SHAPE, 42)
        out = torch.empty_like(reference)
        config = CONFIG_2 if args.profile_one == 2 else CONFIG_3
        trusted_two_slab_kernel.launch(a, b, out, config)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
        print(f"profiled stage {args.profile_one}: correctness PASS")
        return 0
    result = run(args.out_dir, args.rounds, args.aa_rounds, args.iterations, args.warmups)
    print(f"{result['status']}: {args.out_dir / 'result.json'}")
    return 0 if result["status"] == "screening_complete_unconfirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
