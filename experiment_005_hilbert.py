"""Explicit stages for Experiment 005; import never launches CUDA work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import time

import torch
import triton

import benchmark_core
import experiment_005_evidence as evidence
import experiment_004_evidence as correctness_policy
import hilbert_matmul
from mask_hoisting_memory import _sections
from muse_wild_lab import load_accepted as load_wild_accepted
from stage_reduction_experiment import environment
from trusted_compute_experiment import _inputs
from mask_hoisting_m128_experiment import capture

ROOT = Path(__file__).resolve().parent
SOURCES = evidence.SOURCE_FILES
SEEDS = (42, 43)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seal(folder, schema):
    write_json(folder / "manifest.json", evidence.manifest(folder, schema=schema))


def device_check():
    if not torch.cuda.is_available() or "RTX 4060 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Experiment 005 requires the RTX 4060 Ti CUDA device")


def shape_key(shape):
    return "x".join(map(str, shape))


def map_for(shape):
    tiles_m = triton.cdiv(shape[0], evidence.CONFIG["block_m"])
    tiles_n = triton.cdiv(shape[1], evidence.CONFIG["block_n"])
    start = time.perf_counter()
    order = hilbert_matmul.tile_order("hilbert", tiles_m, tiles_n)
    mapping = torch.tensor(order, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - start) * 1000
    return mapping, order, setup_ms


def config(schedule, mapping=None):
    return {**evidence.CONFIG, "schedule": schedule, "tile_map": mapping}


def has_valid_screening(run_id):
    path = ROOT / evidence.SECTION / run_id / "screening"
    if not path.is_dir() or path.is_symlink():
        return False
    for attempt in path.iterdir():
        if not attempt.is_dir() or attempt.is_symlink():
            continue
        try:
            evidence.validate_timing(ROOT, run_id, "screening", attempt.name)
            return True
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return False


def verify():
    device_check()
    accepted = load_wild_accepted(ROOT)
    if not accepted or accepted["ideas"]["ideas"][1]["name"] != "Hilbert Tile Traversal":
        raise ValueError("accepted Hilbert Wild Research Lab proposal unavailable")
    run_id = stamp()
    folder = ROOT / evidence.SECTION / run_id / "verify"
    folder.mkdir(parents=True, exist_ok=False)
    proposal_raw = (ROOT / "evaluations/muse-wild-research-lab/v1/accepted.json").read_bytes()
    (folder / "wild_lab_accepted.json").write_bytes(proposal_raw)
    result = {"experiment": "005_hilbert_tile_traversal", "status": "incomplete",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": run_id, "shapes": [list(x) for x in evidence.SHAPES],
              "config": evidence.CONFIG, "schedules": list(evidence.SCHEDULES),
              "seeds": list(SEEDS), "environment": environment(),
              "source_hashes": {name: evidence.sha((ROOT / name).read_bytes()) for name in SOURCES},
              "wild_lab_accepted_sha256": evidence.sha(proposal_raw),
              "tile_mappings": [], "correctness": [], "compiler": {},
              "screening": "not_run", "confirmation": "not_run",
              "limitations": ["Representative FP16 GEMMs, not verified production-model workloads.",
                              "Program-ID assignment does not guarantee physical CTA execution order.",
                              "Compiler differences alone do not establish speed or cache-locality benefit."]}
    try:
        for shape in evidence.SHAPES:
            key = shape_key(shape)
            sub = folder / key
            sub.mkdir()
            mapping, order, setup_ms = map_for(shape)
            tiles_m = triton.cdiv(shape[0], evidence.CONFIG["block_m"])
            tiles_n = triton.cdiv(shape[1], evidence.CONFIG["block_n"])
            write_json(sub / "hilbert_order.json", {"tiles_m": tiles_m, "tiles_n": tiles_n,
                                                    "order": order, "setup_ms": setup_ms})
            for schedule in evidence.SCHEDULES:
                rows = hilbert_matmul.tile_order(schedule, tiles_m, tiles_n)
                valid = len(rows) == tiles_m * tiles_n and len(set(rows)) == len(rows)
                result["tile_mappings"].append({"shape": key, "schedule": schedule,
                                                "tile_count": len(rows), "valid": valid})
                if not valid:
                    raise ValueError("invalid tile coverage")
            result["compiler"][key] = {}
            for seed in SEEDS:
                a, b, reference, policy = _inputs(shape, seed)
                for schedule in evidence.SCHEDULES:
                    out = torch.full_like(reference, float("nan"))
                    row = {"shape": key, "schedule": schedule, "seed": seed,
                           "reference_precision": policy, "status": "incomplete"}
                    try:
                        hilbert_matmul.launch(a, b, out, config(schedule, mapping))
                        torch.cuda.synchronize()
                        row.update(correctness_policy.check_output(out, reference, shape, rtol=0.02, atol=0.02))
                        row["status"] = "PASS"
                    except Exception as exc:
                        row.update(status="FAIL", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
                        result["correctness"].append(row)
                        raise
                    result["correctness"].append(row)
                if seed == SEEDS[0]:
                    for schedule in evidence.SCHEDULES:
                        handle = hilbert_matmul.matmul_kernel.warmup(
                            a, b, reference, mapping if schedule == "hilbert" else a,
                            *shape, evidence.CONFIG["block_m"], evidence.CONFIG["block_n"],
                            evidence.CONFIG["block_k"], evidence.SCHEDULES.index(schedule), 4,
                            num_warps=evidence.CONFIG["num_warps"], num_stages=evidence.CONFIG["num_stages"],
                            grid=(tiles_n, tiles_m) if schedule == "row_major" else (tiles_m * tiles_n,))
                        compiler = capture(sub, schedule, handle)
                        if compiler["sass_status"] != "captured":
                            raise ValueError("complete SASS unavailable")
                        sass = (sub / f"{schedule}.sass").read_text()
                        compiler["sass_counts"] = correctness_policy.sass_counts(
                            correctness_policy.sass_instructions(sass))
                        compiler["executable_sections"] = {
                            name: {"sha256": evidence.sha(raw), "size_bytes": len(raw)}
                            for name, raw in _sections((sub / f"{schedule}.cubin").read_bytes()).items()
                            if name.startswith(".text.")}
                        result["compiler"][key][schedule] = compiler
                    baseline_count = result["compiler"][key]["row_major"]["sass_counts"]["instructions"]
                    result["compiler"][key]["whole_kernel_instruction_delta_vs_row_major"] = {
                        schedule: result["compiler"][key][schedule]["sass_counts"]["instructions"] - baseline_count
                        for schedule in evidence.SCHEDULES}
                    result["compiler"][key]["mapping_note"] = (
                        "Deltas describe whole-kernel SASS, not isolated mapping instruction counts. "
                        "Hilbert performs two int32 lookup loads per CTA; row-major and grouped compute tile IDs in-kernel.")
        result["status"] = "verified"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1200]}")
    finally:
        for name in SOURCES:
            shutil.copy2(ROOT / name, folder / name)
        write_json(folder / "result.json", result)
        if result["status"] == "verified":
            seal(folder, "experiment_005_verify_v1")
    return run_id, result


def timing_stage(run_id, stage, *, human_approved=False, rounds=8, warmups=10, iterations=100):
    if (stage not in {"screening", "confirmation"} or type(rounds) is not int or rounds < 8
            or rounds % 2 or type(warmups) is not int or warmups < 1
            or type(iterations) is not int or iterations < 1):
        raise ValueError("invalid controlled timing options")
    device_check()
    if stage == "confirmation" and not human_approved:
        raise ValueError("independent confirmation requires --human-approved")
    verify_folder, verified = evidence.validate_verify(ROOT, run_id)
    if environment() != verified["environment"]:
        raise ValueError("environment drift from correctness/compiler verification")
    if stage == "confirmation":
        if not has_valid_screening(run_id):
            raise ValueError("validated screening required before confirmation")
    folder = ROOT / evidence.SECTION / run_id / stage / stamp()
    folder.mkdir(parents=True, exist_ok=False)
    result = {"stage": stage, "status": "incomplete",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": run_id, "verify_manifest_sha256": evidence.sha((verify_folder / "manifest.json").read_bytes()),
              "source_hashes": verified["source_hashes"], "environment": environment(),
              "comparisons": [], "timing_input_seed": SEEDS[0] if stage == "screening" else 44,
              "confirmation_correctness": [],
              "timing_scope": "Persistent Hilbert lookup reused across timed launches; lookup fetch and arithmetic are inside the kernel. One-time map preparation is reported by verification separately.",
              "decision": "screening_unconfirmed" if stage == "screening" else "confirmation_pending_review"}
    try:
        for shape in evidence.SHAPES:
            mapping, _, setup_ms = map_for(shape)
            a, b, reference, policy = _inputs(shape, result["timing_input_seed"])
            configs = {schedule: config(schedule, mapping) for schedule in evidence.SCHEDULES}
            if stage == "confirmation":
                for schedule in evidence.SCHEDULES:
                    out = torch.full_like(reference, float("nan"))
                    hilbert_matmul.launch(a, b, out, configs[schedule])
                    torch.cuda.synchronize()
                    correctness_policy.check_output(out, reference, shape, rtol=0.02, atol=0.02)
                    result["confirmation_correctness"].append({"shape": shape_key(shape),
                        "schedule": schedule, "status": "PASS", "reference_precision": policy})
            for opponent in ("row_major", "grouped_m4"):
                raw = benchmark_core.run_paired(
                    hilbert_matmul.launch, a, b, configs[opponent], configs["hilbert"],
                    rounds=rounds, warmups=warmups, iterations=iterations,
                    seed=43 if stage == "screening" else 47)
                summary = evidence.validate_comparison(raw)
                result["comparisons"].append({"shape": shape_key(shape), "opponent": opponent,
                                               "timing": raw, "summary": summary,
                                               "one_time_map_setup_ms": setup_ms})
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1200]}")
    finally:
        write_json(folder / "result.json", result)
        if result["status"] == "complete":
            seal(folder, f"experiment_005_{stage}_v1")
    return folder, result


def profile_kernel(run_id, shape, schedule, *, human_approved=False):
    if not human_approved:
        raise ValueError("profiling requires --human-approved")
    evidence.validate_verify(ROOT, run_id)
    if not has_valid_screening(run_id):
        raise ValueError("validated screening required before optional profiling")
    if shape not in evidence.SHAPES or schedule not in evidence.SCHEDULES:
        raise ValueError("invalid profile target")
    device_check()
    mapping, _, _ = map_for(shape)
    a, b, _, _ = _inputs(shape, 42)
    out = torch.empty((shape[0], shape[1]), device="cuda", dtype=torch.float16)
    hilbert_matmul.launch(a, b, out, config(schedule, mapping))
    torch.cuda.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for option in ("verify", "screen", "confirm", "profile-kernel"):
        mode.add_argument("--" + option, action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--human-approved", action="store_true")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"))
    parser.add_argument("--schedule", choices=evidence.SCHEDULES)
    args = parser.parse_args(argv)
    if args.verify:
        run_id, result = verify()
        print(f"{result['status']}: {run_id}")
    elif args.profile_kernel:
        if not args.run_id or not args.shape or not args.schedule:
            parser.error("profiling requires --run-id, --shape and --schedule")
        profile_kernel(args.run_id, tuple(args.shape), args.schedule, human_approved=args.human_approved)
        print("profile launch complete")
        return 0
    else:
        if not args.run_id:
            parser.error("--run-id is required")
        folder, result = timing_stage(args.run_id, "screening" if args.screen else "confirmation",
                                      human_approved=args.human_approved, rounds=args.rounds,
                                      warmups=args.warmups, iterations=args.iterations)
        print(f"{result['status']}: {folder}")
    return 0 if result["status"] in {"verified", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
