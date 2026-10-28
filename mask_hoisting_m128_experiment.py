"""Experiment 003: compiler diagnosis and bounded screening of unchanged reviewed code."""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

import torch
import benchmark_core
import benchmark_proposal
import trusted_compute_experiment as trusted
import trusted_compute_kernel
from stage_reduction_experiment import environment


ROOT = Path(__file__).resolve().parent
PLAN = ROOT / "evaluations/experiment-planner/20261028T145028833360Z/plan.json"
PLAN_SHA = "5a5ec80842b9a544f02471bd0517d77c7ce598d82a88355d1980ed418e5fd8b1"
PRIOR = ROOT / "evaluations/trusted-compute/20261027T154233319712Z/experiment.json"
CONFIG = {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3}
SHAPE = (128, 3072, 768)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity():
    prior = json.loads(PRIOR.read_text())
    checks = {
        "trusted_implementation_sha256": inspect.getsource(trusted_compute_kernel.matmul_kernel.fn).encode(),
        "trusted_module_sha256": Path(trusted_compute_kernel.__file__).read_bytes(),
        "baseline_kernel_sha256": inspect.getsource(benchmark_proposal.matmul_kernel.fn).encode(),
        "baseline_launcher_sha256": inspect.getsource(benchmark_proposal.launch).encode(),
    }
    for name, raw in checks.items():
        if sha(raw) != prior[name]:
            raise ValueError(f"trusted source identity changed: {name}")
    if (sha(PLAN.read_bytes()) != PLAN_SHA or prior["numeric_config"] != CONFIG
            or any(row["status"] != "PASS" for row in prior["correctness"])):
        raise ValueError("plan or historical correctness identity changed")
    plan = json.loads(PLAN.read_text())
    if plan["target_shape_mnk"] != list(SHAPE) or plan["experiment_kind"] != "workload_shape":
        raise ValueError("unexpected advisory plan")
    return {**{name: sha(raw) for name, raw in checks.items()},
            "original_muse_source_sha256": prior["muse_candidate_sha256"],
            "exact_generated_source_evaluated": False,
            "plan_path": str(PLAN.relative_to(ROOT)), "plan_sha256": PLAN_SHA,
            "prior_path": str(PRIOR.relative_to(ROOT)), "prior_sha256": sha(PRIOR.read_bytes()),
            "runner_sha256": sha(Path(__file__).read_bytes()),
            "reference_precision_sha256": sha(Path(trusted.reference_precision.__file__).read_bytes()),
            "environment_helper_sha256": sha(Path(inspect.getsourcefile(environment)).read_bytes()),
            "benchmark_core_sha256": sha(Path(benchmark_core.__file__).read_bytes())}


def executable_sections(cubin):
    """Extract complete ELF64 code bytes, including instruction scheduling encodings."""
    if cubin[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64 cubin")
    offset = struct.unpack_from("<Q", cubin, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", cubin, 58)
    headers = [struct.unpack_from("<IIQQQQIIQQ", cubin, offset + index * size)
               for index in range(count)]
    names = headers[names_index]
    strings = cubin[names[4]:names[4] + names[5]]
    result = {}
    for header in headers:
        name = strings[header[0]:].split(b"\0", 1)[0].decode()
        if header[2] & 4 and header[1] == 1:
            result[name] = cubin[header[4]:header[4] + header[5]]
    if not result:
        raise ValueError("no executable code sections in cubin")
    return result


def normalized_ptx(text):
    # Preserve predicates, operands, labels, and instruction order.
    lines = []
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        if line and not line.startswith((".file", ".loc")):
            lines.append(re.sub(r"\s+", " ", line))
    return lines


def capture(folder, name, handle):
    handle._init_handles()
    artifacts = {}
    for kind in ("ttir", "ttgir", "llir", "ptx", "cubin"):
        content = handle.asm.get(kind)
        if content is None:
            continue
        raw = content if isinstance(content, bytes) else content.encode()
        path = folder / f"{name}.{kind}"
        path.write_bytes(raw)
        artifacts[kind] = {"path": path.name, "sha256": sha(raw), "size_bytes": len(raw)}
    tool = shutil.which("cuobjdump")
    sass_status = "cuobjdump unavailable"
    if tool:
        command = [tool, "--dump-sass", str(folder / f"{name}.cubin")]
        result = subprocess.run(command, capture_output=True, timeout=30)
        (folder / f"{name}.sass.stderr").write_bytes(result.stderr)
        if result.returncode == 0:
            path = folder / f"{name}.sass"
            path.write_bytes(result.stdout)
            artifacts["sass"] = {"path": path.name, "sha256": sha(result.stdout)}
            sass_status = "captured"
        else:
            sass_status = f"cuobjdump failed: {result.returncode}"
    return {"artifacts": artifacts, "registers_per_thread": handle.n_regs,
            "spills": handle.n_spills, "shared_memory_bytes": handle.metadata.shared,
            "sass_status": sass_status}


def diagnose(folder):
    source = identity()
    folder.mkdir(parents=True, exist_ok=False)
    result = {"experiment": "003_mask_hoisting_m128", "status": "incomplete",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_identity": source,
              "shape_mnk": list(SHAPE), "dtype": "fp16", "config": CONFIG,
              "grid": [2, 24], "partial_boundary_tiles": False, "environment": environment()}
    try:
        a = torch.empty((128, 768), device="cuda", dtype=torch.float16)
        b = torch.empty((768, 3072), device="cuda", dtype=torch.float16)
        out = torch.empty((128, 3072), device="cuda", dtype=torch.float16)
        result["compiler"] = {}
        for name, kernel in (("baseline", benchmark_proposal.matmul_kernel),
                             ("trusted", trusted_compute_kernel.matmul_kernel)):
            handle = kernel.warmup(a, b, out, *SHAPE, 64, 128, 32,
                                  num_warps=4, num_stages=3, grid=(2, 24))
            result["compiler"][name] = capture(folder, name, handle)
        codes = {name: executable_sections((folder / f"{name}.cubin").read_bytes())
                 for name in ("baseline", "trusted")}
        result["executable_sections"] = {
            name: {key: {"sha256": sha(raw), "size_bytes": len(raw)} for key, raw in sections.items()}
            for name, sections in codes.items()}
        result["code_bytes_identical"] = codes["baseline"] == codes["trusted"]
        streams = {name: normalized_ptx((folder / f"{name}.ptx").read_text())
                   for name in ("baseline", "trusted")}
        result["ptx_order_and_operands_identical_ignoring_debug"] = streams["baseline"] == streams["trusted"]
        (folder / "ptx.diff").write_text("\n".join(difflib.unified_diff(
            streams["baseline"], streams["trusted"], fromfile="baseline", tofile="trusted")) + "\n")
        if all((folder / f"{name}.sass").exists() for name in codes):
            sass = {}
            for name in codes:
                raw = (folder / f"{name}.sass").read_text()
                sass[name] = [re.sub(r"/\*.*?\*/", "", line).strip() for line in raw.splitlines()
                              if re.match(r"\s*/\*[0-9a-fA-F]+\*/\s+[^/\s]", line)]
            result["sass_instruction_stream_identical"] = sass["baseline"] == sass["trusted"]
            result["sass_instruction_counts"] = {name: len(lines) for name, lines in sass.items()}
            (folder / "sass.diff").write_text("\n".join(difflib.unified_diff(
                sass["baseline"], sass["trusted"], fromfile="baseline", tofile="trusted")) + "\n")
        result["status"] = "diagnostic_complete"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        write_json(folder / "diagnostic.json", result)
        for file in (Path(__file__), Path(trusted_compute_kernel.__file__), Path(benchmark_proposal.__file__)):
            shutil.copy2(file, folder / file.name)
        (folder / "COMMANDS.txt").write_text(
            f"# Run from {ROOT}; choose a fresh artifact directory for reproduction.\n"
            ".venv/bin/python mask_hoisting_m128_experiment.py diagnose --artifact-dir <fresh-directory>\n"
            ".venv/bin/python mask_hoisting_m128_experiment.py screen --artifact-dir <same-directory>\n"
            "# SASS: cuobjdump --dump-sass <directory>/baseline.cubin (and trusted.cubin)\n")
    return result


def screen(folder):
    diagnostic = json.loads((folder / "diagnostic.json").read_text())
    if diagnostic["status"] != "diagnostic_complete" or diagnostic["source_identity"] != identity():
        raise ValueError("compiler diagnostic or source identity incompatible")
    path = folder / "screening.json"
    if path.exists():
        raise FileExistsError(path)
    result = {"status": "incomplete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_identity": identity(), "diagnostic_sha256": sha((folder / "diagnostic.json").read_bytes()),
              "shape_mnk": list(SHAPE), "configurations": {"baseline": CONFIG, "trusted": CONFIG},
              "environment": environment(), "correctness": [], "tolerances": {"rtol": 0.02, "atol": 0.02}}
    try:
        primary = None
        for shape in (SHAPE, *trusted.CORRECTNESS_SHAPES):
            for seed in trusted.CORRECTNESS_SEEDS:
                a, b, reference, policy = trusted._inputs(shape, seed)
                for name, launch in (("baseline", benchmark_proposal.launch), ("trusted", trusted_compute_kernel.launch)):
                    out = torch.empty_like(reference)
                    launch(a, b, out, CONFIG)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
                    result["correctness"].append({"implementation": name, "shape_mnk": list(shape),
                        "seed": seed, "status": "PASS", "reference_precision": policy})
                if shape == SHAPE and seed == 42:
                    primary = a, b
        a, b = primary
        common = {"rounds": 20, "warmups": 10, "iterations": 100}
        result["aa_baseline"] = benchmark_core.run_aa_calibration(
            benchmark_proposal.launch, a, b, CONFIG, seed=42, **common)

        def dispatch(x, y, out, config):
            (benchmark_proposal.launch if config["implementation"] == "baseline"
             else trusted_compute_kernel.launch)(x, y, out, CONFIG)

        result["comparison"] = benchmark_core.run_paired(
            dispatch, a, b, {**CONFIG, "implementation": "baseline"},
            {**CONFIG, "implementation": "trusted"}, seed=43, **common)
        result["aa_trusted"] = benchmark_core.run_aa_calibration(
            trusted_compute_kernel.launch, a, b, CONFIG, seed=44, **common)
        comparison = result["comparison"]
        effect = abs(comparison["paired_ratios"]["median"] - 1)
        noise = max(result[key]["aa_variability"]["median_absolute_ratio_deviation"]
                    for key in ("aa_baseline", "aa_trusted"))
        result["interpretation"] = {"absolute_ratio_effect": effect, "largest_aa_median_deviation": noise,
            "exceeds_typical_noise_heuristic": effect > max(noise, comparison["paired_ratios"]["mad"]),
            "rule": "Effect must exceed both contemporaneous A/A median deviations and paired-ratio MAD; heuristic, not statistical significance."}
        result["status"] = "screening_complete_unconfirmed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnose", "screen"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.artifact_dir) if args.mode == "diagnose" else screen(args.artifact_dir)
    print(f"{result['status']}: {args.artifact_dir}")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
