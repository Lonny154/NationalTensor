"""Prepare a hash-bound Nsight Compute study; GPU capture requires --run-one.

Only the three fixed, human-authored GEMM modules can be selected. Importing this
module and --prepare-only never import Torch or launch a CUDA kernel.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PLAN_RUN = "20261028T002953332983Z"
PLAN_SHA = "1e773b3c45929d5d4c288db29a058afecd995281012023cf0ab287fe44fd57aa"
PLAN_DIR = ROOT / "evaluations/experiment-planner" / PLAN_RUN
OUTPUT_BASE = ROOT / "evaluations/planned-profiling"
SHAPE = [512, 3072, 768]
GPU = "NVIDIA GeForce RTX 4060 Ti"
CONFIG = {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3}
CORRECTNESS_CASES = {(tuple(shape), seed) for shape in
                     ([512, 3072, 768], [509, 3071, 769], [65, 129, 33])
                     for seed in (42, 43)}
BASELINE_SHA = "4fc8041c761226f10ccffbb0e5588b73e769372bb72f8354c0a845a66d0df877"
LAUNCHER_SHA = "349aed3fb91b4f0e59f28417d241ddd9b2fa9359f33b95905c443b039b073999"
KERNELS = {
    "baseline": ("benchmark_proposal.py", BASELINE_SHA, None),
    "mask_hoisting": ("trusted_compute_kernel.py", "951ec924a4cfa6fb9abd4f9a7d12fbb3e187e56a69b77b8df546cf89b15e63f2", "trusted:trusted-compute:20261027T154233319712Z"),
    "two_slab": ("trusted_two_slab_kernel.py", "5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0", "trusted:trusted-two-slab:20261027T161358957319Z"),
}
METRICS = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved active warps relative to peak, percent",
    "smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active": "Long-scoreboard stalled warps per active cycle",
    "smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active": "Short-scoreboard stalled warps per active cycle",
    "smsp__warps_issue_stalled_barrier.avg.per_cycle_active": "Barrier-stalled warps per active cycle",
    "smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": "Tensor-pipe instruction rate relative to sustained peak, percent",
    "dram__bytes.sum.per_second": "DRAM throughput, bytes/second",
    "lts__t_bytes.sum.per_second": "L2 traffic rate, bytes/second",
    "l1tex__t_bytes.sum.per_second": "L1/texture traffic rate, bytes/second",
    "launch__occupancy_limit_registers": "Resident block limit from register allocation",
    "launch__occupancy_limit_shared_mem": "Resident block limit from shared-memory allocation",
    "launch__registers_per_thread": "Allocated registers per thread",
    "launch__shared_mem_per_block": "Allocated shared memory per block",
}
LAUNCH_METRICS = {name for name in METRICS if name.startswith("launch__")}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def function_hash(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = next((node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == name), None)
    if node is None:
        raise ValueError(f"missing reviewed function: {path.name}:{name}")
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return sha("".join(source.splitlines(keepends=True)[start - 1:node.end_lineno]).encode())


def verify_inputs(root: Path = ROOT) -> dict:
    """Check plan, original citations, source identities, and saved correctness."""
    from experiment_planner import load_saved_plans

    root = Path(root).resolve()
    if not (root / "evaluations/experiment-planner" / PLAN_RUN).is_dir():
        raise ValueError("saved advisory plan is missing")
    plans, _ = load_saved_plans(root)
    record = next((item for item in plans if item["run_id"] == PLAN_RUN), None)
    if record is None or record["plan_sha256"] != PLAN_SHA:
        raise ValueError("saved advisory plan failed hash or audit validation")
    plan = record["plan"]
    if plan["experiment_kind"] != "profiling" or plan["target_shape_mnk"] != SHAPE:
        raise ValueError("plan kind or target shape changed")
    prompt = json.loads((root / "evaluations/experiment-planner" / PLAN_RUN /
                         "proposal_audit.jsonl").read_text().splitlines()[-1])["prompt"]
    context = json.loads(prompt.split("\n", 1)[1])
    if (context.get("gpu") != GPU or context.get("dtype") != "fp16"
            or context.get("current_numeric_config") != CONFIG
            or context.get("baseline_source_sha256") != BASELINE_SHA
            or context.get("launcher_source_sha256") != LAUNCHER_SHA):
        raise ValueError("saved planner hardware or source context changed")
    expected_ids = {spec[2] for spec in KERNELS.values() if spec[2]}
    citations = {row["id"]: row for row in plan["supporting_evidence"]}
    if set(citations) != expected_ids:
        raise ValueError("required trusted evidence is missing from plan")
    provenance = {row["id"]: row for row in record["evidence_provenance"]}
    identities = {}
    if function_hash(root / "benchmark_proposal.py", "launch") != LAUNCHER_SHA:
        raise ValueError("baseline launcher identity changed")
    for key, (filename, expected, evidence_id) in KERNELS.items():
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"reviewed source path is missing or linked: {key}")
        actual = function_hash(path, "matmul_kernel")
        if actual != expected:
            raise ValueError(f"reviewed kernel source identity changed: {key}")
        identities[key] = {"path": filename, "kernel_sha256": actual,
                           "file_sha256": sha(path.read_bytes()),
                           "provenance": "reviewed_source", "evidence_id": evidence_id}
        if evidence_id:
            citation = citations[evidence_id]
            original = provenance[evidence_id]
            if (citation["artifact_path"] != original["artifact_path"]
                    or original["trusted_implementation_sha256"] != actual):
                raise ValueError("trusted evidence path or source hash changed")
            folder = root / citation["artifact_path"]
            if (not folder.is_dir() or folder.is_symlink()
                    or not folder.resolve().is_relative_to(root / "evaluations")):
                raise ValueError("trusted evidence path is missing or unsafe")
            file = folder / ("experiment.json" if key == "mask_hoisting" else "check.json")
            if file.is_symlink():
                raise ValueError("trusted evidence file is a symlink")
            saved = json.loads(file.read_text())
            checks = saved.get("correctness")
            valid_checks = (isinstance(checks, list) and len(checks) == 6
                and all(isinstance(row, dict) and row.get("status") == "PASS"
                        and isinstance(row.get("shape_mnk"), list)
                        and len(row["shape_mnk"]) == 3
                        and all(type(value) is int for value in row["shape_mnk"])
                        and type(row.get("input_seed")) is int
                        and isinstance(row.get("reference_precision"), dict)
                        and row["reference_precision"].get("policy")
                            == "fp16_torch_mm_without_reduced_precision_reduction"
                        for row in checks))
            if valid_checks:
                valid_checks = ({(tuple(row["shape_mnk"]), row["input_seed"])
                                 for row in checks} == CORRECTNESS_CASES)
            if (saved.get("trusted_implementation_sha256") != actual
                    or saved.get("trusted_module_sha256") != identities[key]["file_sha256"]
                    or saved.get("baseline_kernel_sha256") != BASELINE_SHA
                    or saved.get("baseline_launcher_sha256") != LAUNCHER_SHA
                    or saved.get("gpu") != GPU
                    or saved.get("exact_generated_source_evaluated") is not False
                    or saved.get("muse_candidate_sha256", saved.get("original_muse_candidate_sha256"))
                       != original["original_muse_source_sha256"]
                    or saved.get("status") != ("correctness_and_compiler_captured"
                                                if key == "two_slab" else "screening_promising")
                    or not valid_checks):
                raise ValueError("saved trusted correctness or provenance is incompatible")
            identities[key]["historical_correctness"] = {"path": str(file.relative_to(root)),
                "sha256": sha(file.read_bytes()), "status": "historical PASS; rerun before capture"}
            identities[key]["original_muse_source_sha256"] = original["original_muse_source_sha256"]
    return {"record": record, "identities": identities}


def discover_profiler(*, executable: str | None = None, chip: str = "ad106") -> dict:
    executable = executable or shutil.which("ncu")
    if not executable:
        return {"status": "unavailable", "reason": "ncu executable not found", "metrics": [],
                "unsupported_metrics": list(METRICS)}
    version = subprocess.run([executable, "--version"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip().splitlines()[-1]
    queries = [subprocess.run([executable, "--query-metrics", "--query-metrics-mode", "all",
                              "--query-metrics-collection", collection, "--chips", chip],
                             capture_output=True, text=True, timeout=30)
               for collection in ("profiling", "launch")]
    if any(q.returncode or "==ERROR==" in q.stdout for q in queries):
        return {"status": "query_failed", "executable": executable, "version": version,
                "chip": chip, "reason": "\n".join(q.stderr + q.stdout[:300] for q in queries if q.returncode)[:300],
                "metrics": [], "unsupported_metrics": list(METRICS)}
    available = {}
    for query in queries:
        for line in query.stdout.splitlines():
            parts = line.split(maxsplit=3)
            if len(parts) >= 3 and parts[0] in METRICS:
                available[parts[0]] = parts[2] if len(parts) == 4 else None
    supported = [name for name in METRICS if name in available]
    return {"status": "ready" if supported else "no_selected_metrics",
            "executable": executable, "version": version, "chip": chip,
            "metrics": supported, "unsupported_metrics": [name for name in METRICS if name not in available],
            "metric_units": available,
            "query_method": "ncu --query-metrics --query-metrics-mode all --chips ad106 (profiling and launch collections); no kernel launch",
            "capability_scope": "offline chip catalog; runtime permissions and counter collection unverified"}


def prepare(root: Path = ROOT, output_base: Path | None = None,
            profiler: dict | None = None, now: str | None = None) -> tuple[Path, str]:
    root = Path(root).resolve()
    checked = verify_inputs(root)
    profiler = profiler if profiler is not None else discover_profiler()
    if profiler["status"] != "ready":
        raise ValueError("profiler capability query did not identify supported metrics")
    base = Path(output_base) if output_base is not None else root / "evaluations/planned-profiling"
    base.mkdir(parents=True, exist_ok=True)
    run_id = now or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"\d{8}T\d{12}Z", run_id):
        raise ValueError("invalid preparation run ID")
    folder = base / run_id
    folder.mkdir(exist_ok=False)
    captures = [{"kernel": key, "capture": number, "status": "not_run",
                 "raw_csv": f"{key}-{number}.csv", "command_log": f"{key}-{number}.log"}
                for number in (1, 2) for key in KERNELS]
    def package_version(name):
        try:
            return package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            return None

    manifest = {
        "schema_version": 1, "status": "prepared_only", "plan_run_id": PLAN_RUN,
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "plan_path": f"evaluations/experiment-planner/{PLAN_RUN}/plan.json",
        "plan_sha256": PLAN_SHA, "research_question": checked["record"]["plan"]["research_question"],
        "advisory_only": True, "human_approval_required_for_capture": True,
        "workload": {"shape_mnk": SHAPE, "dtype": "fp16", "gpu": GPU,
                     "input_seed": 42, "numeric_config": CONFIG},
        "source_identities": checked["identities"], "baseline_launcher_sha256": LAUNCHER_SHA,
        "correctness_prerequisite": "Historical six-case PASS is source-verified; each capture process must check FC1 seed 42 against scoped FP16 torch.mm before profiling.",
        "profiler": profiler, "metrics": [{"name": name, "definition": METRICS[name],
                                         "value": None, "unit": profiler.get("metric_units", {}).get(name),
                                         "status": "not_measured"}
                                        for name in profiler["metrics"]],
        "other_measurements": {"theoretical_occupancy": "launch occupancy limits selected where available; full theoretical occupancy percentage not selected",
                               "spills": "historical static compiler reports only; runtime spill counters not selected"},
        "captures": captures, "capture_order": [f"{x['kernel']}-{x['capture']}" for x in captures],
        "command_template": "python planned_profile.py --run-one --manifest <manifest-path> --manifest-sha256 <sha256> --kernel <baseline|mask_hoisting|two_slab> --capture <1|2>",
        "limits": {"ncu_launch_count": 1, "worker_gemm_launches_after_profiler_start": 1,
                   "warmup_launches": 10, "subprocess_timeout_seconds": 170,
                   "outer_timeout_seconds": 180, "max_csv_bytes": 10_000_000},
        "environment_at_preparation": {"python": sys.version.split()[0],
            "platform": {"sysname": os.uname().sysname, "release": os.uname().release,
                         "machine": os.uname().machine}, "ncu_version": profiler.get("version"),
            "torch_version": package_version("torch"), "triton_version": package_version("triton"),
            "driver_version": "not captured during preparation; verify before interpreting counters"},
        "evidence_schema": {"observation": "raw profiler CSV metrics with units, one selected launch per process",
            "interpretation": None, "speculation": None, "measurement_method": "Nsight Compute kernel replay; distinct from CUDA-event latency"},
        "limitations": ["Offline metric availability does not verify runtime counter permissions.",
            "Profiler replay, cache state, and clock behavior may perturb measurements.",
            "No captured values or performance improvement are claimed.",
            "Current source hashes match historical correctness artifacts, but hardware/software state must be checked again before capture."],
    }
    path = folder / "manifest.json"
    raw = json_bytes(manifest)
    path.write_bytes(raw)
    digest = sha(raw)
    commands = [f"python -m json.tool {shlex.quote(str(path))}"]
    commands += [" ".join(["timeout", "180s", shlex.quote(sys.executable),
        shlex.quote(str(root / "planned_profile.py")), "--run-one", "--manifest",
        shlex.quote(str(path)), "--manifest-sha256", digest, "--kernel", row["kernel"],
        "--capture", str(row["capture"])]) for row in captures]
    (folder / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    return path, digest


def checked_manifest(path: Path, expected_sha: str) -> dict:
    path = Path(path).resolve()
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not path.is_relative_to(OUTPUT_BASE.resolve())
            or path.name != "manifest.json" or path.parent.is_symlink()):
        raise ValueError("invalid manifest path or digest")
    raw = path.read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError("manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "prepared_only"
            or manifest.get("plan_sha256") != PLAN_SHA
            or manifest.get("runner_sha256") != sha(Path(__file__).read_bytes())
            or manifest.get("workload") != {"shape_mnk": SHAPE, "dtype": "fp16", "gpu": GPU,
                                           "input_seed": 42, "numeric_config": CONFIG}
            or manifest.get("limits") != {"ncu_launch_count": 1,
                "worker_gemm_launches_after_profiler_start": 1, "warmup_launches": 10,
                "subprocess_timeout_seconds": 170, "outer_timeout_seconds": 180,
                "max_csv_bytes": 10_000_000}):
        raise ValueError("manifest identity mismatch")
    profiler = manifest.get("profiler", {})
    metrics = profiler.get("metrics", [])
    installed = shutil.which("ncu")
    if (profiler.get("status") != "ready" or profiler.get("chip") != "ad106"
            or not installed or Path(profiler.get("executable", "")).resolve() != Path(installed).resolve()
            or not isinstance(metrics, list) or not 1 <= len(metrics) <= len(METRICS)
            or len(set(metrics)) != len(metrics) or any(name not in METRICS for name in metrics)):
        raise ValueError("profiler executable or selected metrics changed")
    checked = verify_inputs(ROOT)
    if manifest.get("source_identities") != checked["identities"]:
        raise ValueError("source identities changed since preparation")
    return manifest


def capture_worker(manifest: dict, kernel: str) -> None:
    # The hidden worker is still explicit and hash-bound, with its own time limit.
    signal.alarm(160)
    import importlib
    import torch
    import triton
    import reference_precision

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != GPU:
        raise RuntimeError("expected RTX 4060 Ti is unavailable")
    module = importlib.import_module(Path(KERNELS[kernel][0]).stem)
    m, n, k = SHAPE
    generator = torch.Generator(device="cuda")
    generator.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16, generator=generator)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    reference, _ = reference_precision.fp16_mm_reference(a, b)
    module.launch(a, b, out, CONFIG)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
    for _ in range(manifest["limits"]["warmup_launches"]):
        module.launch(a, b, out, CONFIG)
    torch.cuda.synchronize()
    print(json.dumps({"marker": "PROFILE_TARGET", "kernel": kernel,
                      "shape_mnk": SHAPE, "seed": 42, "correctness": "PASS",
                      "torch": torch.__version__, "triton": triton.__version__}), file=sys.stderr, flush=True)
    runtime = torch.cuda.cudart()
    runtime.cudaProfilerStart()
    try:
        module.launch(a, b, out, CONFIG)
        torch.cuda.synchronize()
    finally:
        runtime.cudaProfilerStop()


def run_one(path: Path, digest: str, kernel: str, capture: int) -> Path:
    manifest = checked_manifest(path, digest)
    if kernel not in KERNELS or capture not in (1, 2):
        raise ValueError("invalid fixed capture selection")
    profiler = manifest["profiler"]
    if profiler.get("status") != "ready" or not profiler.get("metrics"):
        raise ValueError("no verified profiler metrics")
    executable = profiler["executable"]
    if not Path(executable).is_file():
        raise ValueError("recorded ncu executable is unavailable")
    folder = Path(path).resolve().parent
    stem = f"{kernel}-{capture}"
    csv_path, log_path = folder / f"{stem}.csv", folder / f"{stem}.log"
    for output in (csv_path, log_path):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"capture output already exists: {output.name}")
    command = [executable, "--target-processes", "application-only", "--profile-from-start", "off",
               "--launch-count", "1", "--cache-control", "none", "--clock-control", "none",
               "--kernel-name", "regex:.*matmul_kernel.*",
               "--metrics", ",".join(profiler["metrics"]), "--csv", "--page", "raw",
               sys.executable, str(ROOT / "planned_profile.py"), "--worker", "--manifest",
               str(Path(path).resolve()), "--manifest-sha256", digest, "--kernel", kernel]
    with csv_path.open("xb") as csv_file, log_path.open("xb") as log_file:
        log_file.write((shlex.join(command) + "\n").encode())
        log_file.flush()
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=csv_file,
                                       stderr=log_file, timeout=170, check=False)
        except subprocess.TimeoutExpired:
            log_file.write(b"\nCAPTURE_TIMEOUT\n")
            raise
        if completed.returncode:
            raise RuntimeError(f"ncu capture failed with exit code {completed.returncode}; see {log_path}")
    if csv_path.stat().st_size > 10_000_000:
        raise RuntimeError("raw profiler CSV exceeded 10 MB bound")
    return csv_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hash-bound planned Nsight Compute capture")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-one", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--kernel", choices=tuple(KERNELS))
    parser.add_argument("--capture", type=int, choices=(1, 2))
    args = parser.parse_args(argv)
    if args.prepare_only:
        path, digest = prepare()
        print(f"Prepared only: {path}\nManifest SHA-256: {digest}\nCommands: {path.parent / 'COMMANDS.txt'}")
        return 0
    if not args.manifest or not args.manifest_sha256 or not args.kernel:
        parser.error("capture requires --manifest, --manifest-sha256, and --kernel")
    if args.run_one:
        if args.capture is None:
            parser.error("--run-one requires --capture")
        print(f"Saved raw profile: {run_one(args.manifest, args.manifest_sha256, args.kernel, args.capture)}")
    else:
        capture_worker(checked_manifest(args.manifest, args.manifest_sha256), args.kernel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
