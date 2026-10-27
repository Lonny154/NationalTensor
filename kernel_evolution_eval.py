"""Isolated worker for explicitly approved generated Triton kernels.

Run only inside an externally configured GPU container with no network,
resource limits, a read-only root and candidate mount, and a separate writable
artifact mount. Local checks fail closed but are not themselves a sandbox.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import tempfile

import torch
import triton
import triton.language as tl

import benchmark_core
import benchmark_proposal
import kernel_evolution as proposal
import optimize
import reference_precision


BASELINE = dict(optimize.BASELINE)
PRIMARY_SHAPE = proposal.SHAPE
CORRECTNESS_SHAPES = (PRIMARY_SHAPE, (509, 3071, 769), (65, 129, 33))
CORRECTNESS_SEEDS = (42, 43)


def _finite_cgroup_limit(path, maximum):
    try:
        text = Path(path).read_text().strip()
        return text != "max" and 0 < int(text) <= maximum
    except (OSError, ValueError):
        return False


def _cpu_cgroup_limited():
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return quota != "max" and 0 < int(quota) / int(period) <= 8
    except (OSError, ValueError, ZeroDivisionError):
        return False


def _root_mount_readonly():
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            if len(fields) > 5 and fields[4] == "/":
                return "ro" in fields[5].split(",")
    except OSError:
        pass
    return False


def verify_isolation(candidate_dir, approval_file, output):
    """Fail closed on missing operational controls; external review is still required."""
    failures = []
    if os.geteuid() == 0:
        failures.append("worker must run as a non-root user")
    try:
        interfaces = {path.name for path in Path("/sys/class/net").iterdir()}
    except OSError:
        interfaces = set()
    if interfaces != {"lo"}:
        failures.append("network namespace must expose only loopback")
    if not _finite_cgroup_limit("/sys/fs/cgroup/memory.max", 8 * 1024**3):
        failures.append("cgroup memory.max must be finite and <= 8 GiB")
    if not _finite_cgroup_limit("/sys/fs/cgroup/pids.max", 256):
        failures.append("cgroup pids.max must be finite and <= 256")
    if not _cpu_cgroup_limited():
        failures.append("cgroup cpu.max must cap execution at <= 8 CPUs")
    if not _root_mount_readonly():
        failures.append("container root filesystem must be read-only")
    for label, path in (("trusted code", Path(__file__)),
                        ("candidate", candidate_dir), ("approval", approval_file)):
        try:
            if not os.statvfs(path).f_flag & os.ST_RDONLY:
                failures.append(f"{label} mount must be read-only")
        except OSError:
            failures.append(f"cannot verify {label} mount")
    try:
        if os.statvfs(output.parent).f_flag & os.ST_RDONLY:
            failures.append("output mount must be writable")
    except OSError:
        failures.append("cannot verify output mount")
    if output.resolve().is_relative_to(candidate_dir.resolve()):
        failures.append("output must be separate from candidate source")
    if Path("/var/run/docker.sock").exists():
        failures.append("container control socket must not be mounted")
    for name in ("MODEL_API_KEY", "OPENAI_API_KEY", "AWS_ACCESS_KEY_ID",
                 "AWS_SECRET_ACCESS_KEY"):
        if os.getenv(name):
            failures.append(f"secret environment variable {name} must be absent")
    try:
        status = Path("/proc/self/status").read_text()
        if "Seccomp:\t2" not in status:
            failures.append("seccomp filtering must be active")
    except OSError:
        failures.append("cannot verify seccomp status")
    cpu_soft, _ = resource.getrlimit(resource.RLIMIT_CPU)
    if cpu_soft == resource.RLIM_INFINITY or cpu_soft > 900:
        failures.append("CPU time rlimit must be <= 900 seconds")
    nofile_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if nofile_soft == resource.RLIM_INFINITY or nofile_soft > 1024:
        failures.append("open-file rlimit must be <= 1024")
    if not torch.cuda.is_available():
        failures.append("isolated worker has no CUDA GPU")
    if failures:
        raise RuntimeError("isolation unavailable: " + "; ".join(failures))
    return {"network_interfaces": sorted(interfaces), "memory_max_bytes": int(
                Path("/sys/fs/cgroup/memory.max").read_text()),
            "pids_max": int(Path("/sys/fs/cgroup/pids.max").read_text()),
            "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "seccomp": "filtering", "root_mount": "read_only",
            "gpu": torch.cuda.get_device_name(0)}


def verify_approval(candidate_dir, approval_file):
    """Bind human approval to exact saved source and the current launcher."""
    metadata = json.loads((candidate_dir / "metadata.json").read_text())
    source = (candidate_dir / "candidate.py").read_text()
    approval = json.loads(approval_file.read_text())
    if metadata.get("status") != "awaiting_human_review":
        raise ValueError("candidate is not ready for review")
    digest = proposal.source_sha256(source)
    if digest != metadata.get("candidate_sha256"):
        raise ValueError("candidate source hash differs from proposal metadata")
    if metadata.get("baseline_sha256") != proposal.source_sha256(
            proposal.baseline_source()):
        raise ValueError("baseline source changed since proposal")
    if metadata.get("launcher_sha256") != proposal.source_sha256(
            inspect.getsource(benchmark_proposal.launch)):
        raise ValueError("launcher source changed since proposal")
    if metadata.get("shape_mnk") != list(PRIMARY_SHAPE):
        raise ValueError("unsupported workload shape")
    if (not isinstance(approval, dict)
            or set(approval) != {"scope", "candidate_sha256", "approved_by",
                                     "approved_at"}
            or approval["scope"] != "kernel_evolution_v1"
            or approval["candidate_sha256"] != digest
            or type(approval["approved_by"]) is not str
            or not approval["approved_by"].strip()):
        raise ValueError("explicit approval for this exact source hash is required")
    try:
        datetime.fromisoformat(approval["approved_at"])
    except (TypeError, ValueError):
        raise ValueError("approval timestamp must be ISO 8601") from None
    findings = proposal.inspect_kernel_source(source)
    return metadata, source, approval, findings


def load_reviewed_kernel(source):
    """Import a private copy only after approval and isolation checks pass."""
    temporary = tempfile.TemporaryDirectory(prefix="approved_triton_")
    path = Path(temporary.name) / "candidate.py"
    path.write_text(source)
    path.chmod(0o400)
    spec = importlib.util.spec_from_file_location("approved_triton_candidate", path)
    module = importlib.util.module_from_spec(spec)
    module.triton = triton
    module.tl = tl
    spec.loader.exec_module(module)
    return module.matmul_kernel, temporary


def candidate_launch(kernel, a, b, out, config):
    """Trusted copy of the fixed launch grid and signature."""
    m, k = a.shape
    _, n = b.shape
    bm, bn, bk = (config[name] for name in ("block_m", "block_n", "block_k"))
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    kernel[grid](a, b, out, m, n, k, bm, bn, bk,
                 num_warps=config["num_warps"], num_stages=config["num_stages"])


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


def check_correctness(kernel):
    checks = []
    primary = None
    for shape in CORRECTNESS_SHAPES:
        for seed in CORRECTNESS_SEEDS:
            a, b, reference, policy = _inputs(shape, seed)
            out = torch.empty((shape[0], shape[1]), device="cuda", dtype=torch.float16)
            record = {"shape_mnk": list(shape), "input_seed": seed,
                      "reference_precision": policy}
            try:
                candidate_launch(kernel, a, b, out, BASELINE)
                torch.cuda.synchronize()
            except Exception as exc:
                record.update({"status": "FAIL", "failure_stage": "compilation_or_execution",
                               "error": f"{type(exc).__name__}: {str(exc)[:1000]}"})
                checks.append(record)
                return checks, None
            try:
                torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
            except Exception as exc:
                record.update({"status": "FAIL", "failure_stage": "correctness",
                               "error": f"{type(exc).__name__}: {str(exc)[:1000]}"})
                checks.append(record)
                return checks, None
            record["status"] = "PASS"
            checks.append(record)
            if shape == PRIMARY_SHAPE and seed == 42:
                primary = (a, b, out)
    return checks, primary


def measure_paired(kernel, primary, *, confirmation):
    a, b, candidate_out = primary
    baseline_out = torch.empty_like(candidate_out)

    def launcher(x, y, out, config):
        if out is baseline_out:
            benchmark_proposal.launch(x, y, out, config)
        elif out is candidate_out:
            candidate_launch(kernel, x, y, out, config)
        else:
            raise ValueError("unexpected benchmark output buffer")

    timing = optimize.compare(
        a, b, baseline_out, candidate_out, BASELINE,
        14 if confirmation else 8, 2026 if confirmation else 43,
        warmups=10, iterations=200 if confirmation else 100,
        launcher=launcher, baseline_config=BASELINE)
    timing["candidate_launches"] = ["reviewed_generated_gemm"]
    return timing


def _write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def evaluate(candidate_dir, approval_file, output, *, mode, prior_evaluation=None):
    """No candidate import before exact approval and verified isolation."""
    metadata, source, approval, findings = verify_approval(candidate_dir, approval_file)
    isolation = verify_isolation(candidate_dir, approval_file, output)
    if mode == "confirm":
        if prior_evaluation is None:
            raise ValueError("confirmation requires a prior screening artifact")
        previous = json.loads(prior_evaluation.read_text())
        if (previous.get("candidate_sha256") != metadata["candidate_sha256"]
                or previous.get("baseline_sha256") != metadata["baseline_sha256"]
                or previous.get("launcher_sha256") != metadata["launcher_sha256"]
                or previous.get("shape_mnk") != list(PRIMARY_SHAPE)
                or previous.get("numeric_config") != BASELINE
                or previous.get("status") != "screening_promising"
                or previous.get("screening", {}).get("measurement_mode")
                != "python_launch_cuda_event"):
            raise ValueError("prior screening is incompatible or not promising")
    result = {"status": "incomplete", "mode": mode,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_dir": str(candidate_dir), "candidate_sha256":
              metadata["candidate_sha256"], "baseline_sha256":
              metadata["baseline_sha256"], "launcher_sha256":
              metadata["launcher_sha256"], "source_findings": findings,
              "hypothesis": metadata["hypothesis"], "approval": approval,
              "isolation": isolation, "gpu": torch.cuda.get_device_name(0),
              "shape_mnk": list(PRIMARY_SHAPE), "numeric_config": BASELINE,
              "measurement_method": optimize.benchmark_method(8, 7, 1.02),
              "correctness": []}
    temporary = None
    try:
        kernel, temporary = load_reviewed_kernel(source)
        result["compilation"] = {"import": "PASS", "first_launch": "not_attempted"}
        checks, primary = check_correctness(kernel)
        result["correctness"] = checks
        result["compilation"]["first_launch"] = (
            "FAIL" if checks[0]["status"] == "FAIL"
            and checks[0].get("failure_stage") == "compilation_or_execution"
            else "PASS")
        if primary is None:
            result["status"] = "failed"
            result["failure_stage"] = checks[-1].get("failure_stage")
            result["error"] = checks[-1].get("error")
            return result
        a, b, _ = primary
        result["aa"] = benchmark_core.run_aa_calibration(
            benchmark_proposal.launch, a, b, BASELINE,
            rounds=20, warmups=10, iterations=100, seed=42)
        timing = measure_paired(kernel, primary, confirmation=mode == "confirm")
        if mode == "confirm":
            decision = optimize.confirmation_decision(timing)
            result["confirmation"] = timing
            result["confirmation_decision"] = decision
            result["status"] = ("heuristically_confirmed" if decision["confirmed"]
                                else "unconfirmed")
        else:
            result["screening"] = timing
            result["status"] = ("screening_promising"
                                if timing["median_paired_speedup"] > 1.0
                                else "screening_no_gain")
    except Exception as exc:
        result["status"] = "failed"
        result["failure_stage"] = "import" if temporary is None else "measurement"
        if temporary is None:
            result["compilation"] = {"import": "FAIL", "first_launch": "not_attempted"}
        result["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
    finally:
        if temporary is not None:
            temporary.cleanup()
        _write_result(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--prior-evaluation", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output file already exists")
    result = evaluate(args.candidate_dir, args.approval_file, args.output,
                      mode=args.mode, prior_evaluation=args.prior_evaluation)
    print(f"Saved {result['status']} evaluation: {args.output}")
    return 0 if result["status"] not in {"failed", "incomplete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
