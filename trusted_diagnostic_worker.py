"""Fixed approval-gated adapter for Experiment 003; never imports artifact Python."""

import argparse
import json
from pathlib import Path
import shutil
import signal
import shlex

from research_workflow import RUNNER_ID, REGISTRY, JOB_ID, digest, safe_path, verify_approval, registered_command


ROOT = Path(__file__).resolve().parent


def run(approval, output):
    approval = approval.resolve()
    run_id = approval.parent.name
    expected_approval = safe_path(ROOT, f"evaluations/research-workflow/reviews/{run_id}/review.json")
    if approval != expected_approval:
        raise ValueError("approval must be the registered workflow review")
    review, approval_sha = verify_approval(ROOT, run_id)
    binding = review["binding"]
    if binding["runner_id"] != RUNNER_ID:
        raise ValueError("unregistered worker")
    if (output.name != "artifacts" or not JOB_ID.fullmatch(output.parent.name)
            or output.parent.parent != ROOT / "evaluations/research-workflow/jobs"
            or output.exists()):
        raise ValueError("worker requires a fresh server-owned artifact directory")
    safe_path(ROOT, str(output.relative_to(ROOT)))
    # Import only after exact plan, fixed phases, and all reviewed source hashes pass.
    import torch
    import mask_hoisting_m128_experiment as experiment
    from mask_hoisting_memory import _sections
    diagnostic = experiment.diagnose(output)
    if diagnostic["status"] != "diagnostic_complete":
        raise RuntimeError("compiler diagnostic failed; see diagnostic.json")
    source_folder = output / "sources"
    source_folder.mkdir()
    for name, expected in binding["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError("approved source changed during execution")
        shutil.copy2(ROOT / name, source_folder / name)
    for name, expected in binding["input_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError("approved input artifacts changed during execution")
        destination = output / "inputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    checks = {"status": "incomplete", "shape_mnk": list(experiment.SHAPE),
              "tolerances": {"rtol": 0.02, "atol": 0.02}, "checks": [],
              "verifier_file": Path(__file__).name, "verifier_sha256": digest(Path(__file__).read_bytes())}
    try:
        for seed in (42, 43):
            a, b, reference, policy = experiment.trusted._inputs(experiment.SHAPE, seed)
            for name, launch in (("baseline", experiment.benchmark_proposal.launch),
                                 ("trusted", experiment.trusted_compute_kernel.launch)):
                out = torch.empty_like(reference)
                launch(a, b, out, experiment.CONFIG)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
                checks["checks"].append({"implementation": name, "seed": seed,
                                        "status": "PASS", "reference_precision": policy})
        checks["status"] = "PASS"
    finally:
        experiment.write_json(output / "correctness.json", checks)
    binary = {name: _sections((output / f"{name}.cubin").read_bytes()) for name in ("baseline", "trusted")}
    elf = {}
    for key in (".text.matmul_kernel", ".nv.constant0.matmul_kernel", ".nv.info", ".nv.info.matmul_kernel"):
        a, b = binary["baseline"][key], binary["trusted"][key]
        elf[key] = {"identical": a == b, "baseline_sha256": digest(a), "trusted_sha256": digest(b),
                    "size_bytes": len(a), "trusted_size_bytes": len(b)}
    experiment.write_json(output / "elf_comparison.json", elf)
    experiment.write_json(output / "result.json", {
        "status": "diagnostic_complete", "phase_2": {"status": "not_run", "median_paired_speedup": None,
            "raw_latency_measurements": [], "reason": "Approval permits compiler diagnosis and correctness only; runtime screening is intentionally skipped."},
        "performance_conclusion": "Not measured; no runtime performance conclusion.",
        "limitations": ["Workflow validation replay of completed Experiment 003; not independent research evidence.",
                        "Compiler equivalence does not establish equal measured runtime."]})
    (output / "COMMANDS.txt").write_text(
        "Workflow validation replay. Run only through a separately approved workflow job.\n"
        "Registered phases: compiler diagnosis and correctness. Runtime screening was not executed.\n"
        + shlex.join(registered_command(ROOT, output, approval)) + "\n")
    manifest = {"binding": binding, "approval_sha256": approval_sha, "purpose": "workflow_validation",
                "files": {str(path.relative_to(output)): digest(path.read_bytes())
                          for path in sorted(output.rglob("*")) if path.is_file()}}
    experiment.write_json(output / "manifest.json", manifest)
    print("Trusted diagnostic and four FP16 correctness checks completed; screening skipped.")


if __name__ == "__main__":
    # An OS-level deadline still terminates the trusted worker if the server exits.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(REGISTRY[RUNNER_ID]["timeout_seconds"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.approval, args.output.resolve())
