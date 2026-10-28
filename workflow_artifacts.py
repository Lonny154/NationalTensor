"""Validate manifested trusted diagnostics and expose completed workflow evidence."""

import json
from pathlib import Path
import re

from grouped_confirmation_memory import _read
import mask_hoisting_memory
from research_workflow import SECTION, REGISTRY, RUNNER_ID, SOURCE_FILES, INPUT_FILES, JOB_ID, digest, read_json


def validate_job_artifacts(root, job):
    root = Path(root).resolve()
    if not JOB_ID.fullmatch(job["id"]):
        raise ValueError("invalid job ID")
    if job["status"] == "completed" and (
            not re.fullmatch(r"[0-9a-f]{64}", job.get("manifest_sha256", ""))
            or set(job.get("log_sha256", {})) != {"stdout.log", "stderr.log"}):
        raise ValueError("completed job lacks manifest or execution-log fingerprints")
    bound = job["binding"]
    spec = REGISTRY.get(bound["runner_id"])
    if (spec is None or bound["runner_id"] != RUNNER_ID or bound["plan_sha256"] != spec["plan_sha256"]
            or bound["shape_mnk"] != spec["shape_mnk"] or bound["config"] != spec["config"]
            or bound["dtype"] != "fp16" or bound["timeout_seconds"] != spec["timeout_seconds"]
            or bound["phases"] != spec["phases"] or bound["purpose"] != "workflow_validation"
            or bound["independent_research_evidence"] is not False
            or set(bound["source_sha256"]) != set(SOURCE_FILES)
            or set(bound["input_sha256"]) != set(INPUT_FILES)):
        raise ValueError("job is outside the single registered diagnostic scope")
    approval_path = f"{SECTION}/reviews/{job['plan_run_id']}/review.json"
    review = json.loads(_read(root, approval_path, job["approval_sha256"]))
    if (job["approval_path"] != approval_path or review["decision"] != "approved"
            or review["binding"] != bound or review["plan_run_id"] != job["plan_run_id"]
            or bound["plan_run_id"] != job["plan_run_id"]):
        raise ValueError("approval/job binding mismatch")
    from research_workflow import saved_plan
    if saved_plan(root, job["plan_run_id"])["plan_sha256"] != bound["plan_sha256"]:
        raise ValueError("approved plan changed")
    folder = f"{SECTION}/jobs/{job['id']}/artifacts/"
    manifest_raw = _read(root, folder + "manifest.json", job.get("manifest_sha256"))
    manifest = json.loads(manifest_raw)
    if (manifest["approval_sha256"] != job["approval_sha256"]
            or manifest["binding"] != bound or manifest["purpose"] != "workflow_validation"):
        raise ValueError("artifact manifest binding mismatch")
    files = manifest["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 100:
        raise ValueError("invalid artifact inventory")
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid artifact path")
        _read(root, folder + name, expected)
    actual = {str(path.relative_to(root / folder)) for path in (root / folder).rglob("*") if path.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("unmanifested or missing artifacts")
    for name, expected in bound["source_sha256"].items():
        _read(root, folder + "sources/" + name, expected)
    for name, expected in bound["input_sha256"].items():
        _read(root, folder + "inputs/" + name, expected)
    hashes = {name: files[name] for name in mask_hoisting_memory.HASHES}
    correctness = json.loads(_read(root, folder + "correctness.json"))
    if (correctness.get("verifier_file") != spec["script"]
            or correctness["verifier_sha256"] != bound["source_sha256"][spec["script"]]):
        raise ValueError("correctness was not produced by the approved trusted adapter")
    entry, warnings = mask_hoisting_memory.validate_directory(
        root, folder, hashes, source_root=root / folder / "sources", require_equivalent=False)
    if not entry or warnings:
        raise ValueError("; ".join(warnings) or "diagnostic evidence unavailable")
    source = entry["diagnostic_evidence"]["source_identity"]
    if (source["plan_sha256"] != bound["plan_sha256"]
            or source["runner_sha256"] != bound["source_sha256"]["mask_hoisting_m128_experiment.py"]
            or source["prior_sha256"] != bound["input_sha256"][source["prior_path"]]
            or source["trusted_module_sha256"] != bound["source_sha256"]["trusted_compute_kernel.py"]):
        raise ValueError("executed diagnostic identity differs from approval")
    for name, expected in job.get("log_sha256", {}).items():
        if name not in {"stdout.log", "stderr.log"}:
            raise ValueError("invalid log path")
        _read(root, f"{SECTION}/jobs/{job['id']}/{name}", expected)
    entry.update(id=f"workflow:{job['id']}", workflow_validation=True, independent_research_evidence=False)
    entry["provenance"] += "; workflow validation replay, not new independent research evidence"
    entry["diagnostic_evidence"]["workflow"] = {
        "job_id": job["id"], "approval_path": job["approval_path"],
        "approval_sha256": job["approval_sha256"], "purpose": "workflow_validation"}
    return entry, digest(manifest_raw)


def load_completed(root):
    root = Path(root).resolve()
    base = root / SECTION / "jobs"
    entries, warnings = [], []
    if not base.exists():
        return entries, warnings
    for directory in sorted(base.iterdir()):
        if not JOB_ID.fullmatch(directory.name):
            continue
        try:
            job = read_json(root, f"{SECTION}/jobs/{directory.name}/job.json")
            if job["id"] != directory.name:
                raise ValueError("job directory identity mismatch")
            if job["status"] == "completed":
                entry, _ = validate_job_artifacts(root, job)
                entries.append(entry)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Workflow artifact rejected ({directory.name}): {exc}")
    return entries, warnings
