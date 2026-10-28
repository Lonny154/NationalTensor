"""Auditable, advisory-only Muse research plans built from experiment memory.

No plan is executable. A live request requires --request-from-muse; importing
this module or preparing a prompt never contacts Muse or touches the GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from dashboard_data import ROOT, RUN_ID, load_history
import experiment_memory


MODEL = "muse-spark-1.3"
API_BASE_URL = "https://api.meta.ai/v1"
PLAN_SECTION = "evaluations/experiment-planner"
MAX_FILE_BYTES = 256_000
MAX_CONTEXT = 8
KINDS = {
    "kernel_transformation", "workload_shape", "numeric_tuning",
    "memory_movement", "profiling", "do_not_repeat",
}
OBSERVATIONS = {
    "correctness", "compilation", "performance_interpretation", "screening",
    "compiler_evidence", "compiler_resources", "confirmation", "provenance",
    "outcome", "shape_mnk",
}
FIELDS = {
    "research_question", "experiment_kind", "target_shape_mnk",
    "optimization_mechanism", "technical_hypothesis", "supporting_evidence",
    "reasoning_from_evidence", "difference_from_previous",
    "expected_performance_mechanism", "risks_and_uncertainties",
    "correctness_plan", "measurement_plan", "support_criteria",
    "contradict_criteria",
}
EVIDENCE_FIELDS = {"id", "artifact_path", "observation_keys", "why_relevant"}


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _audit_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _text(value, name: str, *, limit=1200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    if "```" in value or "@triton.jit" in value:
        raise ValueError(f"{name} must describe research, not contain source code")
    return value


def _shape(value) -> list[int]:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(number) is not int or not 1 <= number <= 8192 for number in value)):
        raise ValueError("target_shape_mnk must contain three positive dimensions at most 8192")
    return value


def _observation_available(entry: dict, key: str) -> bool:
    if key in {"screening", "compiler_evidence", "compiler_resources"}:
        return bool(entry.get(key))
    return entry.get(key) is not None


def parse_plan(raw: str, evidence: list[dict]) -> dict:
    """Validate exact JSON schema and every citation against supplied memory."""
    if not isinstance(raw, str) or len(raw) > 32_000:
        raise ValueError("model response is not bounded JSON text")
    try:
        plan = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("model response is not valid JSON") from exc
    if not isinstance(plan, dict) or set(plan) != FIELDS:
        raise ValueError("plan must contain exactly the advisory schema fields")
    if not isinstance(plan["experiment_kind"], str) or plan["experiment_kind"] not in KINDS:
        raise ValueError("unsupported experiment_kind")
    plan["target_shape_mnk"] = _shape(plan["target_shape_mnk"])
    for name in FIELDS - {"experiment_kind", "target_shape_mnk",
                          "supporting_evidence", "risks_and_uncertainties"}:
        plan[name] = _text(plan[name], name)
    risks = plan["risks_and_uncertainties"]
    if not isinstance(risks, list) or not 1 <= len(risks) <= 5:
        raise ValueError("risks_and_uncertainties must contain one to five items")
    plan["risks_and_uncertainties"] = [_text(item, "risk", limit=500) for item in risks]
    references = plan["supporting_evidence"]
    if not isinstance(references, list) or not 1 <= len(references) <= 5:
        raise ValueError("supporting_evidence must contain one to five references")
    indexed = {item["id"]: item for item in evidence}
    seen = set()
    for ref in references:
        if not isinstance(ref, dict) or set(ref) != EVIDENCE_FIELDS:
            raise ValueError("evidence reference has unsupported fields")
        identifier = ref["id"]
        item = indexed.get(identifier) if isinstance(identifier, str) else None
        if item is None or identifier in seen or ref["artifact_path"] != item["artifact_path"]:
            raise ValueError("evidence reference is absent, repeated, or has a mismatched path")
        seen.add(identifier)
        keys = ref["observation_keys"]
        if (not isinstance(keys, list) or not 1 <= len(keys) <= 4
                or len(set(map(str, keys))) != len(keys)
                or any(not isinstance(key, str) or key not in OBSERVATIONS
                       or not _observation_available(item, key) for key in keys)):
            raise ValueError("evidence observation key is unavailable")
        ref["why_relevant"] = _text(ref["why_relevant"], "why_relevant", limit=500)
    return plan


def direction_key(plan: dict) -> tuple:
    """Conservative exact-direction comparison; related ideas remain eligible."""
    mechanism = re.sub(r"\s+", " ", plan["optimization_mechanism"].casefold()).strip()
    return (plan["experiment_kind"], tuple(plan["target_shape_mnk"]), mechanism)


def _evidence_provenance(plan: dict, evidence: list[dict]) -> list[dict]:
    by_id = {row["id"]: row for row in evidence}
    return [{"id": ref["id"], "artifact_path": ref["artifact_path"],
             "original_muse_source_sha256":
                 by_id[ref["id"]].get("original_muse_source_sha256"),
             "trusted_implementation_sha256":
                 by_id[ref["id"]].get("trusted_implementation_sha256")}
            for ref in plan["supporting_evidence"]]


def build_context(root: Path = ROOT) -> dict:
    history = load_history(root)
    memory = experiment_memory.from_history(history)
    if not memory["entries"]:
        raise ValueError("No verified experiment memory is available")
    proposals = [item for item in history["proposals"]
                 if item["kind"] == "kernel" and item["source_hash_valid"] is True]
    if not proposals:
        raise ValueError("No verified generated proposal supplies the FC1 context")
    latest = max(proposals, key=lambda item: item["created_utc"])
    gpu = latest["gpu"]
    numeric = latest.get("config")
    shapes = sorted({tuple(item["shape_mnk"]) for item in memory["entries"]
                     if isinstance(item.get("shape_mnk"), list)
                     and len(item["shape_mnk"]) == 3})
    arguments = {"dtype": "fp16", "gpu": gpu, "method": "python_launch_cuda_event",
                 "baseline_sha256": latest.get("baseline_sha256"),
                 "launcher_sha256": latest.get("launcher_sha256"),
                 "numeric_config": numeric}
    selected = experiment_memory.retrieve(memory, shape=(512, 3072, 768),
                                          limit=6, **arguments)
    for shape in shapes:
        if shape == (512, 3072, 768) or len(selected) >= MAX_CONTEXT:
            continue
        other = experiment_memory.retrieve(memory, shape=shape, limit=1, **arguments)
        if other and all(row["id"] != other[0]["id"] for row in selected):
            selected.append(other[0])
    return {"workload_focus_mnk": [512, 3072, 768], "dtype": "fp16",
            "gpu": gpu, "hardware": latest.get("hardware"),
            "current_numeric_config": numeric,
            "baseline_source_sha256": latest.get("baseline_sha256"),
            "launcher_source_sha256": latest.get("launcher_sha256"),
            "available_workload_shapes_mnk": [list(shape) for shape in shapes],
            "measurement_method": "python_launch_cuda_event",
            "relevant_experiment_memory": selected,
            "allowed_observation_keys": sorted(OBSERVATIONS),
            "memory_warnings": memory["warnings"][:8],
            "allowed_experiment_kinds": sorted(KINDS),
            "constraints": [
                "Recommend one advisory research step; do not return code or launch instructions.",
                "The recommendation may be profiling, a different workload, numeric tuning, data movement, a kernel transformation, or a decision not to repeat a low-value direction.",
                "Use only supplied experiment IDs, artifact paths, and available observation keys in supporting_evidence.",
                "Separate measured observations from predicted effects; explain why this differs from prior attempts.",
                "Trusted results apply to independently reviewed implementations, not their original Muse-generated sources.",
                "Screening is not confirmation. A/A variation and timing-method compatibility limit conclusions; do not pool absolute timings across runs.",
                "State correctness checks, bounded measurement or profiling method, risks, and criteria that would support or contradict the hypothesis.",
                "A single screening loss does not rule out an entire optimization family; scope conclusions to the tested implementation, configuration, workload, and method.",
                "Matching PTX instruction multisets or resource counts do not prove identical runtime performance, overlap, or stalls.",
                "Do not use A/A timing variation as a measure of profiler-counter uncertainty or claim an unmeasured bottleneck is established.",
            ]}


def build_prompt(context: dict) -> str:
    evidence = context["relevant_experiment_memory"]
    if not evidence:
        raise ValueError("A planner prompt requires at least one saved evidence record")
    cited = evidence[0]
    observation = next(key for key in ("screening", "correctness", "outcome")
                       if _observation_available(cited, key))
    example = {
        "research_question": "Which measured bottleneck should guide the next FC1 experiment?",
        "experiment_kind": "profiling",
        "target_shape_mnk": list(context["workload_focus_mnk"]),
        "optimization_mechanism": "Profile existing reviewed kernels before another change",
        "technical_hypothesis": "An unmeasured execution bottleneck may explain the saved screening outcome.",
        "supporting_evidence": [{
            "id": cited["id"], "artifact_path": cited["artifact_path"],
            "observation_keys": [observation],
            "why_relevant": "This saved record supplies an observation for the selected workload."}],
        "reasoning_from_evidence": "The cited record is an observation; the bottleneck is a prediction to test.",
        "difference_from_previous": "This profiles reviewed implementations instead of changing kernel source.",
        "expected_performance_mechanism": "A measured bottleneck could guide a later optimization; no speedup is claimed.",
        "risks_and_uncertainties": ["Profiling changes execution conditions and does not establish latency."],
        "correctness_plan": "Use the existing FP16 reference policy for any changed implementation.",
        "measurement_plan": "Keep profiler observations separate from paired CUDA-event screening.",
        "support_criteria": "A reproducible profiler observation identifies a specific bottleneck to test.",
        "contradict_criteria": "Profiler observations do not support the proposed bottleneck.",
    }
    parse_plan(json.dumps(example), evidence)
    kinds = ", ".join(f'"{kind}"' for kind in sorted(KINDS))
    rules = (
        "Return ONLY one JSON object with exactly the example's keys and no code. "
        "experiment_kind MUST be one JSON string, not an array; choose exactly one of: "
        + kinds + ". "
        "target_shape_mnk MUST be a JSON array of exactly three positive integers at most 8192, "
        "not a quoted string. supporting_evidence MUST be a JSON array of 1-5 objects; "
        "each object must have exactly id (string), artifact_path (string), "
        "observation_keys (array of 1-4 allowed strings), and why_relevant (string). "
        "Use only the exact id and artifact_path pairs and available observation keys "
        "from the supplied experiment memory. risks_and_uncertainties is an array of "
        "1-5 strings; every other field is a nonempty string. "
        "This example shows JSON types only; its research content is not an observation "
        "or a requested recommendation. VALID_JSON_EXAMPLE: "
    )
    return ("Recommend one next NationalTensor experiment for human review only. "
            + rules + json.dumps(example, sort_keys=True, allow_nan=False) + "\n"
            + json.dumps(context, sort_keys=True, allow_nan=False))


def _folder(base: Path) -> Path:
    folder = base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _read_bounded(path: Path) -> bytes | None:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_bytes()
    except OSError:
        return None


def load_saved_plans(root: Path = ROOT) -> tuple[list[dict], list[str]]:
    """Load only hash-verified advisory plans from the fixed artifact section."""
    root = Path(root).resolve()
    base = (root / PLAN_SECTION).resolve()
    plans, warnings = [], []
    if (not base.is_relative_to(root) or (root / "evaluations").is_symlink()
            or (root / PLAN_SECTION).is_symlink()):
        return [], ["Planner artifact directory escapes the repository"]
    try:
        folders = sorted(p for p in base.iterdir() if p.is_dir() and not p.is_symlink()
                         and RUN_ID.fullmatch(p.name) and p.resolve().is_relative_to(base))
    except OSError:
        return [], []
    for folder in folders:
        metadata_raw = _read_bounded(folder / "metadata.json")
        if not metadata_raw:
            warnings.append(f"Missing planner metadata: {folder.name}")
            continue
        try:
            metadata = json.loads(metadata_raw)
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
        except (ValueError, TypeError):
            warnings.append(f"Malformed planner metadata: {folder.name}")
            continue
        if metadata.get("status") != "advisory_saved":
            continue
        plan_raw = _read_bounded(folder / "plan.json")
        audit_raw = _read_bounded(folder / "proposal_audit.jsonl")
        if not metadata_raw or not plan_raw or not audit_raw:
            warnings.append(f"Incomplete planner artifact: {folder.name}")
            continue
        try:
            plan = json.loads(plan_raw)
            audit = json.loads(audit_raw.splitlines()[-1])
            supplied = json.loads(audit["prompt"].split("\n", 1)[1])
            validated = parse_plan(audit["raw_response"],
                                   supplied["relevant_experiment_memory"])
            if (not isinstance(plan, dict)
                    or metadata.get("advisory_only") is not True
                    or metadata.get("model") != MODEL
                    or hashlib.sha256(plan_raw).hexdigest() != metadata.get("plan_sha256")
                    or validated != plan
                    or metadata.get("evidence_provenance") != _evidence_provenance(
                        plan, supplied["relevant_experiment_memory"])):
                raise ValueError("mismatched planner record")
            plans.append({"run_id": folder.name, "created_utc": metadata.get("created_utc"),
                          "plan": plan, "plan_sha256": metadata["plan_sha256"],
                          "model": metadata.get("model"),
                          "evidence_provenance": metadata["evidence_provenance"],
                          "artifact_path": f"{PLAN_SECTION}/{folder.name}/plan.json"})
        except (ValueError, TypeError, KeyError, IndexError):
            warnings.append(f"Malformed or mismatched planner artifact: {folder.name}")
    return plans, warnings


def propose_to_directory(client, prompt: str, context: dict, root: Path = ROOT) -> tuple[Path, dict]:
    """Make one mocked or explicitly authorized call; audit failures too."""
    base = Path(root) / PLAN_SECTION
    folder = _folder(base)
    metadata = {"status": "request_failed", "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": MODEL, "advisory_only": True,
                "evidence_ids": [row["id"] for row in context["relevant_experiment_memory"]]}
    started = time.perf_counter()
    try:
        response = client.responses.create(model=MODEL, input=prompt)
    except Exception as exc:
        audit = {"attempt": 1, "prompt": prompt, "error_type": type(exc).__name__,
                 "api_wall_s": time.perf_counter() - started}
        (folder / "proposal_audit.jsonl").write_bytes(_audit_bytes(audit))
        metadata["error_type"] = type(exc).__name__
    else:
        raw = getattr(response, "output_text", None)
        audit = {"attempt": 1, "prompt": prompt, "raw_response": raw,
                 "response_id": getattr(response, "id", None),
                 "api_wall_s": time.perf_counter() - started}
        try:
            plan = parse_plan(raw, context["relevant_experiment_memory"])
            prior, _ = load_saved_plans(root)
            match = next((item for item in prior
                          if direction_key(item["plan"]) == direction_key(plan)), None)
            if match:
                metadata.update(status="duplicate_direction",
                                duplicate_of=match["artifact_path"])
                audit["duplicate_of"] = match["artifact_path"]
            else:
                content = _json_bytes(plan)
                (folder / "plan.json").write_bytes(content)
                metadata["evidence_provenance"] = _evidence_provenance(
                    plan, context["relevant_experiment_memory"])
                metadata.update(status="advisory_saved",
                                plan_sha256=hashlib.sha256(content).hexdigest())
        except (ValueError, TypeError, KeyError) as exc:
            metadata["status"] = "invalid_response"
            audit["rejection"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        (folder / "proposal_audit.jsonl").write_bytes(_audit_bytes(audit))
    (folder / "metadata.json").write_bytes(_json_bytes(metadata))
    return folder, metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Advisory NationalTensor experiment planner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="save prompt, no API call")
    mode.add_argument("--request-from-muse", action="store_true", help="one live advisory request")
    args = parser.parse_args(argv)
    context = build_context(ROOT)
    prompt = build_prompt(context)
    if args.prepare_only:
        folder = _folder(ROOT / PLAN_SECTION)
        (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
        (folder / "metadata.json").write_bytes(_json_bytes({
            "status": "prepared_only", "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL, "advisory_only": True}))
        print(f"Saved advisory request without API call: {folder}")
        return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        parser.error("MODEL_API_KEY is required for --request-from-muse")
    client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0)
    folder, metadata = propose_to_directory(client, prompt, context, ROOT)
    print(f"Saved advisory planner artifacts: {folder}; status: {metadata['status']}")
    return 0 if metadata["status"] == "advisory_saved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
