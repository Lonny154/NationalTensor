"""Read-only, bounded loaders for the saved Day 2 research artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SECTIONS = (
    "structural-loop", "kernel-evolution", "trusted-compute",
    "trusted-two-slab", "day2-reliability",
)
RUN_ID = re.compile(r"\d{8}T\d{12}Z\Z")
MAX_TEXT_BYTES = 4_000_000


def _read_text(root: Path, section: str, relative: str, warnings: list[str],
               *, required: bool = False) -> str | None:
    """Read only a known section; never follow a link outside it."""
    if section not in SECTIONS:
        raise ValueError("unknown artifact section")
    base = (root / "evaluations" / section).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        warnings.append(f"Blocked path outside {section}: {relative}")
        return None
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            warnings.append(f"Artifact too large: {section}/{relative}")
            return None
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError):
        if required:
            warnings.append(f"Missing or unreadable: {section}/{relative}")
        return None


def _read_json(root: Path, section: str, relative: str,
               warnings: list[str], *, required: bool = False) -> dict | None:
    text = _read_text(root, section, relative, warnings, required=required)
    if text is None:
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        warnings.append(f"Malformed JSON: {section}/{relative}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"Expected JSON object: {section}/{relative}")
        return None
    return value


def _read_jsonl(root: Path, section: str, relative: str,
                warnings: list[str], *, required: bool = False) -> list[dict]:
    text = _read_text(root, section, relative, warnings, required=required)
    if text is None:
        return []
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("expected object")
            rows.append(value)
        except ValueError:
            warnings.append(f"Malformed JSONL: {section}/{relative}:{number}")
    return rows


def _runs(root: Path, section: str) -> list[str]:
    base = (root / "evaluations" / section).resolve()
    try:
        return sorted(entry.name for entry in base.iterdir()
                      if RUN_ID.fullmatch(entry.name) and entry.is_dir()
                      and entry.resolve().is_relative_to(base))
    except OSError:
        return []


def _prompt_context(prompt: str | None) -> dict:
    if not isinstance(prompt, str) or "\n" not in prompt:
        return {}
    try:
        value = json.loads(prompt.split("\n", 1)[1])
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def _method(record: dict) -> str:
    method = record.get("measurement_method", {})
    if isinstance(method, dict):
        return str(method.get("measurement_mode", "unknown"))
    return str(method or "unknown")


def _correctness(rows: object) -> tuple[str, int, int]:
    if not isinstance(rows, list):
        return "unknown", 0, 0
    passed = sum(isinstance(row, dict) and row.get("status") == "PASS"
                 for row in rows)
    return ("PASS" if rows and passed == len(rows) else "FAIL",
            passed, len(rows))


def _structural(root: Path, warnings: list[str]) -> list[dict]:
    proposals = []
    for run in _runs(root, "structural-loop"):
        prefix = f"{run}/"
        metadata = _read_json(root, "structural-loop", prefix + "metadata.json",
                              warnings) or {}
        experiments = _read_jsonl(root, "structural-loop",
                                  prefix + "experiments.jsonl", warnings,
                                  required=True)
        audits = _read_jsonl(root, "structural-loop",
                             prefix + "proposal_audit.jsonl", warnings)
        aa = _read_json(root, "structural-loop", prefix + "aa.json", warnings)
        for index, row in enumerate(experiments):
            number = row.get("number")
            if not isinstance(number, int) or number < 1:
                warnings.append(f"Invalid structural experiment number: {run}")
                continue
            audit = audits[index] if index < len(audits) else {}
            context = _prompt_context(audit.get("prompt"))
            timing = row.get("timing") if isinstance(row.get("timing"), dict) else None
            outcome = row.get("optimization_status") or "not_screened"
            shape = row.get("shape_mnk") if isinstance(row.get("shape_mnk"), list) else []
            proposals.append({
                "id": f"structural:{run}:{number}", "kind": "structural",
                "run_id": run, "number": number,
                "created_utc": row.get("created_utc") or metadata.get("created_utc") or "",
                "label": f"{row.get('schedule', 'unknown')} · {'×'.join(map(str, shape))}",
                "strategy": row.get("schedule"), "shape": shape,
                "dtype": metadata.get("dtype"), "config": row.get("config"),
                "gpu": row.get("gpu") or metadata.get("gpu"),
                "hypothesis": row.get("hypothesis", ""),
                "expected_mechanism": row.get("expected_mechanism", ""),
                "reasoning": row.get("reasoning_from_history", ""),
                "source": None, "source_sha256": None,
                "source_fingerprints": row.get("source_sha256"),
                "source_hash_valid": None, "static_status": "reviewed configuration",
                "generated_status": "reviewed strategy selected",
                "correctness": row.get("correctness", "unknown"),
                "outcome": outcome, "confirmed": False,
                "timing": timing, "aa": aa,
                "measurement_method": (timing or {}).get("measurement_mode")
                    or _method(row),
                "input_seed": row.get("input_seed"),
                "raw_prompt": audit.get("prompt"),
                "raw_response": audit.get("response"),
                "prompt_context": context,
                "prior_results_in_prompt": context.get("current_run_experiments", []),
                "related_trusted": [], "feedback_link": None,
                "artifact_path": f"evaluations/structural-loop/{run}/experiments.jsonl",
            })
    return proposals


def _kernel(root: Path, warnings: list[str]) -> tuple[list[dict], int]:
    proposals = []
    rejected = 0
    for run in _runs(root, "kernel-evolution"):
        prefix = f"{run}/"
        metadata = _read_json(root, "kernel-evolution", prefix + "metadata.json",
                              warnings, required=True)
        if metadata is None or metadata.get("status") != "awaiting_human_review":
            continue  # Prepared-only requests are not source proposals.
        source = _read_text(root, "kernel-evolution", prefix + "candidate.py",
                            warnings, required=True)
        if source is None:
            continue
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        hash_valid = source_hash == metadata.get("candidate_sha256")
        if not hash_valid:
            warnings.append(f"Generated source SHA-256 mismatch: {run}")
        audits = _read_jsonl(root, "kernel-evolution",
                             prefix + "proposal_audit.jsonl", warnings)
        rejected += sum("rejection" in row for row in audits)
        accepted = {}
        for row in audits:
            raw = row.get("raw_response")
            try:
                response = json.loads(raw) if isinstance(raw, str) else {}
            except ValueError:
                response = {}
            if (isinstance(response, dict)
                    and response.get("kernel_source") == source
                    and "rejection" not in row):
                accepted = row
                break
        if not accepted:
            warnings.append(f"Missing accepted proposal audit: {run}")
        context = _prompt_context(accepted.get("prompt"))
        proposals.append({
            "id": f"kernel:{run}", "kind": "kernel", "run_id": run,
            "created_utc": metadata.get("created_utc", ""),
            "label": f"Generated kernel · {run[9:15]}",
            "strategy": "generated Triton", "shape": metadata.get("shape_mnk"),
            "dtype": "fp16", "config": metadata.get("numeric_config"),
            "baseline_sha256": metadata.get("baseline_sha256"),
            "launcher_sha256": metadata.get("launcher_sha256"),
            "gpu": metadata.get("gpu"),
            "hypothesis": metadata.get("hypothesis", ""),
            "expected_mechanism": metadata.get("expected_mechanism", ""),
            "reasoning": metadata.get("reasoning_from_feedback", ""),
            "source": source, "source_sha256": metadata.get("candidate_sha256"),
            "source_hash_valid": hash_valid,
            "static_status": (metadata.get("static_findings") or {}).get(
                "ast_check", "unknown"),
            "static_findings": metadata.get("static_findings") or {},
            "audit_attempts": [{"attempt": row.get("attempt"),
                                "response_id": row.get("response_id"),
                                "api_wall_s": row.get("api_wall_s"),
                                "rejection": row.get("rejection")}
                               for row in audits],
            "generated_status": metadata.get("status"),
            "correctness": "not evaluated", "outcome": "unevaluated",
            "confirmed": False, "timing": None, "aa": None,
            "measurement_method": metadata.get("measurement_method", "not measured"),
            "input_seed": None, "raw_prompt": accepted.get("prompt"),
            "raw_response": accepted.get("raw_response"),
            "prompt_context": context,
            "prior_results_in_prompt": [], "related_trusted": [],
            "feedback_link": None,
            "review_from": metadata.get("review_from"),
            "artifact_path": f"evaluations/kernel-evolution/{run}/candidate.py",
        })
    return proposals, rejected


def _trusted(root: Path, warnings: list[str]) -> list[dict]:
    results = []
    for section in ("trusted-compute", "trusted-two-slab"):
        for run in _runs(root, section):
            prefix = f"{run}/"
            if section == "trusted-compute":
                check = _read_json(root, section, prefix + "experiment.json",
                                   warnings, required=True)
                screen = check
                generated_key = "muse_candidate_sha256"
            else:
                check = _read_json(root, section, prefix + "check.json",
                                   warnings, required=True)
                screen = _read_json(root, section, prefix + "screening.json",
                                    warnings)
                generated_key = "original_muse_candidate_sha256"
            if check is None:
                continue
            if screen and (screen.get(generated_key) != check.get(generated_key)
                           or screen.get("trusted_implementation_sha256")
                           != check.get("trusted_implementation_sha256")):
                warnings.append(f"Trusted artifact provenance mismatch: {section}/{run}")
                screen = None
            feedback = _read_json(root, section, prefix + "feedback.json", warnings)
            feedback_hash = None
            if feedback:
                raw = _read_text(root, section, prefix + "feedback.json", warnings)
                feedback_hash = hashlib.sha256(raw.encode()).hexdigest() if raw else None
            generated_hash = check.get(generated_key)
            trusted_hash = check.get("trusted_implementation_sha256")
            if feedback and (feedback.get("original_muse_candidate_sha256") != generated_hash
                             or feedback.get("trusted_implementation_sha256") != trusted_hash
                             or feedback.get("exact_generated_source_evaluated") is not False):
                warnings.append(f"Trusted feedback provenance mismatch: {section}/{run}")
                feedback = None
                feedback_hash = None
            correctness, passed, total = _correctness(check.get("correctness"))
            timing = screen.get("screening") if screen else None
            aa = screen.get("aa") if screen else None
            compiler = check.get("compiler") if isinstance(check.get("compiler"), dict) else {}
            results.append({
                "id": f"trusted:{section}:{run}", "kind": section, "run_id": run,
                "created_utc": (screen or check).get("created_utc", ""),
                "label": ("Mask/pipeline trusted test" if section == "trusted-compute"
                          else "Two-slab trusted test"),
                "gpu": check.get("gpu"), "shape": check.get("shape_mnk")
                    or [512, 3072, 768],
                "dtype": "fp16",
                "generated_sha256": generated_hash,
                "trusted_sha256": trusted_hash,
                "provenance": check.get("provenance"),
                "exact_generated_source_evaluated":
                    check.get("exact_generated_source_evaluated"),
                "correctness": correctness, "correctness_passed": passed,
                "correctness_total": total, "correctness_records": check.get("correctness", []),
                "status": (screen or check).get("status", "unknown"),
                "screening": timing, "aa": aa, "confirmed": False,
                "measurement_method": (timing or {}).get("measurement_mode")
                    or _method(screen or check),
                "compiler_resources": {name: {key: compiler.get(name, {}).get(key)
                                       for key in ("registers_per_thread", "spills",
                                                   "shared_memory_bytes")}
                                       for name in ("baseline", "trusted")},
                "ptx_instruction_counts": {name: compiler.get(name, {}).get(
                    "ptx_instruction_counts") for name in ("baseline", "trusted")},
                "compiler_review": (feedback or {}).get("compiler_review", {}),
                "dynamic_profile": (feedback or {}).get("dynamic_profile"),
                "feedback": feedback, "feedback_sha256": feedback_hash,
                "related_proposal": None,
                "artifact_path": f"evaluations/{section}/{run}/",
            })
    return results


def _reliability(root: Path, warnings: list[str]) -> dict:
    captures = []
    for name in ("aa_python_first_20261027.json", "aa_graph_first_20261027.json"):
        data = _read_json(root, "day2-reliability", name, warnings)
        if not data:
            continue
        metrics = data.get("comparison", {}).get("metrics", {})
        captures.append({"name": name, "order": data.get("comparison", {}).get(
            "mode_order", []), "metrics": metrics,
            "interpretation": data.get("comparison", {}).get("interpretation", ""),
            "created_utc": data.get("captured_utc", "")})
    settling = _read_json(root, "day2-reliability",
                          "idle_settling_20261027.json", warnings)
    analysis = settling.get("analysis", {}) if settling else {}
    return {"captures": captures, "settling": {
        "cells": analysis.get("cells", {}), "aa": analysis.get("aa", {}),
        "interpretation": analysis.get("interpretation", ""),
        "method": settling.get("method", {}) if settling else {},
    }}


def load_history(root: Path = ROOT) -> dict:
    """Load small records only; full Matmul Inspector comparisons stay lazy."""
    root = Path(root).resolve()
    warnings: list[str] = []
    structural = _structural(root, warnings)
    kernel, rejected = _kernel(root, warnings)
    trusted = _trusted(root, warnings)
    proposals = sorted(structural + kernel, key=lambda item: item["created_utc"])
    by_hash = {item["source_sha256"]: item for item in kernel
               if item["source_hash_valid"]}
    trusted_by_feedback = {item["feedback_sha256"]: item for item in trusted
                           if item["feedback_sha256"]}
    for item in trusted:
        proposal = by_hash.get(item["generated_sha256"])
        if proposal and item["exact_generated_source_evaluated"] is False:
            item["related_proposal"] = proposal["id"]
            proposal["related_trusted"].append(item["id"])
    for item in kernel:
        external = item["prompt_context"].get("external_trusted_implementation_feedback")
        if isinstance(external, dict):
            linked = trusted_by_feedback.get(external.get("feedback_sha256"))
            if (linked and external.get("original_muse_candidate_sha256")
                    == linked["generated_sha256"]
                    and external.get("trusted_implementation_sha256")
                    == linked["trusted_sha256"]):
                item["feedback_link"] = linked["id"]
    checkpoint = ""
    checkpoint_path = (root / "docs" / "day2_checkpoint.md").resolve()
    if checkpoint_path.is_relative_to((root / "docs").resolve()):
        try:
            checkpoint = checkpoint_path.read_text(encoding="utf-8")[:30_000]
        except (OSError, UnicodeError):
            warnings.append("Missing Day 2 checkpoint note")
    correctness_passed = sum(item["correctness_passed"] for item in trusted)
    correctness_total = sum(item["correctness_total"] for item in trusted)
    overview = {
        "muse_proposals": len(proposals), "structural_proposals": len(structural),
        "kernel_proposals": len(kernel), "rejected_responses": rejected,
        "trusted_implementations": len(trusted),
        "trusted_correctness_passed": correctness_passed,
        "trusted_correctness_total": correctness_total,
        "confirmed_improvements": sum(item["confirmed"] for item in trusted)
            + sum(item["confirmed"] for item in structural),
        "gpus": sorted({str(item["gpu"]) for item in proposals + trusted if item.get("gpu")}),
        "shapes": sorted({tuple(item["shape"]) for item in proposals + trusted
                          if isinstance(item.get("shape"), list)}),
    }
    timeline = sorted(proposals + trusted, key=lambda item: item["created_utc"])
    return {"proposals": proposals, "trusted": trusted, "timeline": timeline,
            "reliability": _reliability(root, warnings), "overview": overview,
            "checkpoint": checkpoint, "warnings": warnings}


def load_compiler_detail(root: Path, trusted_record: dict) -> dict:
    """Load large comparison JSON only for a selected trusted two-slab run."""
    if trusted_record.get("kind") != "trusted-two-slab":
        return {}
    run = trusted_record.get("run_id", "")
    if not RUN_ID.fullmatch(run):
        return {}
    warnings: list[str] = []
    result = {}
    for label, scope in (("ptx", "kernel"), ("sass", "function")):
        data = _read_json(Path(root).resolve(), "trusted-two-slab",
                          f"{run}/{label}-compare/comparison.json", warnings)
        if not data:
            continue
        sides = data.get("comparison", {}).get(scope, {})
        baseline = sides.get("baseline", {})
        candidate = sides.get("candidate", {})
        if isinstance(baseline, dict) and isinstance(candidate, dict):
            baseline_mnemonics = baseline.get("instruction_mnemonics", {})
            trusted_mnemonics = candidate.get("instruction_mnemonics", {})
            if not isinstance(baseline_mnemonics, dict):
                baseline_mnemonics = {}
            if not isinstance(trusted_mnemonics, dict):
                trusted_mnemonics = {}
            groups = (("cp.async (all forms)", "cp.async"), ("mma.sync", "mma.sync"),
                      ("ldmatrix", "ldmatrix")) if label == "ptx" else (
                      ("HMMA", "HMMA"), ("LDGSTS", "LDGSTS"), ("LDSM", "LDSM"))
            operations = [{"name": title,
                           "baseline": sum(value for key, value in baseline_mnemonics.items()
                                           if key.startswith(prefix) and isinstance(value, int)),
                           "trusted": sum(value for key, value in trusted_mnemonics.items()
                                          if key.startswith(prefix) and isinstance(value, int))}
                          for title, prefix in groups]
            result[label] = {"baseline_instructions": baseline.get("instruction_count"),
                             "trusted_instructions": candidate.get("instruction_count"),
                             "operations": operations}
    return {"comparisons": result, "warnings": warnings}
