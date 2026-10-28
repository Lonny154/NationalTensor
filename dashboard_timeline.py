"""Read-only, renderer-neutral activity projection over existing verified research records."""

import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import experiment_memory
from dashboard_data import load_history
from dashboard_interpretation import status_for
from experiment_planner import load_saved_plans
from grouped_confirmation_memory import _read
from research_workflow import SECTION, JOB_ID, WorkflowManager, verify_approval


FILTERS = {"all": "All activity", "scientific": "Scientific experiments",
           "compiler": "Compiler diagnostics", "decisions": "Research decisions", "workflow": "Workflow validation"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _current_sources(root):
    """Mirror the existing source fingerprints without importing any GPU modules."""
    parsed = {}

    def source(filename, name):
        if filename not in parsed:
            text = _read(root, filename).decode()
            parsed[filename] = (text.splitlines(keepends=True), ast.parse(text))
        lines, tree = parsed[filename]
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return "".join(lines[start - 1:node.end_lineno])

    def functions(*specs):
        return _sha("\n".join(source(*spec) for spec in specs).encode())

    return source, functions


def _validate_timing(timing):
    if not timing:
        return
    rows = timing["paired_rounds"]
    if not rows or len(rows) % 2:
        raise ValueError("incomplete paired timing rounds")
    if timing["method"].get("balanced_order") is not True or any(
            first["order"] != list(reversed(second["order"])) for first, second in zip(rows[::2], rows[1::2])):
        raise ValueError("unbalanced timing protocol")
    ratios = []
    for row in rows:
        if sorted(row["order"]) != ["baseline", "candidate"]:
            raise ValueError("invalid timing order")
        values = [row[key] for key in ("baseline_ms", "candidate_ms", "baseline_over_candidate")]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("invalid timing value")
        if not math.isclose(values[0] / values[1], values[2], rel_tol=1e-9):
            raise ValueError("paired ratio contradicts raw latency")
        ratios.append(values[2])
    for side in ("baseline", "candidate"):
        samples = [row[side + "_ms"] for row in rows]
        if samples != timing["samples_ms"][side] or not math.isclose(statistics.median(samples), timing[side + "_ms"], rel_tol=1e-9):
            raise ValueError("latency summary contradicts raw samples")
    if not math.isclose(statistics.median(ratios), timing["median_paired_speedup"], rel_tol=1e-9):
        raise ValueError("speedup summary contradicts raw pairs")


def _validate_legacy(root, raw, source, functions):
    """Keep legacy records on their existing source and timing identity contracts."""
    if raw["kind"] == "kernel":
        if not raw.get("source_hash_valid") or not raw.get("raw_response"):
            raise ValueError("generated proposal source/audit unavailable")
        return
    if raw["kind"] == "structural":
        folder = Path(raw["artifact_path"]).parent
        metadata = json.loads(_read(root, str(folder / "metadata.json")))
        recorded = raw["source_fingerprints"]
        if recorded != metadata["source_sha256"]:
            raise ValueError("structural source identity mismatch")
        if (raw["shape"] != metadata["shape_mnk"] or raw["gpu"] != metadata["gpu"]
                or raw["input_seed"] != metadata["input_seed"]
                or any(raw["config"].get(key) != value for key, value in metadata["numeric_config"].items())):
            raise ValueError("structural workload/configuration mismatch")
        specs = {
            "direct": [("benchmark_proposal.py", "launch"), ("benchmark_proposal.py", "matmul_kernel")],
            "grouped": [("grouped_matmul.py", name) for name in ("grouped_matmul_kernel", "launch_grouped", "launch_scheduled")],
            "split_k": [("split_k_matmul.py", name) for name in ("partial_kernel", "reduction_kernel", "allocate_workspace", "launch_split_k")],
            "loop_dispatch": [("structural_loop.py", name) for name in ("launch_strategy", "checked_output", "compare_config", "evaluate_gpu")] + [("optimize.py", "compare")],
            "loop_policy": [("structural_loop.py", name) for name in ("parse_proposal", "build_prompt", "propose", "run_experiment_loop")],
            "confirmation_rule": [("optimize.py", "confirmation_decision")],
        }
        if any(recorded[key] != functions(*value) for key, value in specs.items()):
            raise ValueError("structural source hashes differ from reviewed code")
        for key, filename in (("benchmark_core_file", "benchmark_core.py"), ("reference_precision_file", "reference_precision.py")):
            if recorded[key] != _sha(_read(root, filename)):
                raise ValueError("measurement helper identity mismatch")
        _validate_timing(raw.get("timing"))
    else:
        folder = Path(raw["artifact_path"])
        check_name = "experiment.json" if raw["kind"] == "trusted-compute" else "check.json"
        check = json.loads(_read(root, str(folder / check_name)))
        module = "trusted_compute_kernel.py" if raw["kind"] == "trusted-compute" else "trusted_two_slab_kernel.py"
        if (check["trusted_module_sha256"] != _sha(_read(root, module))
                or check["trusted_implementation_sha256"] != _sha(source(module, "matmul_kernel").encode())
                or check["baseline_kernel_sha256"] != _sha(source("benchmark_proposal.py", "matmul_kernel").encode())
                or check["baseline_launcher_sha256"] != _sha(source("benchmark_proposal.py", "launch").encode())):
            raise ValueError("trusted source identity mismatch")
        for compiler in check["compiler"].values():
            for artifact in compiler["artifacts"].values():
                _read(root, str(folder / artifact["path"]), artifact["sha256"])
        if raw["kind"] == "trusted-two-slab" and raw.get("screening"):
            screen = json.loads(_read(root, str(folder / "screening.json")))
            for key in ("trusted_module_sha256", "trusted_implementation_sha256", "baseline_kernel_sha256", "baseline_launcher_sha256"):
                if screen[key] != check[key]:
                    raise ValueError("screening source identity mismatch")
        _validate_timing(raw.get("screening"))


def _event(identifier, date, title, event_type, outcome, filters, evidence, **extra):
    date_label = date[:19].replace("T", " ") + " UTC" if "T" in date else (date + " · time not recorded" if date else "Date not recorded")
    return {"id": identifier, "created_utc": date, "date_label": date_label, "title": title,
            "experiment_type": event_type, "outcome": outcome, "filters": filters, "evidence": evidence,
            "workflow_validation": "workflow" in filters, "measurement": None,
            "measurement_label": "Not measured", "correctness": None, "interpretation": "",
            "description": "", "artifact_path": evidence["artifact_path"], "tone": "neutral", **extra}


def filter_events(events, selected="all"):
    if selected not in FILTERS:
        raise ValueError("unknown timeline filter")
    return [row for row in events if selected == "all" or selected in row["filters"]]


def build_activity(root, *, history=None, memory=None):
    """No database or cache: every event is a projection of current validated evidence."""
    root = Path(root).resolve()
    history = load_history(root) if history is None else history
    memory = experiment_memory.reconstruct(root) if memory is None else memory
    warnings = list(memory["warnings"])
    raw_records = {row["id"]: row for row in history["timeline"]}
    source, functions = _current_sources(root)
    events = []
    for entry in memory["entries"]:
        raw = raw_records.get(entry["id"])
        try:
            if raw:
                _validate_legacy(root, raw, source, functions)
        except (OSError, ValueError, KeyError, TypeError, StopIteration, SyntaxError, IndexError) as exc:
            warnings.append(f"Timeline record excluded ({entry['id']}): {exc}")
            continue
        kind = entry["kind"]
        replay = entry.get("workflow_validation") is True
        filters = ["workflow", "compiler"] if replay else ["decisions"] if kind == "generated_proposal" else ["scientific"]
        if kind == "compiler_diagnostic" and not replay:
            filters.append("compiler")
        titles = {"compiler_diagnostic": "Mask hoisting on M=128 — compiler diagnostic",
                  "numeric_tuning_screening": "Two-slab pipeline stage reduction",
                  "profiling_study": "Reviewed-kernel profiling study",
                  "confirmation_history": "Grouped-M4 confirmation history"}
        title = titles.get(kind, (raw or {}).get("label") or entry["hypothesis"])
        evidence = {"kind": "experiment" if raw else "memory", "id": entry["id"], "artifact_path": entry["artifact_path"]}
        row = _event(entry["id"], entry["created_utc"], title,
                     "Workflow validation replay" if replay else kind.replace("_", " "),
                     entry["outcome"], filters, evidence, correctness=entry["correctness"],
                     interpretation=entry["performance_interpretation"], description=entry["hypothesis"],
                     shape_mnk=entry["shape_mnk"], tone=status_for(raw)["performance"]["tone"] if raw else "neutral")
        screening = entry.get("screening") or {}
        values = [screening.get(key) for key in ("baseline_median_ms", "candidate_median_ms", "median_paired_speedup")]
        if not replay and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values):
            row["measurement"] = dict(zip(("baseline_ms", "candidate_ms", "median_paired_speedup"), values))
            row["measurement"]["method"] = screening.get("measurement_method")
            row["measurement_label"] = "Paired CUDA-event screening · unconfirmed" if entry["confirmation"] != "confirmed" else "Paired timing · confirmed"
        if kind != "confirmation_history":
            events.append(row)
        else:
            for attempt in entry["confirmation_evidence"]["attempts"]:
                stamp = re.search(r"\d{8}T\d{12}Z", attempt["artifact_path"]).group()
                date = datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").isoformat() + "+00:00"
                events.append(_event("confirmation:" + stamp, date, "Grouped-M4 on FC1 — confirmation attempt",
                    "performance confirmation", attempt["status"], ["scientific"], evidence,
                    measurement_label="Recorded CUDA-event confirmation · unconfirmed",
                    correctness=attempt["correctness"], interpretation="Unconfirmed; ratios belong to this saved run only.",
                    description=f"{attempt['median_verification_pair_speedup']:.5f}× median verification-pair ratio; {attempt['pairs_favoring_candidate']}/7 pairs favorable.",
                    artifact_path=attempt["artifact_path"]))
            review = entry["review_status"]
            events.append(_event("review:" + entry["id"], entry["created_utc"], "Grouped CTA ordering — human review",
                "human review", review["status"], ["decisions"], evidence, description=review["reason"],
                interpretation=review["scope"]))
    plans, plan_warnings = load_saved_plans(root)
    warnings.extend(plan_warnings)
    completed = {row["plan_run_id"]: row for row in memory.get("completed_advisory_decisions", [])}
    for saved in plans:
        closure = completed.get(saved["run_id"])
        events.append(_event("plan:" + saved["run_id"], saved["created_utc"] or "", saved["plan"]["research_question"],
            "completed advisory decision" if closure else "advisory proposal", closure["direction_status"] if closure else "advisory_saved",
            ["decisions"], {"kind": "plan", "id": saved["run_id"], "artifact_path": saved["artifact_path"]},
            description=closure["reason"] if closure else saved["plan"]["optimization_mechanism"],
            interpretation="Advisory only; no execution implied."))
        try:
            review = WorkflowManager(root).review(saved["run_id"])
            if review is None:
                continue
            if (review["plan_run_id"] != saved["run_id"] or review["plan_sha256"] != saved["plan_sha256"]
                    or review["decision"] not in {"approved", "rejected"}
                    or review["decision"] == "rejected" and review["binding"] is not None):
                raise ValueError("review identity mismatch")
            if review["decision"] == "approved":
                # Completed replays already verify the archived approval digest in memory.
                linked = [row for row in memory["entries"] if row.get("workflow_validation") and
                          row["diagnostic_evidence"]["workflow"]["approval_path"] == f"{SECTION}/reviews/{saved['run_id']}/review.json"]
                if not linked:
                    verify_approval(root, saved["run_id"])
            events.append(_event("approval:" + saved["run_id"], review["reviewed_at"],
                saved["plan"]["research_question"], "human review", review["decision"], ["decisions"],
                {"kind": "plan", "id": saved["run_id"], "artifact_path": f"{SECTION}/reviews/{saved['run_id']}/review.json"},
                description="Human decision for the registered workflow; approval alone does not execute a job."))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Timeline review excluded ({saved['run_id']}): {exc}")
    try:
        for job in WorkflowManager(root).jobs():
            if job["status"] == "completed":
                continue  # Only the validated memory entry may represent a completed result.
            try:
                if not JOB_ID.fullmatch(job["id"]) or job["status"] not in {"queued", "running", "validating", "failed", "interrupted"}:
                    raise ValueError("invalid workflow job identity/status")
                review, approval_sha = verify_approval(root, job["plan_run_id"])
                if approval_sha != job["approval_sha256"] or review["binding"] != job["binding"]:
                    raise ValueError("workflow job/approval mismatch")
                date = job.get("finished_utc") or job["created_utc"]
                events.append(_event("workflow-job:" + job["id"], date, "Trusted diagnostic workflow job",
                    "workflow validation attempt", job["status"], ["workflow"],
                    {"kind": "job", "id": job["id"], "artifact_path": f"{SECTION}/jobs/{job['id']}/job.json"},
                    description=job.get("error") or "No completed, validated research result is available for this job.",
                    interpretation="Workflow status only; no scientific or performance conclusion."))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f"Timeline job excluded: {exc}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.append(f"Workflow job records unavailable: {exc}")
    events.sort(key=lambda row: (row["created_utc"], row["id"]), reverse=True)
    return {"events": events, "warnings": warnings, "filters": FILTERS,
            "performance": [row for row in events if row["measurement"] is not None and not row["workflow_validation"]]}
