"""Read-only presentation of already validated memory, plans, and diagnostic artifacts."""

import json
from pathlib import Path

from grouped_confirmation_memory import _read
from research_workflow import SECTION, saved_plan


def evidence_summary(root, entry):
    root = Path(root).resolve()
    evidence = entry.get("diagnostic_evidence") or {}
    source = evidence.get("source_identity") or {}
    workflow = evidence.get("workflow") or {}
    summary = {
        "question": None, "hypothesis": entry.get("hypothesis"), "mechanism": entry.get("mechanism"),
        "approval": None, "job": None, "plan": None, "checks": [], "diagnostic": {},
        "workflow_validation": entry.get("workflow_validation") is True,
        "independent": entry.get("independent_research_evidence"), "warnings": [],
        "resources": (entry.get("compiler") or {}).get("resources") or {},
        "limits": evidence.get("limitations") or (entry.get("profiling_evidence") or {}).get("limitations") or [],
        "runtime_measured": evidence.get("latency_measured"),
        "skip_reason": evidence.get("screening_skip_reason"),
        "tolerances": evidence.get("tolerances") or {},
        "executable": evidence.get("executable_section") or {},
        "constant_metadata_identical": evidence.get("constant_data_and_launch_metadata_identical"),
        "correctness_cases": entry.get("correctness_cases") or evidence.get("correctness_cases"),
    }
    if evidence.get("artifact_hashes"):
        folder = Path(entry["artifact_path"]).parent
        for filename, target in (("correctness.json", "checks"), ("diagnostic.json", "diagnostic")):
            try:
                data = json.loads(_read(root, str(folder / filename), evidence["artifact_hashes"][filename]))
                summary[target] = data["checks"] if target == "checks" else data
            except (OSError, ValueError, KeyError, TypeError) as exc:
                summary["warnings"].append(f"Supporting {filename} unavailable: {exc}")
    if workflow.get("job_id"):
        try:
            job = json.loads(_read(root, f"{SECTION}/jobs/{workflow['job_id']}/job.json"))
            approval = json.loads(_read(root, workflow["approval_path"], workflow["approval_sha256"]))
            if job["id"] != workflow["job_id"] or job["approval_sha256"] != workflow["approval_sha256"]:
                raise ValueError("job identity differs from validated evidence")
            summary.update(job=job, approval=approval)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary["warnings"].append(f"Approval/execution details unavailable: {exc}")
    plan_path = source.get("plan_path")
    if plan_path:
        try:
            plan = saved_plan(root, Path(plan_path).parent.name)
            if plan["plan_sha256"] != source["plan_sha256"]:
                raise ValueError("plan identity differs from validated evidence")
            summary.update(plan=plan, question=plan["plan"]["research_question"],
                           hypothesis=plan["plan"]["technical_hypothesis"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            summary["warnings"].append(f"Original advisory unavailable: {exc}")
    if summary["runtime_measured"] is None and entry.get("screening"):
        summary["runtime_measured"] = True
    summary["resources"] = {name: values for name, values in summary["resources"].items()
                            if isinstance(values, dict)}
    return summary
