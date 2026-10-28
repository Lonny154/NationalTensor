"""Presentation-only workflow captions; no approvals, execution, or science decisions."""


def mission_state(plan, review, state, blocked, verified_evidence):
    jobs = [job for job in state["jobs"] if plan and job["plan_run_id"] == plan["run_id"]]
    job = max(jobs, key=lambda row: row["created_utc"], default=None)
    active = next((row for row in state["jobs"] if row["status"] in {"queued", "running", "validating"}), None)
    canonical_id = "workflow:" + job["id"] if job else None
    evidence = verified_evidence.get(canonical_id)
    verified = bool(job and job["status"] == "completed" and job.get("memory_id") == canonical_id and evidence)
    phases = [
        ("01", "Decode", "requesting" if state["planning"]["status"] == "requesting" else
         "Saved advisory" if plan else state["planning"]["status"]),
        ("02", "Authorize", review["decision"] if review else "Awaiting human review"),
        ("03", "Execute", job["status"] if job else "Not started"),
        ("04", "Validate", "Verified memory evidence" if verified else "No validated result linked"),
    ]
    if blocked:
        title, narrative, action, anchor = "The vault stays locked.", blocked, "Inspect the identity block", "authorization"
    elif active:
        title, narrative, action, anchor = "The operation is on the wire.", f"Job {active['id']}: {active['status']}. No invented countdown. The hardware has declined to provide a screenplay.", "Inspect live dispatch", "dispatch"
    elif state["planning"]["status"] == "requesting":
        title, narrative, action, anchor = "The cryptographer is on the line.", "An advisory request is already in progress. Inspect its status; planning does not authorize or execute GPU work.", "Inspect advisory request status", "planning"
    elif verified:
        title, narrative, action, anchor = "The compiler got there first." if "equivalent" in evidence["outcome"] else "The dossier is sealed.", "Validated diagnostic artifacts are in memory. Workflow validation is a replay, not a new discovery. Runtime was not measured.", "Read the validated debrief", "dispatch"
    elif job:
        title, narrative, action, anchor = "Check the escape route.", f"Recorded job state: {job['status']}. A job record alone is not validated scientific evidence. Inspect its logs before drawing conclusions.", "Inspect dispatch and logs", "dispatch"
    elif review and review["decision"] == "rejected":
        title, narrative, action, anchor = "The operation was declined.", "Human review said no. The GPU has not been sent on this errand.", "Inspect the review", "authorization"
    elif review and review["decision"] == "approved":
        title, narrative, action, anchor = "Permission is not ignition.", "The human signed the paperwork. Starting the fixed trusted diagnostic still requires its own explicit action.", "Review approved execution", "authorization"
    elif plan:
        title, narrative, action, anchor = "A clue is not a clearance.", "Muse supplied a hypothesis. Read the evidence, inspect the source identity, and decide whether this operation deserves your signature.", "Review authorization", "authorization"
    else:
        title, narrative, action, anchor = "The map has a blank space.", "No saved advisory is available. An explicit planning request can ask Muse for one; it will not start a GPU job.", "Inspect advisory planning", "planning"
    return {"title": title, "narrative": narrative, "action": action, "anchor": anchor,
            "phases": phases, "job": job, "verified": verified}
