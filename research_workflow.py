"""One explicitly approved trusted diagnostic, with durable reviews and jobs."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import uuid

from grouped_confirmation_memory import _read


SECTION = "evaluations/research-workflow"
RUNNER_ID = "mask_hoisting_m128_diagnostic_v1"
PLAN_SHA = "5a5ec80842b9a544f02471bd0517d77c7ce598d82a88355d1980ed418e5fd8b1"
CONFIG = {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3}
REGISTRY = {RUNNER_ID: {"script": "trusted_diagnostic_worker.py", "shape_mnk": [128, 3072, 768],
                       "config": CONFIG, "phases": ["compiler_diagnostic", "correctness"],
                       "timeout_seconds": 180, "plan_sha256": PLAN_SHA}}
SOURCE_FILES = (
    "research_workflow.py", "trusted_diagnostic_worker.py", "workflow_artifacts.py", "mask_hoisting_memory.py",
    "mask_hoisting_m128_experiment.py", "trusted_compute_kernel.py", "benchmark_proposal.py",
    "trusted_compute_experiment.py", "benchmark_core.py", "reference_precision.py",
    "stage_reduction_experiment.py", "trusted_two_slab_experiment.py", "trusted_two_slab_kernel.py",
    "optimize.py", "grouped_matmul.py", "split_k_matmul.py", "grouped_confirmation_memory.py",
    "experiment_planner.py", "experiment_memory.py", "dashboard_data.py",
)
INPUT_FILES = (
    "evaluations/experiment-planner/20261028T145028833360Z/plan.json",
    "evaluations/experiment-planner/20261028T145028833360Z/metadata.json",
    "evaluations/experiment-planner/20261028T145028833360Z/proposal_audit.jsonl",
    "evaluations/trusted-compute/20261027T154233319712Z/experiment.json",
)
PINNED = {
    "trusted_compute_kernel.py": "8ad5ab3c4b1c9057429724bbe2622120473a008c950ef5eae6b3fac73bed4f9d",
    "benchmark_proposal.py": "9f7bbffd3180f0825364dd440845976e4ea6eb2dcbdf2fdcba3ac7e4b2c61288",
    "mask_hoisting_m128_experiment.py": "38637079ea6c40ad08d02ee2c27c45f388203b0508e566e149617551f905900c",
}
PLAN_ID = re.compile(r"\d{8}T\d{12}Z\Z")
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
ACTIVE = {"queued", "running", "validating"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p != root):
        raise ValueError("workflow path is missing or linked outside its root")
    return path


def save(root, relative, data, *, exclusive=False):
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if exclusive:
        with path.open("xb") as stream:
            stream.write(raw)
    else:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
        os.replace(temporary, path)


def read_json(root, relative):
    data = json.loads(_read(Path(root).resolve(), relative))
    if not isinstance(data, dict):
        raise ValueError("expected workflow record")
    return data


def acquire(root, name):
    path = safe_path(root, SECTION + "/" + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise ValueError("another workflow action is active") from None
    return stream


@contextmanager
def transaction(root):
    lock = acquire(root, "state.lock")
    try:
        yield
    finally:
        lock.close()


def saved_plan(root, run_id):
    from experiment_planner import load_saved_plans
    if not isinstance(run_id, str) or not PLAN_ID.fullmatch(run_id):
        raise ValueError("invalid plan ID")
    plans, _ = load_saved_plans(root)
    record = next((row for row in plans if row["run_id"] == run_id), None)
    if record is None:
        raise ValueError("plan failed saved audit/hash validation")
    return record


def binding(root, run_id, runner_id=RUNNER_ID):
    if runner_id not in REGISTRY:
        raise ValueError("runner is not registered")
    spec = REGISTRY[runner_id]
    plan = saved_plan(root, run_id)
    if (plan["plan_sha256"] != spec["plan_sha256"]
            or plan["plan"]["target_shape_mnk"] != spec["shape_mnk"]
            or plan["plan"]["experiment_kind"] != "workload_shape"):
        raise ValueError("this plan has no registered execution mapping; advisory only")
    sources = {name: digest(_read(Path(root).resolve(), name)) for name in SOURCE_FILES}
    if any(sources[name] != expected for name, expected in PINNED.items()):
        raise ValueError("registered trusted kernel or runner source changed")
    return {"plan_run_id": run_id, "plan_sha256": plan["plan_sha256"], "runner_id": runner_id,
            "source_sha256": sources, "shape_mnk": list(spec["shape_mnk"]), "dtype": "fp16",
            "input_sha256": {name: digest(_read(Path(root).resolve(), name)) for name in INPUT_FILES},
            "config": dict(spec["config"]), "phases": list(spec["phases"]), "timeout_seconds": spec["timeout_seconds"],
            "python_executable": sys.executable, "purpose": "workflow_validation",
            "independent_research_evidence": False}


def review_path(run_id):
    if not PLAN_ID.fullmatch(run_id):
        raise ValueError("invalid plan ID")
    return f"{SECTION}/reviews/{run_id}/review.json"


def verify_approval(root, run_id):
    path = review_path(run_id)
    review = read_json(root, path)
    if review.get("decision") != "approved" or review.get("binding") != binding(root, run_id):
        raise ValueError("approval is absent, changed, or no longer matches plan/runner identity")
    return review, digest(_read(Path(root).resolve(), path))


def registered_command(root, output, approval):
    return [sys.executable, str(root / REGISTRY[RUNNER_ID]["script"]),
            "--approval", str(approval), "--output", str(output)]


def execute_trusted(root, output, approval, stdout, stderr, lock_fd):
    """Only a fixed registered local script is launched; model text is never argv."""
    run_id = approval.parent.name
    verify_approval(root, run_id)
    spec = REGISTRY[RUNNER_ID]
    command = registered_command(root, output, approval)
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "HOME", "USER", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH"}}
    env["TRITON_CACHE_DIR"] = str(output.parent / "triton-cache")
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=out, stderr=err,
                                   start_new_session=True, pass_fds=(lock_fd,))
        try:
            code = process.wait(timeout=spec["timeout_seconds"])
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise RuntimeError("trusted diagnostic exceeded its 180 second timeout") from None
    if code:
        raise RuntimeError(f"trusted diagnostic exited with code {code}")


class WorkflowManager:
    def __init__(self, root, *, executor=None):
        self.root = Path(root).resolve()
        self.executor = executor or execute_trusted

    def review(self, run_id):
        path = review_path(run_id)
        return read_json(self.root, path) if safe_path(self.root, path).exists() else None

    def decide(self, run_id, decision, runner_id=RUNNER_ID):
        if decision not in {"approved", "rejected"}:
            raise ValueError("invalid human decision")
        with transaction(self.root):
            plan = saved_plan(self.root, run_id)
            if self.review(run_id):
                raise ValueError("plan already reviewed; review records are immutable")
            record = {"decision": decision, "reviewed_by": "local_dashboard_user", "reviewed_at": now(),
                      "plan_run_id": run_id, "plan_sha256": plan["plan_sha256"],
                      "binding": binding(self.root, run_id, runner_id) if decision == "approved" else None}
            save(self.root, review_path(run_id), record, exclusive=True)
        return record

    def jobs(self):
        base = safe_path(self.root, SECTION + "/jobs")
        rows = []
        if base.exists():
            for path in sorted(base.iterdir()):
                if JOB_ID.fullmatch(path.name):
                    rows.append(read_json(self.root, f"{SECTION}/jobs/{path.name}/job.json"))
        return sorted(rows, key=lambda row: row["created_utc"], reverse=True)

    def recover(self):
        """A dead server never automatically resumes GPU work."""
        if not safe_path(self.root, SECTION + "/jobs").exists():
            return
        try:
            lock = acquire(self.root, "execution.lock")
        except ValueError:
            return
        try:
            for job in self.jobs():
                if job["status"] in ACTIVE:
                    job.update(status="interrupted", finished_utc=now(),
                               error="Worker owner exited; retained artifacts require review. No automatic retry.")
                    save(self.root, f"{SECTION}/jobs/{job['id']}/job.json", job)
        finally:
            lock.close()

    def start_job(self, run_id, *, background=True):
        self.recover()
        lock = acquire(self.root, "execution.lock")
        try:
            review, approval_sha = verify_approval(self.root, run_id)
            if any(row["plan_run_id"] == run_id for row in self.jobs()):
                raise ValueError("this approval already has an execution attempt; automatic repeats are disabled")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "plan_run_id": run_id, "status": "queued", "created_utc": now(),
                   "approval_path": review_path(run_id), "approval_sha256": approval_sha,
                   "binding": review["binding"], "purpose": "workflow_validation",
                   "independent_research_evidence": False}
            directory = safe_path(self.root, f"{SECTION}/jobs/{job_id}")
            job["command"] = registered_command(self.root, directory / "artifacts", self.root / review_path(run_id))
            save(self.root, f"{SECTION}/jobs/{job_id}/job.json", job, exclusive=True)
            if background:
                threading.Thread(target=self._work, args=(job, lock), daemon=True, name="trusted-diagnostic").start()
            else:
                self._work(job, lock)
            return job_id
        except Exception:
            lock.close()
            raise

    def _work(self, job, lock):
        relative = f"{SECTION}/jobs/{job['id']}"
        directory = safe_path(self.root, relative)
        try:
            _, digest_now = verify_approval(self.root, job["plan_run_id"])
            if digest_now != job["approval_sha256"]:
                raise ValueError("approval file changed after queueing")
            job.update(status="running", started_utc=now())
            save(self.root, relative + "/job.json", job)
            self.executor(self.root, directory / "artifacts", self.root / job["approval_path"],
                          directory / "stdout.log", directory / "stderr.log", lock.fileno())
            job["status"] = "validating"
            save(self.root, relative + "/job.json", job)
            from workflow_artifacts import validate_job_artifacts
            entry, manifest_sha = validate_job_artifacts(self.root, job)
            job.update(status="completed", outcome=entry["outcome"], memory_id=entry["id"],
                       manifest_sha256=manifest_sha, finished_utc=now(),
                       log_sha256={name: digest(_read(self.root, relative + "/" + name))
                                   for name in ("stdout.log", "stderr.log")})
        except Exception as exc:
            job.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}", finished_utc=now())
        finally:
            try:
                save(self.root, relative + "/job.json", job)
            finally:
                lock.close()

    def planning_status(self):
        path = SECTION + "/planning.json"
        state = read_json(self.root, path) if safe_path(self.root, path).exists() else {"status": "ready"}
        if state["status"] == "requesting":
            try:
                lock = acquire(self.root, "planning.lock")
            except ValueError:
                return state
            try:
                state = read_json(self.root, path)
                if state["status"] == "requesting":
                    state = {"status": "interrupted", "message": "Advisory request owner exited; no automatic retry."}
                    save(self.root, path, state)
            finally:
                lock.close()
        return state

    def request_advisory(self):
        lock = acquire(self.root, "planning.lock")
        try:
            save(self.root, SECTION + "/planning.json", {"status": "requesting", "created_utc": now()})
            threading.Thread(target=self._plan, args=(lock,), daemon=True, name="advisory-planning").start()
        except Exception:
            lock.close()
            raise

    def _plan(self, lock):
        try:
            from dotenv import load_dotenv
            from openai import OpenAI
            import experiment_planner as planner
            load_dotenv(dotenv_path=self.root / ".env")
            key = os.environ.get("MODEL_API_KEY")
            if not key:
                raise ValueError("MODEL_API_KEY missing")
            context = planner.build_context(self.root)
            client = OpenAI(base_url=planner.API_BASE_URL, api_key=key, timeout=30, max_retries=0)
            folder, metadata = planner.propose_to_directory(client, planner.build_prompt(context), context, self.root)
            state = {"status": metadata["status"], "plan_run_id": folder.name, "finished_utc": now()}
        except Exception:
            state = {"status": "failed", "message": "Advisory request failed; no execution was approved.", "finished_utc": now()}
        finally:
            try:
                save(self.root, SECTION + "/planning.json", state)
            finally:
                lock.close()
