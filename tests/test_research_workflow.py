"""Controlled workflow tests: all execution is mocked; no CUDA or live Muse requests."""

import json
import os
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlencode
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dashboard_app import create_app
import experiment_memory
import experiment_planner
import research_workflow as workflow
import workflow_artifacts
import mask_hoisting_memory


ROOT = Path(__file__).resolve().parents[1]
PLAN_RUN = "20261028T145028833360Z"
BASE_URL = "http://127.0.0.1:8501"


class RenderedForms(HTMLParser):
    """Read successful form inputs from the actual rendered page, not hardcoded fields."""
    def __init__(self, html):
        super().__init__()
        self.forms = []
        self.current = None
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "form":
            self.current = {"action": attrs.get("action"), "method": attrs.get("method", "get"), "fields": []}
            self.forms.append(self.current)
        elif tag == "input" and self.current is not None and attrs.get("name") and "disabled" not in attrs:
            self.current["fields"].append((attrs["name"], attrs.get("value", "")))

    def handle_endtag(self, tag):
        if tag == "form":
            self.current = None


def fake_execution(root, output, approval, stdout, stderr, lock_fd):
    """Use saved scientific artifacts only as mock output, never execute their source."""
    review = json.loads(approval.read_text())
    shutil.copytree(ROOT / mask_hoisting_memory.FOLDER, output)
    sources = output / "sources"
    sources.mkdir()
    for name in review["binding"]["source_sha256"]:
        shutil.copy2(root / name, sources / name)
    for name in review["binding"]["input_sha256"]:
        destination = output / "inputs" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    worker = "trusted_diagnostic_worker.py"
    shutil.copy2(root / worker, output / worker)
    correctness = json.loads((output / "correctness.json").read_text())
    correctness.update(verifier_file=worker, verifier_sha256=review["binding"]["source_sha256"][worker])
    (output / "correctness.json").write_text(json.dumps(correctness))
    manifest = {"binding": review["binding"], "purpose": "workflow_validation",
                "approval_sha256": workflow.digest(approval.read_bytes()),
                "files": {str(path.relative_to(output)): workflow.digest(path.read_bytes())
                          for path in output.rglob("*") if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest))
    stdout.write_text("Mock executor: no GPU was used.\n")
    stderr.write_text("")


class ControlledWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in workflow.SOURCE_FILES:
            shutil.copy2(ROOT / name, self.root / name)
        plan = Path("evaluations/experiment-planner") / PLAN_RUN
        (self.root / plan).parent.mkdir(parents=True)
        shutil.copytree(ROOT / plan, self.root / plan)
        prior = Path("evaluations/trusted-compute/20261027T154233319712Z/experiment.json")
        (self.root / prior).parent.mkdir(parents=True)
        shutil.copy2(ROOT / prior, self.root / prior)
        self.manager = workflow.WorkflowManager(self.root, executor=fake_execution)

    def approve(self):
        return self.manager.decide(PLAN_RUN, "approved")

    def test_approval_separates_execution_and_completed_diagnostic_reaches_memory(self):
        with self.assertRaises((ValueError, OSError)):
            self.manager.start_job(PLAN_RUN, background=False)
        review = self.approve()
        self.assertEqual(self.manager.jobs(), [])
        bound = review["binding"]
        self.assertEqual(bound["plan_sha256"], workflow.PLAN_SHA)
        self.assertEqual(bound["phases"], ["compiler_diagnostic", "correctness"])
        self.assertEqual(bound["config"], workflow.CONFIG)
        self.assertEqual(bound["purpose"], "workflow_validation")
        job_id = self.manager.start_job(PLAN_RUN, background=False)
        job, = self.manager.jobs()
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["outcome"], "compiler_equivalent_screening_skipped")
        self.assertEqual(job["id"], job_id)
        reloaded = workflow.WorkflowManager(self.root)
        self.assertEqual(reloaded.jobs(), [job])
        with self.assertRaisesRegex(ValueError, "already has an execution"):
            reloaded.start_job(PLAN_RUN, background=False)
        entries, warnings = workflow_artifacts.load_completed(self.root)
        self.assertEqual(warnings, [])
        entry, = entries
        self.assertTrue(entry["workflow_validation"])
        self.assertFalse(entry["independent_research_evidence"])
        self.assertIsNone(entry["screening"])
        remembered = experiment_memory.reconstruct(self.root)
        self.assertIn(job["memory_id"], [row["id"] for row in remembered["entries"]])
        selected = experiment_memory.retrieve(remembered, shape=[128, 3072, 768], dtype="fp16",
                                               gpu=entry["gpu"], method="python_launch_cuda_event")
        self.assertNotIn(job["memory_id"], [row["id"] for row in selected])

    def test_changed_plan_source_approval_and_invalid_runner_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.manager.decide(PLAN_RUN, "approved", runner_id="generated_python")
        self.approve()
        review_path = self.root / workflow.review_path(PLAN_RUN)
        original_review = review_path.read_bytes()
        forged = json.loads(original_review)
        forged["binding"]["phases"].append("screening")
        review_path.write_text(json.dumps(forged))
        with self.assertRaisesRegex(ValueError, "approval"):
            self.manager.start_job(PLAN_RUN, background=False)
        review_path.write_bytes(original_review)
        source = self.root / "trusted_diagnostic_worker.py"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "approval"):
            self.manager.start_job(PLAN_RUN, background=False)
        shutil.copy2(ROOT / source.name, source)
        plan = self.root / "evaluations/experiment-planner" / PLAN_RUN / "plan.json"
        plan.write_bytes(plan.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "validation"):
            self.manager.start_job(PLAN_RUN, background=False)
        self.assertEqual(self.manager.jobs(), [])

    def test_rejected_review_cannot_be_executed_or_replaced(self):
        self.manager.decide(PLAN_RUN, "rejected")
        with self.assertRaisesRegex(ValueError, "already reviewed"):
            self.approve()
        with self.assertRaisesRegex(ValueError, "approval"):
            self.manager.start_job(PLAN_RUN, background=False)

    def test_only_one_active_job_even_across_manager_instances(self):
        started, finish = threading.Event(), threading.Event()

        def held(*args):
            started.set()
            if not finish.wait(3):
                raise RuntimeError("test timeout")
            fake_execution(*args)

        self.manager.executor = held
        self.approve()
        self.manager.start_job(PLAN_RUN)
        self.assertTrue(started.wait(2))
        try:
            other = workflow.WorkflowManager(self.root)
            with self.assertRaisesRegex(ValueError, "active"):
                other.start_job(PLAN_RUN)
            other.recover()
            self.assertEqual(other.jobs()[0]["status"], "running")
        finally:
            finish.set()
            for _ in range(100):
                if self.manager.jobs()[0]["status"] not in workflow.ACTIVE:
                    break
                time.sleep(0.01)
        self.assertEqual(self.manager.jobs()[0]["status"], "completed")

    def test_process_failure_and_invalid_artifacts_keep_logs_and_are_not_ingested(self):
        def fail(root, output, approval, stdout, stderr, lock_fd):
            stdout.write_text("partial run")
            stderr.write_text("simulated execution failure")
            raise RuntimeError("trusted process failed")

        self.manager.executor = fail
        self.approve()
        job_id = self.manager.start_job(PLAN_RUN, background=False)
        job = self.manager.jobs()[0]
        self.assertEqual(job["status"], "failed")
        self.assertEqual(workflow_artifacts.load_completed(self.root), ([], []))
        self.assertTrue((self.root / workflow.SECTION / "jobs" / job_id / "stderr.log").exists())

    def test_invalid_compiler_output_never_becomes_completed_memory(self):
        def corrupt(*args):
            fake_execution(*args)
            output = args[1]
            (output / "trusted.cubin").write_bytes(b"bad cubin")

        self.manager.executor = corrupt
        self.approve()
        self.manager.start_job(PLAN_RUN, background=False)
        self.assertEqual(self.manager.jobs()[0]["status"], "failed")
        self.assertIn("hash mismatch", self.manager.jobs()[0]["error"])
        self.assertEqual(workflow_artifacts.load_completed(self.root), ([], []))

    def test_completed_artifact_tampering_is_detected_on_memory_reload(self):
        self.approve()
        job_id = self.manager.start_job(PLAN_RUN, background=False)
        directory = self.root / workflow.SECTION / "jobs" / job_id
        (directory / "artifacts/correctness.json").write_text("{}")
        entries, warnings = workflow_artifacts.load_completed(self.root)
        self.assertEqual(entries, [])
        self.assertTrue(warnings)

    def test_completed_job_requires_manifest_and_log_fingerprints(self):
        self.approve()
        self.manager.start_job(PLAN_RUN, background=False)
        job = self.manager.jobs()[0]
        job.pop("manifest_sha256")
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            workflow_artifacts.validate_job_artifacts(self.root, job)

    def test_timeout_kills_process_group_and_fixed_command_has_no_model_arguments(self):
        self.approve()
        directory = self.root / "timeout-check"
        directory.mkdir()
        process = Mock(pid=1234)
        process.wait.side_effect = [subprocess.TimeoutExpired("trusted", 180), 0]
        with patch("research_workflow.subprocess.Popen", return_value=process) as popen, \
             patch("research_workflow.os.killpg") as kill, \
             patch.dict(os.environ, {"MODEL_API_KEY": "never-forward-this"}):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                workflow.execute_trusted(self.root, directory / "artifacts",
                    self.root / workflow.review_path(PLAN_RUN), directory / "stdout.log", directory / "stderr.log", 99)
        command = popen.call_args.args[0]
        self.assertEqual(command[1], str(self.root / "trusted_diagnostic_worker.py"))
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertNotIn("MODEL_API_KEY", popen.call_args.kwargs["env"])
        self.assertEqual(popen.call_args.kwargs["pass_fds"], (99,))
        kill.assert_called_once()

    def test_stale_jobs_are_interrupted_without_automatic_retry(self):
        self.approve()
        job_id = "a" * 32
        workflow.save(self.root, f"{workflow.SECTION}/jobs/{job_id}/job.json", {
            "id": job_id, "plan_run_id": PLAN_RUN, "status": "running", "created_utc": workflow.now()})
        self.manager.recover()
        self.assertEqual(self.manager.jobs()[0]["status"], "interrupted")
        with self.assertRaisesRegex(ValueError, "already has an execution"):
            self.manager.start_job(PLAN_RUN, background=False)

    def test_dashboard_approval_execution_result_and_csrf(self):
        app = create_app(self.root)
        manager = app.extensions["research_workflow"]
        manager.executor = fake_execution
        start = manager.start_job
        manager.start_job = lambda run: start(run, background=False)
        client = app.test_client()
        page = client.get("/research", base_url=BASE_URL)
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Approve fixed diagnostic", page.data)
        token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()

        def post(action, data=None):
            return client.post(f"/research/plans/{PLAN_RUN}/{action}", base_url=BASE_URL,
                               headers={"Origin": BASE_URL}, data=data or {"csrf_token": token})

        self.assertEqual(post("approve", {"csrf_token": "wrong"}).status_code, 403)
        self.assertEqual(post("approve", {"csrf_token": token, "runner": "generated"}).status_code, 403)
        self.assertEqual(post("execute").status_code, 409)
        self.assertEqual(post("approve").status_code, 303)
        self.assertEqual(manager.jobs(), [])
        self.assertIn(b"Execute approved diagnostic", client.get("/research", base_url=BASE_URL).data)
        self.assertEqual(post("execute").status_code, 303)
        self.assertEqual(post("execute").status_code, 409)
        page = client.get("/research", base_url=BASE_URL)
        self.assertIn(b"compiler_equivalent_screening_skipped", page.data)
        job = manager.jobs()[0]
        evidence = client.get(f"/research/evidence/{job['memory_id']}", base_url=BASE_URL)
        self.assertEqual(evidence.status_code, 200)
        self.assertIn(b"not new independent research evidence", evidence.data)
        log = client.get(f"/research/jobs/{job['id']}/stdout.log", base_url=BASE_URL)
        self.assertEqual(log.status_code, 200)
        self.assertEqual(client.get(f"/research/jobs/{job['id']}/candidate.py", base_url=BASE_URL).status_code, 404)

    def test_native_rendered_approval_form_policy_and_request_validation(self):
        app = create_app(self.root)
        client = app.test_client()
        manager = app.extensions["research_workflow"]
        manager.executor = Mock(side_effect=AssertionError("GPU execution on approval"))
        page_url = f"/research?plan={PLAN_RUN}"
        page = client.get(page_url, base_url=BASE_URL)
        # This policy is essential: no-referrer makes Chromium's native POST Origin null.
        self.assertEqual(page.headers["Referrer-Policy"], "same-origin")
        forms = RenderedForms(page.get_data(as_text=True)).forms
        approval = next(form for form in forms if form["action"].endswith("/approve"))
        self.assertEqual(approval["method"].lower(), "post")
        self.assertEqual([name for name, _ in approval["fields"]], ["csrf_token"])
        with client.session_transaction(base_url=BASE_URL) as saved_session:
            self.assertEqual(dict(approval["fields"])["csrf_token"], saved_session["csrf_token"])

        def submit(fields, origin=BASE_URL, mime="application/x-www-form-urlencoded"):
            # A native form navigates with a URL-encoded body and its existing session cookie.
            headers = {"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "same-origin",
                       "Referer": BASE_URL + page_url}
            if origin is not None:
                headers["Origin"] = origin
            return client.open(approval["action"], method=approval["method"].upper(),
                base_url=BASE_URL, data=urlencode(fields), content_type=mime, headers=headers)

        invalid_fields = [[], [("csrf_token", "wrong")], approval["fields"] * 2,
                          approval["fields"] + [("unexpected", "value")],
                          approval["fields"] + [("runner_id", "generated_python")],
                          approval["fields"] + [("plan_sha256", "0" * 64)],
                          approval["fields"] + [("source_sha256", "0" * 64)],
                          approval["fields"] + [("phases", "screening")]]
        for fields in invalid_fields:
            with self.subTest(fields=[name for name, _ in fields]):
                self.assertEqual(submit(fields).status_code, 403)
        for origin in (None, "null", "https://forged.example", "http://localhost:8501"):
            with self.subTest(origin=origin):
                self.assertEqual(submit(approval["fields"], origin=origin).status_code, 403)
        self.assertEqual(submit(approval["fields"], mime="application/json").status_code, 415)
        self.assertIsNone(manager.review(PLAN_RUN))
        with patch.object(manager, "start_job", side_effect=AssertionError("Execution on approval")) as start:
            result = submit(approval["fields"])
        self.assertEqual(result.status_code, 303)
        start.assert_not_called()
        review = manager.review(PLAN_RUN)
        self.assertEqual(review["decision"], "approved")
        self.assertEqual(review["binding"], workflow.binding(self.root, PLAN_RUN))
        self.assertEqual(manager.jobs(), [])
        manager.executor.assert_not_called()

    @unittest.skipUnless(os.environ.get("NT_APPROVAL_BROWSER"), "opt-in native browser integration")
    def test_real_browser_native_approval_preserves_origin_and_creates_no_job(self):
        from flask import request, session
        import secrets
        from werkzeug.serving import make_server
        import dashboard_app

        app = create_app(self.root)
        manager = app.extensions["research_workflow"]
        manager.executor = Mock(side_effect=AssertionError("GPU forbidden in browser test"))
        seen = []

        def capture():
            if request.method == "POST":
                seen.append({"origin": request.headers.get("Origin"), "host": request.host,
                             "mime": request.mimetype, "fields": sorted(request.form),
                             "csrf_matches": secrets.compare_digest(request.form.get("csrf_token", ""),
                                                                    session.get("csrf_token", ""))})

        app.before_request_funcs[None].insert(0, capture)

        @app.get("/test-click-approve.js")
        def click_approve():
            return app.response_class(
                "const form = [...document.forms].find(f => f.action.endsWith('/approve')); "
                "if (form) form.querySelector('button').click();", mimetype="application/javascript")

        @app.after_request
        def inject_browser_click(response):
            if request.path == "/research":
                response.set_data(response.get_data().replace(b"</head>",
                    b'<script src="/test-click-approve.js" defer></script></head>'))
            return response

        server = make_server("127.0.0.1", 0, app, threaded=True)
        origin = f"http://127.0.0.1:{server.server_port}"
        with patch.object(dashboard_app, "ALLOWED_HOSTS", dashboard_app.ALLOWED_HOSTS | {origin[7:]}), \
             tempfile.TemporaryDirectory(prefix="nt-approval-browser-") as profile:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                browser = subprocess.run([os.environ["NT_APPROVAL_BROWSER"], "--headless", "--no-sandbox",
                    "--disable-gpu", "--disable-dev-shm-usage", "--no-proxy-server", "--no-first-run",
                    "--no-default-browser-check", "--disable-background-networking", "--user-data-dir=" + profile,
                    "--dump-dom", "--timeout=15000", "--virtual-time-budget=4000",
                    origin + f"/research?plan={PLAN_RUN}"], capture_output=True, text=True, timeout=35)
            finally:
                server.shutdown()
                thread.join(timeout=3)
                server.server_close()
        self.assertEqual(browser.returncode, 0, browser.stderr[-2000:])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["origin"], origin)
        self.assertEqual(seen[0]["mime"], "application/x-www-form-urlencoded")
        self.assertEqual(seen[0]["fields"], ["csrf_token"])
        self.assertTrue(seen[0]["csrf_matches"])
        self.assertEqual(manager.review(PLAN_RUN)["decision"], "approved")
        self.assertEqual(manager.jobs(), [])
        manager.executor.assert_not_called()

    def test_advisory_dashboard_action_never_approves_or_executes(self):
        app = create_app(self.root)
        client = app.test_client()
        page = client.get("/research", base_url=BASE_URL)
        token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()
        manager = app.extensions["research_workflow"]
        with patch.object(manager, "request_advisory") as request, patch.object(manager, "start_job") as execute:
            response = client.post("/research/advisory", base_url=BASE_URL,
                headers={"Origin": BASE_URL}, data={"csrf_token": token})
        self.assertEqual(response.status_code, 303)
        request.assert_called_once()
        execute.assert_not_called()
        self.assertIsNone(manager.review(PLAN_RUN))

    def test_planning_worker_reuses_planner_and_does_not_execute_a_saved_response(self):
        context = experiment_planner.build_context(ROOT)
        plan = json.loads((ROOT / "examples/experiment_planner_plan.json").read_text())
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(id="mock-advisory", output_text=json.dumps(plan))
        self.manager.executor = Mock(side_effect=AssertionError("GPU execution during planning"))
        lock = workflow.acquire(self.root, "planning.lock")
        with patch("openai.OpenAI", return_value=client), \
             patch("experiment_planner.build_context", return_value=context), \
             patch.dict(os.environ, {"MODEL_API_KEY": "mock"}):
            self.manager._plan(lock)
        client.responses.create.assert_called_once()
        self.manager.executor.assert_not_called()
        self.assertIsNone(self.manager.review(PLAN_RUN))
        self.assertEqual(self.manager.jobs(), [])
        self.assertEqual(self.manager.planning_status()["status"], "advisory_saved")

    def test_source_is_reverified_after_queueing_before_executor_is_called(self):
        self.approve()
        threads = []

        class DeferredThread:
            def __init__(self, target, args, **kwargs):
                self.target, self.args = target, args
                threads.append(self)

            def start(self):
                pass

        self.manager.executor = Mock(side_effect=AssertionError("changed source executed"))
        with patch("research_workflow.threading.Thread", DeferredThread):
            self.manager.start_job(PLAN_RUN)
        source = self.root / "trusted_diagnostic_worker.py"
        source.write_bytes(source.read_bytes() + b"\n")
        threads[0].target(*threads[0].args)
        self.assertEqual(self.manager.jobs()[0]["status"], "failed")
        self.manager.executor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
