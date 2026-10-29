"""Mission data boundaries and optional browser scenes; no GPU or Muse requests."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from dashboard_app import create_app
import dashboard_app
from dashboard_data import ROOT
from dashboard_mission import HISTORICAL_ID, classify_ending
import research_workflow as workflow
import test_research_workflow as fixtures


class MissionTests(unittest.TestCase):
    def test_endings_require_recorded_evidence(self):
        equivalent = {"outcome": "compiler_equivalent_screening_skipped", "correctness": "PASS",
                      "diagnostic_evidence": {"executable_section": {"identical": True}}}
        self.assertEqual(classify_ending(equivalent, False), "equivalent")
        self.assertEqual(classify_ending(equivalent, True), "inconclusive")
        self.assertEqual(classify_ending({"confirmed": True}, False), "inconclusive")
        self.assertEqual(classify_ending({"confirmed": True}, True), "inconclusive")
        self.assertEqual(classify_ending({"confirmed": True, "correctness": "PASS",
                                          "screening": {"median_paired_speedup": 1.04}}, True), "improvement")
        self.assertEqual(classify_ending({"correctness": "PASS",
                                          "screening": {"median_paired_speedup": 0.94}}, True), "slowdown")
        self.assertEqual(classify_ending({"correctness": "PASS",
                                          "screening": {"median_paired_speedup": 1.04}}, True), "inconclusive")
        self.assertEqual(classify_ending({"correctness": "FAIL", "confirmed": True,
                                          "screening": {"median_paired_speedup": 1.04}}, True), "inconclusive")
        self.assertEqual(classify_ending({}, None), "inconclusive")

    def test_replay_is_read_only_and_uses_verified_historical_numbers(self):
        app = create_app(ROOT)
        client = app.test_client()
        historical = ROOT / "evaluations/experiment-003-mask-hoisting-m128/20261028T145501296300Z/result.json"
        before = hashlib.sha256(historical.read_bytes()).hexdigest()
        jobs_before = app.extensions["research_workflow"].jobs()
        page = client.get("/research?plan=" + fixtures.PLAN_RUN).get_data(as_text=True)
        self.assertIn("Replay Experiment 003", page)
        self.assertIn('data-replay="{', page)
        self.assertIn("form-action 'self'", client.get("/research").headers["Content-Security-Policy"])
        self.assertNotIn("'unsafe-inline'", client.get("/research").headers["Content-Security-Policy"])
        with patch.object(dashboard_app, "mission_evidence", return_value={
                "hypothesis": '<img src=x onerror=alert(1)>', "id": HISTORICAL_ID}):
            escaped = client.get("/research?plan=" + fixtures.PLAN_RUN).get_data(as_text=True)
        self.assertNotIn('<img src=x onerror=alert(1)>', escaped)
        self.assertIn(r'\u003cimg src=x onerror=alert(1)\u003e', escaped)
        result = client.get("/api/research/mission/evidence/" + HISTORICAL_ID)
        self.assertEqual(result.status_code, 200)
        data = result.json
        self.assertEqual(data["correctness_cases"], [4, 4])
        self.assertEqual(data["executable_bytes"], 6912)
        self.assertTrue(data["executable_identical"])
        self.assertEqual(data["ending"], "equivalent")
        self.assertFalse(data["runtime_measured"])
        self.assertIsNone(data["screening"])
        self.assertEqual(app.extensions["research_workflow"].jobs(), jobs_before)
        self.assertEqual(hashlib.sha256(historical.read_bytes()).hexdigest(), before)

    def test_live_evidence_requires_validated_memory_and_missing_evidence_stays_unavailable(self):
        fixture = fixtures.ControlledWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        app = create_app(fixture.root)
        client = app.test_client()
        self.assertEqual(client.get("/api/research/mission/evidence/workflow:missing").status_code, 404)
        fixture.approve()
        fixture.manager.start_job(fixtures.PLAN_RUN, background=False)  # saved fixture output, no GPU
        job = fixture.manager.jobs()[0]
        result = client.get("/api/research/mission/evidence/" + job["memory_id"])
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.json["workflow_validation"])
        self.assertFalse(result.json["runtime_measured"])
        self.assertEqual(result.json["outcome"], job["outcome"])
        self.assertEqual(client.get("/api/research/mission/evidence/not-a-diagnostic").status_code, 404)

    @unittest.skipUnless(os.environ.get("NT_MISSION_BROWSER"), "opt-in Chromium mission screenshots")
    def test_chromium_replay_scenes_and_reduced_motion(self):
        from flask import request
        from werkzeug.serving import make_server
        import dashboard_app

        app = create_app(ROOT)
        posts = []
        app.before_request(lambda: posts.append(request.path) if request.method == "POST" else None)

        @app.get("/mission-smoke.js")
        def smoke_script():
            return app.response_class('''
document.addEventListener('DOMContentLoaded', () => {
  const step = new URL(location).searchParams.get('scene');
  document.querySelector('[data-presentation-choice="hollywood"]').click();
  if (step === 'live-complete') {
    document.querySelector('[data-mission-open="live"]').click();
    const next = () => document.querySelector('#mission-controls button').click();
    next(); next();
    setTimeout(() => {
      const overlay = document.querySelector('#mission-overlay');
      if (overlay.dataset.vault !== 'open' || !document.querySelector('#mission-body').textContent.includes('workflow validation')) throw Error('completed live evidence was not revealed');
      document.documentElement.dataset.missionSmoke = 'passed';
    }, 250);
    return;
  }
  document.querySelector('[data-mission-open="replay"]').click();
  const next = () => document.querySelector('#mission-controls button').click();
  if (step === 'hack' || step === 'vault') { next(); next(); }
  if (step === 'vault') next();
  const expected = step === 'hack' ? 'hack' : step === 'vault' ? 'vault' : 'briefing';
  if (document.querySelector('#mission-overlay').dataset.scene !== expected) throw Error('wrong scene');
  if (step === 'vault' && !document.querySelector('#mission-body').textContent.includes('6,912')) throw Error('unverified number');
  if (matchMedia('(prefers-reduced-motion: reduce)').matches &&
      getComputedStyle(document.querySelector('.vault-dial')).animationName !== 'none') throw Error('reduced motion ignored');
  document.documentElement.dataset.missionSmoke = 'passed';
});
''', mimetype="application/javascript")

        @app.after_request
        def inject(response):
            if request.path == "/research":
                response.set_data(response.get_data().replace(
                    b"</head>", b'<script src="/mission-smoke.js" defer></script></head>'))
            return response

        screenshots = Path(os.environ.get("NT_MISSION_SCREENSHOTS",
            str(ROOT / "evaluations/dashboard-previews/operation-stolen-cycles")))
        screenshots.mkdir(parents=True, exist_ok=True)
        server = make_server("127.0.0.1", 0, app, threaded=True)
        origin = f"http://127.0.0.1:{server.server_port}"
        with patch.object(dashboard_app, "ALLOWED_HOSTS", dashboard_app.ALLOWED_HOSTS | {origin[7:]}), \
                tempfile.TemporaryDirectory(prefix="nt-mission-browser-") as profile:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for scene in ("briefing", "hack", "vault", "live-complete"):
                    name = {"briefing": "locked", "hack": "archive-scanning", "vault": "revealed",
                            "live-complete": "live-complete"}[scene]
                    result = subprocess.run([os.environ["NT_MISSION_BROWSER"], "--headless", "--no-sandbox",
                        "--disable-gpu", "--disable-dev-shm-usage", "--no-proxy-server", "--no-first-run",
                        "--disable-background-networking", "--force-prefers-reduced-motion", "--user-data-dir=" + profile,
                        "--window-size=1440,1000", "--dump-dom", "--virtual-time-budget=1200",
                        "--screenshot=" + str(screenshots / (name + ".png")),
                        origin + "/research?plan=" + fixtures.PLAN_RUN + "&scene=" + scene],
                        capture_output=True, text=True, timeout=35)
                    self.assertEqual(result.returncode, 0, result.stderr[-1200:])
                    self.assertIn('data-mission-smoke="passed"', result.stdout, result.stdout[-2000:])
                    self.assertTrue((screenshots / (name + ".png")).is_file())
            finally:
                server.shutdown()
                thread.join(timeout=3)
                server.server_close()
        self.assertEqual(posts, [])

    @unittest.skipUnless(os.environ.get("NT_MISSION_BROWSER"), "opt-in Chromium live-state smoke")
    def test_chromium_live_queued_failed_and_missing_evidence(self):
        from flask import request
        from werkzeug.serving import make_server

        fixture = fixtures.ControlledWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.approve()
        job_id = "b" * 32
        job_path = f"{workflow.SECTION}/jobs/{job_id}/job.json"
        app = create_app(fixture.root)
        app.extensions["research_workflow"].recover = lambda: None  # keep synthetic queued state for viewing
        posts = []
        app.before_request(lambda: posts.append(request.path) if request.method == "POST" else None)

        @app.get("/mission-live-smoke.js")
        def smoke_script():
            return app.response_class('''
document.addEventListener('DOMContentLoaded', () => {
  const scenario = new URL(location).searchParams.get('scenario');
  document.querySelector('[data-presentation-choice="hollywood"]').click();
  document.querySelector('[data-mission-open="live"]').click();
  const next = () => document.querySelector('#mission-controls button').click();
  next(); next();
  setTimeout(() => {
    const overlay = document.querySelector('#mission-overlay');
    const expected = scenario === 'queued' || scenario === 'active-close' ? 'scanning' : scenario === 'failed' ? 'failed' : 'locked';
    if (overlay.dataset.vault !== expected) throw Error('incorrect live vault state: ' + overlay.dataset.vault);
    if (scenario === 'missing' && overlay.dataset.scene === 'vault') throw Error('missing evidence opened vault');
    if (scenario === 'queued' && !document.querySelector('#mission-body').textContent.includes('queued')) throw Error('job state absent');
    if (scenario === 'failed' && !document.querySelector('#mission-body').textContent.includes('GPU fixture failure')) throw Error('error absent');
    if (scenario === 'active-close') {
      document.querySelector('#mission-effects').click();
      if (overlay.dataset.effects !== 'off' || document.querySelector('#mission-sound').getAttribute('aria-pressed') !== 'false') throw Error('effects and sound not independent');
      document.querySelector('#mission-close').click();
      if (!overlay.hidden || document.documentElement.hasAttribute('data-mission-active')) throw Error('close did not clear overlay');
      document.querySelector('[data-presentation-choice="research"]').click();
      if (document.documentElement.dataset.presentation !== 'research') throw Error('mode switch failed');
    }
    document.documentElement.dataset.liveSmoke = 'passed';
  }, 180);
});
''', mimetype="application/javascript")

        @app.after_request
        def inject(response):
            if request.path == "/research":
                response.set_data(response.get_data().replace(
                    b"</head>", b'<script src="/mission-live-smoke.js" defer></script></head>'))
            return response

        screenshots = Path(os.environ.get("NT_MISSION_SCREENSHOTS",
            str(ROOT / "evaluations/dashboard-previews/operation-stolen-cycles")))
        screenshots.mkdir(parents=True, exist_ok=True)
        server = make_server("127.0.0.1", 0, app, threaded=True)
        origin = f"http://127.0.0.1:{server.server_port}"
        with patch.object(dashboard_app, "ALLOWED_HOSTS", dashboard_app.ALLOWED_HOSTS | {origin[7:]}), \
                tempfile.TemporaryDirectory(prefix="nt-mission-live-") as profile:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for scenario, status in (("queued", "queued"), ("active-close", "queued"),
                                         ("failed", "failed"), ("missing", "completed")):
                    workflow.save(fixture.root, job_path, {
                        "id": job_id, "plan_run_id": fixtures.PLAN_RUN, "status": status,
                        "created_utc": workflow.now(), "error": "GPU fixture failure" if status == "failed" else None,
                        "memory_id": "workflow:" + job_id if status == "completed" else None})
                    result = subprocess.run([os.environ["NT_MISSION_BROWSER"], "--headless", "--no-sandbox",
                        "--disable-gpu", "--disable-dev-shm-usage", "--no-proxy-server", "--no-first-run",
                        "--disable-background-networking", *(["--force-prefers-reduced-motion"] if scenario != "queued" else []), "--user-data-dir=" + profile,
                        "--window-size=1440,1000", "--dump-dom", "--virtual-time-budget=700",
                        "--screenshot=" + str(screenshots / (("scanning" if scenario == "queued" else scenario) + ".png")),
                        origin + "/research?plan=" + fixtures.PLAN_RUN + "&scenario=" + scenario],
                        capture_output=True, text=True, timeout=35)
                    self.assertEqual(result.returncode, 0, result.stderr[-1000:])
                    self.assertIn('data-live-smoke="passed"', result.stdout, result.stdout[-1500:])
            finally:
                server.shutdown()
                thread.join(timeout=3)
                server.server_close()
        self.assertEqual(posts, [])


if __name__ == "__main__":
    unittest.main()
