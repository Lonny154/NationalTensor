"""Presentation isolation and opt-in Chromium smoke tests; no GPU or Muse calls."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import urlencode
from unittest.mock import Mock, patch

from dashboard_app import create_app
import experiment_memory
import test_research_workflow as fixtures


class HollywoodModeTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ControlledWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.app = create_app(self.root)
        self.manager = self.app.extensions["research_workflow"]
        self.manager.executor = Mock(side_effect=AssertionError("GPU forbidden"))
        self.client = self.app.test_client()
        self.url = "/research?plan=" + fixtures.PLAN_RUN

    def test_modes_share_forms_routes_and_csrf_while_approval_never_executes(self):
        pages = [self.client.get(self.url + "&mode=" + mode, base_url=fixtures.BASE_URL)
                 for mode in ("research", "hollywood")]
        forms = [fixtures.RenderedForms(page.get_data(as_text=True)).forms for page in pages]
        self.assertEqual(forms[0], forms[1])
        for page in pages:
            html = page.get_data(as_text=True)
            self.assertIn('type="button" data-presentation-choice="hollywood"', html)
            self.assertIn("Approve fixed diagnostic", html)
            self.assertIn("Authorize the Operation", html)
            self.assertIn("form-action 'self'", page.headers["Content-Security-Policy"])
            self.assertNotIn("'unsafe-inline'", page.headers["Content-Security-Policy"])
        form = next(form for form in forms[0] if form["action"].endswith("/approve"))
        token = dict(form["fields"])["csrf_token"]
        headers = {"Origin": fixtures.BASE_URL}
        for fields in ({}, {"csrf_token": "forged"}, {"csrf_token": token, "mode": "hollywood"},
                       {"csrf_token": token, "runner_id": "forged"}):
            self.assertEqual(self.client.post(form["action"], data=fields, headers=headers,
                                             content_type="application/x-www-form-urlencoded",
                                             base_url=fixtures.BASE_URL).status_code, 403)
        self.assertEqual(self.client.post(form["action"], data={"csrf_token": token},
                                         headers={"Origin": "http://attacker.invalid"},
                                         base_url=fixtures.BASE_URL).status_code, 403)
        response = self.client.post(form["action"], data=urlencode(form["fields"]),
                                    content_type="application/x-www-form-urlencoded", headers=headers,
                                    base_url=fixtures.BASE_URL)
        self.assertEqual(response.status_code, 303)
        review = self.manager.review(fixtures.PLAN_RUN)
        self.assertEqual(review["decision"], "approved")
        approved = self.client.get(self.url, base_url=fixtures.BASE_URL).get_data(as_text=True)
        self.assertIn("Execute approved diagnostic", approved)
        self.assertIn("Initiate the Heist", approved)
        self.assertEqual(self.manager.jobs(), [])
        self.manager.executor.assert_not_called()

    def test_reading_modes_cannot_change_science_or_workflow_state(self):
        self.fixture.approve()
        self.fixture.manager.start_job(fixtures.PLAN_RUN, background=False)  # fixture artifacts only
        original = experiment_memory.reconstruct(self.root)
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        memory_id = self.fixture.manager.jobs()[0]["memory_id"]
        for mode in ("research", "hollywood"):
            for route in (self.url, "/?view=timeline&filter=workflow", "/research/evidence/" + memory_id):
                response = self.client.get(route + "&mode=" + mode if "?" in route else route + "?mode=" + mode,
                                           base_url=fixtures.BASE_URL)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"workflow validation", response.data.lower())
            page = self.client.get("/research/evidence/" + memory_id, base_url=fixtures.BASE_URL).data
            self.assertIn(b"Not measured", page)
            self.assertIn(b"No latency, paired speedup, or performance improvement was established", page)
        self.assertEqual(original, experiment_memory.reconstruct(self.root))
        self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()})
        self.manager.executor.assert_not_called()

    @unittest.skipUnless(os.environ.get("NT_HOLLYWOOD_BROWSER"), "opt-in Chromium visual smoke")
    def test_browser_toggle_preserves_dom_url_selection_and_reload_preference(self):
        from flask import request
        from werkzeug.serving import make_server
        import dashboard_app

        posts = []
        self.app.before_request(lambda: posts.append(request.path) if request.method == "POST" else None)

        @self.app.get("/mode-smoke.js")
        def script():
            return self.app.response_class('''
document.addEventListener('DOMContentLoaded', () => {
const check = (condition, message) => { if (!condition) throw Error(message); };
const mode = () => document.documentElement.dataset.presentation;
const click = value => document.querySelector(`[data-presentation-choice="${value}"]`).click();
const requested = new URL(location).searchParams.get('preview');
const initial = mode();
const url = location.href;
const select = document.querySelector('#plan-select');
const chosen = select.value;
const forms = [...document.forms].map(f => f.outerHTML).sort().join('');
const details = document.querySelector('details');
if (details) details.open = true;
click('hollywood');
check(mode() === 'hollywood', 'toggle failed');
check(getComputedStyle(document.querySelector('.hollywood-experience')).display !== 'none', 'layout missing');
click('research');
check(getComputedStyle(document.querySelector('.hollywood-experience')).display === 'none', 'research changed');
check(location.href === url && select.value === chosen, 'navigation or selection reset');
check([...document.forms].map(f => f.outerHTML).sort().join('') === forms, 'forms changed');
check(!details || details.open, 'details reset');
if (details) details.open = false;
if (requested === 'persisted') check(initial === 'hollywood', 'preference lost on reload');
click(requested === 'research' ? 'research' : 'hollywood');
document.documentElement.dataset.smoke = 'passed';
});
''', mimetype="application/javascript")

        @self.app.after_request
        def inject(response):
            if request.path == "/research":
                response.set_data(response.get_data().replace(b"</head>",
                    b'<script src="/mode-smoke.js" defer></script></head>'))
            return response

        screenshots = Path(os.environ.get("NT_HOLLYWOOD_SCREENSHOTS",
            str(fixtures.ROOT / "evaluations/dashboard-previews/hollywood-v1")))
        screenshots.mkdir(parents=True, exist_ok=True)
        server = make_server("127.0.0.1", 0, self.app, threaded=True)
        origin = f"http://127.0.0.1:{server.server_port}"
        with patch.object(dashboard_app, "ALLOWED_HOSTS", dashboard_app.ALLOWED_HOSTS | {origin[7:]}), \
                tempfile.TemporaryDirectory(prefix="nt-mode-browser-") as profile:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for mode in ("research", "hollywood", "persisted"):
                    result = subprocess.run([os.environ["NT_HOLLYWOOD_BROWSER"], "--headless", "--no-sandbox",
                        "--disable-gpu", "--disable-dev-shm-usage", "--no-proxy-server", "--no-first-run",
                        "--disable-background-networking", "--user-data-dir=" + profile, "--window-size=1440,1100",
                        "--dump-dom", "--virtual-time-budget=1500", "--screenshot=" + str(screenshots / (mode + ".png")),
                        origin + self.url + "&preview=" + mode], capture_output=True, text=True, timeout=35)
                    self.assertEqual(result.returncode, 0, result.stderr[-1500:])
                    self.assertIn('data-smoke="passed"', result.stdout, result.stdout[-2000:])
                    self.assertTrue((screenshots / (mode + ".png")).is_file(), result.stderr[-2000:])
            finally:
                server.shutdown()
                thread.join(timeout=3)
                server.server_close()
        self.assertEqual(posts, [])
        self.assertEqual(self.manager.jobs(), [])
        self.manager.executor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
