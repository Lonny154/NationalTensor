"""Dedicated presentation parity, narrative scope, and opt-in browser comparisons."""
from copy import deepcopy
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from dashboard_app import create_app, _fmt_ms, _fmt_ratio, _fmt_pct
from dashboard_data import load_history
from dashboard_hollywood import mission_state
from dashboard_timeline import build_activity, FILTERS
import test_research_workflow as fixtures


class Layouts(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class HollywoodLayoutTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ControlledWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        fixture.approve()
        fixture.manager.start_job(fixtures.PLAN_RUN, background=False)  # saved-artifact mock, never CUDA
        for section in ("kernel-evolution", "structural-loop", "trusted-compute", "trusted-two-slab",
                        "planned-profiling", "stage-reduction", "experiment-003-mask-hoisting-m128",
                        "experiment-planner", "grouped-m4", "day2-reliability"):
            shutil.copytree(fixtures.ROOT / "evaluations" / section, self.root / "evaluations" / section, dirs_exist_ok=True)
        shutil.copytree(fixtures.ROOT / "runs/20261027T010046401977Z", self.root / "runs/20261027T010046401977Z")
        shutil.copytree(fixtures.ROOT / "docs/research_reviews", self.root / "docs/research_reviews")
        shutil.copy2(fixtures.ROOT / "docs/day2_checkpoint.md", self.root / "docs/day2_checkpoint.md")
        for filename in ("structural_loop.py", "planned_profile.py"):
            shutil.copy2(fixtures.ROOT / filename, self.root / filename)
        self.app = create_app(self.root)
        self.manager = self.app.extensions["research_workflow"]
        self.manager.executor = Mock(side_effect=AssertionError("GPU forbidden"))
        self.client = self.app.test_client()

    def get(self, url):
        response = self.client.get(url, base_url=fixtures.BASE_URL)
        self.assertEqual(response.status_code, 200)
        return response.get_data(as_text=True)

    def test_dedicated_layouts_and_exact_verified_scientific_metrics(self):
        for page in ("overview", "timeline", "performance", "reliability"):
            html = self.get("/?view=" + page)
            self.assertIn('data-mission-layout="' + page + '"', html)
            self.assertIn('class="research-presentation has-mission-layout"', html)
        activity = build_activity(self.root)
        archive = self.get("/?view=timeline")
        self.assertEqual(len(activity["events"]), 29)
        for row in activity["events"]:
            self.assertIn('data-mission-activity="' + row["id"] + '"', archive)
        for key in FILTERS:
            filtered = self.get("/?view=timeline&filter=" + key)
            expected = [r for r in activity["events"] if key == "all" or key in r["filters"]]
            self.assertEqual(filtered.count('data-mission-activity='), len(expected))
        self.assertIn("The compiler got there first.", archive)
        self.assertIn("The trail goes cold.", archive)
        self.assertIn("Not independent research evidence", archive)
        performance = self.get("/?view=performance")
        self.assertEqual(performance.count('data-performance-id='), 9)
        for row in activity["performance"]:
            values = row["measurement"]
            for value in (_fmt_ms(values["baseline_ms"]), _fmt_ms(values["candidate_ms"]),
                          _fmt_ratio(values["median_paired_speedup"])):
                self.assertIn(value, performance)
            self.assertIn('%.5f' % values["uncertainty"]["paired_ratio_mad"], performance)
        self.assertNotIn('data-performance-id="workflow:', performance)
        self.assertIn("Not Measured".upper(), performance.upper())
        reliability = self.get("/?view=reliability")
        for capture in load_history(self.root)["reliability"]["captures"]:
            for metrics in capture["metrics"].values():
                for field in ("median_absolute_ratio_deviation", "baseline_cv", "candidate_cv"):
                    self.assertIn(_fmt_pct(metrics[field]), reliability)
        self.assertIn("do not establish GPU clock drift", reliability)
        self.manager.executor.assert_not_called()

    def test_dossier_model_text_is_escaped_in_both_presentations(self):
        activity = build_activity(self.root)
        row = activity["events"][0]
        row["title"] = '<script>alert("model")</script>'
        row["description"] = '<img src=x onerror="alert(1)">'
        with patch("dashboard_app.build_activity", return_value=activity):
            html = self.get("/?view=timeline")
        self.assertNotIn(row["title"], html)
        self.assertNotIn(row["description"], html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)

    def test_heist_has_one_set_of_original_protected_controls_and_polling_node(self):
        html = self.get("/research?plan=" + fixtures.PLAN_RUN)
        tags = Layouts(html).tags
        self.assertEqual(sum(attrs.get("id") == "workflow-status" for _, attrs in tags), 1)
        self.assertEqual(sum(attrs.get("id") == "plan-select" for _, attrs in tags), 1)
        sources = {attrs["data-hollywood-source"] for _, attrs in tags if "data-hollywood-source" in attrs}
        slots = {attrs["data-hollywood-slot"] for _, attrs in tags if "data-hollywood-slot" in attrs}
        self.assertEqual(sources, slots)
        self.assertEqual(sources, {"planning", "intelligence", "authorization", "dispatch"})
        forms = fixtures.RenderedForms(html).forms
        self.assertEqual(sum(form["action"] == "/research/advisory" for form in forms), 1)
        self.assertIn("Verified memory evidence", html)
        self.assertIn("workflow validation", html.lower())
        self.assertNotIn("100%", html)

    def test_workflow_copy_is_derived_from_recorded_states_without_mutation(self):
        plan = {"run_id": "plan"}
        state = {"planning": {"status": "idle"}, "jobs": []}
        original = deepcopy(state)
        self.assertIn("clue", mission_state(plan, None, state, None, {})["title"].lower())
        approved = {"decision": "approved"}
        self.assertIn("ignition", mission_state(plan, approved, state, None, {})["title"])
        self.assertIn("declined", mission_state(plan, {"decision": "rejected"}, state, None, {})["title"])
        self.assertIn("locked", mission_state(plan, approved, state, "Hash changed", {})["title"])
        self.assertEqual(state, original)
        state["planning"]["status"] = "requesting"
        planning = mission_state(plan, approved, state, None, {})
        self.assertEqual(planning["anchor"], "planning")
        self.assertEqual(planning["phases"][0][2], "requesting")
        state["planning"]["status"] = "idle"
        for status in ("queued", "running", "validating", "failed", "interrupted", "completed"):
            job = {"id": "job", "plan_run_id": "plan", "created_utc": "2026-10-28", "status": status,
                   "memory_id": "workflow:job", "outcome": "compiler_equivalent_screening_skipped"}
            state["jobs"] = [job]
            result = mission_state(plan, approved, state, None, {})
            self.assertFalse(result["verified"])
            self.assertEqual(result["phases"][2][2], status)
            if status == "completed":
                self.assertNotIn("compiler got there", result["title"])
                result = mission_state(plan, approved, state, None,
                                       {"workflow:job": {"outcome": "compiler_equivalent_screening_skipped"}})
                self.assertTrue(result["verified"])
                self.assertIn("compiler got there", result["title"])
        # A completed replay on another plan cannot imply completion of the selected plan.
        self.assertIsNone(mission_state({"run_id": "other"}, None, state, None, {})["job"])
        job["memory_id"] = "diagnostic:other"
        self.assertFalse(mission_state(plan, approved, state, None,
                                      {"diagnostic:other": {"outcome": job["outcome"]}})["verified"])

    @unittest.skipUnless(os.environ.get("NT_HOLLYWOOD_BROWSER"), "opt-in Chromium layout comparisons")
    def test_browser_v2_page_comparisons_and_state_preservation(self):
        from flask import request, send_file
        from werkzeug.serving import make_server
        import dashboard_app

        posts = []
        self.app.before_request(lambda: posts.append(request.path) if request.method == "POST" else None)
        screenshots = Path(os.environ.get("NT_HOLLYWOOD_V2_SCREENSHOTS",
            str(fixtures.ROOT / "evaluations/dashboard-previews/hollywood-v2")))
        screenshots.mkdir(parents=True, exist_ok=True)
        views = {"overview": "/?view=overview", "reliability": "/?view=reliability",
                 "workflow": "/research?plan=" + fixtures.PLAN_RUN,
                 "timeline": "/?view=timeline", "performance": "/?view=performance"}

        @self.app.get("/v2-smoke.js")
        def smoke_script():
            return self.app.response_class('''
document.addEventListener('DOMContentLoaded', () => {
  const check = (ok, why) => { if (!ok) throw Error(why); };
  const root = document.documentElement;
  const url = location.href;
  const forms = [...document.forms];
  const before = forms.map(f => f.outerHTML);
  const select = document.querySelector('#plan-select');
  const selected = select && select.value;
  const status = document.querySelector('#workflow-status');
  const snapshot = status && status.dataset.snapshot;
  const toggle = mode => document.querySelector(`[data-presentation-choice="${mode}"]`).click();
  toggle('hollywood');
  check(getComputedStyle(document.querySelector('.has-mission-layout')).display === 'none', 'research not hidden');
  check(getComputedStyle(document.querySelector('.hollywood-experience')).display !== 'none', 'mission missing');
  if (status) check(status.closest('[data-hollywood-slot="planning"]'), 'shared controls not moved');
  toggle('research');
  if (status) check(status.closest('.research-presentation'), 'original position lost');
  check(getComputedStyle(document.querySelector('.hollywood-experience')).display === 'none', 'mission not hidden');
  toggle(new URL(url).searchParams.get('preview'));
  check(location.href === url, 'navigation changed');
  check(forms.every((form, i) => form.isConnected && form.outerHTML === before[i]), 'controls changed or cloned');
  check(!select || select.value === selected, 'selection reset');
  check(!status || status.dataset.snapshot === snapshot, 'job state reset');
  check(document.querySelectorAll('#plan-select').length <= 1, 'duplicate controls');
  root.dataset.smoke = 'passed';
});
''', mimetype="application/javascript")

        @self.app.after_request
        def inject(response):
            if request.path in {"/", "/research"}:
                response.set_data(response.get_data().replace(b"</head>", b'<script src="/v2-smoke.js" defer></script></head>'))
            return response

        @self.app.get("/comparison/<page>")
        def comparison(page):
            if page not in {"overview", "reliability", "workflow"}:
                return "Unknown comparison", 404
            # Browser-composed contact sheet, using only the captured screenshots.
            return '<!doctype html><html><head><link rel="stylesheet" href="/comparison.css"></head><body><section><h1>RESEARCH MODE / ' + page + '</h1><img src="/preview/research-' + page + '"></section><section><h1>HOLLYWOOD MODE / ' + page + '</h1><img src="/preview/hollywood-' + page + '"></section></body></html>'

        @self.app.get("/comparison.css")
        def comparison_css():
            return self.app.response_class("body{margin:0;display:flex;background:#080b0a;color:#ead9ae;font:18px monospace}section{width:1440px}h1{height:60px;box-sizing:border-box;margin:0;padding:20px;font-size:18px}img{display:block;width:1440px}", mimetype="text/css")

        @self.app.get("/preview/<name>")
        def screenshot_file(name):
            allowed = {mode + "-" + view for mode in ("research", "hollywood") for view in views}
            if name not in allowed:
                return "Unknown image", 404
            return send_file(screenshots / (name + ".png"), mimetype="image/png")

        server = make_server("127.0.0.1", 0, self.app, threaded=True)
        origin = f"http://127.0.0.1:{server.server_port}"
        with patch.object(dashboard_app, "ALLOWED_HOSTS", dashboard_app.ALLOWED_HOSTS | {origin[7:]}), \
                tempfile.TemporaryDirectory(prefix="nt-v2-browser-") as profile:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def capture(path, name, comparison=False):
                result = subprocess.run([os.environ["NT_HOLLYWOOD_BROWSER"], "--headless", "--no-sandbox",
                    "--disable-gpu", "--disable-dev-shm-usage", "--no-proxy-server", "--no-first-run",
                    "--disable-background-networking", "--user-data-dir=" + profile,
                    "--window-size=" + ("2880,1160" if comparison else "1440,1100"),
                    "--dump-dom", "--virtual-time-budget=1500", "--screenshot=" + str(screenshots / (name + ".png")),
                    origin + path], capture_output=True, text=True, timeout=35)
                self.assertEqual(result.returncode, 0, result.stderr[-1500:])
                if not comparison:
                    self.assertIn('data-smoke="passed"', result.stdout, result.stdout[-1500:])
                self.assertTrue((screenshots / (name + ".png")).is_file(), result.stderr[-1500:])

            try:
                for page, path in views.items():
                    for mode in ("research", "hollywood"):
                        capture(path + "&preview=" + mode, mode + "-" + page)
                for page in ("overview", "reliability", "workflow"):
                    capture("/comparison/" + page, "side-by-side-" + page, comparison=True)
            finally:
                server.shutdown()
                thread.join(timeout=3)
                server.server_close()
        self.assertEqual(posts, [])
        self.manager.executor.assert_not_called()
