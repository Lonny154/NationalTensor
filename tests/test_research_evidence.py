"""Read-only evidence presentation; mock execution produces the fixture, never GPU work."""

from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from dashboard_app import create_app
import experiment_memory
import test_research_workflow as fixtures


class EvidencePage(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.in_raw = False
        self.raw_collapsed = False
        self.visible = []
        self.raw = []
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "details" and attrs.get("id") == "raw-evidence":
            self.in_raw = True
            self.raw_collapsed = "open" not in attrs

    def handle_endtag(self, tag):
        if tag == "details":
            self.in_raw = False

    def handle_data(self, data):
        if self.in_raw:
            self.raw.append(data)
        else:
            self.visible.append(data)

    @property
    def text(self):
        return " ".join(" ".join(self.visible).split())


class ResearchEvidenceTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ControlledWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        fixture.approve()
        fixture.manager.start_job(fixtures.PLAN_RUN, background=False)
        self.job = fixture.manager.jobs()[0]
        self.memory = experiment_memory.reconstruct(self.root)
        self.entry = next(row for row in self.memory["entries"] if row["id"] == self.job["memory_id"])

    def test_workflow_summary_shows_all_findings_and_keeps_complete_json_collapsed(self):
        snapshot = {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in self.root.rglob("*") if path.is_file()}
        with patch("research_workflow.execute_trusted", side_effect=AssertionError("Execution from evidence view")) as execute:
            client = create_app(self.root).test_client()
            response = client.get("/research/evidence/" + self.entry["id"], base_url=fixtures.BASE_URL)
        execute.assert_not_called()
        self.assertEqual(response.status_code, 200)
        page = EvidencePage(response.get_data(as_text=True))
        self.assertTrue(page.raw_collapsed)
        self.assertIn("Workflow validation — not independent research evidence", page.text)
        self.assertIn("Does the reviewed mask-hoisting kernel show a paired screening difference on M=128", page.text)
        self.assertIn(self.entry["hypothesis"], page.text)
        for heading in ("Original Muse hypothesis", "Approval and execution", "GPU correctness",
                        "Compiler resource comparison", "Executable equivalence", "Runtime benchmarking",
                        "Outcome and research interpretation", "Limitations"):
            self.assertIn(heading, page.text)
        self.assertIn("Human approval Approved", page.text)
        self.assertIn("Execution Completed", page.text)
        self.assertIn("4/4 checks passed", page.text)
        self.assertIn("rtol=0.02, atol=0.02", page.text)
        for implementation in ("Baseline", "Trusted"):
            for seed in (42, 43):
                self.assertIn(f"{implementation} {seed} PASS", page.text)
        self.assertIn("Registers / thread 120 120", page.text)
        self.assertIn("Shared memory / block (bytes) 24576 24576", page.text)
        self.assertIn("Compiler spills 0 0", page.text)
        self.assertIn("6,912 bytes", page.text)
        self.assertIn("byte-for-byte identical", page.text)
        self.assertIn("PTX differences do not survive", page.text)
        self.assertIn("Benchmarking was intentionally skipped", page.text)
        self.assertIn("No latency, paired speedup, or performance improvement was established", page.text)
        self.assertIn("compiler equivalent screening skipped", page.text)
        raw = "".join(page.raw)
        # The details section contains its summary label followed by the unchanged full JSON.
        self.assertEqual(json.loads(raw[raw.index("{"):]), self.entry)
        self.assertNotIn('"source_identity"', page.text)
        self.assertEqual(snapshot, {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in self.root.rglob("*") if path.is_file()})

    def test_legacy_or_incomplete_summary_is_safe_and_does_not_invent_results(self):
        entry = deepcopy(self.entry)
        entry.update(id="legacy:unmeasured", workflow_validation=False, independent_research_evidence=None,
                     hypothesis='<script>alert("hypothesis")</script>', mechanism='<img src=x onerror="alert(1)">',
                     correctness="not evaluated", correctness_cases=None, compiler=None,
                     diagnostic_evidence=None, outcome="unevaluated", performance_interpretation="Not measured")
        with patch("dashboard_app.experiment_memory.reconstruct", return_value={"entries": [entry]}):
            response = create_app(self.root).test_client().get("/research/evidence/" + entry["id"], base_url=fixtures.BASE_URL)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        page = EvidencePage(html)
        self.assertNotIn('<script>alert("hypothesis")</script>', html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("No approval record linked", page.text)
        self.assertIn("No compiler resource comparison was recorded", page.text)
        self.assertIn("No executable-equivalence comparison was recorded", page.text)
        self.assertIn("No runtime measurement is recorded", page.text)
        self.assertNotIn("4/4 checks passed", page.text)
        self.assertNotIn("INDEPENDENT RESEARCH EVIDENCE", page.text)
        self.assertTrue(page.raw_collapsed)

    def test_unknown_evidence_remains_not_found(self):
        response = create_app(self.root).test_client().get("/research/evidence/unknown", base_url=fixtures.BASE_URL)
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
