"""Page hierarchy and evidence display use saved fixtures only."""

import copy
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from dashboard_app import _dialogue_lines, create_app
import muse_debrief


ROOT = Path(__file__).resolve().parents[1]


class DashboardSimplificationTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(ROOT).test_client()

    def test_primary_navigation_is_consistent_across_routes(self):
        evidence_id = muse_debrief.evidence_package(ROOT)["evidence_id"]
        pages = ["/?view=overview", "/?view=timeline", "/?view=performance",
                 "/?view=reliability", "/research", "/research/evidence/" + evidence_id]
        labels = ["Overview", "Timeline", "Performance", "Reliability", "Research workflow"]
        for route in pages:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)
                html = response.get_data(as_text=True)
                nav = re.search(r'<nav class="top-nav" aria-label="Sections">(.*?)</nav>', html, re.S)
                self.assertIsNotNone(nav)
                links = re.findall(r'<a\b[^>]*>\s*([^<]+)\s*</a>', nav.group(1))
                self.assertEqual([label.strip() for label in links], labels)
                self.assertEqual(nav.group(1).count('aria-current="page"'), 1)
                self.assertIn('href="#main-content">Skip to content</a>', html)

    def test_critical_findings_remain_visible_and_full_records_expand(self):
        overview = self.client.get("/").get_data(as_text=True)
        self.assertIn("What needs attention?", overview)
        self.assertIn('class="primary-link" href="/research"', overview)
        self.assertIn("Research totals and test environment</summary>", overview)
        performance = self.client.get("/?view=performance").get_data(as_text=True)
        self.assertIn("These are saved paired screenings, not confirmed improvements", performance)
        self.assertRegex(performance, r'<details class="progressive-section"><summary>Compare \d+ measured screenings')
        evidence_id = muse_debrief.evidence_package(ROOT)["evidence_id"]
        evidence = self.client.get("/research/evidence/" + evidence_id).get_data(as_text=True)
        self.assertIn("<strong>Recorded outcome:</strong>", evidence)
        self.assertIn("<strong>Runtime:</strong> Not measured", evidence)
        self.assertIn("Inspect correctness, compiler, and provenance details</summary>", evidence)
        self.assertIn("Complete raw evidence JSON</summary>", evidence)
        self.assertIn("No latency, paired speedup, or performance improvement was established", evidence)
        workflow = self.client.get("/research").get_data(as_text=True)
        self.assertIn('href="#workflow-jobs">Inspect saved job', workflow)
        self.assertIn('id="workflow-jobs"', workflow)

    def test_muse_dialogue_renders_separate_escaped_character_lines(self):
        raw = "VAULT GUARD: The compiler is sealed. ROOKIE: Did it steal <cycles>?"
        self.assertEqual(_dialogue_lines(raw), [
            {"speaker": "VAULT GUARD", "line": "The compiler is sealed."},
            {"speaker": "ROOKIE", "line": "Did it steal <cycles>?"}])
        saved = json.loads((ROOT / muse_debrief.SECTION / "accepted.json").read_text())
        fixture = copy.deepcopy(saved)
        fixture["interpretation"]["hollywood_debrief"]["text"] = raw
        with patch.object(muse_debrief, "load_cinematic_revision", return_value=fixture):
            html = self.client.get("/research").get_data(as_text=True)
        self.assertIn('class="dialogue-lines"', html)
        self.assertIn("<strong>VAULT GUARD</strong>", html)
        self.assertIn("<strong>ROOKIE</strong>", html)
        self.assertIn("Did it steal &lt;cycles&gt;?", html)
        self.assertNotIn("Did it steal <cycles>?", html)
        self.assertIn("Verified observation:", html)
        self.assertIn("Muse interpretation:", html)
        self.assertIn("Fictional dialogue", html)


if __name__ == "__main__":
    unittest.main()
