"""Mocked local proposal requests; no Muse API, GPU, or candidate execution."""

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
import unittest
from unittest.mock import patch

from dashboard_app import create_app
import kernel_evolution


REPOSITORY = Path(__file__).resolve().parents[1]
PRIOR_RUN = "20261027T163630745455Z"
BASE_URL = "http://127.0.0.1:8501"


def revised_source():
    return kernel_evolution.baseline_source().replace(
        "for offset in range(tl.cdiv(K, BK)):",
        "for offset in tl.range(0, tl.cdiv(K, BK), num_stages=3):")


def reply(source=None):
    return SimpleNamespace(id="mock-response", output_text=json.dumps({
        "kernel_source": source or revised_source(),
        "hypothesis": "Change K-loop scheduling for lower overhead",
        "expected_mechanism": "Change load and dot pipeline instructions",
        "reasoning_from_feedback": "Earlier trusted tests did not confirm a gain",
    }))


class ProposalDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        prior = self.root / "evaluations" / "kernel-evolution" / PRIOR_RUN
        prior.mkdir(parents=True)
        source = REPOSITORY / "evaluations" / "kernel-evolution" / PRIOR_RUN
        for name in ("metadata.json", "candidate.py", "proposal_audit.jsonl"):
            shutil.copy2(source / name, prior / name)
        self.app = create_app(self.root)
        self.client = self.app.test_client()
        page = self.client.get("/", base_url=BASE_URL)
        self.token = re.search(rb'name="csrf_token" value="([^"]+)"',
                               page.data).group(1).decode()

    def post(self, *, origin=BASE_URL, data=None):
        return self.client.post(
            "/api/proposals", base_url=BASE_URL,
            headers={"Origin": origin},
            data={"csrf_token": self.token} if data is None else data)

    def finish(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.client.get("/api/proposals/status", base_url=BASE_URL).json
            if not status["active"]:
                return status
            time.sleep(0.01)
        self.fail("mocked proposal worker did not finish")

    def test_success_saves_proposal_and_refreshes_timeline_without_execution(self):
        calls = []

        def model(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(responses=SimpleNamespace(
                create=lambda **request: reply()))

        with patch.dict(os.environ, {"MODEL_API_KEY": "test-key"}), \
             patch("dashboard_proposals.OpenAI", side_effect=model), \
             patch("torch.cuda.is_available", side_effect=AssertionError("GPU used")):
            started = self.post()
            self.assertEqual(started.status_code, 202)
            status = self.finish()
        self.assertEqual(status["state"], "saved_for_review")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timeout"], 30.0)
        self.assertEqual(calls[0]["max_retries"], 0)
        self.assertNotIn("test-key", str(status))
        detail_path = urlsplit(status["detail_url"]).path
        self.assertTrue(detail_path.startswith("/experiments/kernel:"))
        run_id = detail_path.split("kernel:", 1)[1]
        output = self.root / "evaluations" / "kernel-evolution" / run_id
        metadata = json.loads((output / "metadata.json").read_text())
        audit = json.loads((output / "proposal_audit.jsonl").read_text())
        source = (output / "candidate.py").read_text()
        self.assertEqual(metadata["candidate_sha256"], kernel_evolution.source_sha256(source))
        self.assertEqual(metadata["status"], "awaiting_human_review")
        self.assertEqual(metadata["review_from"], str(
            self.root / "evaluations" / "kernel-evolution" / PRIOR_RUN))
        self.assertEqual(metadata["static_findings"]["ast_check"],
                         "accepted_for_review_only")
        self.assertEqual(audit["raw_response"], reply().output_text)
        self.assertIn("prior_unevaluated_review", audit["prompt"])
        page = self.client.get(status["detail_url"], base_url=BASE_URL)
        self.assertEqual(page.status_code, 200)
        self.assertIn(metadata["candidate_sha256"].encode(), page.data)
        self.assertIn(b"Static review findings", page.data)
        self.assertIn(b"Awaiting review", page.data)
        self.assertIn(b"Proposal audit trail", page.data)
        self.assertIn(b"Change K-loop scheduling", page.data)

    def test_model_failure_and_timeout_hide_exception_text(self):
        for error, expected in ((RuntimeError("secret request header"),
                                 "Muse request failed"),
                                (TimeoutError("secret timeout"),
                                 "Muse request timed out")):
            with self.subTest(error=type(error).__name__):
                with patch.dict(os.environ, {"MODEL_API_KEY": "test-key"}), \
                     patch("dashboard_proposals.OpenAI", return_value=SimpleNamespace(
                         responses=SimpleNamespace(create=lambda **_kw: (_ for _ in ()).throw(error)))):
                    self.assertEqual(self.post().status_code, 202)
                    status = self.finish()
                self.assertEqual(status["state"], "failed")
                self.assertIn(expected, status["message"])
                self.assertNotIn("secret", str(status))
                runs = [p for p in (self.root / "evaluations" /
                                    "kernel-evolution").iterdir() if p.name != PRIOR_RUN]
                self.assertEqual(len(runs), 1 if isinstance(error, RuntimeError) else 2)
                artifact = max(runs)
                self.assertNotIn("secret", (artifact / "metadata.json").read_text())
                self.assertNotIn("secret", (artifact / "proposal_audit.jsonl").read_text())

    def test_invalid_response_is_audited_once_without_candidate(self):
        requested = []
        bad = SimpleNamespace(id="invalid", output_text='{"hypothesis":"missing source"}')

        def create(**kwargs):
            requested.append(kwargs)
            return bad

        with patch.dict(os.environ, {"MODEL_API_KEY": "test-key"}), \
             patch("dashboard_proposals.OpenAI", return_value=SimpleNamespace(
                 responses=SimpleNamespace(create=create))):
            self.assertEqual(self.post().status_code, 202)
            status = self.finish()
        self.assertEqual(status["state"], "failed")
        self.assertIn("static review", status["message"])
        self.assertEqual(len(requested), 1)
        runs = [p for p in (self.root / "evaluations" / "kernel-evolution").iterdir()
                if p.name != PRIOR_RUN]
        self.assertEqual(len(runs), 1)
        self.assertFalse((runs[0] / "candidate.py").exists())
        audit = json.loads((runs[0] / "proposal_audit.jsonl").read_text())
        self.assertEqual(audit["raw_response"], bad.output_text)
        self.assertIn("rejection", audit)

    def test_duplicate_request_is_rejected_while_first_is_active(self):
        entered, release = threading.Event(), threading.Event()

        def create(**_kwargs):
            entered.set()
            release.wait(2)
            return reply()

        with patch.dict(os.environ, {"MODEL_API_KEY": "test-key"}), \
             patch("dashboard_proposals.OpenAI", return_value=SimpleNamespace(
                 responses=SimpleNamespace(create=create))):
            self.assertEqual(self.post().status_code, 202)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.client.get("/api/proposals/status", base_url=BASE_URL)
                             .json["state"], "requesting_proposal")
            duplicate = self.post()
            self.assertEqual(duplicate.status_code, 409)
            self.assertTrue(duplicate.json["active"])
            release.set()
            self.assertEqual(self.finish()["state"], "saved_for_review")

    def test_validating_state_is_visible_before_static_review_finishes(self):
        entered, release = threading.Event(), threading.Event()
        original = kernel_evolution.parse_response

        def held_validation(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original(*args, **kwargs)

        with patch.dict(os.environ, {"MODEL_API_KEY": "test-key"}), \
             patch("dashboard_proposals.OpenAI", return_value=SimpleNamespace(
                 responses=SimpleNamespace(create=lambda **_kw: reply()))), \
             patch("kernel_evolution.parse_response", side_effect=held_validation):
            self.assertEqual(self.post().status_code, 202)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.client.get("/api/proposals/status", base_url=BASE_URL)
                             .json["state"], "validating_response")
            release.set()
            self.assertEqual(self.finish()["state"], "saved_for_review")

    def test_csrf_origin_host_and_request_shape(self):
        with patch("dashboard_proposals.OpenAI",
                   side_effect=AssertionError("model called")):
            self.assertEqual(self.post(origin="http://evil.example").status_code, 403)
            self.assertEqual(self.post(data={"csrf_token": "wrong"}).status_code, 403)
            self.assertEqual(self.post(data={"csrf_token": self.token,
                                             "out_dir": "/tmp"}).status_code, 403)
            self.assertEqual(self.client.post("/api/proposals", base_url=BASE_URL,
                                               data={"csrf_token": self.token}).status_code, 403)
            self.assertEqual(self.client.get("/", base_url="http://evil.example").status_code,
                             400)
            self.assertEqual(self.client.post(
                "/api/proposals", base_url=BASE_URL, headers={"Origin": BASE_URL},
                data={"csrf_token": self.token, "padding": "x" * 2000}).status_code, 413)
        self.assertEqual(self.app.extensions["proposal_manager"].snapshot()["state"], "ready")

    def test_existing_views_and_cli_guard_remain_available(self):
        with patch("dashboard_proposals.OpenAI",
                   side_effect=AssertionError("model called")):
            for view in ("overview", "timeline", "performance", "reliability"):
                response = self.client.get(f"/?view={view}", base_url=BASE_URL)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"NationalTensor", response.data)
        with patch("kernel_evolution.OpenAI",
                   side_effect=AssertionError("CLI called model")):
            with self.assertRaises(SystemExit) as error:
                kernel_evolution.main(["propose", "--gpu", "mock GPU"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
