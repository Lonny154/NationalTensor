"""Muse debrief tests use fake API responses and never touch a GPU."""

import copy
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mask_hoisting_memory
import muse_debrief
from dashboard_app import create_app

ROOT = Path(__file__).resolve().parents[1]


def valid_response(package):
    identifier = package["evidence_id"]
    def field(text, *keys):
        return {"text": text, "citations": [
            {"evidence_id": identifier, "observation_key": key} for key in keys]}
    return {
        "technical_interpretation": field("The compiled variants match for this recorded configuration.", "executable_comparison", "scope_and_limitations"),
        "supporting_evidence": field("Four checks passed and both sides use 120 registers per thread.", "correctness", "compiler_resources"),
        "limitations_uncertainties": field("Runtime was not measured, so performance remains uncertain.", "screening"),
        "recommended_next_action": field("Review a distinct workload where the compiler may retain a difference.", "scope_and_limitations"),
        "nontechnical_explanation": field("Two recipes made the same machine instructions; no stopwatch was used.", "executable_comparison", "screening"),
        "hollywood_debrief": field("The compiler gave both suspects the same costume. The stopwatch stayed in its trailer.", "executable_comparison", "screening"),
    }


class MuseDebriefTests(unittest.TestCase):
    def setUp(self):
        self.package = muse_debrief.evidence_package(ROOT)

    def test_deterministic_verified_package_records_no_timing(self):
        self.assertEqual(self.package, muse_debrief.evidence_package(ROOT))
        observations = self.package["observations"]
        self.assertEqual(observations["correctness"]["passed"], 4)
        self.assertEqual(observations["compiler_resources"]["baseline"]["registers_per_thread"], 120)
        self.assertTrue(observations["executable_comparison"]["complete_executable_section"]["identical"])
        self.assertFalse(observations["screening"]["latency_measured"])
        self.assertFalse(observations["screening"]["speedup_measured"])
        self.assertIsNone(observations["screening"]["median_paired_speedup"])
        self.assertIn("No meaningful compiler-level difference", observations["screening"]["reason"])
        self.assertFalse(observations["source_hashes"]["exact_generated_source_evaluated"])

    def test_prompt_example_uses_appropriate_local_keys(self):
        prompt = muse_debrief.prompt_for(self.package)
        schema = json.loads(prompt.split("\nSCHEMA\n")[1].split("\nEVIDENCE\n")[0])
        muse_debrief.parse_response(json.dumps(schema), self.package)
        keys = {c["observation_key"] for field in schema.values() for c in field["citations"]}
        self.assertTrue({"correctness", "compiler_resources", "executable_comparison",
                         "scope_and_limitations", "screening"}.issubset(keys))

    def test_supported_message_structures_and_sdk_convenience_property(self):
        from openai.types.responses import Response
        raw = json.dumps(valid_response(self.package))
        payload = {"id": "resp_supported", "model": muse_debrief.MODEL, "status": "completed",
                   "output": [{"type": "reasoning", "summary": []},
                              {"type": "message", "role": "assistant", "status": "completed",
                               "content": [{"type": "output_text", "text": raw[:100]},
                                           {"type": "output_text", "text": raw[100:]}]}]}
        sdk = Response.model_construct(**{
            **payload, "output": [SimpleNamespace(type="reasoning"), SimpleNamespace(
                type="message", role="assistant", status="completed", content=[
                    SimpleNamespace(type="output_text", text=raw)])]})
        self.assertEqual(sdk.output_text, raw)
        for response in (payload, sdk, SimpleNamespace(status="completed", output_text=raw)):
            with self.subTest(response=type(response).__name__):
                text, metadata = muse_debrief.inspect_response(response)
                muse_debrief.require_complete_text(text, metadata)
                self.assertEqual(muse_debrief.parse_response(text, self.package), valid_response(self.package))

    def test_empty_incomplete_refusal_and_unsupported_responses_are_distinct(self):
        base = {"id": "resp_empty", "model": muse_debrief.MODEL, "status": "completed", "output": []}
        cases = [
            (base, "empty_output"),
            ({**base, "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
              "usage": {"output_tokens": 2500, "output_tokens_details": {"reasoning_tokens": 2500}}}, "incomplete_output"),
            ({**base, "output": [{"type": "message", "status": "incomplete", "content": []}]}, "incomplete_output"),
            ({**base, "status": "failed", "error": {"code": "server_error", "message": "secret must not be logged"}}, "api_error"),
            ({**base, "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "private refusal"}]}]}, "refusal"),
            ({**base, "output": [{"type": "function_call", "arguments": "private arguments"}]}, "unsupported_output"),
        ]
        for response, kind in cases:
            with self.subTest(kind=kind):
                raw, metadata = muse_debrief.inspect_response(response)
                with self.assertRaises(muse_debrief.ResponseHandlingError) as raised:
                    muse_debrief.require_complete_text(raw, metadata)
                self.assertEqual(raised.exception.kind, kind)
                self.assertNotIn("secret", json.dumps(metadata))
                self.assertNotIn("private", json.dumps(metadata))

    def test_incomplete_valid_json_is_never_accepted_and_metadata_is_saved(self):
        raw = json.dumps(valid_response(self.package))
        with tempfile.TemporaryDirectory() as temp:
            client = Mock()
            client.responses.create.return_value = {
                "id": "resp_incomplete", "model": muse_debrief.MODEL, "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": raw, "usage": {"output_tokens": 2500}}
            with patch.object(muse_debrief, "evidence_package", return_value=self.package):
                self.assertEqual(muse_debrief.request_once(client, Path(temp)), "rejected")
            audit = json.loads(next((Path(temp) / muse_debrief.SECTION).glob("*/audit.json")).read_text())
            self.assertEqual(audit["rejection_kind"], "incomplete_output")
            self.assertEqual(audit["raw_response"], raw)
            self.assertEqual(audit["response_metadata"]["usage"]["output_tokens"], 2500)
            self.assertFalse((Path(temp) / muse_debrief.SECTION / "accepted.json").exists())

    def test_empty_text_with_id_is_audited_as_empty_not_malformed(self):
        with tempfile.TemporaryDirectory() as temp:
            client = Mock()
            client.responses.create.return_value = SimpleNamespace(
                id="resp_empty", model=muse_debrief.MODEL, status="completed", output_text="")
            with patch.object(muse_debrief, "evidence_package", return_value=self.package):
                self.assertEqual(muse_debrief.request_once(client, Path(temp)), "rejected")
            audit = json.loads(next((Path(temp) / muse_debrief.SECTION).glob("*/audit.json")).read_text())
            self.assertEqual(audit["rejection_kind"], "empty_output")
            self.assertEqual(audit["response_id"], "resp_empty")
            self.assertEqual(audit["raw_response"], "")
            self.assertNotIn("not JSON", audit["rejection"])
            self.assertEqual(audit["request_options"], muse_debrief.REQUEST_OPTIONS)
            client.responses.create.assert_called_once_with(
                model=muse_debrief.MODEL, input=muse_debrief.prompt_for(self.package), **muse_debrief.REQUEST_OPTIONS)

    def test_valid_citations_and_invalid_references(self):
        response = valid_response(self.package)
        self.assertEqual(muse_debrief.parse_response(json.dumps(response), self.package), response)
        for key, value in (("evidence_id", "fabricated"), ("observation_key", "imaginary")):
            bad = copy.deepcopy(response)
            bad["technical_interpretation"]["citations"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "citation"):
                muse_debrief.parse_response(json.dumps(bad), self.package)

    def test_fabricated_number_and_runtime_claim_rejected(self):
        bad = valid_response(self.package)
        bad["supporting_evidence"]["text"] = "The result achieved a 2.5x speedup."
        with self.assertRaisesRegex(ValueError, "numerical"):
            muse_debrief.parse_response(json.dumps(bad), self.package)
        bad["supporting_evidence"]["text"] = "The test confirmed faster runtime."
        with self.assertRaisesRegex(ValueError, "runtime"):
            muse_debrief.parse_response(json.dumps(bad), self.package)
        bad["supporting_evidence"]["text"] = "The variant delivered a 4x speedup."
        with self.assertRaisesRegex(ValueError, "runtime"):
            muse_debrief.parse_response(json.dumps(bad), self.package)

    def test_invalid_model_response_is_audited_and_retry_can_succeed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock()
            client.responses.create.return_value = SimpleNamespace(status="completed", output_text="not json", model=muse_debrief.MODEL, id="bad")
            with patch.object(muse_debrief, "evidence_package", return_value=self.package):
                self.assertEqual(muse_debrief.request_once(client, root), "rejected")
                audits = list((root / muse_debrief.SECTION).glob("*/audit.json"))
                self.assertEqual(len(audits), 1)
                audit = json.loads(audits[0].read_text())
                self.assertEqual(audit["raw_response"], "not json")
                self.assertIn("response is not JSON", audit["rejection"])
                client.responses.create.return_value = SimpleNamespace(
                    status="completed", output_text=json.dumps(valid_response(self.package)), model=muse_debrief.MODEL, id="good")
                self.assertEqual(muse_debrief.request_once(client, root), "accepted")
                self.assertEqual(muse_debrief.request_once(client, root), "already_accepted")
                self.assertEqual(client.responses.create.call_count, 2)
                accepted = muse_debrief.load_accepted(root, self.package)
                self.assertEqual(accepted["response_model"], muse_debrief.MODEL)
                self.assertEqual(accepted["prompt"], muse_debrief.prompt_for(self.package))
                self.assertEqual(accepted["interpretation"], valid_response(self.package))

    def test_mismatched_model_identity_is_rejected_and_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = Mock()
            client.responses.create.return_value = SimpleNamespace(
                status="completed", output_text=json.dumps(valid_response(self.package)), model="other-model", id="wrong")
            with patch.object(muse_debrief, "evidence_package", return_value=self.package):
                self.assertEqual(muse_debrief.request_once(client, root), "rejected")
                self.assertIsNone(muse_debrief.load_accepted(root, self.package))
            audit_path = next((root / muse_debrief.SECTION).glob("*/audit.json"))
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit["response_model"], "other-model")
            self.assertIn("model identity", audit["rejection"])

    def test_simultaneous_request_does_not_call_muse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / muse_debrief.SECTION
            base.mkdir(parents=True)
            client = Mock()
            with (base / "request.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(muse_debrief, "evidence_package", return_value=self.package):
                    self.assertEqual(muse_debrief.request_once(client, root), "request_in_progress")
            client.responses.create.assert_not_called()

    def test_missing_evidence_and_source_hash_mismatch_block_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                muse_debrief.evidence_package(root)
            folder = root / mask_hoisting_memory.FOLDER
            folder.parent.mkdir(parents=True)
            shutil.copytree(ROOT / mask_hoisting_memory.FOLDER, folder)
            prior = ROOT / "evaluations/trusted-compute/20261027T154233319712Z/experiment.json"
            plan = ROOT / "evaluations/experiment-planner/20261028T145028833360Z/plan.json"
            for source in (prior, plan, ROOT / "trusted_compute_kernel.py", ROOT / "reference_precision.py",
                           ROOT / "benchmark_core.py", ROOT / "benchmark_proposal.py"):
                target = root / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            (root / "trusted_compute_kernel.py").write_text("# changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                muse_debrief.evidence_package(root)

    def test_dashboard_both_modes_and_explicit_action(self):
        app = create_app(ROOT)
        page = app.test_client().get("/research").get_data(as_text=True)
        self.assertIn("Request one live Muse Research Debrief", page)
        self.assertIn("No latency or speedup was measured", page)
        self.assertIn("/research/muse-debrief", page)
        accepted = {"interpretation": valid_response(self.package), "response_model": muse_debrief.MODEL,
                    "prompt": muse_debrief.prompt_for(self.package), "raw_response": json.dumps(valid_response(self.package))}
        with patch.object(muse_debrief, "load_accepted", return_value=accepted):
            page = app.test_client().get("/research").get_data(as_text=True)
        self.assertIn("Muse interpretation", page)
        self.assertIn("Theatrical dialogue", page)
        self.assertIn("Limitations and uncertainty", page)


if __name__ == "__main__":
    unittest.main()
