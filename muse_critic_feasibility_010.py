"""Experiment 010: one explicit Muse v3 P06 feasibility attempt, never retried."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import API_BASE_URL, MODEL, _get, canonical, digest, inspect_response, require_complete_text
from muse_discovery_study import REQUEST_OPTIONS
from muse_research_critic import source_package
from muse_research_critic_v3 import VERSION as PROMPT_VERSION, critique_prompt_v3, parse_critique_v3

SECTION = Path("evaluations/muse-critic-feasibility/v1")
VERSION = "experiment_010_v1"
STUDY_ID = "20261030T142150402688Z"
REVIEW_ID = "20261030T142419308682Z"
PROPOSAL_ID = "P06"


def _folder(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"\d{8}T\d{12}Z", run_id):
        raise ValueError("invalid Experiment 010 run ID")
    root = Path(root).resolve()
    base = root / SECTION
    if any(path.is_symlink() for path in (root / "evaluations", root / "evaluations/muse-critic-feasibility", base)):
        raise ValueError("linked Experiment 010 artifact path")
    return base / run_id


def _write(stage, name, record, schema):
    stage.mkdir(parents=True, exist_ok=False)
    (stage / name).write_bytes(canonical(record))
    (stage / "manifest.json").write_bytes(canonical(manifest(stage, schema=schema)))


def _read(stage, name, schema):
    read_manifest(stage, schema=schema)
    return json.loads((stage / name).read_bytes())


def _source(root):
    return source_package(root, STUDY_ID, REVIEW_ID, PROPOSAL_ID)


def prepare(root):
    source = _source(root)
    prompt = critique_prompt_v3(source)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = {"schema": "experiment_010_request_v1", "run_id": run_id,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "study_id": STUDY_ID, "review_id": REVIEW_ID, "proposal_id": PROPOSAL_ID,
              "proposal_sha256": source["proposal_sha256"], "source_sha256": digest(canonical(source)),
              "source_hashes": source["source_hashes"],
              "evidence_keys": {v["evidence_id"]: sorted(v["observations"])
                                for v in source["validated_evidence"].values()},
              "model": MODEL, "api_base_url": API_BASE_URL, "endpoint": "/responses",
              "request_options": REQUEST_OPTIONS, "temperature": None,
              "temperature_note": "omitted; API default",
              "prompt_version": PROMPT_VERSION, "response_schema": "experiment_008_critic_v3",
              "prompt": prompt, "prompt_sha256": sha(prompt.encode()),
              "prompt_utf8_bytes": len(prompt.encode())}
    _write(_folder(root, run_id) / "setup", "request.json", record, "experiment_010_request_v1")
    return run_id


def load_request(root, run_id):
    record = _read(_folder(root, run_id) / "setup", "request.json", "experiment_010_request_v1")
    source = _source(root)
    if (record.get("schema") != "experiment_010_request_v1" or record.get("run_id") != run_id
            or record.get("proposal_id") != PROPOSAL_ID or record.get("proposal_sha256") != source["proposal_sha256"]
            or record.get("source_sha256") != digest(canonical(source))
            or record.get("source_hashes") != source["source_hashes"]
            or record.get("prompt") != critique_prompt_v3(source)
            or record.get("prompt_sha256") != sha(record["prompt"].encode())
            or record.get("model") != MODEL or record.get("api_base_url") != API_BASE_URL
            or record.get("request_options") != REQUEST_OPTIONS or record.get("temperature") is not None
            or record.get("prompt_version") != PROMPT_VERSION):
        raise ValueError("Experiment 010 request identity mismatch")
    return record, source


def request_once(client, root, run_id):
    request, source = load_request(root, run_id)
    folder = _folder(root, run_id)
    marker = folder / "attempt_started.json"
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Experiment 010 attempt already started; no retry") from exc
    started = datetime.now(timezone.utc)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical({"run_id": run_id, "started_utc": started.isoformat(),
                                "request_sha256": sha((folder / "setup/request.json").read_bytes())}))
    audit = {"schema": "experiment_010_audit_v1", "run_id": run_id,
             "request_sha256": sha((folder / "setup/request.json").read_bytes()),
             "prompt_sha256": request["prompt_sha256"], "started_utc": started.isoformat(),
             "completed_utc": None, "duration_ms": None, "model": MODEL,
             "response_model": None, "response_id": None, "response_metadata": None,
             "raw_response": None, "parsed_response": None, "structural_status": None,
             "scientific_status": "pending_human_review", "first_error": None,
             "error_type": None, "error_field": None, "transport_status_code": None}
    tick = time.monotonic()
    try:
        response = client.responses.create(model=MODEL, input=request["prompt"], **REQUEST_OPTIONS)
        audit["response_id"] = _get(response, "id")
        audit["response_model"] = _get(response, "model")
        audit["raw_response"], audit["response_metadata"] = inspect_response(response)
        require_complete_text(audit["raw_response"], audit["response_metadata"])
        if audit["response_model"] != MODEL:
            raise ValueError("response model mismatch")
        parsed = parse_critique_v3(audit["raw_response"], source)
        audit["parsed_response"] = parsed["response"]
        audit["structural_status"] = parsed["structural_status"]
    except Exception as exc:
        audit["structural_status"] = "rejected"
        audit["error_type"] = type(exc).__name__
        # Avoid storing transport exception text, which can include credentials or request details.
        if audit["response_metadata"] is None:
            audit["first_error"] = "transport_or_response_error"
            code = getattr(exc, "status_code", None)
            audit["transport_status_code"] = code if type(code) is int else None
        else:
            audit["first_error"] = str(exc)[:500]
    audit["duration_ms"] = round((time.monotonic() - tick) * 1000, 3)
    audit["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write(folder / "result", "audit.json", audit, "experiment_010_audit_v1")
    return audit["structural_status"]


def validate(root, run_id):
    request, source = load_request(root, run_id)
    folder = _folder(root, run_id)
    marker = json.loads((folder / "attempt_started.json").read_bytes())
    audit = _read(folder / "result", "audit.json", "experiment_010_audit_v1")
    if (marker.get("run_id") != run_id or marker.get("request_sha256") != sha((folder / "setup/request.json").read_bytes())
            or audit.get("run_id") != run_id or audit.get("request_sha256") != marker["request_sha256"]
            or audit.get("prompt_sha256") != request["prompt_sha256"]
            or audit.get("model") != MODEL or audit.get("started_utc") != marker.get("started_utc")):
        raise ValueError("Experiment 010 audit identity mismatch")
    if audit["raw_response"] is not None:
        try:
            require_complete_text(audit["raw_response"], audit["response_metadata"])
            if audit["response_model"] != MODEL:
                raise ValueError("response model mismatch")
            parsed = parse_critique_v3(audit["raw_response"], source)
            expected_status = parsed["structural_status"]
            expected_parsed = parsed["response"]
        except Exception:
            expected_status, expected_parsed = "rejected", None
        if audit["structural_status"] != expected_status or audit["parsed_response"] != expected_parsed:
            raise ValueError("Experiment 010 validation result mismatch")
    return {"run_id": run_id, "structural_status": audit["structural_status"],
            "scientific_status": audit["scientific_status"],
            "response_metadata": audit["response_metadata"], "first_error": audit["first_error"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experiment 010 one-request Muse v3 feasibility")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--request", action="store_true")
    group.add_argument("--validate", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare(ROOT)); return 0
    if not args.run_id:
        parser.error("--run-id required")
    if args.validate:
        print(json.dumps(validate(ROOT, args.run_id), indent=2)); return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        parser.error("MODEL_API_KEY required for live request")
    client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0)
    status = request_once(client, ROOT, args.run_id)
    print(status)
    return 0 if status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
