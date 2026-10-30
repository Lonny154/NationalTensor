"""One bounded, advisory Muse interpretation of hash-validated Experiment 003."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from dashboard_data import ROOT
from mask_hoisting_memory import load as load_experiment

MODEL = "muse-spark-1.3"
API_BASE_URL = "https://api.meta.ai/v1"
SECTION = "evaluations/muse-research-debrief"
REVISION_VERSION = "cinematic_v2"
# The budget covers private reasoning AND visible JSON, per Meta's Responses API.
REQUEST_OPTIONS = {"max_output_tokens": 8000, "reasoning": {"effort": "low"}}
FIELDS = {"technical_interpretation", "supporting_evidence", "limitations_uncertainties",
          "recommended_next_action", "nontechnical_explanation", "hollywood_debrief"}
REF_FIELDS = {"evidence_id", "observation_key"}
NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\d.])")
PERFORMANCE_CLAIM = re.compile(r"\b(?:measured|achieved|observed|delivered|proved|confirmed)\b.{0,45}\b(?:latency|speedup|slowdown|runtime|performance gain|faster|slower)\b|\b(?:latency|speedup|slowdown|runtime)\b.{0,45}\b(?:measured|achieved|observed|confirmed)\b", re.I)
NUMERIC_PERFORMANCE = re.compile(r"(?:\b\d+(?:\.\d+)?\s*(?:x|×|%|ms|µs|ns)\b.{0,35}\b(?:speedup|slowdown|latency|faster|slower)\b|\b(?:speedup|slowdown|latency|faster|slower)\b.{0,35}\b\d+(?:\.\d+)?\s*(?:x|×|%|ms|µs|ns)\b)", re.I)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def evidence_package(root=ROOT):
    """Read only the fixed validated experiment; never import experiment runners."""
    entry, warnings = load_experiment(root)
    if entry is None:
        raise ValueError("Experiment 003 evidence unavailable: " + "; ".join(warnings or ["missing record"]))
    if (entry["outcome"] != "compiler_equivalent_screening_skipped"
            or entry["correctness"] != "PASS"
            or entry["diagnostic_evidence"]["latency_measured"] is not False
            or entry["screening"] is not None):
        raise ValueError("Experiment 003 is not the validated compiler-equivalence result")
    detail = entry["diagnostic_evidence"]
    observations = {
        "correctness": {"status": "PASS", "passed": 4, "total": 4,
                        "tolerances": detail["tolerances"]},
        "compiler_resources": entry["compiler"]["resources"],
        "executable_comparison": {"complete_executable_section": detail["executable_section"],
                                  "constant_data_and_launch_metadata_identical": detail["constant_data_and_launch_metadata_identical"]},
        "source_hashes": {"baseline_kernel_sha256": entry["baseline_sha256"],
                          "baseline_launcher_sha256": entry["launcher_sha256"],
                          "trusted_implementation_sha256": entry["trusted_implementation_sha256"],
                          "original_muse_source_sha256": entry["original_muse_source_sha256"],
                          "exact_generated_source_evaluated": False},
        "screening": {"status": "skipped", "reason": detail["screening_skip_reason"],
                      "latency_measured": False, "speedup_measured": False,
                      "latency_measurements": [], "median_paired_speedup": None,
                      "performance_conclusion": detail["performance_conclusion"]},
        "scope_and_limitations": {"shape_mnk": entry["shape_mnk"], "dtype": entry["dtype"],
                                  "configuration": entry["configuration"],
                                  "limitations": detail["limitations"]},
    }
    return {"schema": "muse_research_debrief_evidence_v1", "evidence_id": entry["id"],
            "artifact_path": entry["artifact_path"], "artifact_hashes": detail["artifact_hashes"],
            "observations": observations}


def _prompt_for(package, *, cinematic=False):
    examples = {
        "technical_interpretation": ("The compared executable output matches within this recorded scope.", ["executable_comparison", "scope_and_limitations"]),
        "supporting_evidence": ("Correctness checks passed and compiler resource allocations match.", ["correctness", "compiler_resources"]),
        "limitations_uncertainties": ("This scope has no measured latency or speedup result.", ["scope_and_limitations", "screening"]),
        "recommended_next_action": ("Review a workload with partial boundary tiles as a distinct research question.", ["scope_and_limitations"]),
        "nontechnical_explanation": ("The recipes produced matching machine instructions; runtime remains unmeasured.", ["executable_comparison", "screening"]),
        "hollywood_debrief": (("Chief: The compiler gave both suspects the same disguise. Agent: Then who stole the clock cycles? Chief: Nobody knows. The stopwatch stayed in its trailer."
                               if cinematic else "The compiler gave both suspects the same disguise. The stopwatch stayed in its trailer."), ["executable_comparison", "screening"]),
    }
    schema = {name: {"text": text, "citations": [
        {"evidence_id": package["evidence_id"], "observation_key": key} for key in keys]}
        for name, (text, keys) in examples.items()}
    return ("Interpret this one verified GPU compiler diagnostic. Return one JSON object with exactly "
            "the six fields in SCHEMA; each field has text and 1-4 citations. Cite only the supplied "
            "evidence_id and observation keys. Cite every claim locally. Distinguish measured evidence "
            "from interpretation and theatrical dialogue. Do not claim measured latency, speedup, "
            "slowdown, or runtime equivalence. The Hollywood field is fictional dialogue, short and "
            "humorous, with no invented experimental result. "
            + ("Make that field an original, absurd technological conspiracy thriller scene in three to "
               "five brisk exchanges: give fictional characters distinct voices, have them treat GPUs, "
               "compilers, and stolen clock cycles "
               "with wildly excessive seriousness, and let one character cleverly misunderstand a "
               "technical term before a dramatic reveal, unexpected joke, or sharp one-liner. Use cheesy "
               "hacker-movie "
               "energy without quoting any film. Keep all theatrics in this field; the other five "
               "fields must state the science plainly. The scene must never imply a measurement, "
               "speedup, or outcome absent from the evidence. " if cinematic else "")
            + "Recommend only one next research action "
            "for human review; do not request another hypothesis or execute anything. No code. "
            "Numerals in prose must occur in the cited observation values. The example demonstrates "
            "the schema and appropriate local citations; form your own interpretation from EVIDENCE.\nSCHEMA\n"
            + json.dumps(schema, sort_keys=True) + "\nEVIDENCE\n"
            + json.dumps(package, sort_keys=True, allow_nan=False))


def prompt_for(package):
    return _prompt_for(package, cinematic=True)


def _numbers(value):
    # Hash strings are identities, not measured numeric values.
    if isinstance(value, bool) or value is None:
        return set()
    if type(value) in (int, float):
        return {str(value), str(float(value))} if type(value) is int else {str(value)}
    if isinstance(value, str):
        return set(NUMBER.findall(value)) if not re.fullmatch(r"[0-9a-f]{64}", value) else set()
    if isinstance(value, dict):
        return set().union(*(_numbers(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_numbers(v) for v in value)) if value else set()
    return set()


def parse_response(raw, package):
    if not isinstance(raw, str) or len(raw) > 24_000:
        raise ValueError("response is missing or exceeds 24000 characters")
    if not raw.strip():
        raise ValueError("response contains empty output text")
    try:
        result = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("response is not JSON") from exc
    if not isinstance(result, dict) or set(result) != FIELDS:
        raise ValueError("response must contain exactly the six debrief fields")
    for name, item in result.items():
        if not isinstance(item, dict) or set(item) != {"text", "citations"}:
            raise ValueError(f"{name} must have text and citations")
        prose, citations = item["text"], item["citations"]
        if (not isinstance(prose, str) or not prose.strip() or len(prose) > 1200
                or "\x00" in prose or "```" in prose or "@triton.jit" in prose):
            raise ValueError(f"{name} has invalid text")
        if not isinstance(citations, list) or not 1 <= len(citations) <= 4:
            raise ValueError(f"{name} requires 1-4 citations")
        available = set()
        seen = set()
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != REF_FIELDS:
                raise ValueError("invalid citation shape")
            if citation["evidence_id"] != package["evidence_id"] or citation["observation_key"] not in package["observations"]:
                raise ValueError("citation references unavailable evidence")
            key = citation["observation_key"]
            if key in seen:
                raise ValueError("duplicate citation")
            seen.add(key)
            available |= _numbers(package["observations"][key])
        if any(number not in available for number in NUMBER.findall(prose)):
            raise ValueError(f"{name} contains unsupported numerical claim")
        if NUMERIC_PERFORMANCE.search(prose):
            raise ValueError(f"{name} asserts a numerical runtime result that was not measured")
        for sentence in re.split(r"[.!?;]+", prose):
            claim = PERFORMANCE_CLAIM.search(sentence)
            if claim and not re.search(r"\b(?:no|not|never|without|unmeasured)\b", sentence[:claim.end()], re.I):
                raise ValueError(f"{name} implies measured runtime evidence")
    return result


class ResponseHandlingError(ValueError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def inspect_response(response):
    """Extract only Responses message text; retain allowlisted metadata, never reasoning or headers."""
    def label(value):
        return value[:100] if isinstance(value, str) else None

    output = _get(response, "output")
    diagnostics = {
        "status": label(_get(response, "status")),
        "incomplete_reason": label(_get(_get(response, "incomplete_details"), "reason")),
        "finish_reason": label(_get(response, "finish_reason")),
        "error_present": _get(response, "error") is not None,
        "error_code": label(_get(_get(response, "error"), "code")),
        "output_items": [], "refusal_present": False,
        "unsupported_output": False, "unfinished_items": False,
        "extraction_source": None,
    }
    usage = _get(response, "usage")
    diagnostics["usage"] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = _get(usage, key)
        if type(value) is int:
            diagnostics["usage"][key] = value
    for key, child in (("output_tokens_details", "reasoning_tokens"), ("input_tokens_details", "cached_tokens")):
        value = _get(_get(usage, key), child)
        if type(value) is int:
            diagnostics["usage"][key] = {child: value}
    limit = _get(response, "max_output_tokens")
    diagnostics["max_output_tokens"] = limit if type(limit) is int else None
    texts = []
    if output is not None:
        if not isinstance(output, list) or len(output) > 100:
            diagnostics["unsupported_output"] = True
        else:
            diagnostics["extraction_source"] = "output.message.content.output_text"
            for item in output:
                kind, status = _get(item, "type"), _get(item, "status")
                record = {"type": label(kind), "status": label(status),
                          "finish_reason": label(_get(item, "finish_reason")), "content_types": []}
                diagnostics["output_items"].append(record)
                if status is not None and status != "completed":
                    diagnostics["unfinished_items"] = True
                if kind == "reasoning":
                    continue  # Reasoning summaries/encrypted content are never a debrief.
                if kind != "message" or _get(item, "role", "assistant") != "assistant":
                    diagnostics["unsupported_output"] = True
                    continue
                content = _get(item, "content")
                if not isinstance(content, list) or len(content) > 100:
                    diagnostics["unsupported_output"] = True
                    continue
                for part in content:
                    part_type = _get(part, "type")
                    record["content_types"].append(label(part_type))
                    if part_type == "refusal":
                        diagnostics["refusal_present"] = True
                    elif part_type == "output_text" and isinstance(_get(part, "text"), str):
                        texts.append(_get(part, "text"))
                    else:
                        diagnostics["unsupported_output"] = True
    else:
        # Convenience-only adapters and decoded mappings, not Chat Completions shapes.
        text = _get(response, "output_text")
        if isinstance(text, str):
            texts.append(text)
            diagnostics["extraction_source"] = "output_text"
        else:
            diagnostics["unsupported_output"] = True
    raw = "".join(texts)
    diagnostics["text_characters"] = len(raw)
    return raw, diagnostics


def require_complete_text(raw, diagnostics):
    if diagnostics["error_present"] or diagnostics["status"] == "failed":
        raise ResponseHandlingError("api_error", "API response reported failure; see safe response metadata")
    if diagnostics["refusal_present"]:
        raise ResponseHandlingError("refusal", "API response contains a refusal")
    if (diagnostics["incomplete_reason"] or diagnostics["unfinished_items"]
            or diagnostics["status"] != "completed"
            or diagnostics["finish_reason"] not in (None, "stop")
            or any(item["finish_reason"] not in (None, "stop") for item in diagnostics["output_items"])):
        raise ResponseHandlingError("incomplete_output", "API response is incomplete or completion is unverified; see safe response metadata")
    if diagnostics["unsupported_output"]:
        raise ResponseHandlingError("unsupported_output", "API response contains an unsupported output structure")
    if not raw.strip():
        raise ResponseHandlingError("empty_output", "API response contains no visible output text; this is not a JSON validation failure")


def _safe_base(root):
    root = Path(root).resolve()
    base = root / SECTION
    if (root / "evaluations").is_symlink() or base.is_symlink() or not base.resolve().is_relative_to(root):
        raise ValueError("debrief artifact path is linked or escapes root")
    return base


def load_accepted(root=ROOT, package=None):
    package = package or evidence_package(root)
    path = _safe_base(root) / "accepted.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 100_000:
        return None
    try:
        saved = json.loads(path.read_bytes())
        if (saved["status"] != "accepted"
                or saved["evidence_sha256"] != digest(canonical(package)) or saved["model"] != MODEL
                or saved["response_model"] != MODEL
                or saved["prompt"] not in (prompt_for(package), _prompt_for(package))
                or saved["interpretation"] != parse_response(saved["raw_response"], package)):
            return None
        require_complete_text(saved["raw_response"], saved["response_metadata"])
        return saved
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _revision_base(root):
    base = _safe_base(root) / "revisions" / REVISION_VERSION
    if ((_safe_base(root) / "revisions").is_symlink() or base.is_symlink()
            or not base.resolve().is_relative_to(_safe_base(root))):
        raise ValueError("revision artifact path is linked or escapes root")
    return base


def load_cinematic_revision(root=ROOT, package=None):
    package = package or evidence_package(root)
    original = load_accepted(root, package)
    if original is None:
        return None
    path = _revision_base(root) / "accepted.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 100_000:
        return None
    try:
        saved = json.loads(path.read_bytes())
        if (saved["status"] != "accepted" or saved["prompt_version"] != REVISION_VERSION
                or saved["source_accepted_sha256"] != digest((_safe_base(root) / "accepted.json").read_bytes())
                or saved["source_response_id"] != original["response_id"]
                or saved["evidence_sha256"] != digest(canonical(package))
                or saved["model"] != MODEL or saved["response_model"] != MODEL
                or saved["prompt"] != prompt_for(package)
                or saved["interpretation"] != parse_response(saved["raw_response"], package)):
            return None
        require_complete_text(saved["raw_response"], saved["response_metadata"])
        return saved
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_display_accepted(root=ROOT, package=None):
    package = package or evidence_package(root)
    return load_cinematic_revision(root, package) or load_accepted(root, package)


def request_once(client, root=ROOT, *, cinematic_revision=False):
    """One synchronous API call, serialized across processes; rejected raw output is retained."""
    package = evidence_package(root)
    if cinematic_revision and load_accepted(root, package) is None:
        return "original_unavailable"
    base = _revision_base(root) if cinematic_revision else _safe_base(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / "request.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "request_in_progress"
        original = load_accepted(root, package) if cinematic_revision else None
        if cinematic_revision and original is None:
            return "original_unavailable"
        accepted = load_cinematic_revision(root, package) if cinematic_revision else load_accepted(root, package)
        if accepted:
            return "already_accepted"
        prompt = prompt_for(package)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = base / stamp
        folder.mkdir(exist_ok=False)
        audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
                 "prompt": prompt, "evidence_sha256": digest(canonical(package)),
                 "request_options": REQUEST_OPTIONS,
                 "raw_response": None, "response_model": None}
        if cinematic_revision:
            audit.update(prompt_version=REVISION_VERSION,
                         source_accepted_sha256=digest((_safe_base(root) / "accepted.json").read_bytes()),
                         source_response_id=original["response_id"],
                         audit_path=str(Path(SECTION) / "revisions" / REVISION_VERSION / stamp / "audit.json"))
        request_completed = False
        try:
            response = client.responses.create(model=MODEL, input=prompt, **REQUEST_OPTIONS)
            request_completed = True
            audit["raw_response"], audit["response_metadata"] = inspect_response(response)
            audit["response_model"] = _get(response, "model")
            audit["response_id"] = _get(response, "id")
            require_complete_text(audit["raw_response"], audit["response_metadata"])
            if audit["response_model"] != MODEL:
                raise ValueError("response model identity missing or mismatched")
            interpretation = parse_response(audit["raw_response"], package)
            if evidence_package(root) != package:
                raise ValueError("source evidence changed during the request")
            if cinematic_revision and (load_accepted(root, package) is None
                    or digest((_safe_base(root) / "accepted.json").read_bytes()) != audit["source_accepted_sha256"]):
                raise ValueError("original accepted debrief changed during the request")
            audit["interpretation"] = interpretation
            audit["status"] = "accepted"
            (folder / "audit.json").write_bytes(canonical(audit))
            (base / "accepted.json").write_bytes(canonical(audit))
            return "accepted"
        except Exception as exc:
            audit["status"] = "rejected"
            audit["rejection_kind"] = getattr(exc, "kind", "invalid_debrief" if request_completed else "request_error")
            # API exception messages can contain request data; retain only type/status.
            audit["rejection"] = f"{type(exc).__name__}: {str(exc)[:300]}" if request_completed and isinstance(exc, ValueError) else type(exc).__name__
            if type(getattr(exc, "status_code", None)) is int:
                audit["http_status"] = exc.status_code
            (folder / "audit.json").write_bytes(canonical(audit))
            return "rejected"


def request_cinematic_revision_once(client, root=ROOT):
    return request_once(client, root, cinematic_revision=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experiment 003 Muse Research Debrief")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true", help="print deterministic prompt; no API call")
    mode.add_argument("--request-from-muse", action="store_true", help="one live debrief request")
    mode.add_argument("--request-cinematic-revision", action="store_true",
                      help="one versioned cinematic revision without changing the original")
    args = parser.parse_args(argv)
    package = evidence_package(ROOT)
    if args.prepare_only:
        print(prompt_for(package))
        return 0
    if args.request_from_muse and load_accepted(ROOT, package):
        print("already_accepted")
        return 0
    if args.request_cinematic_revision:
        if load_accepted(ROOT, package) is None:
            print("original_unavailable")
            return 1
        if load_cinematic_revision(ROOT, package):
            print("already_accepted")
            return 0
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        parser.error("MODEL_API_KEY is required for a live Muse request")
    status = request_once(OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0),
                          ROOT, cinematic_revision=args.request_cinematic_revision)
    print(status)
    return 0 if status in {"accepted", "already_accepted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
