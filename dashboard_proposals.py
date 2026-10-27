"""One fixed, proposal-only Muse action for the local research console.

The browser supplies no model arguments or paths. Candidate source is only
parsed and saved by kernel_evolution; this module never imports it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading

from dotenv import load_dotenv
from openai import OpenAI

from dashboard_data import RUN_ID, load_history
import kernel_evolution


API_TIMEOUT_SECONDS = 30.0
API_BASE_URL = "https://api.meta.ai/v1"


def _context(root: Path):
    """Use the latest hash-verified FC1 proposal as fixed prompt history."""
    history = load_history(root)
    previous = [item for item in history["proposals"]
                if item["kind"] == "kernel" and item["shape"] == list(kernel_evolution.SHAPE)]
    if not previous:
        raise ValueError("No prior FC1 proposal is available for this request.")
    latest = max(previous, key=lambda item: item["run_id"])
    run_id = latest["run_id"]
    if not RUN_ID.fullmatch(run_id) or not latest["source_hash_valid"]:
        raise ValueError("The latest FC1 proposal did not pass source verification.")
    base = (root / "evaluations" / "kernel-evolution").resolve()
    review_dir = (base / run_id).resolve()
    if not review_dir.is_relative_to(base):
        raise ValueError("The prior proposal path is outside the artifact directory.")
    review = kernel_evolution.reviewed_unevaluated_proposal(review_dir)
    metadata = json.loads((review_dir / "metadata.json").read_text(encoding="utf-8"))
    gpu, hardware = metadata.get("gpu"), metadata.get("hardware")
    if not isinstance(gpu, str) or not gpu.strip() or not isinstance(hardware, dict):
        raise ValueError("The prior FC1 proposal lacks hardware metadata.")
    structural = root / "evaluations" / "structural-loop"
    prompt = kernel_evolution.build_prompt(
        gpu, hardware, kernel_evolution.structural_history(structural), review=review)
    return review_dir, gpu, hardware, prompt


class ProposalManager:
    """In-memory lifecycle for a single dashboard-triggered request."""

    def __init__(self, root: Path, *, api_timeout: float = API_TIMEOUT_SECONDS):
        self.root = Path(root).resolve()
        self.api_timeout = api_timeout
        self._lock = threading.Lock()
        self._active = False
        self._state = "ready"
        self._message = "Ready to request one FC1 kernel proposal."
        self._experiment = None

    def snapshot(self):
        with self._lock:
            return {"state": self._state, "message": self._message,
                    "active": self._active, "experiment": self._experiment}

    def _update(self, state, message, *, active=None, experiment=None):
        with self._lock:
            self._state = state
            self._message = message
            if active is not None:
                self._active = active
            self._experiment = experiment

    def start(self):
        with self._lock:
            if self._active:
                return False
            self._active = True
            self._state = "requesting_proposal"
            self._message = "Requesting one proposal from Muse."
            self._experiment = None
        thread = threading.Thread(target=self._run, name="muse-proposal", daemon=True)
        thread.start()
        return True

    def _run(self):
        try:
            review_dir, gpu, hardware, prompt = _context(self.root)
            load_dotenv(dotenv_path=self.root / ".env")
            key = os.environ.get("MODEL_API_KEY")
            if not key:
                raise ValueError("MODEL_API_KEY is missing from the server environment.")
            client = OpenAI(base_url=API_BASE_URL, api_key=key,
                            timeout=self.api_timeout, max_retries=0)
            output = self.root / "evaluations" / "kernel-evolution"
            folder, metadata = kernel_evolution.propose_to_directory(
                client, prompt, output, review_from=review_dir, gpu=gpu,
                hardware=hardware, max_attempts=1,
                on_response=lambda: self._update(
                    "validating_response", "Validating the saved model response."))
            if metadata["status"] == "awaiting_human_review":
                self._update("saved_for_review", "Proposal saved for human review.",
                             active=False, experiment=f"kernel:{folder.name}")
            elif metadata.get("error") in {"APITimeoutError", "TimeoutError"}:
                self._update("failed", "Muse request timed out. No proposal was accepted.",
                             active=False)
            elif metadata.get("error", "").startswith("no source passed"):
                self._update("failed", "Muse response did not pass static review; its audit was saved.",
                             active=False)
            else:
                self._update("failed", "Muse request failed. No proposal was accepted.",
                             active=False)
        except ValueError as exc:
            # Only local, fixed messages are allowed into the browser.
            message = (str(exc) if str(exc) in {
                "No prior FC1 proposal is available for this request.",
                "The latest FC1 proposal did not pass source verification.",
                "The prior proposal path is outside the artifact directory.",
                "The prior FC1 proposal lacks hardware metadata.",
                "MODEL_API_KEY is missing from the server environment.",
            } else "The saved experiment context could not be validated.")
            self._update("failed", message, active=False)
        except Exception:
            # API and filesystem exception text may contain secrets or paths.
            self._update("failed", "Proposal request failed before a candidate was accepted.",
                         active=False)
