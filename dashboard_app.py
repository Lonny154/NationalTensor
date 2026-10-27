"""Local NationalTensor research dashboard with proposal-only Muse requests.

Run with: .venv/bin/python dashboard_app.py
The server binds only to 127.0.0.1 and never imports generated kernels.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import secrets

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for

from dashboard_data import ROOT, load_compiler_detail, load_history
from dashboard_interpretation import decorate_history
from dashboard_proposals import ProposalManager
import experiment_memory
from kernel_evolution import MODEL as PROPOSAL_MODEL


ALLOWED_HOSTS = {"127.0.0.1", "127.0.0.1:8501", "localhost", "localhost:8501"}
EXPERIMENT_ID = re.compile(
    r"(?:kernel:\d{8}T\d{12}Z|"
    r"structural:\d{8}T\d{12}Z:[1-9]\d*|"
    r"trusted:trusted-(?:compute|two-slab):\d{8}T\d{12}Z)\Z")


def _fmt_ms(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value * 1000:.2f} µs"
    return "—"


def _fmt_ratio(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.5f}×"
    return "—"


def _fmt_pct(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value * 100:.2f}%"
    return "—"


def _sample_chart(timing: dict | None) -> dict | None:
    """Map every saved positive sample to a labelled log-scale SVG plot."""
    if not isinstance(timing, dict):
        return None
    samples = timing.get("samples_ms", {})
    if not isinstance(samples, dict):
        return None
    sides = {}
    for name in ("baseline", "candidate"):
        raw = samples.get(name)
        if not isinstance(raw, list) or not raw:
            return None
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               or value <= 0 for value in raw):
            return None
        sides[name] = raw
    values = sides["baseline"] + sides["candidate"]
    low, high = min(values), max(values)
    if low == high:
        low *= 0.9
        high *= 1.1
    log_low, log_high = math.log(low), math.log(high)

    def x(value):
        return 104 + 600 * (math.log(value) - log_low) / (log_high - log_low)

    points = []
    for side, center in (("baseline", 70), ("candidate", 130)):
        for index, value in enumerate(sides[side]):
            points.append({"side": side, "x": round(x(value), 2),
                           "y": center + ((index % 5) - 2) * 5,
                           "value": _fmt_ms(value), "round": index + 1})
    ticks = [{"x": round(x(value), 2), "label": _fmt_ms(value)}
             for value in (low, math.sqrt(low * high), high)]
    return {"points": points, "ticks": ticks, "count": len(values),
            "baseline_median_x": x(timing.get("baseline_ms", low))
                if isinstance(timing.get("baseline_ms"), (int, float))
                and timing["baseline_ms"] > 0 else None,
            "candidate_median_x": x(timing.get("candidate_ms", low))
                if isinstance(timing.get("candidate_ms"), (int, float))
                and timing["candidate_ms"] > 0 else None}


def _prompt_evidence(selected: dict, records: dict) -> dict:
    """Describe only evidence actually present in the saved proposal prompt."""
    context = selected.get("prompt_context")
    context = context if isinstance(context, dict) else {}
    raw = context.get("relevant_experiment_memory")
    raw = raw if isinstance(raw, list) else []
    trusted, generated = [], []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind not in {"trusted_implementation", "generated_proposal"}:
            continue
        record = records.get(item.get("id")) if isinstance(item.get("id"), str) else None
        original = item.get("original_muse_source_sha256")
        reviewed = item.get("trusted_implementation_sha256")
        verified = bool(
            record and original and
            ((kind == "trusted_implementation"
              and record.get("kind") in {"trusted-compute", "trusted-two-slab"}
              and reviewed and record.get("generated_sha256") == original
              and record.get("trusted_sha256") == reviewed
              and record.get("exact_generated_source_evaluated") is False)
             or (kind == "generated_proposal" and record.get("kind") == "kernel"
                 and record.get("source_hash_valid") is True
                 and record.get("source_sha256") == original)))
        row = {"id": record["id"] if verified else None,
               "label": (record.get("label") or record["id"]) if verified
                        else str(item.get("id", "unknown"))[:100],
               "verified": verified, "original_sha256": original,
               "trusted_sha256": reviewed,
               "performance": item.get("performance_interpretation") if kind == "trusted_implementation" else None}
        (trusted if kind == "trusted_implementation" else generated).append(row)
    prior = context.get("prior_unevaluated_review")
    if isinstance(prior, dict):
        prior_hash = prior.get("candidate_sha256")
        if isinstance(prior_hash, str) and not any(
                row["original_sha256"] == prior_hash for row in generated):
            record = next((row for row in records.values()
                           if row.get("kind") == "kernel"
                           and row.get("source_hash_valid") is True
                           and row.get("source_sha256") == prior_hash), None)
            generated.append({"id": record["id"] if record else None,
                              "label": (record.get("label") or record["id"])
                                       if record else "Prior generated proposal",
                              "verified": bool(record), "original_sha256": prior_hash,
                              "trusted_sha256": None, "performance": None})
    return {"trusted": trusted, "generated": generated,
            "unverified_direct": isinstance(
                context.get("external_trusted_implementation_feedback"), dict)
                and not selected.get("feedback_link")}


def create_app(root: Path = ROOT) -> Flask:
    app = Flask(__name__)
    app.secret_key = secrets.token_bytes(32)
    app.config.update(MAX_CONTENT_LENGTH=1024, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Strict")
    root = Path(root).resolve()
    proposals = ProposalManager(root)
    app.extensions["proposal_manager"] = proposals
    app.jinja_env.filters["ms"] = _fmt_ms
    app.jinja_env.filters["ratio"] = _fmt_ratio
    app.jinja_env.filters["pct"] = _fmt_pct

    @app.before_request
    def validate_local_request():
        if request.host not in ALLOWED_HOSTS:
            abort(400)
        if request.method == "POST":
            if request.headers.get("Origin") != f"http://{request.host}":
                abort(403)
            if request.mimetype != "application/x-www-form-urlencoded":
                abort(415)
            if ("csrf_token" not in session
                    or set(request.form) != {"csrf_token"}
                    or len(request.form.getlist("csrf_token")) != 1
                    or not secrets.compare_digest(
                        request.form["csrf_token"], session.get("csrf_token", ""))):
                abort(403)

    @app.after_request
    def protect(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; img-src 'self'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        return response

    @app.get("/api/proposals/status")
    def proposal_status():
        return _proposal_json(proposals.snapshot())

    @app.post("/api/proposals")
    def request_proposal():
        if not proposals.start():
            return _proposal_json(proposals.snapshot()), 409
        return _proposal_json(proposals.snapshot()), 202

    def _proposal_json(snapshot):
        experiment = snapshot.pop("experiment")
        snapshot["detail_url"] = (url_for("experiment_detail", experiment_id=experiment)
                                  if experiment else None)
        return jsonify(snapshot)

    @app.get("/")
    def home():
        view = request.args.get("view", "overview")
        if view not in {"overview", "timeline", "performance", "reliability"}:
            abort(404)
        selected_id = request.args.get("experiment")
        if selected_id is not None:
            if not EXPERIMENT_ID.fullmatch(selected_id):
                abort(404)
            history = decorate_history(load_history(root))
            if selected_id not in {item["id"] for item in history["timeline"]}:
                abort(404)
            return redirect(url_for("experiment_detail", experiment_id=selected_id))
        return render_dashboard(view)

    @app.get("/experiments/<experiment_id>")
    def experiment_detail(experiment_id):
        if not EXPERIMENT_ID.fullmatch(experiment_id):
            abort(404)
        return render_dashboard("detail", experiment_id)

    def render_dashboard(view, selected_id=None):
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        history = decorate_history(load_history(root))
        records = {item["id"]: item for item in history["timeline"]}
        selected = records.get(selected_id) if selected_id else None
        if selected_id and selected is None:
            abort(404)
        timing = None
        aa = None
        if selected:
            timing = (selected.get("timing") if selected["kind"] == "structural"
                      else selected.get("screening"))
            aa = selected.get("aa")
        compiler = None
        if selected and request.args.get("compiler") == "1":
            compiler = load_compiler_detail(root, selected)
        comparison = None
        comparison_trusted = None
        related_memory = []
        prompt_evidence = None
        mechanism_primary = None
        mechanism_secondary = []
        if selected:
            if selected["kind"].startswith("trusted-"):
                comparison = selected.get("comparison")
                comparison_trusted = selected
            elif selected["kind"] == "kernel":
                prompt_evidence = _prompt_evidence(selected, records)
                memory = experiment_memory.from_history(history)
                current_memory = next((entry for entry in memory["entries"]
                                       if entry["id"] == selected["id"]), None)
                if current_memory:
                    mechanism_primary = experiment_memory.FAMILY_NAMES.get(
                        current_memory["primary_family"])
                    mechanism_secondary = [experiment_memory.FAMILY_NAMES[name]
                                           for name in current_memory["secondary_families"]]
                related_memory = experiment_memory.related_for(memory, selected["id"])
                for related_id in selected.get("related_trusted", []):
                    related = records.get(related_id)
                    if related and related.get("comparison"):
                        comparison = related["comparison"]
                        comparison_trusted = related
                        break
        return render_template(
            "dashboard.html", history=history, records=records, view=view,
            selected=selected, timing=timing, aa=aa,
            chart=_sample_chart(timing), compiler=compiler,
            comparison=comparison, comparison_trusted=comparison_trusted,
            related_memory=related_memory, prompt_evidence=prompt_evidence,
            mechanism_primary=mechanism_primary,
            mechanism_secondary=mechanism_secondary,
            csrf_token=session["csrf_token"], proposal_model=PROPOSAL_MODEL)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8501, debug=False,
                     use_reloader=False)
