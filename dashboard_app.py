"""Local, read-only NationalTensor research dashboard.

Run with: .venv/bin/python dashboard_app.py
The server binds only to 127.0.0.1 and never imports generated kernels.
"""

from __future__ import annotations

import math
from pathlib import Path

from flask import Flask, abort, render_template, request

from dashboard_data import ROOT, load_compiler_detail, load_history
from dashboard_interpretation import decorate_history


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


def create_app(root: Path = ROOT) -> Flask:
    app = Flask(__name__)
    root = Path(root).resolve()
    app.jinja_env.filters["ms"] = _fmt_ms
    app.jinja_env.filters["ratio"] = _fmt_ratio
    app.jinja_env.filters["pct"] = _fmt_pct

    @app.after_request
    def protect(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'self'; img-src 'self'; "
            "form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        return response

    @app.get("/")
    def home():
        history = decorate_history(load_history(root))
        view = request.args.get("view", "overview")
        if view not in {"overview", "timeline", "performance", "reliability"}:
            abort(404)
        selected_id = request.args.get("experiment")
        records = {item["id"]: item for item in
                   history["proposals"] + history["trusted"]}
        selected = records.get(selected_id) if selected_id else None
        if selected_id and selected is None:
            abort(404)
        if view == "performance" and selected is None:
            selected = next((item for item in history["trusted"]
                             if item.get("screening")), None)
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
        if selected:
            if selected["kind"].startswith("trusted-"):
                comparison = selected.get("comparison")
                comparison_trusted = selected
            elif selected["kind"] == "kernel":
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
            comparison=comparison, comparison_trusted=comparison_trusted)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8501, debug=False,
                     use_reloader=False)
