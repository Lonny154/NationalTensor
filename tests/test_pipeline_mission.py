"""The mission only presents a verified portable record."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from unittest.mock import patch

import pytest

from dashboard_app import create_app
from dashboard_pipeline_mission import pipeline_mission


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_replay_uses_verified_public_evidence():
    evidence = json.loads((ROOT / "public_pipeline_evidence/evidence.json").read_text())
    mission = pipeline_mission(ROOT)
    assert mission["confirmation_ratio"] == evidence["discovery"]["confirmation"]["phases"]["comparison"]["median_paired_ratio"]
    assert mission["favorable_pairs"] == 13
    assert mission["total_pairs"] == 14
    assert mission["occupancy_stage2"] == 27.51
    assert mission["occupancy_stage3"] == 15.86
    assert mission["prediction_direction"] == "lower"
    assert mission["target_ratio"] == evidence["heldout"]["target_k1536"]["phases"]["comparison"]["median_paired_ratio"]
    assert mission["prediction_score"] == "inconclusive"
    client = create_app(ROOT).test_client()
    page = client.get("/research").get_data(as_text=True)
    assert 'data-mission-open="pipeline"' in page
    assert 'data-mission-open="replay"' in page
    assert 'data-mission-open="live"' in page
    assert 'data-pipeline="{' in page
    assert client.get("/research/pipeline/source/evidence.json").json == evidence
    assert client.get("/research/pipeline/source/unknown").status_code == 404


def test_tampered_bundle_disables_replay(tmp_path):
    shutil.copytree(ROOT / "public_pipeline_evidence", tmp_path / "public_pipeline_evidence")
    source = tmp_path / "public_pipeline_evidence/evidence.json"
    source.write_text(source.read_text().replace("0.9562214032109078", "0.7562214032109078"))
    with pytest.raises(ValueError, match="file hash"):
        pipeline_mission(tmp_path)


def test_missing_bundle_disables_feature(tmp_path):
    with pytest.raises((OSError, ValueError)):
        pipeline_mission(tmp_path)


def test_featured_tabs_use_verified_measurements_and_keep_existing_records():
    evidence = json.loads((ROOT / "public_pipeline_evidence/evidence.json").read_text())
    confirmation = evidence["discovery"]["confirmation"]["phases"]["comparison"]
    client = create_app(ROOT).test_client()
    performance = client.get("/?view=performance").get_data(as_text=True)
    reliability = client.get("/?view=reliability").get_data(as_text=True)
    assert performance.index("The Pipeline Paradox") < performance.index("Compare ")
    assert f'{confirmation["median_paired_ratio"]:.6f}' in performance
    assert f'{confirmation["baseline_median_ms"] * 1000:.2f} µs' in performance
    assert f'{confirmation["candidate_median_ms"] * 1000:.2f} µs' in performance
    assert "Independent confirmation" in performance
    assert "Earlier screening ratios" in performance
    assert "four selected captures" in reliability
    assert "27.51%" in reliability and "15.86%" in reliability
    assert "INCONCLUSIVE" in reliability
    assert "A/A calibration captures" in reliability


@pytest.mark.parametrize("error", [FileNotFoundError("missing bundle"), ValueError("file hash: evidence.json")])
def test_featured_tabs_fail_closed_when_bundle_cannot_verify(error):
    client = create_app(ROOT).test_client()
    with patch("dashboard_app.pipeline_mission", side_effect=error):
        for view, records in (("performance", "Compare "), ("reliability", "A/A calibration captures")):
            page = client.get(f"/?view={view}").get_data(as_text=True)
            assert "Pipeline Paradox evidence unavailable" in page
            assert "pipeline-hero-metric" not in page
            assert records in page


@pytest.mark.skipif(not os.environ.get("NT_PIPELINE_BROWSER"), reason="opt-in Chromium check")
def test_chromium_featured_tabs_are_read_only_at_recording_size():
    from flask import request
    from werkzeug.serving import make_server
    import dashboard_app

    app = create_app(ROOT)
    methods = []
    app.before_request(lambda: methods.append(request.method))

    @app.get("/pipeline-smoke.js")
    def pipeline_smoke_script():
        return app.response_class('''
document.addEventListener('DOMContentLoaded', () => {
  const feature = document.querySelector('.pipeline-feature');
  const chart = feature && feature.querySelector('.pipeline-chart');
  const bounds = feature && feature.getBoundingClientRect();
  if (!feature || !chart || !bounds || bounds.width < 700 ||
      bounds.right > innerWidth || bounds.top < 0 ||
      getComputedStyle(feature).display === 'none') throw Error('Featured card layout');
  document.documentElement.dataset.pipelineSmoke = 'passed';
});
''', mimetype="application/javascript")

    @app.after_request
    def inject_smoke(response):
        if request.path == "/" and request.args.get("view") in {"performance", "reliability"}:
            response.set_data(response.get_data().replace(
                b"</head>", b'<script src="/pipeline-smoke.js" defer></script></head>'))
        return response
    server = make_server("127.0.0.1", 0, app, threaded=True)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.object(dashboard_app, "ALLOWED_HOSTS", dashboard_app.ALLOWED_HOSTS | {origin[7:]}), \
                tempfile.TemporaryDirectory(prefix="nt-pipeline-browser-") as profile:
            for view, marker in (("performance", "pipeline-performance-title"),
                                 ("reliability", "pipeline-reliability-title")):
                result = subprocess.run([
                    os.environ["NT_PIPELINE_BROWSER"], "--headless", "--no-sandbox", "--disable-gpu",
                    "--disable-dev-shm-usage", "--no-proxy-server", "--no-first-run",
                    "--disable-background-networking", "--user-data-dir=" + profile,
                    "--window-size=1920,1080", "--dump-dom", "--virtual-time-budget=1500",
                    origin + "/?view=" + view,
                ], capture_output=True, text=True, timeout=40)
                assert result.returncode == 0, result.stderr[-1500:]
                assert marker in result.stdout
                assert 'data-pipeline-smoke="passed"' in result.stdout
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
    assert methods and set(methods) == {"GET"}
