"""The mission only presents a verified portable record."""

import json
from pathlib import Path
import shutil

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
