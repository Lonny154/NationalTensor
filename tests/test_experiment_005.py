"""CPU-only tests for Experiment 005 mapping, calculations, and audit integrity."""
import copy
import json
from pathlib import Path
import statistics

import pytest
import torch

import experiment_005_evidence as evidence
import hilbert_matmul


@pytest.mark.parametrize("tiles_m,tiles_n", [(1, 1), (1, 7), (7, 1), (2, 3),
                                             (3, 2), (5, 9), (9, 5), (8, 24), (32, 32)])
def test_every_schedule_covers_rectangular_tiles_once(tiles_m, tiles_n):
    expected = {(m, n) for m in range(tiles_m) for n in range(tiles_n)}
    for schedule in evidence.SCHEDULES:
        order = hilbert_matmul.tile_order(schedule, tiles_m, tiles_n)
        assert len(order) == len(expected)
        assert set(order) == expected
        assert len(set(order)) == len(order)


def test_mapping_rejects_invalid_and_is_nontrivial():
    with pytest.raises(ValueError): hilbert_matmul.tile_order("hilbert", 0, 4)
    with pytest.raises(ValueError): hilbert_matmul.tile_order("hilbert", 1, 65536)
    with pytest.raises(ValueError): hilbert_matmul.tile_order("unknown", 2, 3)
    with pytest.raises(ValueError): hilbert_matmul.hilbert_xy(3, 0)
    assert hilbert_matmul.tile_order("hilbert", 8, 24) != hilbert_matmul.tile_order("row_major", 8, 24)


def test_cpu_tiled_gemm_fixture_same_math_and_boundary_coverage():
    m, n, k = 7, 11, 5
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k) / 7
    b = torch.arange(k * n, dtype=torch.float32).reshape(k, n) / 11
    reference = a @ b
    bm, bn = 3, 4
    tiles_m, tiles_n = 3, 3
    for schedule in evidence.SCHEDULES:
        out = torch.full((m, n), float("nan"))
        for tile_m, tile_n in hilbert_matmul.tile_order(schedule, tiles_m, tiles_n):
            rows = slice(tile_m * bm, min((tile_m + 1) * bm, m))
            cols = slice(tile_n * bn, min((tile_n + 1) * bn, n))
            out[rows, cols] = a[rows, :] @ b[:, cols]
        torch.testing.assert_close(out, reference)


def sample_comparison():
    rows = []
    baseline, candidate, ratios = [], [], []
    for number in range(1, 9):
        b = 2.0 + number / 100
        c = 1.0 + number / 100
        baseline.append(b); candidate.append(c); ratios.append(b / c)
        rows.append({"round": number,
                     "order": ["baseline", "candidate"] if number % 2 else ["candidate", "baseline"],
                     "baseline_ms": b, "candidate_ms": c, "baseline_over_candidate": b / c})
    return {"mode": "comparison", "rounds": rows,
            "samples_ms": {"baseline": baseline, "candidate": candidate},
            "paired_ratios": {"median": statistics.median(ratios)},
            "ratio_of_medians": statistics.median(baseline) / statistics.median(candidate)}


def test_paired_comparison_recomputed_and_tamper_rejected():
    raw = sample_comparison()
    summary = evidence.validate_comparison(raw)
    assert summary["rounds"] == 8
    assert summary["bootstrap_median_ratio_95pct"][0] > 1
    for mutation in ("ratio", "order", "sample", "median"):
        bad = copy.deepcopy(raw)
        if mutation == "ratio": bad["rounds"][0]["baseline_over_candidate"] = 9
        if mutation == "order": bad["rounds"][0]["order"] = ["candidate", "baseline"]
        if mutation == "sample": bad["samples_ms"]["baseline"][0] = 9
        if mutation == "median": bad["paired_ratios"]["median"] = 9
        with pytest.raises(ValueError): evidence.validate_comparison(bad)


def test_stage_manifest_detects_missing_corrupt_and_extra_files(tmp_path):
    folder = tmp_path / "verify"
    folder.mkdir()
    (folder / "result.json").write_text('{"status":"synthetic_fixture"}')
    schema = "experiment_005_verify_v1"
    (folder / "manifest.json").write_text(json.dumps(evidence.manifest(folder, schema=schema)))
    assert evidence.read_manifest(folder, schema=schema)["schema"] == schema
    (folder / "result.json").write_text("tampered")
    with pytest.raises(ValueError): evidence.read_manifest(folder, schema=schema)
    (folder / "result.json").write_text('{"status":"synthetic_fixture"}')
    (folder / "extra.txt").write_text("undeclared")
    with pytest.raises(ValueError): evidence.read_manifest(folder, schema=schema)


def test_dashboard_pending_result(monkeypatch):
    from dashboard_app import create_app
    monkeypatch.setattr(evidence, "latest", lambda root: None)
    client = create_app(Path.cwd()).test_client()
    response = client.get("/research")
    assert response.status_code == 200
    assert b"Experiment 005" in response.data
    assert b"Hilbert Tile Traversal" in response.data
    assert b"Awaiting explicit verification" in response.data


def test_dashboard_validated_result_summary(monkeypatch):
    from dashboard_app import create_app
    summary = {"run_id": "test-run", "verify": {"correctness": [{"status": "PASS"}]},
               "screening": None, "confirmation": None,
               "best_supported_result": None, "hypothesis_support": "Not established"}
    monkeypatch.setattr(evidence, "latest", lambda root: summary)
    response = create_app(Path.cwd()).test_client().get("/research")
    assert response.status_code == 200
    assert b"All 1 recorded" in response.data
    assert b"Not established" in response.data
