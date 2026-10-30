"""CPU-only locality summary and audit tests; Nsight is never invoked."""
import json
from pathlib import Path

import pytest

import experiment_005_evidence as evidence
import experiment_005_locality as locality


def fixture(tmp_path, monkeypatch):
    base = tmp_path / evidence.SECTION / locality.RUN_ID
    verify = base / "verify"
    screen = base / "screening" / locality.SCREEN_ID
    for path in (verify, screen, verify / locality.SHAPE, verify / "128x3072x768"):
        path.mkdir(parents=True, exist_ok=True)
    (verify / "manifest.json").write_text("verified")
    (screen / "manifest.json").write_text("screened")
    (verify / locality.SHAPE / "hilbert_order.json").write_text(json.dumps({"setup_ms": 0.7}))
    (verify / "128x3072x768/hilbert_order.json").write_text(json.dumps({"setup_ms": 130.0}))
    def compiler(instructions):
        return {"executable_sections": {".text.matmul_kernel": {"sha256": "abc"}},
                "sass_counts": {"instructions": instructions},
                "registers_per_thread": 121, "shared_memory_bytes": 24576, "spills": 0}
    verified = {"environment": {"gpu": "fixture"}, "source_hashes": {"kernel": "hash"},
                "wild_lab_accepted_sha256": "proposal-hash",
                "compiler": {locality.SHAPE: {"grouped_m4": compiler(464), "hilbert": compiler(440)}}}
    comparisons = [{"shape": locality.SHAPE, "opponent": opponent,
                    "summary": {"median_paired_ratio": ratio, "paired_ratio_mad": 0.005,
                                "bootstrap_median_ratio_95pct": [0.99, 1.01], "rounds": 8},
                    "one_time_map_setup_ms": 0.8}
                   for opponent, ratio in (("row_major", 0.999), ("grouped_m4", 1.002))]
    comparisons.append({"shape": "128x3072x768", "opponent": "grouped_m4",
                        "one_time_map_setup_ms": 93.22})
    monkeypatch.setattr(evidence, "validate_verify", lambda root, run_id: (verify, verified))
    monkeypatch.setattr(evidence, "validate_timing", lambda root, run_id, stage, stamp:
                        (screen, {"comparisons": comparisons}))
    return base


def test_summary_uses_bound_records_and_keeps_outcomes_separate(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    report = locality.summarize(tmp_path)
    assert report["screening"]["grouped_m4"]["median_paired_ratio_opponent_over_hilbert"] == 1.002
    assert report["compiler"]["hilbert"]["sass_counts"]["instructions"] == 440
    assert report["mapping"]["table_bytes"] == 6144
    assert report["small_workload_setup_ms"] == {"verify": 130.0, "screening": 93.22}
    assert "No cache counters" in report["interpretation"]["cache_causation"]


def test_profiler_failure_marks_metrics_unavailable(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    summary = locality.summarize(tmp_path)
    report = locality.profile_status(summary, version="ncu test",
        attempt_command=["ncu", "--set", "basic"], exit_code=1,
        error="failed to connect to the CUDA driver")
    assert report["profiling"]["status"] == "unavailable"
    assert report["profiling"]["hilbert_attempted"] is False
    assert all(row["value"] is None and row["unit"] is None
               for row in report["profiling"]["metrics"].values())
    with pytest.raises(ValueError): locality.profile_status(summary, version="ncu test",
        attempt_command=[], exit_code=0, error="")


def test_separate_manifest_rejects_tamper_and_overwrite(tmp_path):
    folder = tmp_path / evidence.SECTION / locality.RUN_ID / "locality" / "20261030T020000000000Z"
    folder.mkdir(parents=True)
    (folder / "grouped_m4.stderr.txt").write_text("driver unavailable")
    locality.save_diagnostic(folder, {"status": "unavailable"}, source_root=tmp_path)
    evidence.read_manifest(folder, schema="experiment_005_locality_v1")
    with pytest.raises(ValueError): locality.save_diagnostic(folder, {}, source_root=tmp_path)
    (folder / "report.json").write_text("tampered")
    with pytest.raises(ValueError): evidence.read_manifest(folder, schema="experiment_005_locality_v1")
