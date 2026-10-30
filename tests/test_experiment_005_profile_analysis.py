"""Deterministic parser checks for the bounded Nsight Compute reports."""
import csv

import pytest

import experiment_005_profile_analysis as profile


def write_csv(path, metrics, values):
    fixed = ["Kernel Name", "Grid Size", "Block Size", "Device", "CC"]
    headers = fixed + list(metrics)
    units = ["", "", "", "", ""] + ["%" if name.endswith(".pct") else "sector" for name in metrics]
    data = ["matmul_kernel", "(768, 1, 1)", "(128, 1, 1)", "0", "8.9"] + [str(values.get(name, 1)) for name in metrics]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([headers, units, data])


def fixture(tmp_path):
    for name, hit, total, derived in (("grouped_m4", 967, 1000, 120.0),
                                      ("hilbert", 963, 1001, 96.0)):
        write_csv(tmp_path / f"{name}.metrics.csv", profile.FULL_METRICS,
                  {"lts__t_sector_hit_rate.pct": derived,
                   "lts__t_sectors.sum": total,
                   "profiler__replayer_passes": 3})
        write_csv(tmp_path / f"{name}_l2check.metrics.csv", profile.RAW_METRICS,
                  {"lts__t_sectors_lookup_hit.sum": hit,
                   "lts__t_sectors.sum": total,
                   "lts__t_sectors_srcunit_tex_op_read.sum": 900,
                   "profiler__replayer_passes": 3})


def test_parser_retains_units_and_kernel_identity(tmp_path):
    fixture(tmp_path)
    row = profile.parse_ncu_csv(tmp_path / "grouped_m4.metrics.csv", profile.FULL_METRICS)
    assert row["metrics"]["lts__t_sector_hit_rate.pct"] == {
        "value": 120.0, "unit": "%", "status": "collected"}
    assert row["kernel"] == "matmul_kernel"
    path = tmp_path / "grouped_m4.metrics.csv"
    raw = path.read_text().replace("matmul_kernel", "other_kernel")
    path.write_text(raw)
    with pytest.raises(ValueError, match="kernel"):
        profile.parse_ncu_csv(path, profile.FULL_METRICS)


def test_raw_l2_comparison_flags_impossible_derived_rate(tmp_path):
    fixture(tmp_path)
    row = profile.compare_profiles(tmp_path)
    assert row["derived"]["grouped_m4"]["derived_l2_hit_percent_valid"] is False
    assert row["comparison"]["raw_l2_hit_percentage_point_difference_hilbert_minus_grouped"] < 0
    assert row["comparison"]["l2_sector_difference_hilbert_minus_grouped"] == 1
    (tmp_path / "hilbert_l2check.metrics.csv").unlink()
    with pytest.raises(ValueError, match="missing"):
        profile.compare_profiles(tmp_path)
