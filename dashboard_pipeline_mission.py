"""Read-only mission facts from the verified portable pipeline evidence bundle."""

import json
from pathlib import Path

from verify_pipeline_public_evidence import verify


def pipeline_mission(root):
    bundle = Path(root) / "public_pipeline_evidence"
    verification = verify(bundle)
    data = json.loads((bundle / "evidence.json").read_text())
    confirmation = data["discovery"]["confirmation"]["phases"]["comparison"]
    target = data["heldout"]["target_k1536"]["phases"]["comparison"]
    control = data["heldout"]["control_k768"]["phases"]["comparison"]
    metric = "sm__warps_active.avg.pct_of_peak_sustained_active"
    long_stall = "smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active"
    barrier = "smsp__warps_issue_stalled_barrier.avg.per_cycle_active"
    profiles = data["profiles"]
    return {
        "hardware": "NVIDIA RTX 4060 Ti",
        "shape_mnk": data["discovery"]["confirmation"]["shape_mnk"],
        "confirmation_stage2_us": confirmation["candidate_median_ms"] * 1000,
        "confirmation_stage3_us": confirmation["baseline_median_ms"] * 1000,
        "confirmation_interval": confirmation["bootstrap_interval"],
        "confirmation_correctness": len(data["discovery"]["confirmation"]["correctness"]),
        "screening_ratios": [data["discovery"][name]["phases"]["comparison"]["median_paired_ratio"]
                             for name in ("screening_1", "screening_2")],
        "profile_captures": verification["profile_captures"],
        "compiler_stage2_shared_kib": data["compiler"]["stages_2"]["shared_memory_bytes"] / 1024,
        "compiler_stage3_shared_kib": data["compiler"]["stages_3"]["shared_memory_bytes"] / 1024,
        "confirmation_ratio": confirmation["median_paired_ratio"],
        "favorable_pairs": confirmation["stage3_favorable_pairs"],
        "total_pairs": len(confirmation["rounds"]),
        "occupancy_stage2": profiles[1]["metrics"][metric]["value"],
        "occupancy_stage3": profiles[0]["metrics"][metric]["value"],
        "long_stall_stage2": [profiles[i]["metrics"][long_stall]["value"] for i in (1, 2)],
        "long_stall_stage3": [profiles[i]["metrics"][long_stall]["value"] for i in (0, 3)],
        "barrier_stage2": [profiles[i]["metrics"][barrier]["value"] for i in (1, 2)],
        "barrier_stage3": [profiles[i]["metrics"][barrier]["value"] for i in (0, 3)],
        "compiler_ttir_identical": data["compiler"]["stages_2"]["ttir_sha256"] == data["compiler"]["stages_3"]["ttir_sha256"],
        "prediction_direction": data["prediction"]["direction"],
        "prediction_model": data["prediction"]["model"],
        "reference_ratio": data["prediction"]["reference_ratio"],
        "deadband": data["prediction"]["deadband"],
        "target_ratio": target["median_paired_ratio"],
        "target_interval": target["bootstrap_interval"],
        "control_ratio": control["median_paired_ratio"],
        "prediction_score": verification["prediction_score"],
    }
