"""Read historical M4 confirmations and the separate human review; no GPU imports."""

import hashlib
import json
import math
from pathlib import Path
import statistics


REVIEW = "docs/research_reviews/grouped_cta_003.json"
CONFIG = {"block_m": 64, "block_n": 128, "block_k": 32,
          "num_warps": 4, "num_stages": 3}
DIRECT_SHA = "4fc8041c761226f10ccffbb0e5588b73e769372bb72f8354c0a845a66d0df877"
GROUPED_SHA = "2a7913100da0c4fb8e92184a3dee5eeb8bed7992afab45377a1e430136734040"
LAUNCH_SHA = "349aed3fb91b4f0e59f28417d241ddd9b2fa9359f33b95905c443b039b073999"


def _read(root, relative, digest=None):
    path = root / relative
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or not path.is_file() or path.stat().st_size > 4_000_000):
        raise ValueError("missing, linked, or oversized artifact")
    raw = path.read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"artifact hash mismatch: {relative}")
    return raw


def _decision(rows):
    if len(rows) != 14:
        raise ValueError("expected 14 confirmation rounds")
    ratios = []
    for first, second in zip(rows[::2], rows[1::2]):
        if (first["order"] != list(reversed(second["order"]))
                or sorted(first["order"]) != ["baseline", "candidate"]):
            raise ValueError("confirmation pair is not balanced")
        for row in (first, second):
            for side in ("baseline_ms", "candidate_ms"):
                if type(row[side]) not in (int, float) or not math.isfinite(row[side]) or row[side] <= 0:
                    raise ValueError("invalid latency")
        ratios.append(statistics.median([first["baseline_ms"], second["baseline_ms"]]) /
                      statistics.median([first["candidate_ms"], second["candidate_ms"]]))
    return ratios


def load(root):
    root = Path(root).resolve()
    if not (root / REVIEW).exists():
        return None, []
    try:
        review_raw = _read(root, REVIEW)
        review = json.loads(review_raw)
        _read(root, review["plan_path"], review["plan_sha256"])
        if (review["status"] != "reviewed_not_approved_for_execution"
                or review["future_controlled_replication_allowed"] is not True
                or len(review["confirmation_artifacts"]) != 3):
            raise ValueError("invalid review status or evidence count")
        for path, digest in review["source_files"].items():
            _read(root, path, digest)
        attempts = []
        for spec in review["confirmation_artifacts"]:
            saved = json.loads(_read(root, spec["path"], spec["sha256"]))
            source = saved["source_sha256"]
            if "shapes" in saved:
                if saved["status"] != "complete" or saved["group_size_m"] != 4:
                    raise ValueError("incomplete or different schedule")
                case = next(row for row in saved["shapes"] if row["shape_mnk"] == [512, 3072, 768])
                if case["correctness"] != {"direct_2d": "PASS", "grouped_m4": "PASS"}:
                    raise ValueError("correctness not PASS")
                configs = saved["configurations"]
                if (configs["direct_2d"] != {**CONFIG, "schedule": "direct_2d"}
                        or configs["grouped_m4"] != {**CONFIG, "schedule": "grouped_m4"}
                        or source["direct_kernel"] != DIRECT_SHA or source["grouped_kernel"] != GROUPED_SHA
                        or source["direct_launcher"] != LAUNCH_SHA):
                    raise ValueError("source or configuration mismatch")
                timing = case["confirmation"]["paired"]
                rows = timing["rounds"]
                decision = case["confirmation"]["decision"]
                status = case["confirmation"]["status"]
                gpu = saved["environment"]["gpu"]
                reference = saved.get("reference_precision_policy", "not recorded in this legacy artifact")
            else:
                case = saved["screening_winner"]
                if (saved["run_status"] != "complete" or saved["shape"] != [512, 3072, 768]
                        or case["status"] != "PASS"
                        or case["config"] != {**CONFIG, "schedule": "grouped_m4"}
                        or saved["baseline"] != {**CONFIG, "schedule": "direct_2d"}
                        or source["direct_kernel_and_launcher"] != "f99c4aa0c87b37a6286cd2f2b755e395d9d4956399493cc61b53c6b17e521d53"
                        or source["grouped_kernel_and_launcher"] != "bdee1a1bb879d867dee51d0a8376064cbcf9d3bf12715a70c201620468a20a2b"):
                    raise ValueError("structural source or configuration mismatch")
                timing = saved["confirmation"]
                rows = timing["paired_rounds"]
                decision = saved["confirmation_decision"]
                status = saved["confirmation_status"]
                gpu = saved["gpu"]
                reference = saved["reference_precision_policy"]
            method = timing["method"]
            if (method["clock"] != "cuda_event" or method["iterations_per_side_per_round"] != 200
                    or method["warmups_per_side_per_round"] != 10):
                raise ValueError("confirmation method mismatch")
            ratios = _decision(rows)
            if (status != "unconfirmed" or decision["confirmed"] is not False
                    or decision["required_pairs"] != 7 or decision["minimum_median_speedup"] != 1.02
                    or decision["pairs_favoring_candidate"] != sum(ratio > 1 for ratio in ratios)
                    or not math.isclose(statistics.median(ratios), decision["median_verification_pair_speedup"], rel_tol=1e-10)
                    or all(ratio > 1 for ratio in ratios) and statistics.median(ratios) >= 1.02):
                raise ValueError("confirmation decision contradicts raw rounds")
            attempts.append({"artifact_path": spec["path"], "artifact_sha256": spec["sha256"],
                             "source_sha256": source, "status": status, "correctness": "PASS",
                             "reference_precision_policy": reference, "gpu": gpu,
                             "median_verification_pair_speedup": statistics.median(ratios),
                             "pairs_favoring_candidate": sum(ratio > 1 for ratio in ratios),
                             "verification_pair_speedups": ratios, "method": method,
                             "aa_control": "not recorded in this confirmation artifact"})
        return {
            "id": "grouped_m4:confirmation_history", "kind": "confirmation_history",
            "created_utc": review["reviewed_date"], "shape_mnk": [512, 3072, 768],
            "dtype": "fp16", "gpu": attempts[-1]["gpu"], "configuration": CONFIG,
            "hypothesis": "Does the existing GROUP_M=4 CTA mapping reproducibly improve FC1 latency?",
            "mechanism": review["schedule_distinction"]["grouped_m4"],
            "families": ["grouped_scheduling"], "primary_family": "grouped_scheduling",
            "secondary_families": [], "original_muse_source_sha256": None,
            "trusted_implementation_sha256": GROUPED_SHA,
            "baseline_sha256": DIRECT_SHA, "launcher_sha256": LAUNCH_SHA,
            "compilation": "compiled for saved attempts", "correctness": "historical PASS; reference policy differs by artifact",
            "screening": None, "compiler": None,
            "confirmation": "three unconfirmed attempts", "outcome": "inconclusive_grouped_m4",
            "performance_interpretation": "No confirmed improvement; timings remain scoped to each run",
            "confirmation_evidence": {"attempts": attempts, "pooled_latency": None,
                "limitations": ["No contemporaneous A/A controls recorded in these artifacts.",
                                "Earliest attempt predates the recorded scoped FP16 reference policy.",
                                "Confirmation rule is a heuristic, not statistical significance.",
                                "Idle gaps and host submission affect CUDA-event blocks; no causality or family-wide conclusion."]},
            "review_status": {key: review[key] for key in
                ("status", "reason", "plan_path", "plan_sha256", "schedule_distinction", "scope",
                 "future_controlled_replication_allowed", "replication_guidance")},
            "provenance": "Hash-validated original confirmation artifacts and separate human review",
            "artifact_path": REVIEW, "review_sha256": hashlib.sha256(review_raw).hexdigest(),
        }, []
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        return None, [f"Grouped-M4 confirmation evidence rejected: {exc}"]
