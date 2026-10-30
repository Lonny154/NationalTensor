"""Read-only validation and summary of Experiment 005 stage artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

from dashboard_data import ROOT
from muse_wild_lab import MODEL, REQUEST_OPTIONS, PROMPT_VERSION, parse_response, prompt_for
from muse_debrief import canonical, digest, require_complete_text
from muse_research_plan import evidence_package

SECTION = Path("evaluations/experiment-005-hilbert-tile-traversal")
STAMP = re.compile(r"\d{8}T\d{12}Z\Z")
SHAPES = ((128, 3072, 768), (512, 3072, 768),
          (2048, 3072, 768), (2048, 4096, 1024))
CONFIG = {"block_m": 64, "block_n": 128, "block_k": 32,
          "num_warps": 4, "num_stages": 3}
SCHEDULES = ("row_major", "grouped_m4", "hilbert")
SOURCE_FILES = ("hilbert_matmul.py", "experiment_005_hilbert.py", "experiment_005_evidence.py",
                "benchmark_core.py", "reference_precision.py", "grouped_matmul.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest(folder, *, schema):
    files = {str(path.relative_to(folder)): sha(path.read_bytes())
             for path in folder.rglob("*") if path.is_file() and path.name != "manifest.json"}
    return {"schema": schema, "files": sorted(files), "artifact_hashes": files}


def read_manifest(folder, *, schema):
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("missing or linked stage directory")
    path = folder / "manifest.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 250_000:
        raise ValueError("missing stage manifest")
    saved = json.loads(path.read_bytes())
    if saved.get("schema") != schema or not isinstance(saved.get("artifact_hashes"), dict):
        raise ValueError("stage manifest schema mismatch")
    if any(part.is_symlink() for part in folder.rglob("*")):
        raise ValueError("linked stage artifact")
    actual = manifest(folder, schema=schema)
    if actual != saved:
        raise ValueError("stage artifact hashes or file set mismatch")
    return saved


def validate_comparison(raw):
    """Recompute every paired ratio and reported median from raw samples."""
    if not isinstance(raw, dict) or raw.get("mode") != "comparison":
        raise ValueError("invalid comparison mode")
    rows = raw.get("rounds")
    if not isinstance(rows, list) or len(rows) < 8 or len(rows) % 2:
        raise ValueError("insufficient paired rounds")
    orders = []
    sides = {"baseline": [], "candidate": []}
    ratios = []
    for index, row in enumerate(rows, 1):
        if row.get("round") != index or row.get("order") not in (["baseline", "candidate"], ["candidate", "baseline"]):
            raise ValueError("invalid paired order")
        orders.append(row["order"])
        for side in sides:
            value = row.get(f"{side}_ms")
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid latency sample")
            sides[side].append(value)
        ratio = row["baseline_ms"] / row["candidate_ms"]
        if not math.isclose(row.get("baseline_over_candidate", -1), ratio, rel_tol=1e-9):
            raise ValueError("paired ratio mismatch")
        ratios.append(ratio)
    if orders.count(["baseline", "candidate"]) != orders.count(["candidate", "baseline"]):
        raise ValueError("unbalanced paired order")
    if raw.get("samples_ms") != sides:
        raise ValueError("raw samples mismatch")
    if (not math.isclose(raw["paired_ratios"]["median"], statistics.median(ratios), rel_tol=1e-9)
            or not math.isclose(raw["ratio_of_medians"],
                                statistics.median(sides["baseline"]) / statistics.median(sides["candidate"]),
                                rel_tol=1e-9)):
        raise ValueError("reported comparison median mismatch")
    rng = random.Random(5005)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return {"median_paired_ratio": statistics.median(ratios),
            "paired_ratio_mad": statistics.median(abs(x - statistics.median(ratios)) for x in ratios),
            "bootstrap_median_ratio_95pct": [bootstrap[49], bootstrap[1949]],
            "baseline_median_ms": statistics.median(sides["baseline"]),
            "candidate_median_ms": statistics.median(sides["candidate"]),
            "rounds": len(rows)}


def validate_verify(root, run_id):
    root = Path(root).resolve()
    if not STAMP.fullmatch(run_id):
        raise ValueError("invalid Experiment 005 run ID")
    folder = root / SECTION / run_id / "verify"
    read_manifest(folder, schema="experiment_005_verify_v1")
    result = json.loads((folder / "result.json").read_bytes())
    if (result.get("status") != "verified" or result.get("shapes") != [list(x) for x in SHAPES]
            or result.get("config") != CONFIG or result.get("schedules") != list(SCHEDULES)
            or any(result.get("source_hashes", {}).get(name) != sha((root / name).read_bytes())
                   or sha((folder / name).read_bytes()) != result.get("source_hashes", {}).get(name)
                   for name in SOURCE_FILES)):
        raise ValueError("Experiment 005 verify identity mismatch")
    wild_raw = (folder / "wild_lab_accepted.json").read_bytes()
    if result.get("wild_lab_accepted_sha256") != sha(wild_raw):
        raise ValueError("Wild Research Lab provenance mismatch")
    accepted = json.loads(wild_raw)
    context = accepted["context"]
    current_evidence = evidence_package(root)
    if (accepted["status"] != "accepted" or accepted["model"] != MODEL or accepted["response_model"] != MODEL
            or accepted["prompt_version"] != PROMPT_VERSION or accepted["request_options"] != REQUEST_OPTIONS
            or accepted["context_sha256"] != digest(canonical(context))
            or context["validated_evidence_sha256"] != digest(canonical(current_evidence))
            or accepted["prompt"] != prompt_for(context)
            or accepted["ideas"] != parse_response(accepted["raw_response"], context)):
        raise ValueError("Wild Research Lab snapshot invalid")
    require_complete_text(accepted["raw_response"], accepted["response_metadata"])
    if accepted["ideas"]["ideas"][1]["name"] != "Hilbert Tile Traversal":
        raise ValueError("accepted proposal is not Hilbert Tile Traversal")
    correctness = result.get("correctness", [])
    if (not all(row.get("status") == "PASS" for row in correctness)
            or {(row.get("shape"), row.get("schedule"), row.get("seed")) for row in correctness}
            != {(f"{m}x{n}x{k}", schedule, seed) for m, n, k in SHAPES
                for schedule in SCHEDULES for seed in (42, 43)}
            or len(correctness) != len(SHAPES) * len(SCHEDULES) * 2):
        raise ValueError("correctness is incomplete")
    mappings = result.get("tile_mappings", [])
    for row in mappings:
        if not row.get("valid"):
            raise ValueError("invalid tile mapping")
    if (len(mappings) != len(SHAPES) * len(SCHEDULES)
            or {(row.get("shape"), row.get("schedule")) for row in mappings}
            != {(f"{m}x{n}x{k}", schedule) for m, n, k in SHAPES for schedule in SCHEDULES}):
        raise ValueError("tile mapping is incomplete")
    from hilbert_matmul import tile_order
    from experiment_004_evidence import sass_counts, sass_instructions
    from mask_hoisting_memory import _sections
    for shape in SHAPES:
        key = "x".join(map(str, shape))
        tiles_m = (shape[0] + CONFIG["block_m"] - 1) // CONFIG["block_m"]
        tiles_n = (shape[1] + CONFIG["block_n"] - 1) // CONFIG["block_n"]
        saved_order = json.loads((folder / key / "hilbert_order.json").read_bytes())
        if (saved_order["tiles_m"] != tiles_m or saved_order["tiles_n"] != tiles_n
                or saved_order["order"] != [list(x) for x in tile_order("hilbert", tiles_m, tiles_n)]):
            raise ValueError("Hilbert tile map contradicts deterministic mapping")
        compiler = result["compiler"][key]
        for schedule in SCHEDULES:
            report = compiler[schedule]
            if report["sass_status"] != "captured" or any(
                    type(report[field]) is not int or report[field] < 0
                    for field in ("registers_per_thread", "shared_memory_bytes", "spills")):
                raise ValueError("compiler resource report invalid")
            for kind in ("ttir", "ttgir", "llir", "ptx", "cubin", "sass"):
                path = folder / key / f"{schedule}.{kind}"
                if report["artifacts"][kind]["sha256"] != sha(path.read_bytes()):
                    raise ValueError("compiler artifact identity mismatch")
            if report["sass_counts"] != sass_counts(sass_instructions(
                    (folder / key / f"{schedule}.sass").read_text())):
                raise ValueError("SASS counts contradict decoded artifact")
            sections = _sections((folder / key / f"{schedule}.cubin").read_bytes())
            executable = {name: {"sha256": sha(raw), "size_bytes": len(raw)}
                          for name, raw in sections.items() if name.startswith(".text.")}
            if not executable or executable != report["executable_sections"]:
                raise ValueError("executable section identity mismatch")
    return folder, result


def validate_timing(root, run_id, stage, stamp):
    if stage not in {"screening", "confirmation"} or not STAMP.fullmatch(stamp):
        raise ValueError("invalid timing stage")
    verify_folder, _ = validate_verify(root, run_id)
    folder = Path(root).resolve() / SECTION / run_id / stage / stamp
    read_manifest(folder, schema=f"experiment_005_{stage}_v1")
    result = json.loads((folder / "result.json").read_bytes())
    if (result.get("status") != "complete" or result.get("stage") != stage
            or result.get("verify_manifest_sha256") != sha((verify_folder / "manifest.json").read_bytes())
            or result.get("source_hashes") != json.loads((verify_folder / "result.json").read_bytes())["source_hashes"]
            or result.get("environment") != json.loads((verify_folder / "result.json").read_bytes())["environment"]):
        raise ValueError("timing source or verify binding mismatch")
    expected = {f"{shape[0]}x{shape[1]}x{shape[2]}": {"row_major", "grouped_m4"} for shape in SHAPES}
    actual = {}
    for row in result.get("comparisons", []):
        shape, opponent = row.get("shape"), row.get("opponent")
        if shape not in expected or opponent not in expected[shape] or opponent in actual.get(shape, set()):
            raise ValueError("unexpected timing comparison")
        actual.setdefault(shape, set()).add(opponent)
        if row.get("summary") != validate_comparison(row["timing"]):
            raise ValueError("timing summary contradicts raw samples")
    if actual != expected:
        raise ValueError("timing comparison set incomplete")
    if stage == "confirmation":
        rows = result.get("confirmation_correctness", [])
        if (result.get("timing_input_seed") != 44 or len(rows) != len(SHAPES) * len(SCHEDULES)
                or {(row.get("shape"), row.get("schedule"), row.get("status")) for row in rows}
                != {(f"{m}x{n}x{k}", schedule, "PASS")
                    for m, n, k in SHAPES for schedule in SCHEDULES}):
            raise ValueError("independent confirmation correctness incomplete")
    elif result.get("timing_input_seed") != 42 or result.get("confirmation_correctness") != []:
        raise ValueError("screening seed or correctness record mismatch")
    return folder, result


def latest(root=ROOT):
    base = Path(root).resolve() / SECTION
    if not base.is_dir() or base.is_symlink():
        return None
    for candidate in sorted(base.iterdir(), reverse=True):
        if not candidate.is_dir() or candidate.is_symlink() or not STAMP.fullmatch(candidate.name):
            continue
        try:
            _, verified = validate_verify(root, candidate.name)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        result = {"run_id": candidate.name, "verify": verified,
                  "screening": None, "confirmation": None,
                  "research_question": "Can Hilbert tile traversal improve cache locality or execution performance versus row-major and grouped scheduling?"}
        for stage in ("screening", "confirmation"):
            path = candidate / stage
            if path.is_dir() and not path.is_symlink():
                for attempt in sorted(path.iterdir(), reverse=True):
                    try:
                        _, result[stage] = validate_timing(root, candidate.name, stage, attempt.name)
                        break
                    except (OSError, ValueError, KeyError, TypeError, IndexError):
                        continue
        result["best_supported_result"] = None
        result["hypothesis_support"] = "Not established"
        if result["confirmation"] and result["screening"]:
            screening = {(r["shape"], r["opponent"]): r for r in result["screening"]["comparisons"]}
            confirmation = {(r["shape"], r["opponent"]): r for r in result["confirmation"]["comparisons"]}
            for shape in ("x".join(map(str, item)) for item in SHAPES):
                if all(screening[(shape, opponent)]["summary"]["bootstrap_median_ratio_95pct"][0] > 1
                       and confirmation[(shape, opponent)]["summary"]["bootstrap_median_ratio_95pct"][0] > 1
                       for opponent in ("row_major", "grouped_m4")):
                    result["best_supported_result"] = (
                        f"Hilbert showed a repeatable paired latency advantage over both schedules for {shape} "
                        "in screening and independent confirmation; cache cause remains untested.")
                    result["hypothesis_support"] = "Preliminary latency support; human review required"
                    break
        return result
    return None
