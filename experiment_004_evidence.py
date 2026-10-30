"""CPU-only comparison and validation for Experiment 004 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from dashboard_data import ROOT
from grouped_confirmation_memory import _read
from mask_hoisting_memory import _sections, load as load_experiment_003

SECTION = Path("evaluations/experiment-004-partial-boundary-tiles")
SHAPES = {"control": [128, 3072, 768], "partial_m": [130, 3072, 768]}
CONFIG = {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 3}
TOLERANCES = {"rtol": 0.02, "atol": 0.02}
SEEDS = (42, 43)
METADATA = (".nv.constant0.matmul_kernel", ".nv.info", ".nv.info.matmul_kernel")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def grid(shape):
    m, n, _ = shape
    return [(m + CONFIG["block_m"] - 1) // CONFIG["block_m"],
            (n + CONFIG["block_n"] - 1) // CONFIG["block_n"]]


def normalized_ptx(text):
    lines = []
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        if line and not line.startswith((".file", ".loc")):
            lines.append(re.sub(r"\s+", " ", line))
    return lines


def sass_instructions(text):
    return [re.sub(r"/\*.*?\*/", "", line).strip()
            for line in text.splitlines()
            if re.match(r"\s*/\*[0-9a-fA-F]+\*/\s+[^/\s]", line)]


def sass_counts(lines):
    counts = {"instructions": len(lines), "predicate_set": 0,
              "predicated_instructions": 0, "branches": 0}
    for line in lines:
        instruction = re.sub(r"^@!?P\d+\s+", "", line)
        opcode = instruction.split(None, 1)[0].split(".", 1)[0] if instruction else ""
        counts["predicate_set"] += opcode in {"PSETP", "ISETP", "FSETP", "HSETP", "P2R", "R2P"}
        counts["predicated_instructions"] += bool(re.match(r"^@!?P\d+\s+", line))
        counts["branches"] += opcode in {"BRA", "BRX", "JMP", "RET"}
    return counts


def compare_compiler(folder, compiler):
    """Compare complete code bytes and selected constant/launch metadata."""
    folder = Path(folder)
    sections = {name: _sections((folder / f"{name}.cubin").read_bytes())
                for name in ("baseline", "trusted")}
    code_names = {name for name in sections["baseline"] if name.startswith(".text.")}
    if (not code_names or code_names != {name for name in sections["trusted"] if name.startswith(".text.")}
            or any(name not in sections["baseline"] or name not in sections["trusted"]
                   for name in METADATA)):
        raise ValueError("required executable or metadata sections unavailable")
    compared = {}
    for name in sorted(code_names | set(METADATA)):
        a, b = sections["baseline"][name], sections["trusted"][name]
        compared[name] = {"identical": a == b, "baseline_sha256": sha(a),
                          "trusted_sha256": sha(b), "baseline_size_bytes": len(a),
                          "trusted_size_bytes": len(b), "size_bytes": len(a)}
    resources = {name: {key: compiler[name][key] for key in
                        ("registers_per_thread", "shared_memory_bytes", "spills")}
                 for name in ("baseline", "trusted")}
    if any(type(value) is not int or value < 0
           for side in resources.values() for value in side.values()):
        raise ValueError("invalid compiler resources")
    sass = {name: sass_instructions((folder / f"{name}.sass").read_text())
            for name in ("baseline", "trusted")}
    if any(not lines for lines in sass.values()):
        raise ValueError("decoded SASS unavailable")
    ptx = {name: normalized_ptx((folder / f"{name}.ptx").read_text())
           for name in ("baseline", "trusted")}
    executable_identical = all(compared[name]["identical"] for name in code_names)
    metadata_identical = all(compared[name]["identical"] for name in METADATA)
    return {"sections": compared, "resources": resources,
            "executable_identical": executable_identical,
            "metadata_identical": metadata_identical,
            "resources_identical": resources["baseline"] == resources["trusted"],
            "sass_instruction_stream_identical": sass["baseline"] == sass["trusted"],
            "sass_counts": {name: sass_counts(lines) for name, lines in sass.items()},
            "ptx_order_and_operands_identical_ignoring_debug": ptx["baseline"] == ptx["trusted"],
            "meaningful_compiler_difference": not (executable_identical and metadata_identical
                                                    and resources["baseline"] == resources["trusted"])}


def check_output(actual, reference, shape, *, rtol=0.02, atol=0.02):
    """Check full FP16 output and the last M tile; works on CPU fixture tensors."""
    import torch
    if tuple(actual.shape) != tuple(shape[:2]) or tuple(reference.shape) != tuple(shape[:2]):
        raise ValueError(f"output/reference shape mismatch: {tuple(actual.shape)}, {tuple(reference.shape)}")
    full = torch.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=False)
    boundary_start = (shape[0] - 1) // CONFIG["block_m"] * CONFIG["block_m"]
    boundary = full[boundary_start:]
    if not bool(full.all()) or not bool(boundary.all()):
        first = torch.nonzero(~full, as_tuple=False)[0].tolist()
        row, column = int(first[0]), int(first[1])
        difference = (actual.float() - reference.float()).abs()
        raise ValueError(f"FP16 mismatch at ({row}, {column}); actual={actual[row, column].item()}, "
                         f"reference={reference[row, column].item()}, bad_elements={int((~full).sum().item())}, "
                         f"max_abs_error={difference.max().item()}")
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual[boundary_start:], reference[boundary_start:], rtol=rtol, atol=atol)
    return {"output_shape": list(actual.shape), "boundary_start_row": boundary_start,
            "boundary_rows_checked": True, "boundary_row_count": shape[0] - boundary_start}


def validate_directory(root, folder):
    """Return one display entry only after checking every declared artifact."""
    root = Path(root).resolve()
    folder = Path(folder)
    if folder.is_absolute() or len(folder.parts) != len(SECTION.parts) + 1 or folder.parent != SECTION:
        raise ValueError("invalid Experiment 004 artifact path")
    manifest_path = root / folder / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_size > 100_000:
        raise ValueError("missing or oversized Experiment 004 manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["schema"] != "experiment_004_v1" or manifest["status"] != "complete"
            or manifest["shapes"] != SHAPES or manifest["config"] != CONFIG
            or not 20 <= len(manifest["artifact_hashes"]) <= 100
            or set(manifest["artifact_hashes"]) != set(manifest["files"])):
        raise ValueError("Experiment 004 manifest mismatch")
    files = {}
    for name, expected in manifest["artifact_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name == "manifest.json":
            raise ValueError("unsafe artifact name")
        files[name] = _read(root, str(folder / relative), expected)
    present = {str(path.relative_to(root / folder)) for path in (root / folder).rglob("*") if path.is_file()}
    if present != set(files) | {"manifest.json"}:
        raise ValueError("Experiment 004 contains undeclared or missing files")
    for name, source in (("benchmark_proposal.py", "benchmark_proposal.py"),
                         ("trusted_compute_kernel.py", "trusted_compute_kernel.py"),
                         ("reference_precision.py", "reference_precision.py"),
                         ("experiment_004_partial_boundary.py", "experiment_004_partial_boundary.py")):
        if files[name] != _read(root, source):
            raise ValueError("source snapshot differs from current reviewed code")
    original, warnings = load_experiment_003(root)
    if original is None or warnings:
        raise ValueError("Experiment 003 source control unavailable")
    result = json.loads(files["result.json"])
    identity = result["source_identity"]
    if (identity["experiment_003_id"] != original["id"]
            or identity["experiment_003_result_sha256"] != sha(_read(root, original["artifact_path"]))
            or identity["baseline_kernel_sha256"] != original["baseline_sha256"]
            or identity["baseline_launcher_sha256"] != original["launcher_sha256"]
            or identity["trusted_implementation_sha256"] != original["trusted_implementation_sha256"]
            or identity["original_muse_source_sha256"] != original["original_muse_source_sha256"]
            or identity["reference_precision_sha256"] != sha(files["reference_precision.py"])
            or identity["experiment_004_runner_sha256"] != sha(files["experiment_004_partial_boundary.py"])
            or identity["baseline_module_sha256"] != sha(files["benchmark_proposal.py"])
            or identity["trusted_module_sha256"] != sha(files["trusted_compute_kernel.py"])
            or identity["exact_generated_source_evaluated"] is not False
            or identity["variant_tested"] != "unchanged trusted_compute_kernel.py reviewed mask-hoisting kernel; not Muse-generated source"
            or result["shapes"] != SHAPES or result["config"] != CONFIG
            or result["dtype"] != "fp16" or result["input_seeds"] != list(SEEDS)
            or result["status"] != "complete" or result["runtime_screening"] != "not_run"
            or "RTX 4060 Ti" not in result["environment"]["gpu"]["name"]):
        raise ValueError("Experiment 004 source, configuration, or screening identity mismatch")
    for key, shape in SHAPES.items():
        checks = json.loads(files[f"{key}/correctness.json"])
        diagnostic = json.loads(files[f"{key}/diagnostic.json"])
        if (checks["status"] != "PASS" or checks["shape_mnk"] != shape
                or checks["tolerances"] != TOLERANCES or len(checks["checks"]) != 4
                or {(row["implementation"], row["seed"]) for row in checks["checks"]}
                   != {(side, seed) for side in ("baseline", "trusted") for seed in SEEDS}
                or any(row["status"] != "PASS" or row["reference_precision"]["policy"]
                       != "fp16_torch_mm_without_reduced_precision_reduction"
                       or row["output_shape"] != shape[:2] or row["boundary_rows_checked"] is not True
                       or row["boundary_start_row"] != ((shape[0] - 1) // CONFIG["block_m"]) * CONFIG["block_m"]
                       or row["boundary_row_count"] != shape[0] - row["boundary_start_row"]
                       for row in checks["checks"])
                or diagnostic["shape_mnk"] != shape or diagnostic["grid"] != grid(shape)
                or diagnostic["partial_m_boundary"] != (key == "partial_m")):
            raise ValueError("Experiment 004 correctness or boundary record mismatch")
        compiler = diagnostic["compiler"]
        for side in ("baseline", "trusted"):
            if not {"ttir", "ttgir", "llir", "ptx", "sass", "cubin"}.issubset(compiler[side]["artifacts"]):
                raise ValueError("incomplete compiler artifact set")
            for kind, artifact in compiler[side]["artifacts"].items():
                relative = f"{key}/{side}.{kind}"
                if (artifact["path"] != f"{side}.{kind}" or relative not in files
                        or artifact["sha256"] != manifest["artifact_hashes"][relative]
                        or "size_bytes" in artifact and artifact["size_bytes"] != len(files[relative])):
                    raise ValueError("compiler artifact omitted or inconsistent with manifest")
        comparison = compare_compiler(root / folder / key, compiler)
        if comparison != diagnostic["comparison"] or comparison != result["comparisons"][key]:
            raise ValueError("declared compiler comparison contradicts artifacts")
    if any(name.endswith("screening.json") for name in files):
        raise ValueError("runtime screening artifact forbidden")
    boundary = result["comparisons"]["partial_m"]
    outcome = ("control_difference_screening_skipped" if result["comparisons"]["control"]["meaningful_compiler_difference"]
               else "compiler_difference_screening_skipped" if boundary["meaningful_compiler_difference"]
               else "compiler_equivalent_screening_skipped")
    if result["outcome"] != outcome:
        raise ValueError("Experiment 004 outcome mismatch")
    return {"id": f"diagnostic:partial_boundary_tiles:{folder.name}",
            "kind": "compiler_diagnostic", "created_utc": result["created_utc"],
            "shape_mnk": SHAPES["partial_m"], "dtype": "fp16", "gpu": result["environment"]["gpu"]["name"],
            "configuration": CONFIG, "hypothesis": "A partial M tile may change compiled code for the reviewed mask-hoisting variant relative to baseline.",
            "mechanism": "Compare the unchanged baseline and independently reviewed mask-hoisting implementation on M=130.",
            "families": ["mask_hoisting"], "primary_family": "mask_hoisting", "secondary_families": [],
            "original_muse_source_sha256": identity["original_muse_source_sha256"],
            "trusted_implementation_sha256": identity["trusted_implementation_sha256"],
            "baseline_sha256": identity["baseline_kernel_sha256"],
            "launcher_sha256": identity["baseline_launcher_sha256"],
            "compilation": "both variants compiled for control and partial-M workloads",
            "correctness": "PASS", "correctness_cases": [8, 8],
            "screening": None, "confirmation": "not a runtime measurement", "outcome": outcome,
            "performance_interpretation": "Not measured; runtime screening was not run",
            "compiler": {"resources": boundary["resources"], "review": {
                "sass": "Complete executable SASS bytes and decoded instruction streams compared.",
                "ptx": "Ordered PTX and predicate/branch counts retained in artifacts."}},
            "diagnostic_evidence": {"source_identity": identity, "artifact_hashes": manifest["artifact_hashes"],
                "detail_prefix": "partial_m/",
                "executable_section": boundary["sections"][".text.matmul_kernel"],
                "partial_boundary_tiles": True, "control_comparison": result["comparisons"]["control"],
                "constant_data_and_launch_metadata_identical": boundary["metadata_identical"],
                "screening_status": "skipped", "screening_skip_reason": result["screening_decision"],
                "latency_measured": False, "performance_conclusion": "No runtime outcome measured.",
                "correctness_cases": [8, 8], "tolerances": TOLERANCES,
                "limitations": result["limitations"]},
            "provenance": "Independent Experiment 004 compiler diagnostic of unchanged reviewed code; no timings",
            "exact_generated_source_evaluated": False,
            "artifact_path": str(folder / "result.json"),
            "question": "Does a partial M boundary tile change the compiled baseline versus reviewed variant?",
            "title": "Partial M boundary tile — compiler diagnostic"}


def load(root=ROOT):
    root = Path(root).resolve()
    base = root / SECTION
    if not base.is_dir() or base.is_symlink():
        return [], []
    entries, warnings = [], []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.is_symlink() or not re.fullmatch(r"\d{8}T\d{12}Z", folder.name):
            continue
        if not (folder / "manifest.json").exists():
            continue
        try:
            entries.append(validate_directory(root, folder.relative_to(root)))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            warnings.append(f"Experiment 004 rejected ({folder.name}): {exc}")
    return entries, warnings


def with_display_entries(root, memory):
    """Add validated Experiment 004 records to a read-only dashboard projection."""
    entries, warnings = load(root)
    present = {row["id"] for row in memory["entries"]}
    return {**memory,
            "entries": sorted([*memory["entries"], *(row for row in entries if row["id"] not in present)],
                              key=lambda row: (row["created_utc"], row["id"])),
            "warnings": [*memory.get("warnings", []), *warnings]}
