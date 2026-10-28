"""Artifact-only Experiment 003 ingestion; no compiler, CUDA, or runner imports."""

import hashlib
import json
from pathlib import Path
import struct

from grouped_confirmation_memory import _read


FOLDER = "evaluations/experiment-003-mask-hoisting-m128/20261028T145501296300Z/"
HASHES = {
    "diagnostic.json": "e91921804302fbb122eae5354354bdc96159f7bfcf5f7880bb44ab596db047b9",
    "result.json": "487816e4dda18827f7582b7b0c4847652f6b68ce2dfc0d634f2dfb640e5f0a93",
    "correctness.json": "3df3bad65435cb63729cbe38a931192bbb8c25b9eea5675b599825ec0c95525a",
    "elf_comparison.json": "2aa0a162b7898ce1a1b7a43f814688acbe85382c394b9164926ee1ef79942e59",
}


def _sections(raw):
    if raw[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected ELF64 little-endian cubin")
    offset = struct.unpack_from("<Q", raw, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", raw, 58)
    if size != 64 or offset + size * count > len(raw):
        raise ValueError("invalid ELF section table")
    headers = [struct.unpack_from("<IIQQQQIIQQ", raw, offset + i * size) for i in range(count)]
    names = headers[names_index]
    strings = raw[names[4]:names[4] + names[5]]
    return {strings[h[0]:].split(b"\0", 1)[0].decode(): raw[h[4]:h[4] + h[5]]
            for h in headers if h[1] != 8}


def load(root):
    return validate_directory(root, FOLDER, HASHES)


def validate_directory(root, folder, hashes, *, source_root=None, require_equivalent=True):
    """Validate either immutable historical evidence or a newly manifested trusted run."""
    root = Path(root).resolve()
    folder = str(folder).rstrip("/") + "/"
    source_root = Path(source_root).resolve() if source_root is not None else root
    if not (root / folder / "result.json").exists():
        return None, []
    try:
        if set(hashes) != set(HASHES):
            raise ValueError("incomplete diagnostic manifest")
        files = {name: json.loads(_read(root, folder + name, digest)) for name, digest in hashes.items()}
        diagnostic, result, correctness, elf = (files[name] for name in HASHES)
        identity = diagnostic["source_identity"]
        prior = json.loads(_read(root, identity["prior_path"], identity["prior_sha256"]))
        plan = json.loads(_read(root, identity["plan_path"], identity["plan_sha256"]))
        if (diagnostic["status"] != "diagnostic_complete" or diagnostic["shape_mnk"] != [128, 3072, 768]
                or diagnostic["config"] != prior["numeric_config"]
                or plan["target_shape_mnk"] != diagnostic["shape_mnk"]
                or diagnostic["partial_boundary_tiles"] is not False
                or result["phase_2"]["status"] != "not_run"
                or result["phase_2"]["raw_latency_measurements"] != []
                or result["phase_2"]["median_paired_speedup"] is not None
                or (root / folder / "screening.json").exists()):
            raise ValueError("diagnostic/screening status mismatch")
        for key in ("trusted_implementation_sha256", "trusted_module_sha256",
                    "baseline_kernel_sha256", "baseline_launcher_sha256"):
            if identity[key] != prior[key]:
                raise ValueError("historical source identity mismatch")
        for key, path in (("trusted_module_sha256", "trusted_compute_kernel.py"),
                          ("runner_sha256", "mask_hoisting_m128_experiment.py")):
            _read(root, folder + path, identity[key])
        _read(source_root, "trusted_compute_kernel.py", identity["trusted_module_sha256"])
        _read(source_root, "reference_precision.py", identity["reference_precision_sha256"])
        _read(source_root, "benchmark_core.py", identity["benchmark_core_sha256"])
        if _read(source_root, "benchmark_proposal.py") != _read(root, folder + "benchmark_proposal.py"):
            raise ValueError("baseline source snapshot mismatch")
        _read(root, folder + correctness.get("verifier_file", "verify_artifacts.py"), correctness["verifier_sha256"])
        checks = correctness["checks"]
        if (correctness["status"] != "PASS" or correctness["shape_mnk"] != [128, 3072, 768]
                or correctness["tolerances"] != {"rtol": 0.02, "atol": 0.02}
                or len(checks) != 4
                or {(row["implementation"], row["seed"]) for row in checks}
                   != {(name, seed) for name in ("baseline", "trusted") for seed in (42, 43)}
                or any(row["status"] != "PASS" or row["reference_precision"]["policy"]
                       != "fp16_torch_mm_without_reduced_precision_reduction" for row in checks)):
            raise ValueError("correctness records or tolerances mismatch")
        sections, resources = {}, {}
        for name in ("baseline", "trusted"):
            compiler = diagnostic["compiler"][name]
            for kind, artifact in compiler["artifacts"].items():
                if kind not in {"ttir", "ttgir", "llir", "ptx", "sass", "cubin"} or artifact["path"] != f"{name}.{kind}":
                    raise ValueError("unexpected compiler artifact path")
                _read(root, folder + artifact["path"], artifact["sha256"])
            sections[name] = _sections(_read(root, folder + f"{name}.cubin"))
            resources[name] = {key: compiler[key] for key in
                               ("registers_per_thread", "shared_memory_bytes", "spills")}
            if any(type(value) is not int or value < 0 for value in resources[name].values()):
                raise ValueError("invalid compiler resource metric")
            executable = {key: value for key, value in sections[name].items() if key.startswith(".text.")}
            declared = diagnostic["executable_sections"][name]
            if set(executable) != set(declared) or any(
                    len(raw) != declared[key]["size_bytes"] or hashlib.sha256(raw).hexdigest() != declared[key]["sha256"]
                    for key, raw in executable.items()):
                raise ValueError("declared executable sections do not match binary")
        if diagnostic["code_bytes_identical"] != (sections["baseline"][".text.matmul_kernel"] == sections["trusted"][".text.matmul_kernel"]):
            raise ValueError("declared code equivalence contradicts binary")
        equivalent = resources["baseline"] == resources["trusted"]
        if require_equivalent and not equivalent:
            raise ValueError("resource equivalence mismatch")
        for name in (".text.matmul_kernel", ".nv.constant0.matmul_kernel", ".nv.info", ".nv.info.matmul_kernel"):
            a, b = (sections[key][name] for key in ("baseline", "trusted"))
            identical = a == b
            equivalent = equivalent and identical
            if ((require_equivalent and not identical) or elf[name]["identical"] != identical
                    or len(a) != elf[name]["size_bytes"]
                    or hashlib.sha256(a).hexdigest() != elf[name]["baseline_sha256"]
                    or hashlib.sha256(b).hexdigest() != elf[name]["trusted_sha256"]):
                raise ValueError("executable/constant/metadata equivalence mismatch")
        return {
            "id": f"diagnostic:mask_hoisting_m128:{Path(folder).name}",
            "kind": "compiler_diagnostic", "created_utc": diagnostic["created_utc"],
            "shape_mnk": diagnostic["shape_mnk"], "dtype": "fp16",
            "gpu": diagnostic["environment"]["gpu"]["name"], "configuration": diagnostic["config"],
            "hypothesis": plan["technical_hypothesis"], "mechanism": plan["optimization_mechanism"],
            "families": ["mask_hoisting"], "primary_family": "mask_hoisting", "secondary_families": [],
            "original_muse_source_sha256": identity["original_muse_source_sha256"],
            "trusted_implementation_sha256": identity["trusted_implementation_sha256"],
            "baseline_sha256": identity["baseline_kernel_sha256"],
            "launcher_sha256": identity["baseline_launcher_sha256"],
            "compilation": "both compiled; executable bytes and metadata compared",
            "correctness": "PASS", "correctness_cases": [4, 4],
            "screening": None, "confirmation": "not a runtime measurement",
            "outcome": "compiler_equivalent_screening_skipped" if equivalent else "compiler_difference_screening_skipped",
            "performance_interpretation": "Not measured; runtime screening intentionally skipped",
            "compiler": {"resources": resources, "review": {
                "ptx": "Raw PTX and ordered comparison are saved.",
                "sass": "Complete executable sections, constant data, and launch metadata were compared; not inferred from instruction counts."}},
            "diagnostic_evidence": {"source_identity": identity, "artifact_hashes": hashes,
                "executable_section": elf[".text.matmul_kernel"], "partial_boundary_tiles": False,
                "constant_data_and_launch_metadata_identical": all(elf[name]["identical"] for name in
                    (".nv.constant0.matmul_kernel", ".nv.info", ".nv.info.matmul_kernel")),
                "screening_status": "skipped", "screening_skip_reason": result["phase_2"]["reason"],
                "latency_measured": False, "performance_conclusion": result["performance_conclusion"],
                "correctness_cases": [4, 4], "tolerances": correctness["tolerances"],
                "limitations": result["limitations"]},
            "provenance": "Hash-validated compiler diagnostic of unchanged reviewed code; no CUDA-event timings",
            "exact_generated_source_evaluated": False, "artifact_path": folder + "result.json",
        }, []
    except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error) as exc:
        return None, [f"Experiment 003 diagnostic rejected: {exc}"]
