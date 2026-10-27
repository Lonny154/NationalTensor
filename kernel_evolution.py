"""Proposal-only Muse experiment for a replacement Triton GEMM kernel.

Generated source is parsed and saved for human review. This module never
imports, compiles, or runs a candidate. Evaluation has a separate entry point.
"""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI

import benchmark_proposal
import optimize


SHAPE = (512, 3072, 768)
MODEL = optimize.MUSE_MODEL
ROOT = Path("evaluations/kernel-evolution")
FIELDS = {"kernel_source", "hypothesis", "expected_mechanism",
          "reasoning_from_feedback"}
ARGS = ("A", "B", "C", "M", "N", "K", "BM", "BN", "BK")
TL_ATTRIBUTES = {
    "constexpr", "float16", "float32", "int32", "int64", "program_id",
    "cdiv", "arange", "load", "store", "dot", "full", "zeros", "sum",
    "maximum", "minimum", "where", "trans", "broadcast_to", "multiple_of",
    "max_contiguous", "range", "static_range",
}
TL_CALLS = TL_ATTRIBUTES - {"constexpr", "float16", "float32", "int32", "int64"}
ALLOWED_NODES = (
    ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Expr, ast.Pass,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Attribute, ast.Name, ast.Constant, ast.Subscript,
    ast.Tuple, ast.List, ast.Slice, ast.keyword, ast.Load, ast.Store,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.USub, ast.UAdd, ast.Invert, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
)


def baseline_source():
    return inspect.getsource(benchmark_proposal.matmul_kernel.fn)


def source_sha256(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class KernelAstInspector(ast.NodeVisitor):
    """Narrow review filter; never a security boundary."""

    def __init__(self, local_names):
        self.local_names = set(local_names)
        self.calls = set()

    def generic_visit(self, node):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("_") or node.id not in self.local_names | {"tl", "range"}:
            raise ValueError(f"unsupported name: {node.id}")
        if isinstance(node.ctx, ast.Store) and node.id in {"tl", "range", "triton"}:
            raise ValueError("cannot rebind approved globals")

    def visit_Attribute(self, node):
        if not isinstance(node.value, ast.Name):
            raise ValueError("nested attribute access is unsupported")
        if isinstance(node.ctx, ast.Store):
            raise ValueError("attribute assignment is unsupported")
        if node.value.id == "tl" and node.attr in TL_ATTRIBUTES:
            return
        if node.value.id in self.local_names and node.attr == "to":
            return
        raise ValueError(f"unsupported attribute: {node.attr}")

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            if func.id != "range":
                raise ValueError("only range and approved Triton calls are allowed")
            call_name = "range"
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "tl" and func.attr in TL_CALLS:
                call_name = f"tl.{func.attr}"
            elif func.value.id in self.local_names and func.attr == "to":
                call_name = "value.to"
            else:
                raise ValueError("unsupported function call")
        else:
            raise ValueError("indirect function calls are unsupported")
        self.calls.add(call_name)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("_"):
                raise ValueError("unsupported keyword expansion")
            self.visit(keyword.value)

    def visit_For(self, node):
        if (node.orelse or not isinstance(node.target, ast.Name)
                or not isinstance(node.iter, ast.Call)):
            raise ValueError("only ordinary bounded-looking for loops are accepted")
        func = node.iter.func
        if not ((isinstance(func, ast.Name) and func.id == "range")
                or (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "tl"
                    and func.attr in {"range", "static_range"})):
            raise ValueError("for loops must use range, tl.range, or tl.static_range")
        self.generic_visit(node)

    def visit_Assign(self, node):
        def name_target(target):
            return isinstance(target, ast.Name) or (
                isinstance(target, ast.Tuple) and all(
                    isinstance(item, ast.Name) for item in target.elts))
        if not all(name_target(target) for target in node.targets):
            raise ValueError("assignment targets must be local names")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise ValueError("augmented assignment target must be a local name")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if type(node.value) not in (int, float, bool, type(None)):
            raise ValueError("only numeric, boolean, and None literals are supported")


def inspect_kernel_source(source):
    """Require a modified single JIT function with the fixed kernel interface."""
    if type(source) is not str or not source.strip() or len(source) > 20_000:
        raise ValueError("kernel source must be nonempty and at most 20,000 characters")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("source must contain exactly one kernel function")
    function = tree.body[0]
    if function.name != "matmul_kernel" or len(function.decorator_list) != 1:
        raise ValueError("expected one @triton.jit matmul_kernel")
    decorator = function.decorator_list[0]
    if not (isinstance(decorator, ast.Attribute) and decorator.attr == "jit"
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"):
        raise ValueError("only @triton.jit is allowed")
    signature = function.args
    if (tuple(arg.arg for arg in signature.args) != ARGS or signature.defaults
            or signature.kwonlyargs or signature.posonlyargs
            or signature.vararg or signature.kwarg):
        raise ValueError("kernel signature must match the baseline exactly")
    for index, arg in enumerate(signature.args):
        annotation = arg.annotation
        expected_constexpr = index >= 3
        is_constexpr = (isinstance(annotation, ast.Attribute)
                        and annotation.attr == "constexpr"
                        and isinstance(annotation.value, ast.Name)
                        and annotation.value.id == "tl")
        if is_constexpr != expected_constexpr or (annotation is not None and not is_constexpr):
            raise ValueError("kernel argument annotations must match the baseline")
    if function.returns is not None or function.type_params:
        raise ValueError("return annotations and type parameters are unsupported")
    base_tree = ast.parse(baseline_source())
    if ast.dump(tree, include_attributes=False) == ast.dump(base_tree, include_attributes=False):
        raise ValueError("proposal does not modify the baseline kernel")
    locals_written = {node.id for node in ast.walk(function)
                      if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    inspector = KernelAstInspector(set(ARGS) | locals_written)
    for statement in function.body:
        inspector.visit(statement)
    if not {"tl.load", "tl.dot", "tl.store"}.issubset(inspector.calls):
        raise ValueError("kernel must load operands, perform dot products, and store output")
    return {"ast_check": "accepted_for_review_only",
            "calls": sorted(inspector.calls), "source_sha256": source_sha256(source)}


def inspect_compute_change(source):
    """Reject a schedule-only rewrite in the compute-focused proposal mode."""
    function = ast.parse(source).body[0]
    for name, axis in (("pid_m", 0), ("pid_n", 1)):
        assignments = [node for node in ast.walk(function)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == name
                               for target in node.targets)]
        if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Call):
            raise ValueError("compute-focused proposal must keep direct 2D program IDs")
        expected = ast.parse(f"tl.program_id({axis})", mode="eval").body
        if ast.dump(assignments[0].value) != ast.dump(expected):
            raise ValueError("compute-focused proposal must keep direct 2D program IDs")
    program_id_calls = [node for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "tl"
                        and node.func.attr == "program_id"]
    if len(program_id_calls) != 2:
        raise ValueError("compute-focused proposal must keep direct 2D program IDs")

    def compute_nodes(text):
        function = ast.parse(text).body[0]
        return [ast.dump(node, include_attributes=False)
                for node in ast.walk(function)
                if isinstance(node, ast.For)
                or (isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name)
                            and target.id in {"a_ptrs", "b_ptrs", "k"}
                            for target in node.targets))
                or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "tl"
                    and node.func.attr in {"load", "dot"})]

    if compute_nodes(source) == compute_nodes(baseline_source()):
        raise ValueError("compute-focused proposal changes only tile scheduling or indexing")


def parse_response(raw, *, require_compute_change=False):
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != FIELDS:
        raise ValueError("response must contain exactly the four proposal fields")
    for name in FIELDS - {"kernel_source"}:
        if type(obj[name]) is not str or not obj[name].strip() or len(obj[name]) > 2000:
            raise ValueError(f"{name} must be a nonempty string under 2000 characters")
    findings = inspect_kernel_source(obj["kernel_source"])
    if require_compute_change:
        inspect_compute_change(obj["kernel_source"])
    return obj, findings


def structural_history(root):
    """Summarize compatible FC1 structural results without pooling raw methods."""
    rows = []
    if not root.exists():
        return rows
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            meta = json.loads((folder / "metadata.json").read_text())
            if (meta.get("shape_mnk") != list(SHAPE)
                    or meta.get("measurement_method", {}).get("measurement_mode")
                    != "python_launch_cuda_event"):
                continue
            aa_file = folder / "aa.json"
            aa = json.loads(aa_file.read_text()) if aa_file.exists() else {}
            for line in (folder / "experiments.jsonl").read_text().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                timing = record.get("timing") or {}
                if timing and timing.get("measurement_mode") != "python_launch_cuda_event":
                    continue
                rows.append({
                    "source_run": folder.name, "schedule": record.get("schedule"),
                    "source_sha256": (record.get("source_sha256") or {}).get(
                        record.get("schedule", "")),
                    "config": record.get("config"),
                    "hypothesis": record.get("hypothesis"),
                    "status": record.get("status"),
                    "error": record.get("error"),
                    "median_paired_speedup": timing.get("median_paired_speedup"),
                    "paired_ratio_mad": timing.get("statistics", {})
                    .get("paired_ratios", {}).get("mad"),
                    "aa_median_absolute_ratio_deviation": (
                        aa.get("aa_variability", {})
                        .get("median_absolute_ratio_deviation")),
                    "measurement_mode": timing.get("measurement_mode"),
                    "optimization_status": record.get("optimization_status"),
                })
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return rows[-12:]


def reviewed_feedback(candidate_dir, evaluation_path):
    """Read a completed evaluation and exact prior source for the next prompt."""
    metadata = json.loads((candidate_dir / "metadata.json").read_text())
    evaluation = json.loads(evaluation_path.read_text())
    source = (candidate_dir / "candidate.py").read_text()
    if source_sha256(source) != metadata["candidate_sha256"]:
        raise ValueError("previous candidate source changed after proposal")
    if evaluation.get("candidate_sha256") != metadata["candidate_sha256"]:
        raise ValueError("evaluation does not match previous candidate")
    timing = evaluation.get("confirmation") or evaluation.get("screening") or {}
    paired = timing.get("statistics", {}).get("paired_ratios", {})
    return {"candidate_dir": str(candidate_dir),
            "evaluation_path": str(evaluation_path),
            "candidate_sha256": metadata["candidate_sha256"],
            "hypothesis": metadata["hypothesis"],
            "kernel_source": source,
            "evaluation": {
                "status": evaluation.get("status"),
                "gpu": evaluation.get("gpu"),
                "compilation": evaluation.get("compilation"),
                "correctness": evaluation.get("correctness"),
                "failure_stage": evaluation.get("failure_stage"),
                "error": evaluation.get("error"),
                "median_paired_speedup": timing.get("median_paired_speedup"),
                "baseline_ms": timing.get("baseline_ms"),
                "candidate_ms": timing.get("candidate_ms"),
                "paired_ratio_mad": paired.get("mad"),
                "paired_ratio_cv": paired.get("cv"),
                "measurement_mode": timing.get("measurement_mode"),
                "confirmation_decision": evaluation.get("confirmation_decision"),
            }}


def reviewed_unevaluated_proposal(candidate_dir):
    """Read a prior review artifact without implying it was evaluated."""
    metadata = json.loads((candidate_dir / "metadata.json").read_text())
    source = (candidate_dir / "candidate.py").read_text()
    if (metadata.get("status") != "awaiting_human_review"
            or metadata.get("shape_mnk") != list(SHAPE)
            or source_sha256(source) != metadata.get("candidate_sha256")):
        raise ValueError("review source, status, shape, or SHA-256 is incompatible")
    rejected = []
    for line in (candidate_dir / "proposal_audit.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if "rejection" in entry:
            proposal = json.loads(entry["raw_response"])
            rejected.append({"source_sha256": source_sha256(proposal["kernel_source"]),
                             "hypothesis": proposal["hypothesis"],
                             "expected_mechanism": proposal["expected_mechanism"],
                             "static_rejection": entry["rejection"]})
    mapping_review = None
    if "pid = pid_m0 * num_pid_n + pid_n0" in source:
        mapping_review = (
            "CPU arithmetic shows unique full tile coverage for all three workloads, "
            "but pid_m0 * num_pid_n + pid_n0 disagrees with the 2D launch grid's "
            "dimension-0-fastest flattening; intended Grouped-M4 launch locality "
            "is not established.")
    static_filter_note = None
    if any("AnnAssign" in item["static_rejection"] for item in rejected):
        static_filter_note = (
            "A local annotated assignment (AnnAssign) was rejected as "
            "unsupported syntax. That syntax alone does not show unsafe host "
            "behavior; the review filter remains unchanged.")
    return {"candidate_dir": str(candidate_dir),
            "candidate_sha256": metadata["candidate_sha256"],
            "status": metadata["status"],
            "hypothesis": metadata["hypothesis"],
            "expected_mechanism": metadata["expected_mechanism"],
            "kernel_source": source,
            "measurement_status": "not compiled, checked for correctness, or timed",
            "rejected_attempts": rejected,
            "mapping_review": mapping_review,
            "static_filter_note": static_filter_note}


def reviewed_trusted_feedback(feedback_path, review):
    """Bind external trusted measurements to, but never equate them with, Muse source."""
    report_text = feedback_path.read_text()
    report = json.loads(report_text)
    if not all(isinstance(report.get(name), dict) for name in
               ("screening", "aa_calibration", "compiler_review")):
        raise ValueError("trusted feedback lacks screening, A/A, or compiler evidence")
    if "check_artifact" in report:
        return reviewed_split_trusted_feedback(feedback_path, report_text, report, review)
    experiment_name = report.get("experiment_artifact")
    if (report.get("report_type") != "trusted_implementation_feedback"
            or report.get("provenance") != "model_suggested_human_reviewed_implementation"
            or report.get("exact_generated_source_evaluated") is not False
            or report.get("original_muse_candidate_sha256")
            != review["candidate_sha256"]
            or report.get("trusted_implementation_sha256")
            == review["candidate_sha256"]
            or type(experiment_name) is not str
            or Path(experiment_name).name != experiment_name):
        raise ValueError("trusted feedback provenance does not match prior Muse proposal")
    experiment_path = feedback_path.parent / experiment_name
    experiment_text = experiment_path.read_text()
    experiment = json.loads(experiment_text)
    screening = experiment.get("screening") or {}
    aa = experiment.get("aa") or {}
    compiler = experiment.get("compiler") or {}
    if (experiment.get("provenance") != report["provenance"]
            or experiment.get("exact_generated_source_evaluated") is not False
            or experiment.get("muse_candidate_sha256")
            != review["candidate_sha256"]
            or experiment.get("trusted_implementation_sha256")
            != report["trusted_implementation_sha256"]
            or experiment.get("shape_mnk") != list(SHAPE)
            or report.get("workload_mnk") != list(SHAPE)
            or experiment.get("gpu") != report.get("gpu")
            or screening.get("measurement_mode") != "python_launch_cuda_event"
            or screening.get("median_paired_speedup")
            != report.get("screening", {}).get("median_paired_speedup")
            or screening.get("baseline_ms")
            != report.get("screening", {}).get("baseline_median_ms")
            or screening.get("candidate_ms")
            != report.get("screening", {}).get("trusted_median_ms")
            or screening.get("statistics", {}).get("paired_ratios", {}).get("mad")
            != report.get("screening", {}).get("paired_ratio_mad")
            or aa.get("aa_variability", {}).get("median_absolute_ratio_deviation")
            != report.get("aa_calibration", {}).get("median_absolute_ratio_deviation")
            or aa.get("aa_variability", {}).get("max_absolute_ratio_deviation")
            != report.get("aa_calibration", {}).get("max_absolute_ratio_deviation")
            or not all(item in compiler for item in ("baseline", "trusted"))):
        raise ValueError("trusted feedback does not match its experiment artifact")
    return {"feedback_path": str(feedback_path),
            "feedback_sha256": source_sha256(report_text),
            "experiment_path": str(experiment_path),
            "experiment_sha256": source_sha256(experiment_text),
            "provenance": report["provenance"],
            "exact_generated_source_evaluated": False,
            "original_muse_candidate_sha256": review["candidate_sha256"],
            "trusted_implementation_sha256": report["trusted_implementation_sha256"],
            "workload_mnk": report["workload_mnk"],
            "correctness": report["correctness"],
            "correctness_records": experiment.get("correctness"),
            "screening": report["screening"],
            "aa_calibration": report["aa_calibration"],
            "compiler_review": report["compiler_review"],
            "compiler_resources": {
                name: {key: compiler[name].get(key) for key in
                       ("registers_per_thread", "spills", "shared_memory_bytes")}
                for name in ("baseline", "trusted")},
            "feedback_for_next_proposal": report["feedback_for_next_proposal"]}


def reviewed_split_trusted_feedback(feedback_path, report_text, report, review):
    """Bind the two-stage trusted run and existing Inspector outputs to a proposal."""
    if (report.get("report_type") != "trusted_implementation_feedback"
            or report.get("provenance") != "model_suggested_human_reviewed_implementation"
            or report.get("exact_generated_source_evaluated") is not False
            or report.get("original_muse_candidate_sha256") != review["candidate_sha256"]
            or report.get("trusted_implementation_sha256") == review["candidate_sha256"]
            or report.get("workload_mnk") != list(SHAPE)):
        raise ValueError("trusted feedback provenance does not match prior Muse proposal")

    artifacts = {}
    for label in ("check", "screening", "ptx_comparison", "sass_comparison",
                  "dynamic_profile"):
        name = report.get(f"{label}_artifact")
        if (type(name) is not str or not name or Path(name).is_absolute()
                or ".." in Path(name).parts):
            raise ValueError(f"invalid trusted {label} artifact path")
        path = feedback_path.parent / name
        raw = path.read_bytes()
        artifacts[label] = {"path": str(path),
                            "sha256": hashlib.sha256(raw).hexdigest(), "raw": raw}
    check = json.loads(artifacts["check"]["raw"])
    run = json.loads(artifacts["screening"]["raw"])
    ptx = json.loads(artifacts["ptx_comparison"]["raw"])
    sass = json.loads(artifacts["sass_comparison"]["raw"])
    screening = run.get("screening", {})
    aa = run.get("aa", {})
    compiler = check.get("compiler", {})
    for item in (check, run):
        if (item.get("provenance") != report["provenance"]
                or item.get("exact_generated_source_evaluated") is not False
                or item.get("original_muse_candidate_sha256") != review["candidate_sha256"]
                or item.get("trusted_implementation_sha256")
                != report["trusted_implementation_sha256"]
                or item.get("gpu") != report.get("gpu")):
            raise ValueError("trusted feedback does not match its experiment artifacts")
    correctness = check.get("correctness", [])
    expected_cases = {(tuple(shape), seed) for shape in
                      ((512, 3072, 768), (509, 3071, 769), (65, 129, 33))
                      for seed in (42, 43)}
    observed_cases = {(tuple(row["shape_mnk"]), row["input_seed"])
                      for row in correctness if row.get("status") == "PASS"}
    pairs = screening.get("paired_rounds", [])
    if (check.get("status") != "correctness_and_compiler_captured"
            or run.get("status") != "screening_complete_unconfirmed"
            or run.get("prior_check_sha256") != artifacts["check"]["sha256"]
            or observed_cases != expected_cases or len(correctness) != 6
            or run.get("measurement_method", {}).get("measurement_mode")
            != "python_launch_cuda_event"
            or screening.get("measurement_mode") != "python_launch_cuda_event"
            or screening.get("median_paired_speedup")
            != report["screening"].get("median_paired_speedup")
            or screening.get("baseline_ms") != report["screening"].get("baseline_median_ms")
            or screening.get("candidate_ms") != report["screening"].get("trusted_median_ms")
            or screening.get("statistics", {}).get("paired_ratios", {}).get("mad")
            != report["screening"].get("paired_ratio_mad")
            or len(pairs) != 8
            or all(pair.get("baseline_over_candidate", 1) < 1 for pair in pairs)
            is not report["screening"].get("all_eight_pairs_favor_baseline")
            or aa.get("aa_variability") != report["aa_calibration"]
            or ptx.get("tool") != "ptx-inspector" or ptx.get("mode") != "compare"
            or sass.get("tool") != "sass-inspector" or sass.get("mode") != "compare"
            or not all(name in compiler for name in ("baseline", "trusted"))):
        raise ValueError("trusted feedback does not match its experiment artifacts")

    profile_rows = csv.DictReader(
        line for line in artifacts["dynamic_profile"]["raw"].decode().splitlines()
        if line.startswith('"'))
    profile = {"baseline": {}, "trusted": {}}
    for row in profile_rows:
        side = {"0": "baseline", "1": "trusted"}.get(row["ID"])
        if side:
            profile[side][row["Metric Name"]] = int(row["Metric Value"])
    if any(profile[side] != report["dynamic_profile"][side]
           for side in ("baseline", "trusted")):
        raise ValueError("trusted dynamic profile does not match profiler artifact")

    resources = {name: {key: compiler[name][key] for key in
                        ("registers_per_thread", "spills", "shared_memory_bytes")}
                 for name in ("baseline", "trusted")}
    static_counts = {}
    for label, comparison, scope in (("ptx", ptx, "kernel"),
                                     ("sass", sass, "function")):
        sides = comparison["comparison"][scope]
        static_counts[label] = {
            "baseline": sides["baseline"]["instruction_count"],
            "trusted": sides["candidate"]["instruction_count"]}
    return {"feedback_path": str(feedback_path),
            "feedback_sha256": source_sha256(report_text),
            "experiment_path": artifacts["screening"]["path"],
            "experiment_sha256": artifacts["screening"]["sha256"],
            "evidence_artifacts": {key: {"path": value["path"],
                                          "sha256": value["sha256"]}
                                   for key, value in artifacts.items()},
            "provenance": report["provenance"],
            "exact_generated_source_evaluated": False,
            "original_muse_candidate_sha256": review["candidate_sha256"],
            "trusted_implementation_sha256": report["trusted_implementation_sha256"],
            "workload_mnk": report["workload_mnk"],
            "correctness": report["correctness"],
            "correctness_records": correctness,
            "screening": report["screening"],
            "aa_calibration": report["aa_calibration"],
            "compiler_review": report["compiler_review"],
            "compiler_resources": resources,
            "matmul_inspector_static_instruction_counts": static_counts,
            "dynamic_profile": report["dynamic_profile"],
            "research_question": report["research_question"],
            "measured_observations": report["measured_observations"],
            "speculative_explanations": report["speculative_explanations"],
            "feedback_for_next_proposal": report["feedback_for_next_proposal"]}


def build_prompt(gpu, hardware, history, feedback=None, review=None,
                 trusted_feedback=None):
    context = {
        "workload": {"M": SHAPE[0], "N": SHAPE[1], "K": SHAPE[2],
                     "dtype": "fp16", "gpu": gpu, "hardware": hardware},
        "fixed_launcher_source": inspect.getsource(benchmark_proposal.launch),
        "baseline_kernel_source": baseline_source(),
        "numeric_config": optimize.BASELINE,
        "previous_structural_results": history,
        "reviewed_generated_candidate_feedback": feedback,
        "prior_unevaluated_review": review,
        "external_trusted_implementation_feedback": trusted_feedback,
        "constraints": [
            "Return one replacement @triton.jit matmul_kernel with exactly the baseline signature.",
            "Keep the existing launcher, one kernel launch, FP16 inputs/output, FP32 accumulation.",
            "No imports, host code, additional kernels, file/network access, or side effects outside tl.store.",
            "Correctness must cover multiple seeds and boundary shapes under the scoped FP16 torch.mm policy.",
            "Screening uses paired eager Python-launch CUDA-event timings; results are noisy and unconfirmed.",
            "Explain the technical change and expected performance mechanism without inventing profiler data.",
        ],
    }
    if review is not None:
        context["constraints"].extend([
            "This is an independent proposal request; the prior generated kernel has no compilation, correctness, or timing result.",
            "Keep the baseline program-ID mapping. Do not propose Grouped-M4, Split-K, or any program-ID-only scheduling swizzle.",
            "Make a meaningful change to the inner computation, K-loop organization, operand-loading strategy, or software pipelining.",
            "Preserve FP16 input and output and FP32 accumulation, the exact kernel signature, and the fixed launcher.",
            "In the hypothesis and expected_mechanism, name the specific load, dot, pointer, mask, or pipeline operations expected to improve.",
            "In reasoning_from_feedback, explain how this implementation differs from both prior generated attempts and reviewed structural kernels; do not claim measured feedback for unevaluated code.",
            "Use plain local assignments: annotated local assignments (AnnAssign) are currently unsupported by the review filter.",
        ])
    if trusted_feedback is not None:
        if review is None:
            raise ValueError("trusted feedback requires the related unevaluated proposal")
        paired_ratio = trusted_feedback["screening"]["median_paired_speedup"]
        aa_deviation = trusted_feedback["aa_calibration"][
            "median_absolute_ratio_deviation"]
        compiler_note = trusted_feedback["compiler_review"]
        resources = trusted_feedback["compiler_resources"]
        context["constraints"].append(
            "The external experiment tested an independently written, human-reviewed implementation, NOT the exact Muse-generated source; retain both distinct SHA-256 values and do not transfer correctness or speedup claims to the generated source.")
        if "research_question" in trusted_feedback:
            context["constraints"].extend([
                f"Research question: {trusted_feedback['research_question']}",
                f"The trusted two-slab K loop passed six correctness cases, but median paired screening was {paired_ratio:.5f}x with all eight pairs favoring baseline; A/A median absolute ratio deviation was {aa_deviation:.2%}. No speedup was confirmed.",
                f"Existing Matmul Inspector evidence: {compiler_note}. Structured resources: {resources}. Dynamic profiler counts: {trusted_feedback['dynamic_profile']}.",
                "Propose ONE substantially different compute-focused mechanism to reduce loop overhead or improve operand scheduling while considering register pressure, shared-memory allocation, and occupancy. Avoid repeating two-slab unrolling or cosmetic source changes.",
                "In hypothesis and expected_mechanism, explain why this change could alter generated instructions, memory movement, or resource use under the fixed launcher.",
                "In reasoning_from_feedback, distinguish measured observations from speculative explanations. Occupancy and the cause of the slower screening were not measured; label any such explanation a hypothesis.",
            ])
        else:
            context["constraints"].extend([
                f"Mask hoisting, one K-offset expression, and tl.range(num_stages=3) passed correctness in the trusted implementation but did not establish a performance improvement: median paired screening was {paired_ratio:.5f}x while A/A median absolute ratio deviation was {aa_deviation:.2%}.",
                f"Compiler evidence: {compiler_note}. Structured resources: {resources}. Treat matching instructions and resources as evidence against a substantive change from that idea, not as proof of identical runtime behavior.",
                "Propose one different compute-focused Triton kernel. Prefer substantive changes to data movement, operand reuse, or matrix computation over cosmetic source rewrites or repeating mask hoisting and the explicit tl.range hint.",
                "Explain which generated instructions, memory transactions, pipeline structure, registers, or shared-memory use could change and why the compiler should retain that difference.",
                "Use the trusted experiment's correctness, paired screening uncertainty, A/A, and compiler evidence in reasoning_from_feedback without calling the screening ratio a confirmed gain.",
            ])
    return (
        "Propose one genuinely modified Triton GEMM implementation for review only. "
        "Return ONLY JSON with exactly kernel_source, hypothesis, "
        "expected_mechanism, reasoning_from_feedback. "
        "kernel_source must contain only the decorated matmul_kernel function.\n"
        + json.dumps(context, sort_keys=True)
    )


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def propose_to_directory(client, prompt, root, *, feedback_from=None,
                         review_from=None, trusted_feedback=None, gpu=None,
                         hardware=None):
    """One accepted candidate at most; rejected responses remain in the audit."""
    folder = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    audit_path = folder / "proposal_audit.jsonl"
    metadata = {"status": "proposal_incomplete", "created_utc":
                datetime.now(timezone.utc).isoformat(), "model": MODEL,
                "shape_mnk": list(SHAPE), "gpu": gpu, "hardware": hardware,
                "baseline_sha256": source_sha256(baseline_source()),
                "launcher_sha256": source_sha256(inspect.getsource(
                    benchmark_proposal.launch)),
                "numeric_config": optimize.BASELINE,
                "measurement_method": "python_launch_cuda_event",
                "feedback_from": str(feedback_from) if feedback_from else None,
                "review_from": str(review_from) if review_from else None,
                "trusted_feedback_path": (trusted_feedback or {}).get("feedback_path"),
                "trusted_feedback_sha256": (trusted_feedback or {}).get("feedback_sha256"),
                "trusted_experiment_sha256":
                    (trusted_feedback or {}).get("experiment_sha256"),
                "trusted_feedback_provenance":
                    (trusted_feedback or {}).get("provenance"),
                "trusted_implementation_sha256":
                    (trusted_feedback or {}).get("trusted_implementation_sha256")}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    current_prompt = prompt
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            response = client.responses.create(model=MODEL, input=current_prompt)
        except Exception as exc:
            append_jsonl(audit_path, {"attempt": attempt, "prompt": current_prompt,
                                      "error_type": type(exc).__name__,
                                      "api_wall_s": time.perf_counter() - started})
            metadata["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            break
        raw = response.output_text
        audit = {"attempt": attempt, "prompt": current_prompt, "raw_response": raw,
                 "response_id": getattr(response, "id", None),
                 "api_wall_s": time.perf_counter() - started}
        try:
            obj, findings = parse_response(
                raw, require_compute_change=review_from is not None)
        except (ValueError, SyntaxError, TypeError) as exc:
            audit["rejection"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            append_jsonl(audit_path, audit)
            current_prompt += f"\nPrevious proposal rejected: {exc}. Return a valid replacement."
            continue
        append_jsonl(audit_path, audit)
        (folder / "candidate.py").write_text(obj["kernel_source"])
        metadata.update({"status": "awaiting_human_review",
                         "candidate_sha256": findings["source_sha256"],
                         "hypothesis": obj["hypothesis"],
                         "expected_mechanism": obj["expected_mechanism"],
                         "reasoning_from_feedback": obj["reasoning_from_feedback"],
                         "static_findings": findings})
        break
    if metadata["status"] == "proposal_incomplete" and "error" not in metadata:
        metadata["error"] = "no source passed the review filter in three attempts"
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder, metadata


def prepare_request(prompt, root, *, review, gpu, hardware, trusted_feedback=None):
    """Persist one independent request for inspection without a model call."""
    folder = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "prompt.txt").write_text(prompt)
    (folder / "metadata.json").write_text(json.dumps({
        "status": "prepared_only", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "shape_mnk": list(SHAPE), "gpu": gpu, "hardware": hardware,
        "review_from": review["candidate_dir"],
        "reviewed_candidate_sha256": review["candidate_sha256"],
        "trusted_feedback_path": (trusted_feedback or {}).get("feedback_path"),
        "trusted_feedback_sha256": (trusted_feedback or {}).get("feedback_sha256"),
        "trusted_experiment_sha256":
            (trusted_feedback or {}).get("experiment_sha256"),
        "trusted_feedback_provenance": (trusted_feedback or {}).get("provenance"),
        "trusted_implementation_sha256":
            (trusted_feedback or {}).get("trusted_implementation_sha256"),
        "baseline_sha256": source_sha256(baseline_source()),
        "launcher_sha256": source_sha256(inspect.getsource(benchmark_proposal.launch)),
        "prompt_sha256": source_sha256(prompt),
        "api_call_made": False,
    }, indent=2) + "\n")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("propose", nargs="?", default="propose", choices=["propose"])
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--structural-history-dir", type=Path,
                        default=Path("evaluations/structural-loop"))
    parser.add_argument("--feedback-from", type=Path,
                        help="evaluated prior generated candidate directory")
    parser.add_argument("--feedback-evaluation", type=Path,
                        help="evaluation.json for --feedback-from")
    parser.add_argument("--review-from", type=Path,
                        help="unevaluated prior candidate for an independent compute-focused proposal")
    parser.add_argument("--trusted-feedback", type=Path,
                        help="feedback.json from a separate human-reviewed implementation")
    parser.add_argument("--gpu", help="GPU name if unavailable to this process")
    parser.add_argument("--prepare-only", action="store_true",
                        help="save the independent request locally without a Muse API call")
    parser.add_argument("--request-from-muse", action="store_true",
                        help="Explicitly permit the proposal-only Muse API request")
    args = parser.parse_args(argv)
    if args.prepare_only and (args.request_from_muse or not args.review_from):
        parser.error("--prepare-only requires --review-from and excludes --request-from-muse")
    if args.review_from and (args.feedback_from or args.feedback_evaluation):
        parser.error("--review-from is independent of evaluated --feedback-from")
    if args.trusted_feedback and not args.review_from:
        parser.error("--trusted-feedback requires --review-from for the original Muse source")
    if not (args.request_from_muse or args.prepare_only):
        parser.error("no Muse call made; pass --request-from-muse after reviewing this path")
    if args.request_from_muse:
        load_dotenv()
        if not os.getenv("MODEL_API_KEY"):
            parser.error("MODEL_API_KEY missing")
    if bool(args.feedback_from) != bool(args.feedback_evaluation):
        parser.error("--feedback-from and --feedback-evaluation must be provided together")
    feedback = (reviewed_feedback(args.feedback_from, args.feedback_evaluation)
                if args.feedback_from else None)
    review = reviewed_unevaluated_proposal(args.review_from) if args.review_from else None
    trusted_feedback = (reviewed_trusted_feedback(args.trusted_feedback, review)
                        if args.trusted_feedback else None)
    history = structural_history(args.structural_history_dir)
    gpu = (args.gpu or (feedback or {}).get("evaluation", {}).get("gpu")
           or (json.loads((args.review_from / "metadata.json").read_text()).get("gpu")
               if args.review_from else None))
    # Proposal generation can run without a GPU device exposed to this process.
    hardware = (json.loads((args.review_from / "metadata.json").read_text())
                .get("hardware", {}) if args.review_from else {})
    if not (gpu and hardware):
        try:
            import torch
            if torch.cuda.is_available():
                detected_gpu = torch.cuda.get_device_name(0)
                if not gpu:
                    gpu = detected_gpu
                if gpu == detected_gpu:
                    props = torch.cuda.get_device_properties(0)
                    hardware = {"compute_capability": list(torch.cuda.get_device_capability(0)),
                                "multiprocessor_count": props.multi_processor_count,
                                "total_memory_bytes": props.total_memory}
        except Exception:
            pass
    if not gpu:
        parser.error("GPU unavailable; provide --gpu with the target device name")
    prompt = build_prompt(gpu, hardware, history, feedback, review, trusted_feedback)
    if args.prepare_only:
        folder = prepare_request(prompt, args.out_dir, review=review,
                                 gpu=gpu, hardware=hardware,
                                 trusted_feedback=trusted_feedback)
        print(f"Saved proposal request without API call: {folder}")
        return 0
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    folder, metadata = propose_to_directory(
        client, prompt, args.out_dir, feedback_from=args.feedback_from,
        review_from=args.review_from, trusted_feedback=trusted_feedback,
        gpu=gpu, hardware=hardware)
    print(f"Saved proposal artifacts: {folder}; status: {metadata['status']}")
    return 0 if metadata["status"] == "awaiting_human_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
