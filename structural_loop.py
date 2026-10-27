"""Small Muse feedback loop over reviewed structural GEMM strategies.

Run records live under evaluations/structural-loop, separate from optimize.py's
single-shape numeric and schedule history. Model output is JSON data only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import torch

import benchmark_core
import grouped_matmul
import optimize
import reference_precision
import split_k_matmul


MODE = "autonomous_structural_v1"
SHAPES = {
    "fc1": (512, 3072, 768),
    "small_m": (128, 3072, 768),
    "square": (1024, 1024, 1024),
}
STRATEGIES = ("direct_2d", "grouped_m4", "split_k")
METHOD = optimize.benchmark_method(8, 7, 1.02)
BASELINE = {**optimize.BASELINE, "schedule": "direct_2d"}
REFERENCE_POLICY = "fp16_torch_mm_without_reduced_precision_reduction"
MAX_PROPOSAL_ATTEMPTS = 3


def proposal_key(config):
    return (config["schedule"], config.get("split_factor"))


def valid_config(config):
    if not isinstance(config, dict):
        return False
    schedule = config.get("schedule")
    expected = set(optimize.BASELINE) | {"schedule"}
    if schedule == "split_k":
        expected.add("split_factor")
    if set(config) != expected or schedule not in STRATEGIES:
        return False
    if any(type(config[name]) is not int or config[name] != value
           for name, value in optimize.BASELINE.items()):
        return False
    return (schedule != "split_k" or
            (type(config["split_factor"]) is int
             and config["split_factor"] in split_k_matmul.SPLIT_FACTORS))


def parse_proposal(raw, tried):
    """Validate exact JSON fields; no generated source or numeric tuning."""
    obj = json.loads(raw)
    fields = {"strategy", "parameters", "hypothesis", "expected_mechanism",
              "reasoning_from_history"}
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ValueError("proposal fields must exactly match the structural schema")
    strategy, parameters = obj["strategy"], obj["parameters"]
    if type(strategy) is not str or strategy not in STRATEGIES:
        raise ValueError("unknown structural strategy")
    if strategy == "split_k":
        if (not isinstance(parameters, dict) or set(parameters) != {"split_factor"}
                or type(parameters["split_factor"]) is not int
                or parameters["split_factor"] not in split_k_matmul.SPLIT_FACTORS):
            raise ValueError("Split-K requires split_factor 2 or 4")
        config = {**optimize.BASELINE, "schedule": strategy,
                  "split_factor": parameters["split_factor"]}
    else:
        if parameters != {}:
            raise ValueError("direct and grouped strategies take no parameters")
        config = {**optimize.BASELINE, "schedule": strategy}
    if proposal_key(config) in tried:
        raise ValueError("strategy and parameters already tested in this run")
    for name in ("hypothesis", "expected_mechanism", "reasoning_from_history"):
        if type(obj[name]) is not str or not obj[name].strip() or len(obj[name]) > 2000:
            raise ValueError(f"{name} must be a nonempty string of at most 2000 characters")
    return config, {name: obj[name].strip() for name in
                    ("hypothesis", "expected_mechanism", "reasoning_from_history")}


def source_fingerprints():
    """Exact code identity for all launch paths and the measurement adapter."""
    def source_hash(*functions):
        source = "\n".join(inspect.getsource(fn.fn if hasattr(fn, "fn") else fn)
                           for fn in functions)
        return hashlib.sha256(source.encode()).hexdigest()

    return {
        "direct": optimize.kernel_signature(),
        "grouped": source_hash(grouped_matmul.grouped_matmul_kernel,
                               grouped_matmul.launch_grouped,
                               grouped_matmul.launch_scheduled),
        "split_k": source_hash(split_k_matmul.partial_kernel,
                               split_k_matmul.reduction_kernel,
                               split_k_matmul.allocate_workspace,
                               split_k_matmul.launch_split_k),
        "loop_dispatch": source_hash(launch_strategy, checked_output,
                                     compare_config, evaluate_gpu,
                                     optimize.compare),
        "loop_policy": source_hash(parse_proposal, build_prompt, propose,
                                   run_experiment_loop),
        "benchmark_core_file": hashlib.sha256(
            Path(benchmark_core.__file__).read_bytes()).hexdigest(),
        "reference_precision_file": hashlib.sha256(
            Path(reference_precision.__file__).read_bytes()).hexdigest(),
        "confirmation_rule": source_hash(optimize.confirmation_decision),
    }


def run_identity(shape, gpu, input_seed):
    if tuple(shape) not in SHAPES.values():
        raise ValueError("unsupported workload shape")
    if type(input_seed) is not int or input_seed < 0:
        raise ValueError("input seed must be a nonnegative integer")
    return {
        "search_mode": MODE, "shape_mnk": list(shape), "gpu": gpu,
        "dtype": "fp16", "input_seed": input_seed,
        "numeric_config": dict(optimize.BASELINE), "baseline": dict(BASELINE),
        "strategies": {"direct_2d": {}, "grouped_m4": {},
                       "split_k": {"split_factor": list(split_k_matmul.SPLIT_FACTORS)}},
        "source_sha256": source_fingerprints(),
        "measurement_method": METHOD,
        "reference_precision_policy": REFERENCE_POLICY,
    }


def compatible(metadata, expected):
    """Require exact workload, implementation, and timing identity."""
    return all(metadata.get(name) == expected[name] for name in expected)


def load_compatible_history(root, expected):
    rows = []
    if not root.exists():
        return rows
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name == "confirmations":
            continue
        try:
            metadata = json.loads((folder / "metadata.json").read_text())
            if not compatible(metadata, expected):
                continue
            with (folder / "experiments.jsonl").open() as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if (isinstance(record, dict)
                            and valid_config(record.get("config"))
                            and record.get("measurement_method") ==
                            expected["measurement_method"]
                            and record.get("source_sha256") ==
                            expected["source_sha256"]):
                        rows.append({"run": folder.name, **record})
        except (OSError, ValueError, TypeError):
            continue
    return rows


def feedback_record(record):
    timing = record.get("timing") or {}
    stats = timing.get("statistics") or {}
    return {
        "shape_mnk": record["shape_mnk"], "config": record["config"],
        "hypothesis": record.get("hypothesis"),
        "expected_mechanism": record.get("expected_mechanism"),
        "reasoning_from_history": record.get("reasoning_from_history"),
        "status": record["status"],
        "correctness": record.get("correctness"),
        "failure_stage": record.get("failure_stage"),
        "error": record.get("error"),
        "median_paired_speedup": timing.get("median_paired_speedup"),
        "baseline_ms": timing.get("baseline_ms"),
        "candidate_ms": timing.get("candidate_ms"),
        "paired_ratio_mad": stats.get("paired_ratios", {}).get("mad"),
        "paired_ratio_cv": stats.get("paired_ratios", {}).get("cv"),
        "measurement_mode": timing.get("measurement_mode"),
        "optimization_status": record.get("optimization_status"),
    }


def build_prompt(identity, current, prior, aa, tried):
    context = {
        "workload": {"shape_mnk": identity["shape_mnk"], "dtype": "fp16",
                     "input_seed": identity["input_seed"], "gpu": identity["gpu"]},
        "fixed_numeric_config": identity["numeric_config"],
        "reviewed_strategies": identity["strategies"],
        "tested_this_run": [{"strategy": strategy, "split_factor": factor}
                            for strategy, factor in sorted(tried,
                                                           key=lambda item: str(item))],
        "prior_compatible_experiments": [feedback_record(row) for row in prior[-12:]],
        "current_run_experiments": [feedback_record(row) for row in current],
        "aa_median_absolute_ratio_deviation": (
            aa.get("aa_variability", {}).get("median_absolute_ratio_deviation")
            if aa else None),
        "measurement_method": identity["measurement_method"],
        "measurement_warning": (
            "Eager CUDA-event blocks can include idle gaps. Day 2 A/A captures "
            "showed about 3.7–4.5% median absolute ratio deviations. Screening "
            "is provisional; the confirmation heuristic is not statistical significance. "
            "Do not invent profiler data or claim model learning."
        ),
    }
    return (
        "Propose ONE untested, reviewed structural GEMM experiment for this workload. "
        "Use prior outcomes and failures to explain your choice. Do not provide code. "
        "Return ONLY JSON with exactly: strategy (direct_2d, grouped_m4, split_k), "
        "parameters ({} for direct/grouped or {\"split_factor\": 2 or 4}), "
        "hypothesis, expected_mechanism, reasoning_from_history. "
        "All three explanation fields must be nonempty strings.\n"
        + json.dumps(context, sort_keys=True)
    )


def propose(client, identity, current, prior, aa, tried, audit):
    prompt = build_prompt(identity, current, prior, aa, tried)
    for attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
        import time
        started = time.perf_counter()
        try:
            response = client.responses.create(model=optimize.MUSE_MODEL, input=prompt)
        except Exception as exc:
            audit({"attempt": attempt, "prompt": prompt, "model": optimize.MUSE_MODEL,
                   "error_type": type(exc).__name__,
                   "api_wall_s": time.perf_counter() - started})
            raise
        usage = getattr(response, "usage", None)
        raw = response.output_text
        audit({"attempt": attempt, "prompt": prompt, "response": raw,
               "response_id": getattr(response, "id", None),
               "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
               "model": optimize.MUSE_MODEL,
               "api_wall_s": time.perf_counter() - started})
        try:
            return parse_proposal(raw, tried)
        except (ValueError, TypeError) as exc:
            prompt += f"\nPrevious proposal rejected: {exc}. Choose a valid untested option."
    raise RuntimeError("Muse did not return a valid structural proposal")


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def run_experiment_loop(client, identity, prior, aa, count, evaluate, save_record,
                        audit):
    """Inject evaluation and model client for deterministic CPU-only tests."""
    tried = {("direct_2d", None)}
    current = []
    for number in range(1, count + 1):
        if len(tried) == 4:
            break
        config, explanation = propose(client, identity, current, prior, aa, tried, audit)
        tried.add(proposal_key(config))
        record = {
            "number": number, "created_utc": datetime.now(timezone.utc).isoformat(),
            "shape_mnk": identity["shape_mnk"], "gpu": identity["gpu"],
            "input_seed": identity["input_seed"],
            "numeric_config": identity["numeric_config"],
            "schedule": config["schedule"], "config": config,
            **explanation, "source_sha256": identity["source_sha256"],
            "measurement_method": identity["measurement_method"],
            "reference_precision_policy": identity["reference_precision_policy"],
        }
        try:
            record.update(evaluate(config, number))
        except Exception as exc:
            record.update({"status": "FAIL", "correctness": "FAIL",
                           "failure_stage": "correctness_or_compilation",
                           "error": f"{type(exc).__name__}: {str(exc)[:1000]}"})
        if record["status"] == "PASS":
            ratio = record["timing"]["median_paired_speedup"]
            record["optimization_status"] = (
                "promising_unconfirmed" if ratio > 1.0 else "screening_no_gain")
        else:
            record["optimization_status"] = "failed"
        save_record(record)
        current.append(record)
    return current


def launch_strategy(a, b, out, config, workspaces):
    if not valid_config(config):
        raise ValueError("unsupported structural configuration")
    if config["schedule"] == "split_k":
        workspace = workspaces.get(id(out))
        if workspace is None:
            raise ValueError("Split-K requires a preallocated workspace for this output")
        split_k_matmul.launch_split_k(a, b, out, config, workspace)
    else:
        grouped_matmul.launch_scheduled(a, b, out, config)


def checked_output(a, b, reference, config):
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    workspaces = {}
    if config["schedule"] == "split_k":
        workspaces[id(out)] = split_k_matmul.allocate_workspace(
            a, b, config["split_factor"])
    launch_strategy(a, b, out, config, workspaces)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
    return out, workspaces


def compare_config(a, b, base_out, candidate_out, config, workspaces, *,
                   rounds, iterations, order_seed):
    """The timed candidate call contains both Split-K kernels and no allocation."""
    timing = optimize.compare(
        a, b, base_out, candidate_out, config, rounds, order_seed,
        warmups=optimize.SCREEN_WARMUPS if iterations == optimize.SCREEN_ITERATIONS
        else optimize.CONFIRM_WARMUPS,
        iterations=iterations,
        launcher=lambda x, y, out, cfg: launch_strategy(x, y, out, cfg, workspaces),
        baseline_config=BASELINE,
    )
    timing["baseline_schedule"] = "direct_2d"
    timing["candidate_schedule"] = config["schedule"]
    timing["candidate_launches"] = (["partial_gemm", "fp32_reduction"]
                                    if config["schedule"] == "split_k"
                                    else ["gemm"])
    return timing


def prepare_workload(shape, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16, generator=generator)
    reference, policy = reference_precision.fp16_mm_reference(a, b)
    base_out, _ = checked_output(a, b, reference, BASELINE)
    return a, b, reference, base_out, policy


def measure_aa(a, b):
    return benchmark_core.run_aa_calibration(
        lambda x, y, out, cfg: launch_strategy(x, y, out, cfg, {}),
        a, b, BASELINE, rounds=20, warmups=10, iterations=100, seed=42)


def evaluate_gpu(a, b, reference, base_out, config, number):
    try:
        out, workspaces = checked_output(a, b, reference, config)
    except Exception as exc:
        return {"status": "FAIL", "correctness": "FAIL",
                "failure_stage": "correctness_or_compilation",
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}
    try:
        timing = compare_config(a, b, base_out, out, config, workspaces,
                                rounds=8, iterations=100, order_seed=42 + number)
    except Exception as exc:
        return {"status": "FAIL", "correctness": "PASS",
                "failure_stage": "screening",
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}"}
    return {"status": "PASS", "correctness": "PASS", "timing": timing}


def _new_run_dir(root):
    path = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_search(shape_name, count, seed, root):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    identity = run_identity(SHAPES[shape_name], torch.cuda.get_device_name(0), seed)
    prior = load_compatible_history(root, identity)
    run_dir = _new_run_dir(root)
    metadata = {**identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": optimize.MUSE_MODEL, "requested_experiments": count,
                "compatible_prior_records": len(prior)}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {"status": "incomplete", "experiments_attempted": 0,
               "confirmed_winner": None, "confirmation": "separate process required"}
    try:
        a, b, reference, base_out, policy = prepare_workload(SHAPES[shape_name], seed)
        metadata["reference_precision"] = policy
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        aa = measure_aa(a, b)
        (run_dir / "aa.json").write_text(json.dumps(aa, indent=2) + "\n")
        client = optimize.OpenAI(base_url="https://api.meta.ai/v1",
                                 api_key=os.environ["MODEL_API_KEY"])
        records = run_experiment_loop(
            client, identity, prior, aa, count,
            lambda config, number: evaluate_gpu(
                a, b, reference, base_out, config, number),
            lambda record: append_jsonl(run_dir / "experiments.jsonl", record),
            lambda entry: append_jsonl(run_dir / "proposal_audit.jsonl", entry))
        passing = [row for row in records if row["status"] == "PASS"
                   and row["timing"]["median_paired_speedup"] > 1.0]
        best = max(passing, key=lambda row: row["timing"]["median_paired_speedup"],
                   default=None)
        summary.update({"status": "complete", "experiments_attempted": len(records),
                        "best_screening_experiment": best["number"] if best else None,
                        "best_screening_speedup": (
                            best["timing"]["median_paired_speedup"] if best else None),
                        "interpretation": "Screening only; no confirmed winner"})
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        raise
    finally:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved structural experiment run: {run_dir}")
    return run_dir


def confirm_search_run(run_dir, root):
    """Independent process capture of the best provisional candidate."""
    metadata = json.loads((run_dir / "metadata.json").read_text())
    shape = tuple(metadata["shape_mnk"])
    expected = run_identity(shape, torch.cuda.get_device_name(0), metadata["input_seed"])
    if not compatible(metadata, expected):
        raise ValueError("run is incompatible with current workload, source, or method")
    rows = [json.loads(line) for line in (run_dir / "experiments.jsonl").read_text().splitlines()]
    passing = [row for row in rows if valid_config(row.get("config"))
               and row.get("status") == "PASS"
               and row.get("source_sha256") == expected["source_sha256"]
               and row.get("measurement_method") == expected["measurement_method"]
               and row["timing"]["median_paired_speedup"] > 1.0]
    if not passing:
        raise ValueError("run has no compatible candidate above baseline in screening")
    winner = max(passing, key=lambda row: row["timing"]["median_paired_speedup"])
    out_dir = _new_run_dir(root / "confirmations")
    result = {"status": "incomplete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_run": str(run_dir),
              "source_experiment": winner["number"], "shape_mnk": list(shape),
              "gpu": expected["gpu"], "input_seed": expected["input_seed"],
              "numeric_config": expected["numeric_config"],
              "schedule": winner["config"]["schedule"],
              "config": winner["config"], "source_sha256": expected["source_sha256"],
              "measurement_method": expected["measurement_method"]}
    try:
        a, b, reference, base_out, policy = prepare_workload(shape, expected["input_seed"])
        out, workspaces = checked_output(a, b, reference, winner["config"])
        result["reference_precision"] = policy
        result["aa"] = measure_aa(a, b)
        timing = compare_config(a, b, base_out, out, winner["config"], workspaces,
                                rounds=14, iterations=200, order_seed=2026)
        decision = optimize.confirmation_decision(timing)
        result.update({"timing": timing, "decision": decision,
                       "status": "heuristically_confirmed" if decision["confirmed"]
                       else "unconfirmed"})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        raise
    finally:
        (out_dir / "confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved independent confirmation: {out_dir}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, default="fc1")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("evaluations/structural-loop"))
    parser.add_argument("--confirm-run", type=Path,
                        help="Confirm a previous screening winner in this separate process")
    args = parser.parse_args(argv)
    if not 1 <= args.candidates <= 3 or args.seed < 0:
        parser.error("candidates must be 1–3 and seed must be nonnegative")
    load_dotenv()
    if not args.confirm_run and not os.getenv("MODEL_API_KEY"):
        parser.error("MODEL_API_KEY missing")
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.confirm_run:
        confirm_search_run(args.confirm_run, args.out_dir)
    else:
        run_search(args.shape, args.candidates, args.seed, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
