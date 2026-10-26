"""National Tensor: Muse-guided Triton tuning with persistent experiment history.

Run from the national-tensor project folder alongside benchmark_proposal.py.
Prior runs are stored in runs/<timestamp>/{metadata,experiments,summary}.json*.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics

from dotenv import load_dotenv
from openai import OpenAI
import torch
from benchmark_core import run_paired
from benchmark_proposal import launch, matmul_kernel

M, N, K = 512, 3072, 768
BASELINE = dict(block_m=64, block_n=128, block_k=32,
                num_warps=4, num_stages=3)
ALLOWED = {
    "block_m": [32, 64, 128],
    "block_n": [64, 128, 256],
    "block_k": [32, 64],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
}
FIELDS = tuple(ALLOWED)
SCREEN_WARMUPS = 10
SCREEN_ITERATIONS = 100
CONFIRM_WARMUPS = 10
CONFIRM_ITERATIONS = 200
DEFAULT_CONFIRMATION_PAIRS = 7
DEFAULT_MIN_CONFIRMATION_SPEEDUP = 1.02


def balanced_rounds(requested):
    """Honor the requested minimum while giving each side equal first turns."""
    return requested + requested % 2


def benchmark_method(screen_rounds, confirmation_pairs, min_speedup):
    return {
        "schema_version": 2,
        "measurement_mode": "python_launch_cuda_event",
        "timing_boundary": "CUDA-event block divided by launch count",
        "screening": {"rounds": screen_rounds, "warmups": SCREEN_WARMUPS,
                      "iterations_per_block": SCREEN_ITERATIONS},
        "confirmation": {
            "verification_pairs": confirmation_pairs,
            "rounds": 2 * confirmation_pairs,
            "warmups": CONFIRM_WARMUPS,
            "iterations_per_block": CONFIRM_ITERATIONS,
            "all_pairs_favor_candidate": True,
            "minimum_median_pair_speedup": min_speedup,
            "rule_type": "heuristic; not a statistical significance test",
        },
    }


def key(config):
    return tuple(config[name] for name in FIELDS)


def valid_config(config):
    return isinstance(config, dict) and all(
        type(config.get(name)) is int and config[name] in choices
        for name, choices in ALLOWED.items()
    )


def kernel_signature():
    """Fingerprint the launcher and JIT source to avoid mixing changed kernels."""
    source = inspect.getsource(launch) + "\n" + inspect.getsource(matmul_kernel.fn)
    return hashlib.sha256(source.encode()).hexdigest()


def run_identity(gpu):
    return {
        "gpu": gpu,
        "shape": [M, N, K],
        "dtype": "fp16",
        "baseline": BASELINE,
        "kernel_sha256": kernel_signature(),
    }


def matches_identity(meta, expected):
    """Allow legacy runs without a hash, but label them as unverified."""
    if any(meta.get(k) != expected[k] for k in ("gpu", "shape", "baseline")):
        return False, False
    if meta.get("dtype", "fp16") != expected["dtype"]:
        return False, False
    old_hash = meta.get("kernel_sha256")
    if old_hash is not None and old_hash != expected["kernel_sha256"]:
        return False, False
    return True, old_hash is None


def load_history(runs_dir, expected):
    """Read compatible prior experiments, including failures and raw samples."""
    rows = []
    skipped = 0
    legacy = 0
    if not runs_dir.exists():
        return rows, skipped, legacy

    for folder in sorted(runs_dir.iterdir()):
        if not folder.is_dir():
            continue
        # New runs have metadata at creation time, so interrupted runs still count.
        # Old runs have a summary.json instead.
        meta_file = folder / "metadata.json"
        if not meta_file.exists():
            meta_file = folder / "summary.json"
        experiments_file = folder / "experiments.jsonl"
        if not (meta_file.exists() and experiments_file.exists()):
            continue
        try:
            meta = json.loads(meta_file.read_text())
            compatible, is_legacy = matches_identity(meta, expected)
        except (ValueError, OSError, TypeError):
            skipped += 1
            continue
        if not compatible:
            skipped += 1
            continue
        if is_legacy:
            legacy += 1
        summary = {}
        try:
            summary_file = folder / "summary.json"
            if summary_file.exists():
                summary = json.loads(summary_file.read_text())
        except (ValueError, OSError, TypeError):
            pass
        screened = summary.get("screening_winner") or {}
        try:
            with experiments_file.open() as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if (not valid_config(record.get("config")) or
                                record.get("status") not in {"PASS", "FAIL"}):
                            continue
                        # Each record has source + session identity. Keep numerical
                        # measurements scoped to the run that produced them.
                        rows.append({
                            "run": folder.name,
                            "number": record.get("number", line_num),
                            "config": {n: record["config"][n] for n in FIELDS},
                            "hypothesis": str(record.get("hypothesis", ""))[:900],
                            "status": record["status"],
                            "timing": record.get("timing"),
                            "measurement_mode": (
                                (record.get("timing") or {}).get("measurement_mode")
                                or "legacy_triton_do_bench"
                            ),
                            "optimization_status": (
                                screened.get("optimization_status",
                                             summary.get("confirmation_status", "screening_only"))
                                if screened.get("number") == record.get("number")
                                else "screening_only"
                            ),
                            "error": str(record.get("error", ""))[:500],
                            "legacy_unverified_kernel": is_legacy,
                        })
                    except (ValueError, TypeError, AttributeError, KeyError):
                        # A partial JSONL line can be left after an interrupted write.
                        continue
        except OSError:
            skipped += 1
    return rows, skipped, legacy


def model_history(history, max_records=40):
    """Keep context bounded while retaining all past keys for deduplication."""
    concise = []
    for record in history[-max_records:]:
        t = record.get("timing") or {}
        concise.append({
            "run": record.get("run"),
            "config": record["config"],
            "status": record["status"],
            "hypothesis": record.get("hypothesis", "")[:350],
            "baseline_ms": t.get("baseline_ms"),
            "candidate_ms": t.get("candidate_ms"),
            "speedup": t.get("speedup"),
            "median_paired_speedup": t.get("median_paired_speedup"),
            "measurement_mode": record.get(
                "measurement_mode", t.get("measurement_mode", "legacy_triton_do_bench")),
            "optimization_status": record.get("optimization_status", "screening_only"),
            "error": record.get("error", "")[:200],
            "legacy_unverified_kernel": record.get("legacy_unverified_kernel", False),
        })
    return concise


def propose(client, history, tried):
    context = {
        "gpu": torch.cuda.get_device_name(0),
        "workload": {"M": M, "N": N, "K": K, "dtype": "fp16"},
        "baseline": BASELINE,
        "allowed_values": ALLOWED,
        "recent_experiments": model_history(history),
        "all_previously_tested_configurations": [
            dict(zip(FIELDS, entry)) for entry in sorted(tried)
        ],
        "history_note": (
            "Measurements are per-run and may vary with GPU clocks, load, and "
            "benchmark conditions. Never interpret past-vs-current absolute "
            "latencies as controlled comparisons. Historical runs without a "
            "kernel hash are marked legacy_unverified_kernel. A PASS means "
            "numerical correctness and a completed screening measurement, not "
            "a confirmed performance improvement. Legacy do_bench timings and "
            "new paired CUDA-event timings use different methods. No actual "
            "memory traffic, register counts, or occupancy counters were collected."
        ),
        "method": (
            "Preallocated outputs; compile and warm both sides before timing; "
            "Python-launch CUDA-event blocks divided by launch count; balanced "
            "baseline/candidate order with every order recorded. Numerical "
            "correctness checked against torch.mm. Screening speedups are "
            "provisional; confirmation uses balanced two-round verification blocks."
        ),
    }
    prompt = (
        "You are an evidence-driven GPU optimization research agent. "
        "Propose exactly ONE configuration that has NOT been tested in the "
        "configuration list. Consider successes and failures across sessions. "
        "State a falsifiable hypothesis, not fabricated profiler measurements. "
        "Return ONLY JSON with integer fields block_m, block_n, block_k, "
        "num_warps, num_stages and a string hypothesis.\n"
        + json.dumps(context)
    )
    for attempt in range(3):
        response = client.responses.create(model="muse-spark-1.3", input=prompt)
        raw = response.output_text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1]).strip()
        try:
            obj = json.loads(raw)
            cfg = {name: obj[name] for name in FIELDS}
            if not valid_config(cfg):
                raise ValueError("Invalid/out-of-range configuration")
            if key(cfg) in tried:
                raise ValueError("Configuration already tested in a previous run")
            hypothesis = obj.get("hypothesis")
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise ValueError("Missing hypothesis")
            return cfg, hypothesis.strip()
        except (ValueError, KeyError, TypeError) as exc:
            print(f"  Proposal rejected ({attempt + 1}/3): {exc}")
            prompt += f"\nPrevious response invalid: {exc}. Choose a new valid configuration."
    raise RuntimeError("Muse did not provide a valid new configuration")


def correct_output(a, b, reference, config):
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    launch(a, b, out, config)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
    return out


def compare(a, b, base_out, candidate_out, config, rounds, seed, *,
            warmups=SCREEN_WARMUPS, iterations=SCREEN_ITERATIONS,
            timer=None, synchronize=None):
    """Keep old scalar fields while persisting the full paired evidence."""
    outputs = iter((base_out, candidate_out))
    paired = run_paired(
        launch, a, b, BASELINE, config, rounds=rounds, seed=seed,
        warmups=warmups, iterations=iterations,
        output_factory=lambda _a, _b: next(outputs),
        timer=timer, synchronize=synchronize,
    )
    return {
        "benchmark_schema_version": 2,
        "measurement_mode": "python_launch_cuda_event",
        "baseline_ms": paired["baseline_ms"]["median"],
        "candidate_ms": paired["candidate_ms"]["median"],
        "speedup": paired["ratio_of_medians"],
        "median_paired_speedup": paired["paired_ratios"]["median"],
        "samples_ms": paired["samples_ms"],
        "paired_rounds": paired["rounds"],
        "statistics": {name: paired[name] for name in
                       ("baseline_ms", "candidate_ms", "paired_ratios")},
        "method": {**paired["method"], "measurement_mode": "python_launch"},
    }


def confirmation_decision(timing, required_pairs=DEFAULT_CONFIRMATION_PAIRS,
                          min_speedup=DEFAULT_MIN_CONFIRMATION_SPEEDUP):
    """Apply a conservative descriptive rule to balanced two-round blocks."""
    rows = timing["paired_rounds"]
    if len(rows) != 2 * required_pairs:
        raise ValueError("confirmation must have exactly two rounds per verification pair")
    ratios = []
    for first, second in zip(rows[::2], rows[1::2]):
        if first["order"] != list(reversed(second["order"])):
            raise ValueError("verification pair is not order-balanced")
        baseline = statistics.median((first["baseline_ms"], second["baseline_ms"]))
        candidate = statistics.median((first["candidate_ms"], second["candidate_ms"]))
        ratios.append(baseline / candidate)
    median_ratio = statistics.median(ratios)
    return {
        "confirmed": all(ratio > 1.0 for ratio in ratios)
                     and median_ratio >= min_speedup,
        "verification_pair_speedups": ratios,
        "median_verification_pair_speedup": median_ratio,
        "pairs_favoring_candidate": sum(ratio > 1.0 for ratio in ratios),
        "required_pairs": required_pairs,
        "minimum_median_speedup": min_speedup,
        "rule": "All balanced verification pairs favor candidate and median >= threshold",
        "interpretation": "heuristic, not a statistical significance test",
    }


def select_screening_winner(session):
    passing = [record for record in session if record["status"] == "PASS"
               and record["timing"]["median_paired_speedup"] > 1.0]
    return max(passing, key=lambda record: record["timing"]["median_paired_speedup"],
               default=None)


def screening_report(record):
    timing = record["timing"]
    return (f"PASS: candidate {timing['candidate_ms']:.6f} ms vs "
            f"baseline {timing['baseline_ms']:.6f} ms; "
            f"ratio of medians (baseline/candidate) {timing['speedup']:.4f}x; "
            f"median paired speedup {timing['median_paired_speedup']:.4f}x "
            "(screening selection metric)")


def screening_winner_report(winner):
    return (f"Screening winner: experiment {winner['number']} "
            f"{winner['config']}; median paired speedup "
            f"{winner['timing']['median_paired_speedup']:.4f}x "
            "(screening selection metric)")


def apply_confirmation(summary, winner, timing, required_pairs, min_speedup):
    decision = confirmation_decision(timing, required_pairs, min_speedup)
    summary["confirmation"] = timing
    summary["confirmation_decision"] = decision
    summary["confirmation_improved"] = decision["confirmed"]
    summary["confirmation_status"] = (
        "confirmed" if decision["confirmed"] else "unconfirmed")
    summary["screening_winner"] = {
        **winner, "optimization_status": summary["confirmation_status"]}
    if decision["confirmed"]:
        summary["confirmed_winner"] = summary["screening_winner"]
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--confirmation-pairs", type=int,
                        default=DEFAULT_CONFIRMATION_PAIRS,
                        help="Balanced two-round verification blocks (default: 7)")
    parser.add_argument("--min-confirmation-speedup", type=float,
                        default=DEFAULT_MIN_CONFIRMATION_SPEEDUP,
                        help="Minimum median verification-pair speedup (default: 1.02)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--show-history", action="store_true",
                        help="Inspect compatible past experiments without calling Muse")
    args = parser.parse_args()
    if args.candidates < 1 or args.repeats < 2:
        parser.error("Candidates >= 1 and repeats >= 2 required")
    if args.confirmation_pairs < 1 or not (1.0 < args.min_confirmation_speedup < float("inf")):
        parser.error("Confirmation pairs >= 1 and finite minimum speedup > 1 required")
    screen_rounds = balanced_rounds(args.repeats)
    method = benchmark_method(screen_rounds, args.confirmation_pairs,
                              args.min_confirmation_speedup)
    load_dotenv()
    if not args.show_history and not os.getenv("MODEL_API_KEY"):
        parser.error("MODEL_API_KEY missing; add it to .env")
    if not torch.cuda.is_available():
        parser.error("CUDA not available")

    gpu = torch.cuda.get_device_name(0)
    identity = run_identity(gpu)
    runs_dir = Path(args.runs_dir)
    previous, skipped, legacy = load_history(runs_dir, identity)
    tried = {key(BASELINE)} | {key(r["config"]) for r in previous}
    if args.show_history:
        print(f"GPU: {gpu}")
        print(f"Compatible past experiments: {len(previous)}")
        print(f"Unique previously tested configurations: {len(tried)-1}")
        print(f"Skipped runs: {skipped}; legacy runs: {legacy}")
        for item in previous[-10:]:
            t = item.get("timing") or {}
            speedup = t.get("speedup")
            result = f"{speedup:.4f}x" if isinstance(speedup, (int, float)) else item["error"][:70]
            print(f"  {item['run']}: {item['config']}  {item['status']}  {result}")
        return

    run_dir = runs_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        **identity,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {**method, "requested_screening_rounds": args.repeats},
    }, indent=2))

    print(f"GPU: {gpu}")
    print(f"Results directory: {run_dir}")
    print(f"Loaded {len(previous)} prior experiments; {len(tried)-1} unique prior configs")
    if skipped:
        print(f"Skipped {skipped} incompatible/corrupt historical runs")
    if legacy:
        print(f"NOTE: {legacy} legacy runs lack kernel fingerprint; verify their comparability")

    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=os.environ["MODEL_API_KEY"])

    torch.manual_seed(42)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    reference = torch.mm(a, b)
    base_out = correct_output(a, b, reference, BASELINE)
    print("Baseline correctness: PASS")

    session = []
    history = previous.copy()
    results_path = run_dir / "experiments.jsonl"
    for i in range(1, args.candidates + 1):
        print(f"\n--- Experiment {i}/{args.candidates} ---")
        try:
            cfg, hypothesis = propose(client=client, history=history, tried=tried)
        except Exception as exc:
            print(f"Muse proposal failed: {exc}")
            break
        tried.add(key(cfg))
        print(f"Configuration: {cfg}")
        print(f"Hypothesis: {hypothesis}")
        record = {"number": i, "config": cfg, "hypothesis": hypothesis}
        try:
            out = correct_output(a, b, reference, cfg)
            record["timing"] = compare(a, b, base_out, out, cfg,
                                       screen_rounds, seed=42 + i)
            record["status"] = "PASS"
            record["optimization_status"] = "screening_only"
            print(screening_report(record))
        except Exception as exc:
            record["status"] = "FAIL"
            record["error"] = f"{type(exc).__name__}: {exc}"[:1500]
            print(f"Rejected: {record['error']}")
        with results_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        session.append(record)
        history.append({"run": run_dir.name, **record})

    # Do not choose a losing candidate as the screening winner.
    winner = select_screening_winner(session)
    summary = {**identity,
               "method": {**method, "requested_screening_rounds": args.repeats},
               "historical_experiments_loaded": len(previous),
               "experiments_attempted": len(session), "screening_winner": winner,
               "confirmation_status": "not_run", "confirmed_winner": None}
    if winner:
        print(f"\n{screening_winner_report(winner)}")
        print("Rechecking screening winner...")
        try:
            out = correct_output(a, b, reference, winner["config"])
            confirmation = compare(
                a, b, base_out, out, winner["config"],
                2 * args.confirmation_pairs, seed=2026,
                warmups=CONFIRM_WARMUPS, iterations=CONFIRM_ITERATIONS)
            decision = apply_confirmation(
                summary, winner, confirmation, args.confirmation_pairs,
                args.min_confirmation_speedup)
            print(f"Confirmation median verification-pair speedup: "
                  f"{decision['median_verification_pair_speedup']:.4f}x; "
                  f"{decision['pairs_favoring_candidate']}/"
                  f"{decision['required_pairs']} pairs favor candidate")
            if not decision["confirmed"]:
                print("No confirmed improvement over baseline this session")
        except Exception as exc:
            summary["confirmation_error"] = str(exc)
            summary["confirmation_status"] = "error"
            summary["screening_winner"] = {
                **winner, "optimization_status": "unconfirmed"}
    else:
        print("\nNo candidate beat baseline in screening; baseline remains best")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to {run_dir}")
    print("Confirmation is a heuristic, not a statistical significance test; "
          "reproduce promising results independently.")


if __name__ == "__main__":
    main()
