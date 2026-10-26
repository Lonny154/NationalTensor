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
import random
import statistics

from dotenv import load_dotenv
from openai import OpenAI
import torch
import triton.testing
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
            "numerical correctness, not performance improvement. No actual "
            "memory traffic, register counts, or occupancy counters were collected."
        ),
        "method": (
            "Triton do_bench, warmup=100ms, rep=200ms per sample, "
            "baseline/candidate order randomized each round, medians reported; "
            "numerical correctness checked against torch.mm. "
            "These are screening results, not strong cross-session confirmation."
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


def compare(a, b, base_out, candidate_out, config, rounds, seed):
    samples = {"baseline": [], "candidate": []}
    jobs = {"baseline": (BASELINE, base_out),
            "candidate": (config, candidate_out)}
    rng = random.Random(seed)
    for _ in range(rounds):
        order = ["baseline", "candidate"]
        rng.shuffle(order)
        for name in order:
            cfg, out = jobs[name]
            ms = triton.testing.do_bench(
                lambda cfg=cfg, out=out: launch(a, b, out, cfg),
                warmup=100, rep=200,
            )
            samples[name].append(float(ms))
    b_ms = statistics.median(samples["baseline"])
    c_ms = statistics.median(samples["candidate"])
    return {"baseline_ms": b_ms, "candidate_ms": c_ms,
            "speedup": b_ms / c_ms, "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--show-history", action="store_true",
                        help="Inspect compatible past experiments without calling Muse")
    args = parser.parse_args()
    if args.candidates < 1 or args.repeats < 2:
        parser.error("Candidates >= 1 and repeats >= 2 required")
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
        "method": {"repeats": args.repeats, "warmup_ms": 100,
                   "rep_ms": 200, "randomized_pair_order": True},
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
                                       args.repeats, seed=42 + i)
            record["status"] = "PASS"
            t = record["timing"]
            print(f"PASS: {t['candidate_ms']:.6f} ms vs "
                  f"{t['baseline_ms']:.6f} ms ({t['speedup']:.4f}x)")
        except Exception as exc:
            record["status"] = "FAIL"
            record["error"] = f"{type(exc).__name__}: {exc}"[:1500]
            print(f"Rejected: {record['error']}")
        with results_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        session.append(record)
        history.append({"run": run_dir.name, **record})

    # Do not choose a losing candidate as the screening winner.
    passing = [r for r in session if r["status"] == "PASS"
               and r["timing"]["speedup"] > 1.0]
    winner = max(passing, key=lambda r: r["timing"]["speedup"]) if passing else None
    summary = {**identity,
               "method": {"repeats": args.repeats, "warmup_ms": 100,
                          "rep_ms": 200, "randomized_pair_order": True},
               "historical_experiments_loaded": len(previous),
               "experiments_attempted": len(session), "screening_winner": winner}
    if winner:
        print("\nRechecking screening winner...")
        try:
            out = correct_output(a, b, reference, winner["config"])
            confirmation = compare(
                a, b, base_out, out, winner["config"],
                max(12, args.repeats), seed=2026)
            summary["confirmation"] = confirmation
            summary["confirmation_improved"] = confirmation["speedup"] > 1.0
            print(f"Confirmation: {confirmation['speedup']:.4f}x")
            if not summary["confirmation_improved"]:
                print("No confirmed improvement over baseline this session")
        except Exception as exc:
            summary["confirmation_error"] = str(exc)
    else:
        print("\nNo candidate beat baseline in screening; baseline remains best")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to {run_dir}")
    print("Small speedups are provisional until independently reproduced.")


if __name__ == "__main__":
    main()
