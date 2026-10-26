"""National Tensor: five Muse-guided GEMM configuration experiments.

Requires benchmark_proposal.py (provides launch(a, b, out, config)),
MODEL_API_KEY in .env, and CUDA-enabled PyTorch + Triton.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics

from dotenv import load_dotenv
from openai import OpenAI
import torch
import triton.testing
from benchmark_proposal import launch

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


def key(config):
    return tuple(config[name] for name in ALLOWED)


def propose(client, history, tried):
    context = {
        "gpu": torch.cuda.get_device_name(0),
        "workload": {"M": M, "N": N, "K": K, "dtype": "fp16"},
        "baseline": BASELINE,
        "allowed_values": ALLOWED,
        "prior_experiments": history,
        "already_tried": [dict(zip(ALLOWED, k)) for k in sorted(tried)],
        "method": "CUDA benchmarking via Triton do_bench; each call includes "
                  "warmup, baseline/candidate order randomized per round; "
                  "correctness checked with torch.testing.assert_close. "
                  "Differences may be noise; no profiling counters available.",
    }
    prompt = (
        "Act as a GPU optimization research agent. Suggest ONE distinct "
        "Triton launch configuration worth testing next. Use previous results "
        "and failures to inform your hypothesis. Don't invent measured data. "
        "Return ONLY a JSON object with integer fields block_m, block_n, "
        "block_k, num_warps, num_stages, and a string hypothesis.\n"
        + json.dumps(context)
    )
    for attempt in range(3):
        response = client.responses.create(model="muse-spark-1.3", input=prompt)
        raw = response.output_text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1]).strip()
        try:
            obj = json.loads(raw)
            cfg = {name: obj[name] for name in ALLOWED}
            for name, options in ALLOWED.items():
                if type(cfg[name]) is not int or cfg[name] not in options:
                    raise ValueError(f"Invalid {name}: {cfg[name]}")
            if key(cfg) in tried:
                raise ValueError("Configuration already tested")
            reason = obj.get("hypothesis", "")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Missing hypothesis")
            return cfg, reason
        except (ValueError, KeyError, TypeError) as exc:
            print(f"  Proposal rejected ({attempt + 1}/3): {exc}")
            prompt += f"\nYour last proposal was invalid: {exc}. Try another."
    raise RuntimeError("Muse did not provide a valid new proposal")


def correct_output(a, b, reference, config):
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    launch(a, b, out, config)   # Compilation occurs on first launch.
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
    args = parser.parse_args()
    if args.candidates < 1 or args.repeats < 2:
        parser.error("Candidates >= 1 and repeats >= 2 required")
    load_dotenv()
    if not os.getenv("MODEL_API_KEY"):
        parser.error("MODEL_API_KEY missing; add it to .env")
    if not torch.cuda.is_available():
        parser.error("CUDA not available")

    client = OpenAI(base_url="https://api.meta.ai/v1",
                    api_key=os.environ["MODEL_API_KEY"])
    run_dir = Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Results directory: {run_dir}")
    torch.manual_seed(42)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    reference = torch.mm(a, b)
    base_out = correct_output(a, b, reference, BASELINE)
    print("Baseline correctness: PASS")

    history = []
    tried = {key(BASELINE)}
    results_path = run_dir / "experiments.jsonl"
    for i in range(1, args.candidates + 1):
        print(f"\n--- Experiment {i}/{args.candidates} ---")
        try:
            cfg, hypothesis = propose(client, history, tried)
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
        history.append(record)

    passing = [
        r for r in history
        if r["status"] == "PASS"
        and r["timing"]["speedup"] > 1.0
    ]
    winner = max(passing, key=lambda r: r["timing"]["speedup"]) if passing else None
    summary = {"gpu": torch.cuda.get_device_name(0),
               "shape": [M, N, K], "baseline": BASELINE,
               "method": {"repeats": args.repeats, "warmup_ms": 100,
                          "rep_ms": 200, "randomized_pair_order": True},
               "experiments_attempted": len(history), "screening_winner": winner}
    if winner:
        print("\nRechecking screening winner...")
        try:
            out = correct_output(a, b, reference, winner["config"])
            summary["confirmation"] = compare(
                a, b, base_out, out, winner["config"],
                max(12, args.repeats), seed=2026)
            print(f"Confirmation: {summary['confirmation']['speedup']:.4f}x")
        except Exception as exc:
            summary["confirmation_error"] = str(exc)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to {run_dir}")
    print("Small speedups are provisional until independently reproduced.")


if __name__ == "__main__":
    main()