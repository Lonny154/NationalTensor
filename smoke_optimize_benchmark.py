"""GPU smoke test of optimize.py's paired screening and confirmation paths.

Run from the project root: .venv/bin/python smoke_optimize_benchmark.py
No Muse client or persistent experiment history is used.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import triton

import optimize


AB_CONFIG = {**optimize.BASELINE, "num_warps": 8}


def print_timing(label, timing):
    print(f"  {label}: {len(timing['paired_rounds'])} balanced rounds, "
          f"median paired ratio {timing['median_paired_speedup']:.6f}x")
    print(f"    baseline median {timing['baseline_ms']:.6f} ms, "
          f"CV {timing['statistics']['baseline_ms']['cv']:.3%}; "
          f"candidate median {timing['candidate_ms']:.6f} ms, "
          f"CV {timing['statistics']['candidate_ms']['cv']:.3%}")
    for row in timing["paired_rounds"]:
        print(f"    {row['round']:2d} "
              f"{'→'.join(row['order']):22s} "
              f"baseline {row['baseline_ms']:.6f} ms  "
              f"candidate {row['candidate_ms']:.6f} ms  "
              f"ratio {row['baseline_over_candidate']:.6f}x")


def measure_case(name, config, a, b, reference, base_out):
    candidate_out = optimize.correct_output(a, b, reference, config)
    print(f"\n{name} correctness: PASS")
    screening = optimize.compare(
        a, b, base_out, candidate_out, config,
        optimize.balanced_rounds(8), seed=43,
    )
    print_timing("screening", screening)

    # Measure confirmation for both comparisons, including A/A if it did not
    # win screening, so this diagnostic exercises the same confirmation path.
    candidate_out = optimize.correct_output(a, b, reference, config)
    confirmation = optimize.compare(
        a, b, base_out, candidate_out, config,
        2 * optimize.DEFAULT_CONFIRMATION_PAIRS, seed=2026,
        warmups=optimize.CONFIRM_WARMUPS,
        iterations=optimize.CONFIRM_ITERATIONS,
    )
    print_timing("confirmation", confirmation)

    record = {"number": 1, "config": config, "status": "PASS", "timing": screening}
    selected_in_screening = optimize.select_screening_winner([record]) is not None
    summary = {"confirmed_winner": None}
    decision = optimize.apply_confirmation(
        summary, record, confirmation,
        optimize.DEFAULT_CONFIRMATION_PAIRS,
        optimize.DEFAULT_MIN_CONFIRMATION_SPEEDUP,
    )
    print(f"  screening selected: {selected_in_screening}; "
          f"confirmation: {summary['confirmation_status']}; "
          f"verification pairs favoring candidate: "
          f"{decision['pairs_favoring_candidate']}/{decision['required_pairs']}; "
          f"median verification-pair speedup: "
          f"{decision['median_verification_pair_speedup']:.6f}x")
    return {
        "config": config,
        "correctness": "PASS",
        "screening_selected": selected_in_screening,
        "confirmation_measured_for_diagnostic": True,
        "screening": screening,
        "confirmation": confirmation,
        "confirmation_decision": decision,
        "confirmation_status": summary["confirmation_status"],
        "confirmed_winner": summary["confirmed_winner"] is not None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON artifact path")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    gpu = torch.cuda.get_device_name(0)
    output = args.output or Path("smoke_results") / (
        "phase2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    torch.manual_seed(42)
    a = torch.randn((optimize.M, optimize.K), device="cuda", dtype=torch.float16)
    b = torch.randn((optimize.K, optimize.N), device="cuda", dtype=torch.float16)
    reference = torch.mm(a, b)
    base_out = optimize.correct_output(a, b, reference, optimize.BASELINE)
    print(f"GPU: {gpu}; baseline correctness: PASS")

    artifact = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "workload": optimize.run_identity(gpu),
        "method": optimize.benchmark_method(
            optimize.balanced_rounds(8),
            optimize.DEFAULT_CONFIRMATION_PAIRS,
            optimize.DEFAULT_MIN_CONFIRMATION_SPEEDUP,
        ),
        "comparisons": {},
    }
    for name, config in (("aa", optimize.BASELINE), ("ab", AB_CONFIG)):
        artifact["comparisons"][name] = measure_case(
            name.upper(), config, a, b, reference, base_out)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")
    print(f"\nSaved separate smoke artifact: {output}")
    if artifact["comparisons"]["aa"]["confirmed_winner"]:
        raise SystemExit("FAIL: A/A was falsely confirmed as an improvement")
    print("A/A false-confirmation check: PASS")


if __name__ == "__main__":
    main()
