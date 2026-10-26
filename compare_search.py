"""Fresh-history, counterbalanced Muse versus seeded random search pilot.

The CPU-only --dry-run prints the complete plan without touching CUDA or Muse.
Full runs write only to evaluations/, never to optimize.py's runs/ history.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import optimize


STRATEGIES = ("muse", "random")
MODEL = optimize.MUSE_MODEL
INPUT_SEED = 42
AA_ROUNDS = 20
AA_WARMUPS = 10
AA_ITERATIONS = 100


def search_space():
    """Enumerate the exact allowed space, excluding the baseline."""
    return [dict(zip(optimize.FIELDS, values))
            for values in product(*(optimize.ALLOWED[name] for name in optimize.FIELDS))
            if tuple(values) != optimize.key(optimize.BASELINE)]


def derived_seed(master, trial, purpose):
    digest = hashlib.sha256(f"{master}:{trial}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_plan(trials=4, budget_per_arm=4, seed=20261026):
    space = search_space()
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be an integer >= 1")
    if type(budget_per_arm) is not int or not 1 <= budget_per_arm <= len(space):
        raise ValueError(f"budget_per_arm must be between 1 and {len(space)}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    planned = []
    for index in range(trials):
        number = index + 1
        random_seed = derived_seed(seed, number, "random_configs")
        random_configs = random.Random(random_seed).sample(space, budget_per_arm)
        slots = []
        for slot_index in range(budget_per_arm):
            first = STRATEGIES[(index + slot_index) % 2]
            second = STRATEGIES[1 - (index + slot_index) % 2]
            slots.append({
                "slot": slot_index + 1,
                "order": [first, second],
                "screen_order_seed": derived_seed(
                    seed, number, f"screen_{slot_index + 1}"),
            })
        planned.append({
            "trial": number,
            "random_seed": random_seed,
            "random_configs": random_configs,
            "slots": slots,
            "confirmation_order": list(STRATEGIES if index % 2 == 0
                                       else reversed(STRATEGIES)),
            "confirmation_order_seed": derived_seed(seed, number, "confirmation"),
            "aa_seeds": {stage: derived_seed(seed, number, f"aa_{stage}")
                         for stage in ("start", "end")},
        })
    return {
        "master_seed": seed,
        "input_seed": INPUT_SEED,
        "trials": trials,
        "budget_per_arm": budget_per_arm,
        "available_nonbaseline_configs": len(space),
        "history_policy": "fresh per trial; Muse sees only its own trial results",
        "trial_plans": planned,
    }


def print_plan(plan):
    print(f"DRY RUN: {plan['trials']} paired trials, "
          f"{plan['budget_per_arm']} evaluated configurations per arm; "
          f"{plan['available_nonbaseline_configs']} non-baseline choices")
    print("Fresh histories; no prior runs loaded; no CUDA or Muse calls")
    for trial in plan["trial_plans"]:
        print(f"Trial {trial['trial']}: random seed {trial['random_seed']}")
        for slot in trial["slots"]:
            config = trial["random_configs"][slot["slot"] - 1]
            print(f"  slot {slot['slot']}: {' -> '.join(slot['order'])}; "
                  f"screen order seed {slot['screen_order_seed']}; "
                  f"random config {config}")
        print(f"  confirmation order: {' -> '.join(trial['confirmation_order'])}; "
              f"order seed {trial['confirmation_order_seed']}; "
              f"A/A seeds {trial['aa_seeds']}")


def timed_event_interval_s(timing):
    """Sum CUDA-event intervals from core or optimizer results, not device-busy time."""
    iterations = timing["method"]["iterations_per_side_per_round"]
    rows = timing["paired_rounds"] if "paired_rounds" in timing else timing["rounds"]
    return sum(row[f"{side}_ms"] * iterations / 1000
               for row in rows
               for side in ("baseline", "candidate"))


def _screening_progress(records):
    ratios = [record["timing"]["median_paired_speedup"]
              for record in records if record["status"] == "PASS"]
    best_observed = max(ratios, default=None)
    return {"best_observed_paired_speedup": best_observed,
            "best_vs_baseline": max(1.0, best_observed or 0.0)}


def _failure(exc):
    return f"{type(exc).__name__}: {str(exc)[:500]}"


def run_trial(plan, propose_fn, evaluate_fn, confirm_fn, aa_fn, emit,
              clock=time.perf_counter):
    """Run one trial with injected GPU/API functions for CPU-only tests."""
    started = clock()
    budget = len(plan["slots"])
    trial = {
        "trial": plan["trial"], "status": "running", "incomplete_reason": None,
        "strategy_execution_order": [slot["order"] for slot in plan["slots"]],
        "actual_strategy_execution_order": [],
        "confirmation_order": plan["confirmation_order"],
        "actual_confirmation_order": [],
        "random_seed": plan["random_seed"], "aa_seeds": plan["aa_seeds"],
        "api_calls": 0, "proposal_retries": 0, "api_wall_s": 0.0, "aa": {},
        "arms": {arm: {"candidates": [], "screening_curve": [],
                       "confirmation": None} for arm in STRATEGIES},
    }
    histories = {arm: [] for arm in STRATEGIES}
    tried = {arm: {optimize.key(optimize.BASELINE)} for arm in STRATEGIES}

    def calibrate(stage):
        try:
            result = aa_fn(stage, plan["aa_seeds"][stage])
        except Exception as exc:
            result = {"status": "error", "error": _failure(exc)}
        trial["aa"][stage] = result
        emit({"kind": "aa", "trial": plan["trial"], "stage": stage,
              "result": result})
        return result["status"] == "PASS"

    if not calibrate("start"):
        trial["incomplete_reason"] = "start A/A calibration failed"

    if trial["incomplete_reason"] is None:
        for slot in plan["slots"]:
            slot_number = slot["slot"]
            for arm in slot["order"]:
                attempts = []

                def audit(entry):
                    event = {"kind": "api_attempt", "trial": plan["trial"],
                             "slot": slot_number, "strategy": arm, **entry,
                             "prompt_sha256": hashlib.sha256(
                                 entry["prompt"].encode()).hexdigest()}
                    attempts.append(event)
                    emit(event)

                try:
                    if arm == "muse":
                        config, hypothesis = propose_fn(histories["muse"],
                                                        tried["muse"], audit)
                    else:
                        config = plan["random_configs"][slot_number - 1]
                        hypothesis = "seeded random sample"
                    if not optimize.valid_config(config) or optimize.key(config) in tried[arm]:
                        raise ValueError("proposal is invalid or repeated within its arm")
                except Exception as exc:
                    trial["api_calls"] += len(attempts)
                    trial["proposal_retries"] += max(0, len(attempts) - 1)
                    trial["api_wall_s"] += sum(entry["api_wall_s"] for entry in attempts)
                    trial["incomplete_reason"] = (
                        f"{arm} proposal failed at slot {slot_number}: {type(exc).__name__}")
                    emit({"kind": "proposal_failure", "trial": plan["trial"],
                          "slot": slot_number, "strategy": arm,
                          "error_type": type(exc).__name__})
                    break

                trial["api_calls"] += len(attempts)
                trial["proposal_retries"] += max(0, len(attempts) - 1)
                trial["api_wall_s"] += sum(entry["api_wall_s"] for entry in attempts)
                tried[arm].add(optimize.key(config))
                record = {"number": slot_number, "strategy": arm,
                          "config": config, "hypothesis": hypothesis,
                          "screen_order_seed": slot["screen_order_seed"],
                          "proposal_attempts": len(attempts)}
                try:
                    record.update(evaluate_fn(config, slot["screen_order_seed"]))
                except Exception as exc:
                    record.update({"status": "FAIL", "failure_stage": "evaluation",
                                   "error": _failure(exc), "gpu_work_wall_s": 0.0})
                trial["arms"][arm]["candidates"].append(record)
                trial["actual_strategy_execution_order"].append(
                    {"slot": slot_number, "strategy": arm})
                if arm == "muse":
                    histories["muse"].append({"run": f"trial_{plan['trial']}", **record})
                emit({"kind": "candidate", "trial": plan["trial"],
                      "slot": slot_number, "strategy": arm, "record": record})
            if trial["incomplete_reason"] is not None:
                break
            for arm in STRATEGIES:
                trial["arms"][arm]["screening_curve"].append(
                    _screening_progress(trial["arms"][arm]["candidates"]))

    if trial["incomplete_reason"] is None:
        for arm in plan["confirmation_order"]:
            winner = optimize.select_screening_winner(trial["arms"][arm]["candidates"])
            if winner is None:
                result = {"status": "not_run", "reason": "no screening winner above 1.0x",
                          "best_confirmed_speedup": 1.0, "gpu_work_wall_s": 0.0}
            else:
                try:
                    result = confirm_fn(winner, plan["confirmation_order_seed"])
                except Exception as exc:
                    result = {"status": "error", "error": _failure(exc),
                              "best_confirmed_speedup": 1.0,
                              "gpu_work_wall_s": 0.0}
                result["winner_config"] = winner["config"]
                result["screening_winner_paired_speedup"] = (
                    winner["timing"]["median_paired_speedup"])
            trial["arms"][arm]["confirmation"] = result
            trial["actual_confirmation_order"].append(arm)
            emit({"kind": "confirmation", "trial": plan["trial"],
                  "strategy": arm, "winner_config": winner["config"] if winner else None,
                  "result": result})
            if result["status"] == "error":
                trial["incomplete_reason"] = f"{arm} confirmation failed"
        if not calibrate("end"):
            trial["incomplete_reason"] = trial["incomplete_reason"] or "end A/A calibration failed"

    for arm in STRATEGIES:
        details = trial["arms"][arm]
        records = details["candidates"]
        confirmation = details["confirmation"] or {}
        details["valid_configurations_evaluated"] = len(records)
        details["correctness_failures"] = sum(
            record.get("failure_stage") == "correctness" for record in records)
        details["screened_configurations"] = sum(
            record["status"] == "PASS" for record in records)
        details["best_screening_speedup"] = (
            details["screening_curve"][-1]["best_vs_baseline"]
            if details["screening_curve"] else 1.0)
        details["best_observed_screening_speedup"] = (
            details["screening_curve"][-1]["best_observed_paired_speedup"]
            if details["screening_curve"] else None)
        details["best_confirmed_speedup"] = confirmation.get(
            "best_confirmed_speedup", 1.0)
        details["gpu_work_wall_s"] = (
            sum(record.get("gpu_work_wall_s", 0.0) for record in records)
            + confirmation.get("gpu_work_wall_s", 0.0))

    complete_budget = all(trial["arms"][arm]["valid_configurations_evaluated"] == budget
                          for arm in STRATEGIES)
    if not complete_budget and trial["incomplete_reason"] is None:
        trial["incomplete_reason"] = "unequal or incomplete candidate budget"
    trial["status"] = "complete" if trial["incomplete_reason"] is None else "incomplete"
    trial["total_wall_s"] = clock() - started
    emit({"kind": "trial_end", "trial": plan["trial"], "status": trial["status"],
          "incomplete_reason": trial["incomplete_reason"]})
    return trial


def aggregate_trials(trials, budget_per_arm):
    complete = [trial for trial in trials if trial["status"] == "complete" and all(
        trial["arms"][arm]["valid_configurations_evaluated"] == budget_per_arm
        for arm in STRATEGIES)]
    paired = [{"trial": trial["trial"],
               "muse": trial["arms"]["muse"]["best_confirmed_speedup"],
               "random": trial["arms"]["random"]["best_confirmed_speedup"],
               "muse_minus_random": (
                   trial["arms"]["muse"]["best_confirmed_speedup"]
                   - trial["arms"]["random"]["best_confirmed_speedup"])}
              for trial in complete]
    return {
        "primary_outcome": "best confirmed speedup after budget; 1.0x if none",
        "completed_paired_trials": len(complete),
        "incomplete_trials": [{"trial": trial["trial"],
                               "reason": trial["incomplete_reason"]}
                              for trial in trials if trial not in complete],
        "paired_trial_outcomes": paired,
        "mean_paired_difference": (statistics.mean(row["muse_minus_random"]
                                                   for row in paired) if paired else None),
        "best_screening_curves": {
            arm: [[trial["arms"][arm]["screening_curve"][index]["best_vs_baseline"]
                   for index in range(budget_per_arm)] for trial in complete]
            for arm in STRATEGIES},
        "best_observed_screening_curves": {
            arm: [[trial["arms"][arm]["screening_curve"][index]
                   ["best_observed_paired_speedup"]
                   for index in range(budget_per_arm)] for trial in complete]
            for arm in STRATEGIES},
        "api_calls": sum(trial["api_calls"] for trial in trials),
        "proposal_retries": sum(trial["proposal_retries"] for trial in trials),
        "api_wall_s": sum(trial["api_wall_s"] for trial in trials),
        "correctness_failures": {
            arm: sum(trial["arms"][arm]["correctness_failures"] for trial in trials)
            for arm in STRATEGIES},
        "valid_configurations_evaluated": {
            arm: sum(trial["arms"][arm]["valid_configurations_evaluated"]
                     for trial in trials) for arm in STRATEGIES},
        "gpu_work_wall_s": {
            arm: sum(trial["arms"][arm]["gpu_work_wall_s"] for trial in trials)
            for arm in STRATEGIES},
        "screening_attempts": {
            arm: sum(record["status"] == "PASS"
                     or record.get("failure_stage") == "screening"
                     for trial in trials for record in trial["arms"][arm]["candidates"])
            for arm in STRATEGIES},
        "screening_completed": {
            arm: sum(record["status"] == "PASS"
                     for trial in trials for record in trial["arms"][arm]["candidates"])
            for arm in STRATEGIES},
        "confirmation_attempts": {
            arm: sum((trial["arms"][arm]["confirmation"] or {}).get(
                "winner_config") is not None for trial in trials)
            for arm in STRATEGIES},
        "confirmation_completed": {
            arm: sum((trial["arms"][arm]["confirmation"] or {}).get("timing") is not None
                     for trial in trials) for arm in STRATEGIES},
        "screening_wall_s": {
            arm: sum(record.get("screening_wall_s", 0.0)
                     for trial in trials for record in trial["arms"][arm]["candidates"])
            for arm in STRATEGIES},
        "confirmation_wall_s": {
            arm: sum((trial["arms"][arm]["confirmation"] or {}).get(
                "confirmation_wall_s", 0.0) for trial in trials)
            for arm in STRATEGIES},
        "timed_cuda_event_interval_s": {
            arm: sum(record.get("timed_cuda_event_interval_s", 0.0)
                     for trial in trials for record in trial["arms"][arm]["candidates"])
                 + sum((trial["arms"][arm]["confirmation"] or {}).get(
                     "timed_cuda_event_interval_s", 0.0) for trial in trials)
            for arm in STRATEGIES},
        "aa_gpu_work_wall_s": sum(result.get("gpu_work_wall_s", 0.0)
                                  for trial in trials for result in trial["aa"].values()),
        "total_wall_s": sum(trial["total_wall_s"] for trial in trials),
    }


def measure_candidate(a, b, reference, base_out, config, order_seed):
    """Use the normal correctness gate and screening benchmark."""
    started = time.perf_counter()
    try:
        out = optimize.correct_output(a, b, reference, config)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {"status": "FAIL", "failure_stage": "correctness",
                "error": _failure(exc), "correctness_wall_s": elapsed,
                "gpu_work_wall_s": elapsed}
    correctness_wall_s = time.perf_counter() - started
    started = time.perf_counter()
    try:
        timing = optimize.compare(
            a, b, base_out, out, config, optimize.balanced_rounds(8), order_seed)
    except Exception as exc:
        screening_wall_s = time.perf_counter() - started
        return {"status": "FAIL", "failure_stage": "screening",
                "error": _failure(exc), "correctness_wall_s": correctness_wall_s,
                "screening_wall_s": screening_wall_s,
                "gpu_work_wall_s": correctness_wall_s + screening_wall_s}
    screening_wall_s = time.perf_counter() - started
    return {"status": "PASS", "correctness_status": "PASS", "timing": timing,
            "correctness_wall_s": correctness_wall_s,
            "screening_wall_s": screening_wall_s,
            "gpu_work_wall_s": correctness_wall_s + screening_wall_s,
            "timed_cuda_event_interval_s": timed_event_interval_s(timing)}


def measure_confirmation(a, b, reference, base_out, winner, order_seed):
    """Apply optimize.py's winner confirmation, preserving failures."""
    started = time.perf_counter()
    try:
        out = optimize.correct_output(a, b, reference, winner["config"])
        correctness_wall_s = time.perf_counter() - started
        confirmation_started = time.perf_counter()
        timing = optimize.compare(
            a, b, base_out, out, winner["config"],
            2 * optimize.DEFAULT_CONFIRMATION_PAIRS, order_seed,
            warmups=optimize.CONFIRM_WARMUPS,
            iterations=optimize.CONFIRM_ITERATIONS)
        decision_summary = {"confirmed_winner": None}
        decision = optimize.apply_confirmation(
            decision_summary, winner, timing,
            optimize.DEFAULT_CONFIRMATION_PAIRS,
            optimize.DEFAULT_MIN_CONFIRMATION_SPEEDUP)
    except Exception as exc:
        return {"status": "error", "error": _failure(exc),
                "best_confirmed_speedup": 1.0,
                "gpu_work_wall_s": time.perf_counter() - started}
    return {"status": decision_summary["confirmation_status"],
            "timing": timing, "decision": decision,
            "correctness_wall_s": correctness_wall_s,
            "confirmation_wall_s": time.perf_counter() - confirmation_started,
            "gpu_work_wall_s": time.perf_counter() - started,
            "timed_cuda_event_interval_s": timed_event_interval_s(timing),
            "best_confirmed_speedup": (
                decision["median_verification_pair_speedup"]
                if decision["confirmed"] else 1.0)}


def measure_aa(a, b, seed):
    from benchmark_core import run_aa_calibration

    started = time.perf_counter()
    timing = run_aa_calibration(
        optimize.launch, a, b, optimize.BASELINE,
        rounds=AA_ROUNDS, warmups=AA_WARMUPS,
        iterations=AA_ITERATIONS, seed=seed)
    return {"status": "PASS", "timing": timing,
            "gpu_work_wall_s": time.perf_counter() - started,
            "timed_cuda_event_interval_s": timed_event_interval_s(timing)}


def _write_json_new(path, data):
    with path.open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def run_full(plan, out_dir):
    """GPU/API path. Dry-run returns before this function is entered."""
    run_started = time.perf_counter()
    optimize.load_dotenv()
    if not os.getenv("MODEL_API_KEY"):
        raise SystemExit("MODEL_API_KEY missing; full evaluation not started")
    import torch
    import triton

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; full evaluation not started")

    gpu = torch.cuda.get_device_name(0)
    run_dir = Path(out_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "trials").mkdir()
    events_path = run_dir / "events.jsonl"
    method = optimize.benchmark_method(
        optimize.balanced_rounds(8), optimize.DEFAULT_CONFIRMATION_PAIRS,
        optimize.DEFAULT_MIN_CONFIRMATION_SPEEDUP)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "fresh-history Muse versus seeded random; paired trials",
        "model": MODEL,
        "plan": plan,
        "search_space": {"allowed": optimize.ALLOWED,
                         "baseline_excluded": optimize.BASELINE},
        "workload": optimize.run_identity(gpu),
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("compare_search.py", "optimize.py", "benchmark_core.py")},
        "benchmark_method": method,
        "aa_method": {"mode": "python_launch", "rounds": AA_ROUNDS,
                      "warmups": AA_WARMUPS, "iterations": AA_ITERATIONS,
                      "stages": ["start", "end"]},
        "environment": {"python": sys.version, "torch": str(torch.__version__),
                        "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
                        "gpu": gpu,
                        "compute_capability": list(torch.cuda.get_device_capability(0))},
        "api_key_recorded": False,
    }
    _write_json_new(run_dir / "manifest.json", manifest)

    def emit(event):
        with events_path.open("a") as file:
            file.write(json.dumps(event) + "\n")

    client = optimize.OpenAI(base_url="https://api.meta.ai/v1",
                             api_key=os.environ["MODEL_API_KEY"])
    torch.manual_seed(INPUT_SEED)
    a = torch.randn((optimize.M, optimize.K), device="cuda", dtype=torch.float16)
    b = torch.randn((optimize.K, optimize.N), device="cuda", dtype=torch.float16)
    reference = torch.mm(a, b)
    try:
        base_out = optimize.correct_output(a, b, reference, optimize.BASELINE)
    except Exception as exc:
        emit({"kind": "baseline_correctness", "status": "FAIL",
              "error": _failure(exc)})
        _write_json_new(run_dir / "summary.json", {
            "status": "incomplete", "reason": "baseline correctness failed",
            "completed_paired_trials": 0, "run_total_wall_s": time.perf_counter() - run_started})
        raise SystemExit(f"Baseline correctness failed; artifact saved to {run_dir}") from exc
    emit({"kind": "baseline_correctness", "status": "PASS"})

    def proposer(history, tried, audit):
        return optimize.propose(client, history, tried, audit=audit)

    trials = []
    try:
        for trial_plan in plan["trial_plans"]:
            trial = run_trial(
                trial_plan,
                proposer,
                lambda config, seed: measure_candidate(
                    a, b, reference, base_out, config, seed),
                lambda winner, seed: measure_confirmation(
                    a, b, reference, base_out, winner, seed),
                lambda _stage, seed: measure_aa(a, b, seed),
                emit,
            )
            trials.append(trial)
            _write_json_new(run_dir / "trials" / f"trial_{trial['trial']:02d}.json", trial)
            print(f"Trial {trial['trial']}: {trial['status']}; "
                  f"Muse {trial['arms']['muse']['best_confirmed_speedup']:.4f}x, "
                  f"random {trial['arms']['random']['best_confirmed_speedup']:.4f}x")
    except KeyboardInterrupt:
        emit({"kind": "run_interrupted", "note": "Partial trial events remain in events.jsonl"})
        summary = aggregate_trials(trials, plan["budget_per_arm"])
        summary.update({"status": "interrupted", "run_total_wall_s":
                        time.perf_counter() - run_started})
        _write_json_new(run_dir / "summary.json", summary)
        raise SystemExit(f"Interrupted; partial artifacts saved to {run_dir}") from None

    summary = aggregate_trials(trials, plan["budget_per_arm"])
    summary["run_total_wall_s"] = time.perf_counter() - run_started
    summary["status"] = ("complete" if summary["completed_paired_trials"] == plan["trials"]
                         else "incomplete")
    _write_json_new(run_dir / "summary.json", summary)
    print(f"Saved evaluation to {run_dir}")
    print(f"Completed paired trials: {summary['completed_paired_trials']}/{plan['trials']}; "
          f"API calls: {summary['api_calls']}; retries: {summary['proposal_retries']}")
    print("This pilot is descriptive; it does not establish statistical significance.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--budget-per-arm", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out-dir", default="evaluations/muse-random")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full schedule without CUDA, Muse, or file writes")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.trials, args.budget_per_arm, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print_plan(plan)
        return 0
    run_full(plan, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
