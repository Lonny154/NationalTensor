# Muse versus random search pilot

Run the CPU-only plan check from the project root:

```bash
.venv/bin/python compare_search.py --dry-run
```

The approved GPU/API run, when separately authorized, uses the same defaults:

```bash
.venv/bin/python compare_search.py --trials 4 --budget-per-arm 4 --seed 20261026
```

The runner creates a new timestamped directory under `evaluations/muse-random/`.
It never reads `runs/` or writes `experiments.jsonl`. Dry-run only prints the
precomputed plan and creates no files. `--out-dir` changes the evaluation root.

## Protocol

The search space is `optimize.ALLOWED`: 108 configurations, including the
baseline. Both arms may choose from the same 107 non-baseline configurations.
Each of four independent trials starts with no experiment history. Muse sees
only its own previous candidate records from that trial and excludes only its
own tried configurations. Random search uses `random.Random` with a stored seed
to sample without replacement within its trial; its full four-config sequence
is fixed before any measurements. Cross-arm and cross-trial overlaps are
allowed and are measured independently. Neither arm sees the other's results.
Older `runs/` results, including legacy `do_bench` measurements, are excluded.

The four slots per trial each evaluate one configuration per arm. A valid,
distinct, allowed configuration consumes a candidate slot when correctness is
attempted, even if compilation or correctness fails. Invalid or repeated Muse
responses consume API requests but no GPU candidate slot. Muse has at most
three proposal attempts per slot; failure leaves the trial incomplete and all
partial records intact. Only complete equal-budget trials enter the paired
primary-outcome aggregate. Every trial, including incomplete trials, remains
in the artifacts and cost totals.

Within each trial the first strategy alternates by slot; the starting strategy
reverses in adjacent trials. Confirmation order also reverses by trial. Both
arms share input seed 42, the same FP16 GEMM shape, baseline, correctness gate,
and benchmark settings. Each candidate uses `optimize.correct_output` against
`torch.mm`, then `optimize.compare` with eight balanced paired rounds and 100
Python launches per measured CUDA-event block. Both arms use the same stored
order seed at each candidate slot. The current screening winner is the correct
candidate with the highest **median paired speedup above 1.0**, not the ratio
of baseline/candidate medians. The artifact records the best screening ratio
after every slot and the baseline-clamped incumbent trajectory.

After four slots, each arm follows the current `optimize.py` policy: recheck
the screening winner for correctness and confirm it, or skip confirmation if
there is no screening winner. Confirmation uses 14 balanced rounds, 200 Python
launches per event block, and seven two-round verification blocks. All seven
block ratios must exceed 1.0 and their median must be at least 1.02. The
primary outcome is that confirmed median verification-block speedup; otherwise
it is **1.0x**. Confirmation is a heuristic, not a significance test. A
correctness or confirmation error marks the trial incomplete rather than a
confirmed loss.

An eager Python-launch A/A calibration runs before and after each trial, with
20 balanced rounds and 100 launches per block. These checks are diagnostics,
not candidate evaluations, and no clock correction or post hoc sample
filtering is applied.

## Artifacts and costs

`manifest.json` saves the full strategy plan, random and timing seeds, ordered
random configurations, GPU/runtime/kernel and source fingerprints, model ID,
and measurement policy. `events.jsonl` saves every completed API attempt with
the exact prompt and raw response, a SHA-256 prompt hash, response ID and token
usage when returned, and API latency. It also saves candidate, confirmation,
A/A, failure, and completion events. No API key or authorization header is
recorded. Each `trials/trial_XX.json` contains raw paired timings, actual
within-benchmark execution order, strategy order, screening trajectory,
confirmation decision, failures, and completion status. `summary.json` reports
the primary outcomes, incomplete trials, API calls and retries, correctness
failures, GPU work wall time, screening and confirmation costs, A/A cost, and
total wall time. An interrupted run retains the event log and completed trial
files.

`gpu_work_wall_s` includes host overhead, compilation, warmups, and
synchronization around GPU work. `timed_cuda_event_interval_s` sums measured
CUDA-event blocks, which may include host submission gaps. Neither is a pure
device-busy-time measurement. API latency is reported separately. A complete
pilot has 32 candidate correctness attempts and at most 32 screening
benchmarks, eight confirmations, eight A/A checks, and 16 accepted Muse
proposals; invalid proposals can raise the API-attempt total to 48.

## Limits

Counterbalancing and A/A checks help reveal GPU drift but cannot eliminate
temperature, clock, background-load, host-submission, compilation-cache, or
API-delay effects. The same compiled kernels and fixed inputs may be reused
across trials. A Muse response may vary even with an identical prompt; saved
prompts and responses make the observed sequence auditable, while the random
arm is exactly replayable from its seeds. Four paired trials are descriptive
only. Benchmark rounds are correlated measurements, not independent search
trials. Any claim that Muse is more efficient requires a prespecified primary
comparison and more independent, counterbalanced paired trials, with an
interval computed over trial-level differences.
