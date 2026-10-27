# Paired benchmark timing

`benchmark_core.run_paired(launch, a, b, baseline_config, candidate_config)`
measures a launcher with the signature `launch(a, b, out, config)`. It allocates
one reusable output per side before timing, launches each side once to compile
and prime it, warms both sides, then runs an even number of paired rounds. Each
round measures both sides; within every two rounds, each side runs first once. `order_seed`
determines the reproducible schedule and every round records its actual order.

Each side gets the configured number of excluded warmup launches before the
first timed round and again immediately before each measured block. CUDA events
bracket `iterations` launches and the
event duration is divided by `iterations`. Output allocation, initial
compilation/priming, warmups, and statistics are outside the measured block.
The event interval can still contain idle gaps caused by host submission or GPU
scheduling. It is **CUDA-event block time per launch**, not a guaranteed pure
kernel execution time. This eager timing mode must not be compared as though it
were CUDA Graph replay or host end-to-end timing.

The return value retains every round's baseline and candidate milliseconds,
execution order, ratio, and per-side samples. Summaries include median, mean,
quartiles, MAD, population standard deviation, and CV. Paired ratios are
baseline/candidate within the same round; their median is distinct from the
ratio of the two side medians, which is retained as `ratio_of_medians` for
comparison with older results. No samples are filtered. These descriptive
statistics do not establish statistical significance.

`run_aa_calibration(launch, a, b, config)` uses the same configuration on both
sides with separate output buffers. Its ratio deviations from 1 estimate the
variability of this harness in that session. An A/A result cannot correct clock
behavior or certify a universal noise floor. Repeat captures at separate times
when evaluating small effects.

The standalone `smoke_aa.py` diagnostic runs 20 A/A pairs in each of two
separate modes. `python_launch` times 100 eager Python launches per block.
`cuda_graph` captures those same 100 launches into one graph per preallocated
output, primes each graph with one untimed replay, then times one replay per
block. Both modes use the same inputs, configuration, order seed, and 20 eager
warmup launches per side before each timed block. Compilation, capture,
priming, and warmups are excluded. The graph mode's extra pre-timing replays
and the fixed Python-then-graph mode order can affect GPU state; a single run
does not establish a causal improvement in variability.

For each timed block the JSON saves CUDA-event milliseconds, host submission
milliseconds measured around either the Python launch loop or `graph.replay()`,
and enclosing host-wall milliseconds through end-event synchronization. These
appear under each round's `blocks` entry alongside execution order and paired
timing. The following device-wide synchronization is outside the host-wall
interval. The report compares A/A deviation, CV, absolute bands from the first
A/A trace (<61, 61–<66, ≥66 µs per launch), and bands relative to each mode's
own median. It also prints descriptive host-submission/event correlations.
No clock correction or host-time subtraction is applied. Host-clock reads can
perturb submission slightly; interpret this as a diagnostic capture.

The caller supplies identical prepared inputs to both sides and must verify
numerical correctness before using performance results. `optimize.py` checks
the candidate against `torch.mm` before screening and again before confirmation.
It uses Python-launch CUDA-event timing; CUDA Graph replay remains in the
standalone A/A diagnostic.

The optimizer's `--repeats` option requests screening rounds (default 8).
An odd request runs one additional round to retain exactly balanced order.
Screening uses 100 launches per measured block. Confirmation defaults to seven
verification pairs, each made of two adjacent rounds with opposite execution
orders, and 200 launches per measured block. `--confirmation-pairs` and
`--min-confirmation-speedup` configure the rule. By default, all seven
verification-pair speedups must exceed 1.0 and their median must reach 1.02.
Each pair's speedup is the median baseline latency of its two rounds divided by
the median candidate latency of those rounds. This is a conservative **heuristic,
not a statistical significance test**. A candidate failing it is `unconfirmed`,
even if its screening result was faster. The summary names a `confirmed_winner`
only when the rule passes.

`experiments.jsonl` retains the older `baseline_ms`, `candidate_ms`, `speedup`,
and `samples_ms` fields. New screening records also carry
`benchmark_schema_version: 2`, `measurement_mode: python_launch_cuda_event`,
`paired_rounds`, `statistics`, and `method`. The scalar `speedup` remains the
ratio of side medians for older readers; `median_paired_speedup` is separately
available and used to select the screening winner. `summary.json` stores the
same evidence for confirmation, its decision and status, and the benchmark
policy. Older `do_bench` history remains readable and is labeled
`legacy_triton_do_bench` when loaded. Results from the two methods should not
be pooled as interchangeable measurements.

## Opt-in structural schedule experiment

`python optimize.py --structural` asks Muse for JSON containing only `schedule`
and a testable `hypothesis`. The allowlist is `direct_2d` and `grouped_m4`.
All numeric parameters remain at 64×128×32, 4 warps, and 3 stages. The direct
schedule is the baseline, so one grouped candidate can be evaluated per
compatible history. Use a fresh `--runs-dir` for an independent repeat.
Without `--structural`, the numeric optimization CLI keeps its existing search.

Structural correctness uses unscaled FP16 inputs (seed 42) and `torch.mm`
with FP16 reduced-precision reduction disabled only while creating the
reference. The helper preserves the caller's split-K setting, disables FP16
accumulation when supported, and restores all changed settings afterward,
including on failure. The existing `rtol=0.02, atol=0.02` check applies to
both schedules. The reference policy and observed backend settings are saved
in run metadata, the experiment record, and the summary. These backend flags
are process-wide, so reference generation should remain serial.

Structural runs retain the same paired CUDA-event screening, raw order and
timings, and confirmation heuristic described above. Their metadata carries
an explicit structural search mode and fingerprints for both reviewed kernels,
dispatch, benchmark core, and reference helper. Structural history loading
requires an explicit schedule in each configuration and an exact mode/source
identity. Numeric history, including older records without schedule identity,
is never treated as structural evidence. Idle-associated first-block timing
anomalies remain possible; no settling block or outlier removal is applied.
