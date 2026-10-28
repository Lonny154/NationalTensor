# Advisory experiment planner

`experiment_planner.py` asks Muse for one next research step using the saved
NationalTensor experiment memory. It does not generate kernel code or launch
experiments. Its context includes bounded, artifact-derived summaries of trusted
tests, structural screenings, unevaluated source proposals, compiler and
resource observations, A/A variability, and results from the recorded FP16
GEMM shapes. Generated and trusted source links are verified by hash. Absolute
timings from different runs are never pooled.

Prepare a prompt for inspection without an API call:

```bash
.venv/bin/python experiment_planner.py --prepare-only
```

After reviewing the prompt and supplying `MODEL_API_KEY` in the environment or
repository `.env`, request **one** live advisory plan with:

```bash
.venv/bin/python experiment_planner.py --request-from-muse
```

This command is intentionally separate from the dashboard's kernel-proposal
button. A request uses `muse-spark-1.3`, a 30-second timeout, no SDK retries,
and one response. The planner saves `plan.json`, `metadata.json`, and
`proposal_audit.jsonl` under `evaluations/experiment-planner/<UTC-run-id>/`.
The audit retains the exact prompt and raw response. Metadata records the plan
hash and cited generated/trusted source hashes. Invalid plans, API failures,
and exact repeats of a saved research direction keep an audit but do not
produce an accepted plan. The dashboard reads saved plans; it does not make
planner requests.

The required JSON fields are research question, experiment kind, target MNK
shape, optimization mechanism, technical hypothesis, cited evidence, reasoning
from evidence, difference from prior work, expected performance mechanism,
risks, correctness plan, measurement plan, and criteria supporting or
contradicting the hypothesis. Experiment kinds include kernel transformation,
workload shape, numeric tuning, memory movement, profiling, and a decision
not to repeat a direction. Each citation must match an experiment ID and
artifact path supplied in the saved prompt. Cited observation keys must
actually be available for that record. The dashboard resolves citations
against current local memory and labels missing evidence unavailable.

The prompt now provides a complete valid JSON example using a real supplied
evidence ID and path. It states that `experiment_kind` is one JSON string,
`target_shape_mnk` is an array of three positive integers, and
`supporting_evidence` is an array of validated references. The validator keeps
these types strict; it does not coerce a one-element kind array or a quoted
shape into a valid plan.

The first live planner response in
`evaluations/experiment-planner/20261027T231039939029Z/` was rejected and its
audit remains unchanged. It returned `experiment_kind` as `["profiling"]` and
`target_shape_mnk` as `"[512, 3072, 768]"`. Its Split-K citation also used
`outcome` and `shape_mnk`, which are genuine saved memory fields and are now
allowed observation keys when present. This change validates their references;
it does not endorse the response's free-text interpretation. In particular,
one Split-K factor-2 screening loss does not rule out K-parallelism generally,
and matching PTX instruction multisets do not prove equal runtime performance.
The two-slab implementation screened slower while using more resources, but
resource pressure was not established as the cause. The mask/pipeline screening
was inconclusive, and A/A timing variation is not a measure of profiler-counter
uncertainty.

The first accepted live plan in
`evaluations/experiment-planner/20261028T002953332983Z/` has status
`advisory_saved` and plan SHA-256
`1e773b3c45929d5d4c288db29a058afecd995281012023cf0ab287fe44fd57aa`.
It recommends bounded profiling of already-reviewed kernels on the FC1 shape;
it is **not** a measured performance improvement or an approval to run work.
Its statement that the two-slab screening delta is “within paired noise” is
model interpretation, not a statistical conclusion. The two-slab paired
screening ratio was 0.94736×, with all eight pairs favoring baseline, while
its A/A median absolute ratio deviation was 4.85%. A/A variability alone
cannot establish whether that A/B difference is statistically significant.
The plan's source text and audit remain unchanged; any profiling study still
requires separate review and execution.

[The example JSON](../examples/experiment_planner_plan.json) is illustrative,
not a Muse response or a measured result. It recommends profiling an
unresolved resource-pressure hypothesis rather than generating another kernel.
Its claims about occupancy are explicitly predictions; the saved trusted
experiments did not measure occupancy causality.

Validation checks structure and reference identity, not the truth of every
free-text explanation. A human must review the recommendation and decide
whether to run an experiment. Plans cannot approve, compile, import, or
execute generated kernels. No saved plan is a confirmed optimization.
