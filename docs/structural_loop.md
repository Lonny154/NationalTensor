# Controlled structural experiment loop

`structural_loop.py` is separate from the existing `optimize.py` numeric and
single-shape structural modes. Its three FP16 GEMM workloads are `fc1`
(512×3072×768), `small_m` (128×3072×768), and `square` (1024×1024×1024).
The numeric tile is fixed at 64×128×32, four warps, three stages. The reviewed
choices are `direct_2d` (baseline), `grouped_m4`, and `split_k` with factor 2
or 4. Split-K writes every valid element of a preallocated FP32 workspace,
then an explicit reduction kernel writes the FP16 output. Neither FP16 atomics
nor model supplied code are used.

After setting `MODEL_API_KEY`, a first bounded two-proposal run is:

```bash
.venv/bin/python structural_loop.py --shape small_m --candidates 2 --seed 42
```

Each Muse response must be exact JSON with `strategy`, `parameters`,
`hypothesis`, `expected_mechanism`, and `reasoning_from_history`. The parser
rejects extra fields, generated code, duplicate choices in the current run,
unsupported strategies, and split factors other than 2 or 4. A second prompt
includes the first candidate's correctness, failure stage or full screening
summary, paired-ratio MAD/CV, and the run's A/A calibration. This demonstrates
adaptation to supplied history only; it does not demonstrate model learning.

Each invocation creates a fresh timestamped directory under
`evaluations/structural-loop/`. `metadata.json` records the workload, GPU,
input seed, numeric config, strategies, source fingerprints, and measurement
policy. `aa.json` stores raw A/A rounds. `proposal_audit.jsonl` stores each
prompt and raw response. `experiments.jsonl` stores a record for every accepted
choice, including failures and complete paired timings. `summary.json` names
only a provisional screening best. Prior runs are included in the prompt only
when workload, seed, GPU, source, reference policy, and timing method match
exactly. Choices may be repeated in a new run for independent reproduction.

Correctness uses unscaled FP16 inputs and the existing scoped FP16 `torch.mm`
reference policy. Screening uses eight balanced paired CUDA-event rounds with
100 eager Python launches per block. The measured Split-K candidate call
includes both its partial GEMM and reduction launches; workspace allocation
is excluded just as output allocation is for the baseline. All samples and
actual execution orders are retained. Screening gains are provisional, and
the Day 2 A/A diagnostics found several-percent ratio deviations and unresolved
idle anomalies. No samples are filtered or corrected.

To confirm a provisional winner, start a separate process with the source run
directory shown by the first command:

```bash
.venv/bin/python structural_loop.py --confirm-run evaluations/structural-loop/<run-id>
```

Confirmation rechecks correctness, repeats A/A calibration, and uses the
existing seven-pair, 200-launch confirmation heuristic. It writes a separate
artifact under `evaluations/structural-loop/confirmations/`. Passing that rule
is labeled `heuristically_confirmed`; it is not a statistical significance
result, and close effects still need independent reproduction.
