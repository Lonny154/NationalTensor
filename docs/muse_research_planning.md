# Cross-experiment Muse research planning

`muse_research_plan.py` builds a fixed, read-only evidence package from the validated Experiment 003 and 004 diagnostic loaders. It fails closed if either record is missing, corrupt, incomplete, or inconsistent in source identity. It includes the reviewed implementation identity separately from the original Muse source and never treats the latter as evaluated. The M=128 and M=130 compiler counts describe different workloads; they do not show an optimization benefit. Neither experiment includes runtime timing.

Inspect the evidence, rule-based baseline, and exact prompt without a network request:

```bash
.venv/bin/python muse_research_plan.py --prepare-only
```

Make one explicit live planning request when ready:

```bash
.venv/bin/python muse_research_plan.py --request-from-muse
```

The request uses `muse-spark-1.3` through the Meta Responses API, `reasoning.effort: low`, an 8,000-token output limit, a 30-second timeout, and no SDK retries. `MODEL_API_KEY` is loaded from `.env` if present. No GPU runner or generated kernel is imported or called. Human approval is required before implementing or executing Experiment 005.

Each attempt is saved under `evaluations/muse-research-planning/v1/<UTC timestamp>/audit.json`, with the evidence package and hash, prompt and version, request options, model, safe response metadata, raw visible response, validation outcome, and baseline comparison. A validated accepted response is copied to `evaluations/muse-research-planning/v1/accepted.json`. Rejected and incomplete attempts stay in their own directories and do not replace an accepted record. Repeated requests with the same validated evidence, model, and prompt version return `already_accepted`.

The deterministic baseline proposes a single stage-count comparison on fixed M=130 FP16 GEMM: use the baseline source and hold input, tile sizes, warps, compiler, GPU, and correctness policy fixed while varying `num_stages` from three to two. Compare correctness, executable SASS, registers, shared memory, and spills. A matching executable and resource profile would rule out this stage-count mechanism in the recorded scope. Runtime remains unknown; any later timing requires separate review.

The comparison rubric records evidence grounding, falsifiability, variable control, negative-result value, RTX 4060 Ti feasibility, and avoidance of repeated mask hoisting. It does not score or declare a winner. Structural checks can verify citations and catch selected unsupported claims, but they cannot establish that prose accurately interprets the cited evidence, that controls isolate the mechanism, or that Experiment 005 is feasible. Those judgments remain for human scientific review.
