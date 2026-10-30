# Muse Wild Research Lab

Wild Research Lab is a separate advisory brainstorming workflow. It does not modify the evidence-driven Experiment 005 planner, kernels, approvals, or GPU jobs.

Inspect the concise, validated context and exact prompt without a network request:

```bash
.venv/bin/python muse_wild_lab.py --prepare-only
```

Request one live round explicitly:

```bash
.venv/bin/python muse_wild_lab.py --request-from-muse
```

The request uses `muse-spark-1.3` through the existing Meta Responses API configuration. It asks for three distinct speculative concepts: plausible but overlooked, cross-disciplinary, and high-risk transformative. Each includes a mechanism, assumptions, main obstacle, disproof observation, smallest proof of concept, exact or approximate numerical equivalence, and verified context references. No executable code or benchmark is requested. Human review is required before implementation.

Attempts are saved independently under `evaluations/muse-wild-research-lab/v1/<UTC timestamp>/audit.json`. An accepted round is also exposed through `evaluations/muse-wild-research-lab/v1/accepted.json`. The audit records the prompt, context and hashes, options, model identity, visible raw response, safe completion metadata, and validation outcome. Rejected or incomplete rounds remain auditable and cannot replace an accepted round. Repeating a normal request for the same context, prompt version, and model returns `already_accepted` after acceptance or `already_attempted` after rejection. To ask for a fresh round explicitly, including a retry after rejection, use:

```bash
.venv/bin/python muse_wild_lab.py --request-from-muse --new-round
```

The parser checks structure, citation keys, exact or approximate equivalence labels, duplicate mechanisms, code-shaped output, repeated mask hoisting, and common unsupported performance claims. These checks cannot establish scientific novelty, feasibility, claim-to-citation fit, or numerical safety. The dashboard labels every idea speculative and keeps details behind expandable sections.
