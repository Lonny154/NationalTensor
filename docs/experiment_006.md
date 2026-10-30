# Experiment 006 — Muse Research Reflection

This advisory workflow asks Meta Muse to assess the original accepted Hilbert Tile Traversal idea against validated Experiment 005 verification and screening. It uses the existing stage manifest and timing validators. The evidence package includes the original proposal, per-comparison paired median ratios and bootstrap intervals, experimental limits, missing-stage status, and the evaluation criteria recorded for Experiment 005. Screening alone does not establish a repeatable latency advantage; a locality mechanism needs profiling evidence.

Preview the exact evidence and prompt without making an API call:

```bash
.venv/bin/python muse_research_reflection.py --prepare-only
```

After reviewing the prompt, request one reflection explicitly:

```bash
.venv/bin/python muse_research_reflection.py --request-from-muse
```

The response must choose `abandon`, `revise`, or `investigate_further` and provide five cited assessments and directions. The deterministic validator checks structure, available evidence keys, cited numbers, missing confirmation and profiling acknowledgments, and several mechanically detectable unsupported claims. Claim-to-citation fit and causal reasoning still need human review.

Each request writes `evaluations/muse-research-reflection/v1/TIMESTAMP/audit.json` and a SHA-256 manifest, including rejected responses. The audit retains the prompt, raw output, parsed reflection when valid, model identity, safe response metadata, source hashes, and UTC timestamp. `validate_audit(root, stamp)` checks an accepted audit against the current validated source evidence. No Experiment 005 artifact is changed and no GPU stage runs in this workflow.
