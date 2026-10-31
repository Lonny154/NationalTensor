# National Tensor: public evidence summary

This summary is generated from `evidence.json`. Model judgments and manual scores are labeled separately from hardware observations.

## Experiment 005 — Hilbert traversal

Original hypothesis: **Hilbert Tile Traversal** — Order M-N output tiles along Hilbert space-filling curve borrowed from cartography to maximize 2D DRAM and L2 spatial locality.

Hardware: NVIDIA GeForce RTX 4060 Ti; Python 3.13.11, PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0.

All 24 scoped correctness checks passed across four shapes and three schedules. Eight paired timing comparisons were recorded as **screening only**; independent confirmation and validated profiling are missing.

| Shape | Opponent | Median paired ratio | Descriptive 95% bootstrap interval |
| --- | --- | ---: | --- |
| 128x3072x768 | grouped_m4 | 0.9943× | [0.8707, 1.0641] |
| 128x3072x768 | row_major | 0.9807× | [0.8893, 1.0473] |
| 2048x3072x768 | grouped_m4 | 1.0019× | [0.9977, 1.0271] |
| 2048x3072x768 | row_major | 0.9992× | [0.9987, 1.0042] |
| 2048x4096x1024 | grouped_m4 | 0.9969× | [0.9906, 1.0091] |
| 2048x4096x1024 | row_major | 0.9964× | [0.9959, 1.0098] |
| 512x3072x768 | grouped_m4 | 0.9992× | [0.9985, 1.0003] |
| 512x3072x768 | row_major | 0.9971× | [0.9957, 1.0430] |

No comparison has a bootstrap lower bound above 1. These observations do not establish a repeatable speedup or the proposed cache-locality mechanism.

## Experiment 006 — Muse reflection

Model muse-spark-1.3 (experiment_006_reflection_v1) chose **abandon** as an advisory decision. Observed: timing scope notes persistent lookup reused across timed launches with lookup fetch and arithmetic inside the kernel, and verification notes program assignment does not guarantee physical execution order and compiler differences alone do not establish benefit. Inference: the proposed locality mechanism remains speculative inference only, with no profiling evidence to support cache locality cause.

## Experiment 007 — discovery comparison

Six original hypotheses were blindly scored, then their conditions were revealed. The preserved scores and assignments in `evidence.json` reproduce the recorded means. These are subjective assessments, not statistical evidence of model superiority.

| Condition | novelty | mechanism_specificity | falsifiability | evidence_alignment | implementation_cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 2.00 | 3.67 | 3.33 | 2.67 | 3.33 |
| B | 2.33 | 3.33 | 3.67 | 2.67 | 3.67 |

## Experiment 008 — independent critic

P01 received an **accepted** criticism recommending **deprioritize**. Advisory only, no GPU execution authorized: falsifiable and exact, but prior diagnostics already show compiler equivalence with no runtime signal, and chosen control has no edge tiles. Expected information gain is a predicted null. Deprioritize versus mechanisms with codegen difference or broader tile coverage.
Revision attempts listed in `evidence.json` retain their historical rejection status; no rejected revision is treated as accepted.

## Experiment 009 — critic value pilot

Six proposal-level criticism outcomes are recorded: 1 accepted and 5 rejected. P01 has accepted A and B decision outputs. The six-proposal paired scoring and reveal are **incomplete**. No GPU compute savings were measured.
