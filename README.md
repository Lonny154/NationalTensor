# National Tensor — AI that investigates its own GPU research hypotheses

National Tensor connects **Meta Muse**, reviewed **Triton** kernels, real GPU measurements, and evidence checks into a research workflow. Muse proposes and critiques ideas; people approve experiments; immutable records keep measured results separate from explanations. Our hackathon result is a counterintuitive pipeline-stage latency pattern and a prospective AI prediction tested on held-out hardware.

### Built During the Meta Global AI Developer Hackathon

National Tensor's visible repository work began October 26, 2026. During the event, the project added the Muse-guided proposal and criticism workflow, human approval before trusted GPU runs, experiment memory, paired benchmarking and evidence checks, the pipeline-stage investigation and prospective held-out test, portable evidence bundles, and the Research and Hollywood dashboard views. The four-scene Pipeline Paradox replay and the Performance and Reliability cards present those saved results. The [provenance audit](docs/HACKATHON_PROVENANCE.md) links the code and commits behind each contribution. A commit date is evidence of when work entered this repository, not proof of when every line was first drafted.

### Relationship to Matmul Inspector

[Matmul Inspector](https://github.com/Lonny154/Matmul-Inspector) is an earlier independent GPU performance engineering project. It informed the FC1 workload and baseline configuration, paired timing and A/A controls, correctness gates, hashes, and compiler/profiler analysis. It had studied the same FC1 shape and stage settings before this hackathon, with an inconclusive result against PyTorch/cuBLAS. National Tensor's later stage-3-versus-stage-2 confirmation and Muse test are separate recorded experiments. The [audit](docs/HACKATHON_PROVENANCE.md) found no substantial direct or adapted Matmul Inspector implementation code in the compared tracked snapshots; rewritten or unavailable source cannot be ruled out.

**Demo video:** the three-minute walkthrough has been rehearsed and recorded. Playback review, final export, publication, and hackathon submission remain pending for November 4; no public video link is available yet.

## Featured result: more pipeline stages, faster GEMM

On an **RTX 4060 Ti**, we compared the same reviewed FP16 GEMM at **512×3072×768** with Triton `num_stages=3` and `num_stages=2`. The tile was `64×128×32` with four warps. In a separate, independently prepared confirmation session, stage 3 had a **0.956221 median paired stage-3/stage-2 CUDA-event latency ratio**, with **13 of 14 pairs** favoring stage 3. Two earlier screening processes showed the same direction. Every recorded correctness check passed.

The result is interesting because the variant with **higher measured active-warps occupancy was slower**. In a four-capture A–B–B–A Nsight Compute sweep, stage 2 had 27.51% active warps versus 15.86% for stage 3, while its recorded long-scoreboard and barrier stall pressure was higher. Saved compiler artifacts show different asynchronous-copy buffering and wait policies. These are concrete clues, but **the cause remains unresolved**: the counters do not directly time a particular wait instruction, and cache, code scheduling, and measurement state remain competing explanations.

Read the [consolidated technical report](docs/pipeline_stage_consolidated_report.md) for conditions, uncertainty intervals, compiler observations, original hashes, and limitations. The [portable pipeline evidence bundle](public_pipeline_evidence/README.md) includes raw paired timing and A/A records, selected profiler counters, sanitized compiler text, and a standalone verifier.

## Muse made a prediction before the next measurement

We gave **Muse Wonder 1.0** a neutral package of the confirmed timings, four profiler captures, and compiler observations. It independently selected the stage-depth anomaly, proposed latency hiding and timing-state effects as competing explanations, and committed to a falsifiable direction: the stage-3/stage-2 ratio would **decrease** when K doubled to 1536. It gave no defensible point estimate or range.

| Ratio | Value |
| --- | ---: |
| Frozen historical K=768 reference | **0.953121** |
| First held-out K=1536 target | **0.946341** |
| Fresh K=768 control | **0.951886** |

The target moved in Muse's predicted numerical direction. **The preregistered prediction outcome is inconclusive.** The frozen ±0.04 absolute-ratio scoring rule places the target and its descriptive bootstrap interval in the *near* category, and the movement from the reference is smaller than A/A variation. The target ran before the control in a separate process, so a small K-dependent effect cannot be isolated from measurement state. The model was not told the numerical deadband when it chose the word “lower.” The immutable forecast and scoring rule are documented in the [prospective protocol](docs/prospective_pipeline_hypothesis_protocol.md); the held-out method is in the [K-depth protocol](docs/prospective_k_depth_test_protocol.md).

## The wider research record

- **Negative hardware result:** Experiment 005's Hilbert traversal passed 24 scoped correctness checks, but eight paired screening comparisons did not establish an advantage and independent confirmation is missing. Muse's subsequent *abandon* recommendation was advisory. See the [earlier public evidence summary](public_evidence/summary.md).
- **Evidence-grounded criticism:** Experiments 008–011 developed auditable criticism of research proposals. Experiment 011 received a separate **qualified human scientific acceptance**: its recommendation was defensible, while P06 performance and its proposed mechanism remained untested. See the [human review](docs/experiment_011_scientific_review.md).
- **Decision studies remain incomplete:** The frozen Experiment 009 paired study has only its P01 decision pair. Experiment 012's P03 decisions failed its validator, so it does not establish improved decision-making. Experiment 013 exposed both model attribution errors and a confirmed validator false positive; one response per configuration cannot rank models. See the [009 completion audit](docs/experiment_009_completion_plan.md), [012 protocol](docs/experiment_012_protocol.md), and [013 failure analysis](docs/experiment_013_failure_analysis.md).

These lines of work share a rule: a plausible model explanation is a hypothesis until a discriminating experiment supports it. Prior Matmul Inspector work supplied compiler and profiler context; it is not presented as a new National Tensor speedup.

## Verify and explore

The recorded demo follows the four-scene **Pipeline Paradox** Hollywood parody, then **Reveal the Real Research →** opens Research Overview. Performance presents the verified confirmation, and Reliability presents the occupancy clues and inconclusive held-out forecast. The [dashboard guide](docs/dashboard.md) gives the recording path.

From a source-only clone, both portable bundles verify with **Python's standard library**. These commands check hashes and internal arithmetic; they do not rerun GPU measurements or require local `evaluations/` directories:

```bash
python public_pipeline_evidence/verify.py public_pipeline_evidence
python verify_public_evidence.py public_evidence
```

For a source-only CPU test of bundle portability and tamper detection, install `pytest` and run:

```bash
.venv/bin/python -m pytest -q tests/test_pipeline_public_evidence.py -k "public_bundle_verifies_without_local_evaluations or tampered_pair_or_manifest_is_rejected"
```

In an artifact-bearing development checkout, run the broader CPU-only workflow tests below. Some of these tests validate original local manifests and therefore need the ignored `evaluations/` records:

```bash
.venv/bin/python -m pytest -q tests/test_pipeline_public_evidence.py tests/test_pipeline_stage_discovery.py tests/test_pipeline_stage_profile_sweep.py tests/test_prospective_k_depth_test.py tests/test_muse_pipeline_hypothesis_study.py
```

To explore locally, install `requirements-dashboard.txt` and run `.venv/bin/python dashboard_app.py`, then open `http://127.0.0.1:8501/`. The dashboard displays whichever preserved local records are available; the original run directories are intentionally absent from a source-only clone. The [dashboard guide](docs/dashboard.md) describes its evidence views and demo boundaries. The [project context](docs/PROJECT_CONTEXT.md) is the handoff for contributors.

## Research boundaries

Humans review kernels and explicitly authorize every GPU experiment and Muse request. National Tensor has **not** demonstrated end-to-end autonomous optimization, proven a causal pipeline-overlap mechanism, established general improvement in Muse's reasoning, or shown production-scale or cross-GPU performance. The recorded GEMMs and toolchain are scoped experiments; the raw original runs remain local, while the public bundles let judges inspect the published numerical claims without those directories.

## License

National Tensor is licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Daniel Lonneman; see [NOTICE](NOTICE).
