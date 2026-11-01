# National Tensor — Project Context

**Research handoff for the November 4, 2026 hackathon submission.** Read this with the [root README](../README.md); verify saved artifacts before making a new scientific claim. This handoff summarizes current state and leaves the earlier protocols as historical records.

## Purpose and evidence rules

National Tensor asks whether Meta Muse can help formulate, critique, and test GPU research hypotheses against real hardware. The implemented workflow joins advisory Muse output, human-reviewed Triton kernels, correctness gates, paired timing, compiler/profiler artifacts, SHA-256 manifests, and a local dashboard. Humans authorize every model request and GPU experiment. Model interpretation, measured observations, structural validation, and human scientific review remain separate.

The current judge-facing result is the [pipeline-stage consolidated report](pipeline_stage_consolidated_report.md). A [portable public bundle](../public_pipeline_evidence/README.md) exposes its raw paired and A/A timing samples, selected Nsight counters, sanitized compiler text, frozen forecast fields, and held-out observations. Run `python public_pipeline_evidence/verify.py public_pipeline_evidence` from a source-only clone. It checks export integrity and arithmetic without the ignored local `evaluations/` tree; it does not reproduce GPU runs or authenticate original files that are not supplied.

## New pipeline-stage research sequence

| Step | Recorded state | Boundary |
| --- | --- | --- |
| Discovery and independent confirmation | Run `20261101T012329698902Z` has two screening processes and a separate confirmation. At FP16 `512×3072×768` on the RTX 4060 Ti, stage 3 beat stage 2 in 13/14 confirmation pairs; median stage-3/stage-2 ratio `0.956221`, descriptive bootstrap interval `0.941377–0.966387`. | Correctness passed; CUDA events bracket eager Python launch blocks. The result confirms the measured latency direction under this harness, not its cause. [Protocol](pipeline_stage_discovery_protocol.md). |
| Mechanism investigation | Nsight sweep `20261101T014354664000Z` captured stage order 3–2–2–3. Stage 2 had higher active-warps occupancy (`27.51%` versus `15.86%`) but higher broad long-scoreboard and barrier stall counters. Saved TTIR matched; TTGIR/PTX showed different async-copy buffering and wait policies. | Four selected replayed launches do not prove copy/compute overlap. DRAM rate varied substantially between stage-3 captures. [Offline analysis](pipeline_stage_mechanism_analysis.md); [profile report](pipeline_stage_profile_analysis.md). |
| Prospective Muse forecast | Wonder 1.0 run `20261101T015247349750Z` independently selected the anomaly from a neutral observation package and predicted a **lower** stage-3/stage-2 ratio when K doubled to 1536. The response was structurally valid and immutable before target timing. | No defensible point estimate or range; scientific mechanism review is separate. The prompt did not disclose the numerical scoring deadband. [Protocol](prospective_pipeline_hypothesis_protocol.md). |
| Held-out target and control | Run `20261101T020025480962Z`: first K=1536 target ratio `0.946341` (interval `0.937097–0.951438`), then fresh K=768 control `0.951886` (interval `0.919107–0.986495`); all scoped correctness and stage-agreement checks passed. | Target–control difference `−0.005545` is smaller than A/A variation. Frozen reference `0.953120870845799` and ±`0.04` deadband yield **inconclusive** forecast scoring. Target-before-control order and state differences remain confounders. [Method](prospective_k_depth_test_protocol.md). |

The compiler and profile observations make latency hiding plausible, but another lowered-code scheduling effect, cache behavior, and timing state remain alternatives. No production or cross-GPU generalization is established. The next priority is the hackathon demo and final submission, using these exact labels and limitations, rather than additional experiments or evidence-schema development.

## Other research and historical boundaries

- **Experiments 003–004:** a human-reviewed mask-hoisting source pair passed scoped correctness and compiled equivalently for the tested shapes, so timing was skipped. The original Muse-generated kernel was not executed. See the [debrief](muse_research_debrief.md) and [partial-boundary protocol](experiment_004.md).
- **Experiments 005–006:** Hilbert traversal passed 24 scoped correctness checks, but eight paired screening comparisons had no independently confirmed latency advantage. The validated Experiment 005 pipeline lacked confirmation and validated profiling. Muse's accepted *abandon* reflection was advisory. See [005](experiment_005.md), [006](experiment_006.md), and the [earlier portable evidence](../public_evidence/README.md).
- **Experiments 007–009:** six original proposals received blind subjective review in Experiment 007. Experiment 008's criticism workflow preserved accepted and rejected attempts. Experiment 009 run `20261030T205425148094Z` has accepted bound v1 criticisms for P01/P02 and accepted P01 A/B decisions only; its six-proposal paired study and blind scoring are incomplete. The frozen study cannot use later v2/v3 criticisms as treatment. See [007](experiment_007.md), [008](experiment_008.md), [009](experiment_009.md), the [completion plan](experiment_009_completion_plan.md), and the [rejection audit](rejected_criticism_analysis.md). The older 005–009 public bundle is a historical snapshot and does not reflect every later local attempt.
- **Experiment 011:** one v3.1 criticism passed structural validation. The project owner separately signed a **qualified human scientific acceptance**: the prior screening ratios and advisory `revise` were defensible, while C10 lacked complete claim-level support and P06's mechanism remained untested. The historical audit's `scientific_status` was not rewritten. See the [scientific review and sign-off](experiment_011_scientific_review.md).
- **Experiment 012:** the imperfect-criticism paired decision study remains incomplete. Both P03 A/B decisions failed the frozen validator; preserve them in the attempt accounting. They do not establish improved decision quality. See the [protocol](experiment_012_protocol.md).
- **Experiment 013:** two high-reasoning model decisions were rejected. A read-only audit identified a genuine wrong-shape evidence citation and unsupported prose in Spark, plus a confirmed negation-sensitive validator false positive in Wonder. One response per model cannot rank model capability. See the [failure analysis](experiment_013_failure_analysis.md).

These records do not demonstrate end-to-end autonomous optimization, a causal pipeline mechanism, or general improvement in Muse research reasoning. Prior Matmul Inspector findings are background evidence rather than new National Tensor speedups.

## Repository and verification

The release branch is `hackathon-research-release`, based on local `main` at `385b222` when prepared. The original `evaluations/` directories, API credentials, raw private model requests/responses, and virtual environment are excluded from the public release. Historical protocols remain linked and unchanged unless a separately documented correction is needed.

Run from the repository root:

```bash
python public_pipeline_evidence/verify.py public_pipeline_evidence
python verify_public_evidence.py public_evidence
.venv/bin/python -m pytest -q tests/test_pipeline_public_evidence.py tests/test_pipeline_stage_discovery.py tests/test_pipeline_stage_profile_sweep.py tests/test_prospective_k_depth_test.py tests/test_muse_pipeline_hypothesis_study.py
.venv/bin/python dashboard_app.py
```

The two verifiers and CPU tests make no GPU or Muse request. The listed broader tests validate ignored original records and need an artifact-bearing checkout; source-only clones can run the portable verifiers and the bundle portability/tamper tests selected in the README. The dashboard displays whichever saved local records are present. Reproducing original GPU timings requires compatible hardware, a suitable toolchain, source snapshots, and separately approved runs.

## Submission priority

Prepare a short demo that shows the measured stage inversion, the four-capture hardware clues, Muse's **prospective** forecast, and its **inconclusive** held-out score. Link the consolidated report and both portable bundles in the submission. Before November 4, verify the demo playback, README links, bundle integrity, branch state, and submission text. Do not imply that the profile proves causality or that Muse passed the frozen prediction criterion.
