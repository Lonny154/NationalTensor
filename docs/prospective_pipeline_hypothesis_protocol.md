# Prospective Muse hardware hypothesis — offline request

## Research question and frozen run

Can Muse independently identify an unexpected **measured** GPU performance observation, propose competing explanations, and predict the unmeasured stage-3/stage-2 latency ratio at FP16 `512×3072×1536` before that workload is run?

The separately versioned offline run is `20261101T015247349750Z` under local-only `evaluations/muse-pipeline-hypothesis/v1/`. Its exact request specification (`request/spec.json`) contains the complete prompt bytes, endpoint, model, and settings. Prompt SHA-256: `6a492a625d3be3e743f8dc13f8b131e2748d04c593680d96c0d232d0753d213b`. The neutral evidence package (`evidence/package.json`) has canonical SHA-256 `227358cfe5933a2abbb93751ed6c2c522afa34d60b6b9a1e6ba2e8f0f14ffaa4`. Both artifacts and the setup (`setup/study.json`) have separate SHA-256 manifests. Read-only validation of this preparation stage reported `not_requested`. The [public pipeline bundle](../public_pipeline_evidence/evidence.json) publishes the prompt/evidence hashes, accepted prediction fields, and outcome, but not the exact prompt or full setup.

| Request setting | Frozen value |
| --- | --- |
| Model | `muse-wonder-1.0` |
| Endpoint | Meta Muse Responses API at `https://api.meta.ai/v1` |
| Reasoning effort | `high` |
| `max_output_tokens` | `16000`, shared by reasoning and visible output |
| Timeout | 120 seconds |
| Automatic retries / fallback | 0 / none |
| Prompt version | `prospective_hardware_hypothesis_v1` |
| Prepared authorization | No live request authorized |

Wonder High is a fixed configuration for this one prospective attempt, not a claim that Wonder is scientifically superior. Future server availability and output capacity remain unverified by offline preparation. The request can be made **once** only after a separate explicit authorization. An attempt directory is created before the API call, so transport failure or process interruption cannot silently become a retry; the raw response, metadata/token usage, rejection, and manifest are retained if a result is returned.

## What Muse sees

The prompt contains **eight neutral observation IDs**: three manifested timing sessions (`TIMING_SCREENING_1`, `TIMING_SCREENING_2`, `TIMING_CONFIRMATION`), four individually manifested Nsight captures (`PROFILE_1`–`PROFILE_4`), and `COMPILER`. Each timing observation includes correctness count, per-pair order and CUDA-event ratios, stage medians, the seeded paired bootstrap interval, A/A variation, clock/temperature snapshots, units, and the result/manifest hashes. Each profile observation includes raw counter values and units, replay-pass count, warnings, clock snapshots, and raw CSV/manifest hashes. `COMPILER` includes exact TTIR/TTGIR/LLIR/PTX/cubin hashes, resource allocations, and **static** PTX copy/wait/barrier/MMA site counts. Source hashes identify the reviewed kernel, benchmark method, reference helper, and original stage experiment.

The measured K=768 session-level ratios are `0.953120870845799`, `0.951659464457997`, and `0.9562214032109078`; their median, `0.953120870845799`, is the frozen reference. The target shape is present only as a **not measured** target. The evidence also states that CUDA-event eager-launch blocks and Nsight replay have different boundaries, that bootstrap intervals are descriptive, and that profiling does not prove causality.

The human-authored [mechanism analysis](pipeline_stage_mechanism_analysis.md), [profile interpretation](pipeline_stage_profile_analysis.md), and [discovery sprint](research_discovery_sprint.md) were read for provenance but **are not supplied verbatim**. Muse receives no H1–H4 labels, preferred mechanism, human-predicted K=1536 direction, or K=1536 result. Repository search found the target shape only in planning documents, not an evaluation result. The prepared prompt was checked for those leakage classes before freezing.

## Exact research instruction and response contract

The full instruction is stored without abbreviation in `request/spec.json` under `prompt`; `--preview` prints that exact specification read-only. It asks Muse, before target measurement, to choose the most interesting unexpected observation, explain its interest, give at least two competing hardware or measurement explanations with supporting and weakening observation IDs and falsifiers, choose a prospective stage-3/stage-2 K=1536 ratio direction, give a point/range when defensible or state why not, and propose the minimum discriminating experiment. It instructs Muse to distinguish measured facts from inference and to avoid causal claims from profiler counters alone.

The response is **one JSON object** with `chosen_observation_id`, `why_interesting`, `measured_facts`, two to four `explanations`, `preferred_label`, `prediction`, `discriminating_experiment`, `falsifying_outcome`, and `uncertainties`. Each measured fact and explanation uses exact supplied observation IDs. The prediction identifies the target shape, ratio orientation, `lower|near|higher` direction relative to the frozen reference, a finite positive point ratio or `null`, a valid positive range or `null`, and a rationale. The local parser checks exact field sets, ID membership, categorical values, and numerical finiteness/order. It deliberately does not pretend that a prose regex can validate scientific mechanisms or every restated number; a human reviewer must check claim-level support and numerical fidelity. Structural acceptance will not constitute scientific approval.

## Preregistered evaluation after a separately approved K=1536 experiment

The setup's evaluation rule (local-only `setup/study.json`; see the [public outcome and rule fields](../public_pipeline_evidence/evidence.json)) was frozen before any K=1536 timing. The forecast target is the **first preregistered correctness-passing** K=1536 session's median paired stage-3/stage-2 CUDA-event ratio, with no selection among runs or dropped pairs. Minimum measurement: 14 balanced comparison rounds, 200 launches/block, 20 A/A rounds per variant, ten warmups, the original FP16 `rtol=atol=0.02`, and 2,000 seeded paired-median bootstrap draws. A fresh matched K=768 control is required to interpret a K-depth mechanism, separately from arithmetic forecast scoring.

The reference is `0.953120870845799`. An absolute-ratio deadband of `0.04` was chosen before the prediction, roughly covering the largest observed session-level A/A median deviation. A valid target median below `0.913120870846` is scored **lower**, above `0.993120870846` **higher**, and between those limits **near**. Report point-estimate absolute error and range coverage when provided. Score categorical direction only if the K=1536 bootstrap interval lies wholly within one category and the apparent movement is separated from A/A variation; otherwise record **inconclusive**, not a hit or miss. The chosen Muse prediction and any rejected request remain fixed before target measurement. These are descriptive prospective rules, not a statistical significance test.

The model's hardware explanation requires separate human scientific review: check the observation IDs against their actual fields, distinguish broad stall counters from direct `cp.async` wait timing, assess alternative explanations and falsifiers, and verify that the proposed test can discriminate them. No K=1536 result may be shown to the reviewer until the prospective prediction is recorded and hash-bound.

## Offline verification and stop

```bash
.venv/bin/python muse_pipeline_hypothesis_study.py --validate --run-id 20261101T015247349750Z
.venv/bin/python muse_pipeline_hypothesis_study.py --preview --run-id 20261101T015247349750Z
.venv/bin/python -m pytest -q tests/test_muse_pipeline_hypothesis_study.py
python verify_public_evidence.py public_evidence
```

The CLI has a future `--request-once --human-authorized` path with `max_retries=0`; this offline preparation does **not** authorize invoking it. No Muse request, GPU benchmark, profiler capture, historical edit, commit, or push was made for this protocol.
