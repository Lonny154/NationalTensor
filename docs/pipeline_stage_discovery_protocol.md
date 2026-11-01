# Pipeline-stage discovery — version 1 confirmation protocol

## Purpose and evidence boundary

The saved [stage-reduction screening](../evaluations/stage-reduction/20261028T140056186673Z/result.json) measured a stage-3/stage-2 median paired latency ratio of `0.951883` at FP16 `M=512, N=3072, K=768`: stage 3 was faster in 17 of 20 pairs even though one profile reported lower active-warps occupancy than stage 2. Its stage-3 A/A median absolute ratio deviation was `0.041493`, close to the apparent `0.048117` effect. This is an unresolved screening observation, not a confirmed performance or hardware-mechanism result. The [research sprint](research_discovery_sprint.md) explains competing interpretations.

This harness repeats **the reviewed two-slab kernel** with `64×128×32` tiles and four warps, changing only `num_stages` between 3 (baseline) and 2 (comparison). The GPU target is the RTX 4060 Ti. The runner freezes hashes for itself, the kernel, the established benchmark/timing utilities, the reference helper, and the original stage evidence. A changed source or original record makes offline loading fail. Historical artifacts are never rewritten. No GPU work is authorized by an offline setup.

## Correctness and measurement

Every session starts in a fresh process. Both variants must pass the established FP16 `torch.mm` reference with `rtol=atol=0.02` at `(512,3072,768)`, `(509,3071,769)`, and `(65,129,33)`, using seeds 42 and 43. Confirmation also checks its independent input seed 44 at the primary shape. The reference policy, strides, exact configuration, and failure are recorded. A failure stops timing for the whole session.

| Session | Inputs | Phase order | Paired rounds | Launches per block | Order seed |
| --- | --- | --- | ---: | ---: | ---: |
| `screening_1` | seed 42 | stage-3 A/A, stage-2 A/A, comparison | 20 + 20 + 20 | 100 | 43 |
| `screening_2` | seed 43 | stage-2 A/A, stage-3 A/A, comparison | 20 + 20 + 20 | 100 | 44 |
| `confirmation` | fresh seed 44 | stage-2 A/A, stage-3 A/A, comparison | 20 + 20 + 14 | 200 | 45 |

Each pair is balanced by adjacent opposite orders. Within the comparison, the ratio is always **stage-3 CUDA-event latency / stage-2 CUDA-event latency**. A/A pairs use separate preallocated outputs. Each phase primes once per side, warms each side ten times initially and ten times immediately before each measured block, and retains every timed sample. Stage-2 A/A runs first in screening 2, reversing the overall calibration order; the existing paired utility still primes its comparison baseline first. This is a documented order limitation, with each comparison's measured blocks counterbalanced.

`benchmark_core.run_paired` and `run_aa_calibration` supply preallocation, compilation priming, warmups, balanced order, and raw pairs. `smoke_aa.DiagnosticTimer` and `attach_diagnostics` supply CUDA-event block durations and separate host submission/wall intervals. The new runner adds a timing label for 200-launch confirmation blocks, phase host intervals and gaps, optional `nvidia-smi` clock/temperature snapshots, 2,000 seeded bootstrap resamples of the **median paired ratio**, and descriptive between-session consistency. CUDA events cover eager Python launches and possible submission gaps; they do not isolate pure kernel execution. Host times are diagnostics and are never subtracted from event times. Bootstrap intervals are descriptive and are not a formal significance test.

## Decision rules and mechanism work

The latency inversion is credible for follow-up only if both independent screenings and confirmation show the same direction, confirmation's paired interval is below 1, and its effect is appreciable relative to its own A/A variability. Inspect the full distributions, clock/temperature traces, host intervals, and any discordant pairs before making that judgment. If the confirmation direction changes, its interval includes 1, or A/A variation rivals the effect, report the result as unresolved or null. A repeated timing edge still does not establish a pipeline-overlap mechanism.

The offline compiler inventory reads the original TTIR, TTGIR, LLIR, PTX, cubin, and any available SASS; it records hashes, static wait/barrier markers, registers, shared memory, and spills. Each approved GPU session also captures its compiled artifacts. Static differences are hypotheses for later analysis, not explanations by themselves. **Only after independent confirmation** should a separately approved, bounded Nsight Compute profile compare matched stage-3/stage-2 launches in counterbalanced order. Candidate metrics include registers, shared memory, active warps, tensor-pipe activity, long-scoreboard and barrier stalls, DRAM/L2 traffic, and profiler replay count, only when supported. Profiling must stay separate from latency timing; one replay cannot prove causality. A K-depth variant could subsequently test whether any confirmed edge tracks pipeline depth.

## Offline preparation and verification

From the repository root, with the project virtual environment:

```bash
.venv/bin/python pipeline_stage_discovery.py --prepare
.venv/bin/python pipeline_stage_discovery.py --dry-run --run-id RUN_ID
.venv/bin/python pipeline_stage_discovery.py --validate --run-id RUN_ID
.venv/bin/python pipeline_stage_discovery.py --compiler-inventory --run-id RUN_ID
```

Preparation writes only `evaluations/pipeline-stage-discovery/v1/RUN_ID/setup/` with a SHA-256 manifest. The other commands are read-only and CPU-only. Session results, when separately authorized, are created once under `sessions/SESSION/`, with raw timing, diagnostics, compiler files, result JSON, and manifest. The verifier fails on changed source hashes, missing or malformed samples, incorrect pair orientation, missing correctness cases, wrong timing configuration, or manifest corruption. A failed or interrupted attempt remains in its session path; it is not silently retried or excluded.

## Separately authorized GPU commands

These are **future commands only**. Run each in its own fresh process after explicit human approval of that stage. The CLI approval flags record the operator's authorization; they do not replace it.

```bash
.venv/bin/python pipeline_stage_discovery.py --run-session --run-id RUN_ID --session screening_1 --human-approved
.venv/bin/python pipeline_stage_discovery.py --validate --run-id RUN_ID
.venv/bin/python pipeline_stage_discovery.py --run-session --run-id RUN_ID --session screening_2 --human-approved
.venv/bin/python pipeline_stage_discovery.py --validate --run-id RUN_ID
```

Review both screening records before deciding whether to authorize confirmation. Confirmation has an additional explicit gate:

```bash
.venv/bin/python pipeline_stage_discovery.py --run-session --run-id RUN_ID --session confirmation --human-approved --confirmation-approved
.venv/bin/python pipeline_stage_discovery.py --validate --run-id RUN_ID
```

The expected measured workload is 12,000 timed launches per screening and 21,600 for confirmation, or 45,600 total, **plus** compilation, correctness, warmups, telemetry, and any later profiling. The primary FP16 input matrices occupy about 0.75 MiB and 4.5 MiB; each primary output is 3 MiB, with separate outputs and reference/allocator overhead. CUDA process setup, compilation, and thermal settling may dominate wall time. Compiler artifact capture or GPU telemetry may be unavailable in a particular toolchain; absence is recorded and limits mechanism analysis.

This protocol ends at offline implementation and review. No GPU experiment, Muse request, or Nsight collection has been performed under it.
