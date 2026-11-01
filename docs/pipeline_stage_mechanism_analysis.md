# Pipeline-stage mechanism investigation — offline assessment

## Status and evidence boundary

**The latency phenomenon is independently confirmed under the frozen eager CUDA-event harness; its hardware cause is unknown.** Run `20261101T012329698902Z` has two completed fresh-process screenings and a separate confirmation. Read-only `pipeline_stage_discovery.py --validate` passed for the setup and all three session manifests. An independent CPU recomputation from every raw paired ratio reproduced each session's median, seeded 2,000-draw interval, favorable-pair count, and both A/A median absolute deviations. All correctness records passed (12, 12, and 14 cases). The original stage-reduction result and profile hashes still match the frozen setup.

The confirmation outcome is frozen and separate from this proposed mechanism work. No subsequent profile, scaling benchmark, or Muse request has been run.

| Evidence | Stage-3/stage-2 paired median (seeded interval) | Stage-3 favorable pairs | Stage-3 / stage-2 CUDA-event medians, µs | A/A median absolute ratio deviation, stage 3 / stage 2 |
| --- | --- | ---: | ---: | ---: |
| [Screening 1](../evaluations/pipeline-stage-discovery/v1/20261101T012329698902Z/sessions/screening_1/result.json) | `0.953121` (`0.942540–0.957668`) | 17/20 | `66.91 / 70.05` | `1.98% / 3.75%` |
| [Screening 2](../evaluations/pipeline-stage-discovery/v1/20261101T012329698902Z/sessions/screening_2/result.json) | `0.951659` (`0.950134–0.954090`) | 18/20 | `61.15 / 64.18` | `2.89% / 1.92%` |
| [Independent confirmation](../evaluations/pipeline-stage-discovery/v1/20261101T012329698902Z/sessions/confirmation/result.json) | `0.956221` (`0.941377–0.966387`) | 13/14 | `62.03 / 65.14` | `3.62% / 3.53%` |

All three comparison intervals lie below 1 and their paired effects exceed the respective session's larger A/A median deviation. The intervals are descriptive bootstrap intervals over pairs, not formal significance tests. CUDA-event blocks include 100 eager Python launches in each screening and 200 in confirmation, with possible host submission gaps; they are not isolated single-kernel times. Clocks and temperatures moved within sessions, and A/A maximum deviations reached roughly 6–12%. No raw sample was filtered. These limits matter for mechanism inference even though the direction repeated.

The [original stage-reduction profile](../evaluations/stage-reduction/20261028T140056186673Z/profile_summary.json) contains **one Nsight Compute replay per variant**, not repeated profiling. It reported active warps at `15.85%` for stage 3 and `27.62%` for stage 2. That establishes an occupancy contrast for those captures, not a latency cause or an occupancy distribution. The original single screening ratio was `0.951883`; the three new sessions are separate observations.

## Offline compiler comparison

The reviewed kernel body, FP16 `512×3072×768` workload, `64×128×32` tile, and four warps are shared; only `num_stages` changes in the launch configuration. The new sessions independently captured **byte-identical artifacts for each stage setting** across all three runs; each also matches the corresponding original stage-reduction artifact. Stage 3 and stage 2 have byte-identical TTIR, while TTGIR, LLIR, PTX, and cubin differ. This localizes the first saved difference to the stage-aware compiler lowering; it does not prove which lowered difference controls latency.

| Saved or derived observation | Stage 3 | Stage 2 | Meaning and limit |
| --- | ---: | ---: | --- |
| TTGIR shared descriptors per A/B slab | `2×` buffer dimension | `1×` buffer dimension | Stage 3 has more buffering in the lowered pipeline. [TTGIR 3](../evaluations/stage-reduction/20261028T140056186673Z/stages_3.ttgir), [TTGIR 2](../evaluations/stage-reduction/20261028T140056186673Z/stages_2.ttgir). |
| TTGIR loop async wait operand | `num = 4` | `num = 0` | Different synchronization policy is explicit in the lowered loop; corresponding LLIR/PTX wait operands also differ. It is consistent with a deeper copy/compute pipeline, but elapsed wait time was not measured. |
| Compiler shared memory | `49,152 B` | `24,576 B` | Captured compiler allocation. Nsight's original launch allocation was `50,176 / 25,600 B`, including a further `1,024 B` per block in those captures. |
| Compiler registers / spills per thread | `129 / 0` | `122 / 0` | Stage 3 uses more registers. Original Nsight allocated `136 / 128` registers per thread. Zero compiler-reported spills does not rule out other scheduling effects. |
| PTX static `cp.async` copies / commit groups | `36 / 12` | `24 / 8` | Counts of saved *static PTX sites*, not dynamic instructions or bytes transferred. The larger stage-3 pipeline has more prologue/prefetch code. |
| PTX static wait-group operands | `4`, then `0` | `0`, then `0` | Both drain at the end; loop wait policy differs. [PTX 3](../evaluations/stage-reduction/20261028T140056186673Z/stages_3.ptx), [PTX 2](../evaluations/stage-reduction/20261028T140056186673Z/stages_2.ptx). |
| PTX static barriers / tensor-core MMA sites | `7 / 64` | `6 / 64` | Same `mma.sync.aligned.m16n8k16...f32.f16.f16.f32` instruction form and static MMA-site count; actual dynamic execution and issue rate remain unmeasured. |

The LLIR also contains a wait-group call with operand 4 for stage 3 and 0 for stage 2, alongside different async-copy lowering. The different cubins show that the executable changes, but cubin size or static instruction count alone cannot explain throughput. The original record did **not** save SASS. A CPU-only `nvdisasm --print-code` inspection of the saved cubins was possible: it showed the same static pattern of 64 HMMA sites in each image, with more LDGSTS sites in stage 3, and `DEPBAR` operands corresponding to the PTX wait difference. This was a transient disassembly, not a new manifested SASS artifact or dynamic instruction trace. `nvdisasm` warned that dataflow analysis was disabled during format conversion. No executed instruction count is claimed.

The strongest compiler clue is the combination of extra buffering and the different loop wait operand. It makes a copy/compute-overlap explanation plausible even while the larger allocation reduces occupancy. A distinct code-scheduling effect remains plausible because the full executable differs. The single earlier occupancy capture cannot separate these explanations.

## Competing explanations and discriminators

| Hypothesis | Prediction if it matters | Evidence that would weaken it |
| --- | --- | --- |
| **H1: deeper async-copy overlap hides wait latency.** | On matched launches, stage 2 has more long-scoreboard or async-copy wait pressure despite more active warps; the stage-3 relative advantage persists or grows when K depth doubles at fixed M/N grid. | Repeated profiles show no greater stage-2 wait pressure, and a controlled longer-K comparison shows no stage-depth interaction beyond A/A and between-process variation. Neither result alone proves H1 false because counters can miss a critical path. |
| **H2: another lowered-code scheduling/resource difference dominates.** | A concrete loop-body SASS or executed-instruction/tensor-issue difference predicts the latency direction better than waits do. | Repeated counters and code inspection show no relevant difference beyond buffering/wait policy, or the proposed marker fails to track latency across K-depth variants. Static PTX counts alone cannot decide this. |
| **H3: cache or memory traffic changes with pipeline depth.** | Stage settings yield reproducibly different L2/DRAM traffic or long-scoreboard behavior under the same addresses and work, with the difference tracking latency. | L2/DRAM traffic and cache behavior agree within repeated-profile variability while the latency edge persists. Nsight cache replay policy must be fixed before viewing values. |
| **H4: clock, thermal, or host-submission state creates the apparent gap.** | The ratio changes with order, clocks, temperature, or host timing; a matched fixed-address graph diagnostic removes or materially changes the eager-launch gap. | The direction survives balanced order, matched clock/thermal states, and a separately interpreted graph diagnostic. Three fresh sessions already weaken a simple one-session artifact, but do not eliminate H4. |

Occupancy alone is not a sufficient explanation: the variant with *higher* measured active warps was slower in the repeated eager timing. Neither the occupancy percentage nor the different wait operand establishes a causal mechanism without discriminating observations.

## Proposed bounded Nsight Compute profile — requires separate approval

**Preflight without a capture:** use the installed Nsight Compute's metric-query facility on the target device only when profiling is authorized. Freeze the exact metric list, units, cache policy, clock policy, kernel selector, driver/tool versions, input seed, and capture command before inspecting new values. The old CSV verifies availability only for the previously collected launch resource and active-warps metrics on that earlier toolchain. Treat every other candidate below as **availability unverified** today; if the current device/tool cannot provide it, record `unavailable` with the query error, not zero or an inferred proxy. Do not run a broad default metric set or silently substitute a renamed counter.

| Metric group | Proposed exact counters | Current evidence of availability |
| --- | --- | --- |
| Occupancy/resources | `sm__warps_active.avg.pct_of_peak_sustained_active`, `launch__registers_per_thread`, `launch__registers_per_thread_allocated`, `launch__shared_mem_per_block`, `launch__shared_mem_per_block_allocated` | Present in original stage-reduction CSV; recheck current tool/device before use. |
| Wait pressure | `smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active`, `smsp__warps_issue_stalled_barrier.avg.per_cycle_active` | Unverified; record unavailable if unsupported. These are broad stall categories, not direct measurements of one `cp.async.wait_group`. |
| Tensor pipe | `smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` | Unverified; check unit and replay behavior. |
| Memory | `lts__t_bytes.sum.per_second`, `dram__bytes.sum.per_second` | Unverified; rates are replay-sensitive and should be paired with duration/bytes where available. |
| Replay quality | `profiler__replayer_passes` and per-capture warnings | Original command produced one selected launch per variant; a new multi-metric selection may need more replay passes. |

For one **A–B–B–A** sweep, let A be stage 3 and B stage 2. Use the existing reviewed `stage_reduction_experiment.py --profile-one 3|2` entry point, which creates the same primary inputs and checks correctness before exposing a single selected kernel launch. Run four isolated captures in that order, with one selected `matmul_kernel` launch per capture and a fresh, uniquely named output file under a new mechanism-analysis namespace. Use the *same* verified metric set and Nsight cache/clock controls for all four; record the raw CSV/log, exact command, software/GPU state, warnings, metric units, replay passes, hashes, and failures. Do not delete a failed capture. Counterbalance collection order as prescribed, then compare A1/A2 and B1/B2 spread before interpreting A/B differences. If replay or metric availability makes a counter unreliable, mark that field unavailable and retain the raw capture. A second sweep, if needed, requires another decision; do not expand automatically.

This profile is a **mechanism probe**. Its replay timing must never replace the three independently verified CUDA-event results, and any counter difference needs replication and a falsifiable prediction before causal language is warranted.

## Proposed K-depth scaling test — requires separate approval

At FP16 `512×3072×1536`, keep the reviewed kernel, `64×128×32` tile, four warps, direct-2D **8×24 = 192 CTA output grid**, input distribution, output policy, and stage-3/stage-2 comparison fixed. Doubling K changes the number of 32-wide slabs from 24 to 48 and the two-slab loop count from 12 to 24; it also doubles A/B input bytes and changes compiler specialization and cache footprint. Therefore a K trend would be evidence about the whole longer-K workload, not a one-factor proof of overlap.

First create a **new** versioned run and manifest; do not append to or reinterpret `20261101T012329698902Z`. Inspect fresh compiler artifacts and require both stage settings to pass `torch.mm` reference correctness at the new shape for at least two fresh seeds (proposed 45 and 46) with the original `rtol=atol=0.02`, plus a direct stage-to-stage agreement check. Any failure stops timing. Then use the same preallocated-output CUDA-event method, stage-specific A/A controls, ten initial and per-block warmups, balanced paired order, raw host/event blocks, telemetry, and seeded 2,000-draw paired-median interval. A bounded first session is 14 comparison rounds and 20 A/A rounds per variant at 200 launches/block. Preserve all attempts and never select favorable rounds. A fresh matched K=768 control in the same measurement campaign, with shape order counterbalanced in another process if the first comparison looks promising, is needed before treating a ratio change as a K-depth interaction rather than a day/thermal/compiler-state difference.

H1 gains support if the **relative** stage-3 advantage at K=1536 exceeds a fresh K=768 control by more than each session's A/A variation, remains consistent across reversed shape orders, and stage-2 wait pressure increases in the predicted direction. H1 weakens if the ratio converges toward one or favors stage 2 under those controls, or if wait counters do not follow the predicted pattern. A positive result would still need to distinguish pipeline overlap from changed code generation and cache footprint. A null or failed-correctness result is reportable and must remain in the new run.

## Stop and provenance

This document is an offline analysis and proposed plan for human review. It neither modifies the frozen confirmation nor authorizes profiling or scaling. The only newly created artifact is this report. No Nsight capture, GPU benchmark, Muse call, historical edit, commit, or push was made.
