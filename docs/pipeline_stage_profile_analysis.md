# Pipeline-stage mechanism — one bounded Nsight sweep

## Provenance and scope

The project owner authorized one **A–B–B–A** sweep, A = stage 3 and B = stage 2. The executed mechanism run is `20261101T014354664000Z`: its [frozen plan](../evaluations/pipeline-stage-mechanism/v1/20261101T014354664000Z/setup/plan.json) records eleven exact queried metrics and units, command arguments, source and discovery-run hashes, NVIDIA driver and Nsight versions, and the four-capture order. The current Nsight Compute metric query reported all eleven names available on the RTX 4060 Ti. The selected tensor-pipe counter is marked **deprecated** by the tool; it was retained exactly as preregistered and not silently replaced.

Each of the four isolated commands ran the reviewed `stage_reduction_experiment.py --profile-one` path in its own Python process, selected one `matmul_kernel` launch, and reported correctness PASS. All exit codes were zero, each CSV has one selected kernel row, and all capture manifests pass read-only verification. Raw CSV (including warnings), raw stderr log, timestamps, pre/post clock-temperature snapshots, command, status, and hashes are retained in [A1](../evaluations/pipeline-stage-mechanism/v1/20261101T014354664000Z/captures/01_stage_3/), [B1](../evaluations/pipeline-stage-mechanism/v1/20261101T014354664000Z/captures/02_stage_2/), [B2](../evaluations/pipeline-stage-mechanism/v1/20261101T014354664000Z/captures/03_stage_2/), and [A2](../evaluations/pipeline-stage-mechanism/v1/20261101T014354664000Z/captures/04_stage_3/). Each selected launch required **three profiler replay passes**. The exact policy was kernel replay, one selected launch, `--cache-control none`, `--clock-control none`, and a `matmul_kernel` regex selector; cache and clock warnings were retained. Stage A1 began from a low idle clock (`210 MHz`, P8), whereas later captures began near `2565 MHz`, P0. Temperatures in the pre/post snapshots were `45–47°C`.

An earlier plan, `20261101T014320801381Z`, was **never executed**. A pre-execution review found that it resolved the virtual-environment interpreter path to its underlying binary. The plan remains preserved; a corrected, separately manifested plan was prepared before any profile. The correction has a CPU-only regression check. There was one profiling sweep, four captures total, no retries, and no extra GPU benchmark.

The three discovery timing sessions are untouched. Their independently confirmed CUDA-event result remains stage-3/stage-2 `0.956221` in confirmation, with 13/14 pairs favoring stage 3. Nsight replay durations are not substituted for that timing evidence.

## Counter observations

Values below are **measured profiler counters** from one selected launch per capture. Long-scoreboard and barrier values have the tool's `warp` unit and represent average stalled warps per active cycle, not percentages. L2 and DRAM figures are rates converted from byte/s to decimal GB/s; the sweep did not request total byte counts. Rounding here does not change raw CSV values.

| Counter and unit | A1: stage 3 | B1: stage 2 | B2: stage 2 | A2: stage 3 | Within-variant observation |
| --- | ---: | ---: | ---: | ---: | --- |
| Active warps, % of sustained peak | 15.86 | 27.51 | 27.51 | 15.86 | Repeats matched to reported precision. |
| Registers/thread, compiler / allocated | 129 / 136 | 122 / 128 | 122 / 128 | 129 / 136 | Stable resource difference. |
| Allocated shared memory/block, B | 50,176 | 25,600 | 25,600 | 50,176 | Stable resource difference. |
| Long-scoreboard stalls, warp | 0.10 | 0.67 | 0.64 | 0.09 | Stage 2 higher in both captures; stage-2 spread 0.03, stage-3 spread 0.01. |
| Barrier stalls, warp | 0.08 | 0.31 | 0.31 | 0.09 | Stage 2 higher in both captures. |
| Tensor-pipe activity, % of sustained peak | 11.47 | 10.90 | 10.88 | 11.38 | Stage 3 modestly higher; counter deprecated. |
| L2 request rate, GB/s | 904.53 | 852.95 | 858.81 | 894.76 | Stage 3 higher in these captures; rates alone do not establish byte traffic or locality. |
| DRAM access rate, GB/s | 53.87 | 56.70 | 51.51 | 119.82 | A2 differs markedly from A1; no stable stage comparison. |
| Profiler replay passes, count | 3 | 3 | 3 | 3 | Counters were collected through replay, not a single simultaneous pass. |

The active-warps result reproduces the direction of the original single profile (`15.85%` stage 3 versus `27.62%` stage 2). The new stalled-warp values are **not normalized by the number of active warps**; stage 2 has more resident warps, so an absolute stalled-warp count can rise for that reason. The metric descriptions also matter: `long_scoreboard` concerns L1TEX dependencies and `barrier` concerns sibling warps at a CTA barrier. Neither directly measures time spent at the saved PTX `cp.async.wait_group` instruction. The large DRAM-rate change within stage 3, uncontrolled cache/clock policy, and three replay passes limit any memory-mechanism conclusion.

## Interpretation and K-depth prediction

The sweep supports a **specific, testable prediction**: stage 2 shows more recorded wait-related pressure despite its higher active-warps occupancy, while stage 3 maintains slightly higher tensor-pipe activity in these captures. This pattern is consistent with the compiler clue that stage 3 has extra buffering and a different async-copy wait operand. It does **not** establish that copy/compute overlap caused the confirmed latency difference. Another code-scheduling or cache effect can produce similar counters, and the stalled-warp metrics are broad.

At the proposed `512×3072×1536` test, H1 predicts that the **stage-3 relative latency advantage persists or grows** compared with a fresh matched K=768 control, and that the stage-2 wait-pressure pattern persists under matched profiling conditions. A ratio moving toward one or favoring stage 2, or disappearing wait-pressure separation, would weaken that account. A stronger gap alone would not prove overlap because K also changes compiler specialization and input working set. Fresh correctness, A/A controls, balanced paired timing, and counterbalanced shape order remain required under a separate authorization.

No scaling experiment, further Nsight sweep, Muse request, historical edit, commit, or push was performed here.
