# Pipeline-stage latency inversion: consolidated technical evidence

## Finding and scope

On an RTX 4060 Ti, the reviewed two-slab FP16 GEMM at `M×N×K = 512×3072×768` repeatedly ran faster with Triton `num_stages=3` than `num_stages=2` under paired eager-launch CUDA-event timing. The independent confirmation's median **stage-3/stage-2 latency ratio was `0.956221`** (descriptive seeded interval `0.941377–0.966387`; 13/14 pairs favored stage 3). Four later Nsight Compute captures measured **lower active-warps occupancy for stage 3** despite its latency advantage. Compiler artifacts show a deeper lowered async-copy pipeline. These facts establish a repeatable latency direction under this harness; they do **not** establish which compiler or hardware effect caused it.

Muse Wonder 1.0 then made an immutable prospective **“lower”** directional prediction for the unmeasured K=1536 ratio. The first held-out target session returned `0.946341`; a later fresh K=768 control returned `0.951886`. The target is numerically lower, but the difference is within observed A/A variation. Under the preregistered ±0.04 absolute deadband, the target and its interval are **near** the historical reference, and the directional score remains **inconclusive**. This is one prospective prediction and one target/control campaign, not a powered test of Muse's scientific ability or proof of pipeline overlap.

## Identity, methods, and source integrity

| Record | Run ID | Conditions and role |
| --- | --- | --- |
| Stage discovery | `20261101T012329698902Z` | Two independent fresh-process screenings and one separate confirmation, all K=768. |
| Mechanism profile | `20261101T014354664000Z` | Four isolated Nsight captures, A–B–B–A: stage 3, stage 2, stage 2, stage 3. |
| Prospective Muse request | `20261101T015247349750Z` | One `muse-wonder-1.0` request, reasoning `high`, completed and structurally valid before target timing. |
| Held-out timing | `20261101T020025480962Z` | First K=1536 target, then a separate fresh K=768 control; both correctness-passing. |

The timing comparisons share the human-reviewed `trusted_two_slab_kernel.py` source SHA-256 `98188d25c726c2301cd81e36e6dfd22f02db5f3acbc3d05d385d63ce7cffd06d`, FP16 input policy, `64×128×32` tile, four warps, direct `8×24` CTA output grid, preallocated outputs, and `rtol=atol=0.02` correctness gate. Within each paired comparison the intended configuration difference is `num_stages=3` versus `2`. Target and control use different K values, fresh input seeds, and distinct processes; those differences are part of the K-depth question. All timing rounds are balanced between stage orders. Each screening timed 100 launches per CUDA-event block; confirmation and held-out sessions timed 200. Initial and per-block warmups were ten. The seeded 2,000-draw intervals describe variation among paired rounds; they are not formal significance tests. CUDA events bracket *blocks of eager Python launches*, not isolated individual kernels.

The original discovery recorded 12, 12, and 14 passing correctness cases. Held-out target seeds 45/46 and control seeds 47/48 passed against the `torch.mm` reference for both variants, as did direct stage-to-stage agreement. Read-only validators passed for all five timing sessions, the profiler sweep, and the Muse audit. No historical audit or scoring rule was changed for this report.

## Timing evidence

Ratios are **median of paired stage-3/stage-2 ratios**, not ratios of the displayed marginal medians. Latencies are median CUDA-event milliseconds per launch, rendered here in microseconds. A/A values are median absolute deviations of same-variant paired ratios from 1. Every raw pair, timing block, order, and A/A pair is present in the [portable bundle](../public_pipeline_evidence/README.md).

| Session | Stage 3 / stage 2 median, µs | Paired ratio (descriptive 95% bootstrap interval) | Stage 3 favorable | A/A stage 3 / stage 2 |
| --- | ---: | --- | ---: | ---: |
| Screening 1, K=768 | 66.908 / 70.046 | `0.953121` (`0.942540–0.957668`) | 17/20 | 1.98% / 3.75% |
| Screening 2, K=768 | 61.149 / 64.184 | `0.951659` (`0.950134–0.954090`) | 18/20 | 2.89% / 1.92% |
| Independent confirmation, K=768 | 62.031 / 65.139 | `0.956221` (`0.941377–0.966387`) | 13/14 | 3.62% / 3.53% |
| First held-out target, K=1536 | 119.800 / 126.218 | `0.946341` (`0.937097–0.951438`) | 14/14 | 0.81% / 0.97% |
| Fresh control, K=768 | 60.853 / 64.417 | `0.951886` (`0.919107–0.986495`) | 14/14 | 3.48% / 3.60% |

The historical reference is the median of the three original K=768 session medians: **`0.953120870845799`**. The target minus reference is `−0.006780`; target minus fresh control is `−0.005545`. The fresh control minus historical reference is `−0.001234`, consistent in point estimate with the earlier direction. The target/control difference is smaller than each session's larger A/A median deviation (target `0.009697`, control `0.036046`), and their descriptive intervals overlap. The fixed **target-before-control** process order, different fresh inputs, thermal state, and K-dependent compiler specialization prevent a causal K-depth interaction claim. During the target comparison the GPU was reported around 64–65°C at 2745 MHz SM clock; during control comparison it was around 53°C at 2760 MHz. Host submission and event block diagnostics remain in the export. No pair or outlier was removed.

## Four-capture Nsight mechanism probe

The separate profile sweep used one selected `matmul_kernel` launch per capture, kernel replay with **three passes**, `--cache-control none`, and `--clock-control none`. All selected launches passed correctness. Counters have the units shown; L2 and DRAM are decimal GB/s converted from the saved byte/s rates. Nsight replay timing does not replace the CUDA-event results.

| Counter | A1 stage 3 | B1 stage 2 | B2 stage 2 | A2 stage 3 |
| --- | ---: | ---: | ---: | ---: |
| Active warps, % peak | 15.86 | 27.51 | 27.51 | 15.86 |
| Registers/thread, compiler / allocated | 129 / 136 | 122 / 128 | 122 / 128 | 129 / 136 |
| Allocated shared memory/block, B | 50,176 | 25,600 | 25,600 | 50,176 |
| Long-scoreboard stalled warps/active cycle | 0.10 | 0.67 | 0.64 | 0.09 |
| Barrier stalled warps/active cycle | 0.08 | 0.31 | 0.31 | 0.09 |
| Tensor-pipe activity, % peak | 11.47 | 10.90 | 10.88 | 11.38 |
| L2 request rate, GB/s | 904.53 | 852.95 | 858.81 | 894.76 |
| DRAM rate, GB/s | 53.87 | 56.70 | 51.51 | 119.82 |

Stage 2 had more measured long-scoreboard and barrier stalled warps in both captures despite higher active-warps occupancy. These broad counters are *not* measurements of elapsed time at a particular `cp.async.wait_group`; they are not normalized for the different resident-warp counts. The tensor-pipe counter was reported deprecated by Nsight. Stage-3 DRAM rate changed sharply between its two captures, and uncontrolled caches/clocks plus three-pass replay limit memory-traffic interpretation. The pre-execution plan `20261101T014320801381Z` was never run; the corrected plan above produced exactly four captures.

## Compiler differences and competing explanations

Saved K=768 TTIR is byte-identical between stage settings (SHA-256 `da34e46849480937bf018b590f05a2d18184f978684481b9372eec8901339152`). TTGIR, LLIR, PTX, and cubin differ. The compiler reports stage-3 versus stage-2 shared memory `49,152 / 24,576 B`, registers/thread `129 / 122`, and zero reported spills for both. Stage 3 has a doubled A/B shared-buffer dimension in TTGIR and a loop async-wait operand `4`; stage 2 has operand `0`. Saved PTX contains **36 / 24 static `cp.async` copy sites**, **12 / 8 commit sites**, **7 / 6 barriers**, and **64 / 64 MMA sites** for stage 3 / stage 2. Those are static code sites, not dynamically executed instruction counts. The portable bundle includes source-path-sanitized PTX and TTGIR so these counts and wait operands can be inspected without local run directories. Original compiler artifact hashes are recorded in its evidence file.

Extra buffering and a different wait policy make copy/compute latency hiding plausible; another lowered-code scheduling effect, cache behavior, and timing-state effects remain viable. A true mechanism test needs controlled timing and repeated counters at both K values, ideally with counterbalanced shape order and compiler/cache comparisons. The held-out ratio difference is too small relative to A/A variation to identify a K-depth interaction. The K change also doubles input K extent and changes compiler specialization and cache footprint; it is not a pure pipeline-depth manipulation.

## Muse forecast and held-out evaluation

Muse chose the independent confirmation as the unexpected observation and named **“Extra-stage latency hiding”** and **“Timing-boundary and state artifact”** as competing explanations. Its preferred explanation predicted a `lower` stage-3/stage-2 ratio at `[512, 3072, 1536]` relative to the reference. It explicitly gave `point_ratio: null` and `plausible_range: null`, citing the absence of any supplied K-variation observation. The original audit was structurally accepted; claim-level scientific review remains separate. The request included no K=1536 result and no human-authored predicted direction. Muse was **not told** the numerical ±0.04 deadband, a limitation when interpreting its qualitative word “lower.”

The evaluation rule was frozen before target timing: **lower** below `0.913120870846`, **near** through `0.993120870846`, and **higher** above it. Its scoreable target is the *first* correctness-passing K=1536 session. That session's median `0.946341` and descriptive interval `0.937097–0.951438` are wholly **near**. Its movement from the reference (`0.006780`) is smaller than the larger target A/A median deviation (`0.009697`). The frozen rule therefore yields **inconclusive** rather than a hit or miss for Muse's `lower` forecast. The fresh K=768 control informs mechanism interpretation only; it cannot change this score. A single response cannot establish a general improvement in Muse scientific reasoning.

## Immutable artifact identifiers and portable verification

All hashes below are SHA-256 of original local bytes. The [portable evidence manifest](../public_pipeline_evidence/manifest.json) carries these and the complete source-artifact hash inventory; its own SHA-256 is `ff7f28278eb410b23a555c56e687f5aeecc12f79f63d7b307d9ea9c54f54d3b5` at report preparation.

| Original artifact | Result or raw CSV hash | Manifest hash |
| --- | --- | --- |
| Discovery screening 1 | `d0a40bfecd947c916d77dffde3f50b338d090de1d6c328be70b7cc80b70ae7b7` | `9ddff27ade974ae2913eb3c89479338e4fc12b57dbfb03fed2a2b2a41e65e87e` |
| Discovery screening 2 | `5edd91ca0b0b0ff6a1034a7f8a41508470c95cd6969ff55d1315b129eef19120` | `75d63bbb84cae2c136bd99a5219c109c28049f347e6a407d437bec7e2c550b89` |
| Discovery confirmation | `e18e06cadb27edec24a707c1197914bd0c0fbfe1396392d42472dd8d84831f27` | `bafaf286436e4ec823a26c69b0396d9afdafbdd0f0bb45b2d2eb601de539acff` |
| Profile A1, stage 3 | `2cfba7826f1cecbf11e652d896f5f37137769097a32a6f6da98ea6e48527a86b` | `7dda298b12f9d8e632f6d801899ce515aed631b046d17d6008344b8119ef828d` |
| Profile B1, stage 2 | `d4a58cd8f29705ae49ee926d9a003402e8d163ee39d4e1acf685b52ca672ade2` | `cc1312a9d8b3157fbe0a4ee794351c6810956168a84526af8b27dd490c1f520a` |
| Profile B2, stage 2 | `bc94983c582b747fa590704cfe15dfdcf720af5303008accd8c30ffecf153e91` | `3451d090baf6b2dd8b3daa7ba6e6764d8e19d5f5dd2770eef1a74984007db46f` |
| Profile A2, stage 3 | `1afd0b529aaa7dd99d513e54393ae83ca4a80fbc0fecfd3a8b3fe4b821a175a2` | `7ffbe9ef5f22df95b48d8adb709e9c9bc6bbab940aedd8112e2490e5829baf14` |
| Muse forecast audit | `c384ef6508cec7994155477b80d0cea667093149712c8c26438f773afafaebb5` | `14c0cb9263ab08b6a43b8eecbbf9b8903cf9ef1662dfe975a1d351746224469b` |
| Held-out K=1536 target | `043fdad07bb3336ab4bb9cb97fefa11fe5fe7bb73b7476146a4c933c317fe82f` | `d1185ea0ff2b264c436c31447be1491aae1c1864115432272fc1b7fcbe8a7353` |
| Held-out K=768 control | `1756a9fe34a2f65dc4ffb4df2afcdacc12b855c2d9340898ec22d02f2b6179f5` | `a7b1ab7ac5f0c68bd98143ddc927543843caa759ba841caa089bd7a6f3032832` |

The prospective prompt SHA-256 is `6a492a625d3be3e743f8dc13f8b131e2748d04c593680d96c0d232d0753d213b`; the neutral evidence package SHA-256 is `227358cfe5933a2abbb93751ed6c2c522afa34d60b6b9a1e6ba2e8f0f14ffaa4`. The corrected profile plan SHA-256 is `e001632fbb1297431778a7018a34125d99ce4d6d472d29e14107974c3f727f57`.

Copy [public_pipeline_evidence](../public_pipeline_evidence/README.md) as a directory. From inside it, run `python verify.py .` using only Python's standard library; no `evaluations/` tree, GPU, or Muse credentials are needed. The verifier checks exported file hashes, run identities, all raw paired-ratio and A/A arithmetic, seeded bootstrap intervals, reference arithmetic, profile ordering and units, PTX static counts, and the frozen near/inconclusive outcome. This verifies **bundle integrity and internal consistency**. Original source hashes provide comparison handles if the originals are separately supplied; an unsigned copied bundle cannot independently authenticate its own provenance or reproduce GPU measurements.
