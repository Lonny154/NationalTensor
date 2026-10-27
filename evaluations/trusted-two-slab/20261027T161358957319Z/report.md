# Trusted two-slab K-loop experiment — 2026-10-27

**Decision:** The transformation survived compilation and passed correctness. It did not demonstrate a speedup. Screening suggests a slowdown, but A/A noise remains large and no independent confirmation was run.

## Provenance and method

- Provenance: `model_suggested_human_reviewed_implementation`.
- Original Muse proposal SHA-256: `22ce595f5eb134442dd186de8a3db1b25c9413391daa230dd9e9f87781da43fc`. Its source was read only to verify this hash; it was never imported or executed.
- Independent trusted kernel SHA-256: `5091c2df5c27ede3cce7b0472a157b8787e86c8714f0d384b5cd587babc9f5d0`.
- Baseline: `benchmark_proposal.matmul_kernel`, 64×128×32 tile, 4 warps, 3 launcher stages, direct 2D grid, FP16 inputs/output and FP32 accumulation on NVIDIA GeForce RTX 4060 Ti.
- Correctness: the scoped FP16 `torch.mm` policy with `rtol=atol=0.02`; both seeds 42 and 43 passed at 512×3072×768, 509×3071×769, and 65×129×33. The K slab counts were 24, 25, and 2; the odd 25th slab has one valid K element. Both dots in a pair feed the same FP32 accumulator in the original K order.

## Compiler and instruction evidence

| Evidence | Baseline | Trusted two-slab |
|---|---:|---:|
| TTIR loop trips and dots per trip | 24 × 1 | 12 × 2 |
| TTGIR async-copy sites | 6 | 8 |
| Static PTX instructions | 397 | 498 |
| Static PTX `cp.async.cg.shared.global` | 18 | 36 |
| Static PTX `cp.async.wait_group` | 2 | 2 |
| Static PTX `mma.sync` | 32 | 64 |
| Static PTX `ldmatrix` | 12 | 24 |
| Static SASS instructions | 432 | 496 |
| Static SASS `HMMA.16816.F32` | 32 | 64 |
| Registers per thread, allocated | 120 | 129 |
| Spills | 0 | 0 |
| Shared memory, bytes | 24,576 | 49,152 |

The trusted TTGIR carries more operand buffers through its loop (13 loop-carried values versus 9). Static body counts rise because each iteration contains two K slabs; they are not dynamic totals. The saved IR and PTX do not establish greater load/compute overlap. Matmul Inspector's full PTX and SASS comparisons are in `ptx-compare/` and `sass-compare/`.

Nsight Compute captured one FC1 launch per implementation with zero-valued FP16 inputs. These are dynamic warp-instruction counts, separate from the CUDA-event timing run:

| Dynamic metric | Baseline | Trusted two-slab |
|---|---:|---:|
| `sm__sass_inst_executed.sum` | 2,191,872 | 1,696,512 |
| `sm__sass_inst_executed_op_branch.sum` | 18,432 | 9,216 |
| `sm__inst_executed_pipe_tensor_v2.sum` | 589,824 | 589,824 |

The raw profiler output is `dynamic-ncu.csv`; launch ID 0 is baseline and ID 1 is trusted. The launch order and metric set are recorded by the command used:

```text
ncu --kernel-name regex:matmul_kernel --launch-count 2 --metrics sm__sass_inst_executed.sum,sm__sass_inst_executed_op_branch.sum,sm__inst_executed_pipe_tensor_v2.sum --csv
```

## Timing and interpretation

The existing paired Python-launch CUDA-event method measured eight screening rounds with 10 warmups and 100 launches per block. All eight paired ratios favored the baseline. Baseline median was 0.059484 ms, trusted median was 0.062960 ms, and median paired baseline/trusted ratio was **0.94736×**. The paired-ratio MAD was 0.01589. The 20-round A/A median absolute ratio deviation was **4.85%** (maximum 10.84%). Every raw timing sample and paired order is retained in `screening.json`; no filtering was applied.

The compiled change reduced dynamic branch and total instruction counts, so Muse's proposed loop-overhead mechanism partly held. It also doubled shared memory and raised register allocation. The observed screening result contradicts the predicted speedup. Resource pressure is a plausible explanation, but occupancy and causality were not established. The 0.94736× paired speedup corresponds to about 5.6% longer trusted latency in reciprocal terms, close to the A/A deviation, so it is not a confirmed performance change.
