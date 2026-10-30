# Experiment 005 — Hilbert Tile Traversal

This controlled experiment follows the accepted Wild Research Lab **Hilbert Tile Traversal** idea. It compares three tile assignments for the same FP16 GEMM body: a conventional two-dimensional row-major grid, established grouped-M4, and a Hilbert lookup. The four requested shapes are representative GEMMs; none is asserted to be a production workload. The configuration is 64×128×32, four warps, and three stages on the RTX 4060 Ti.

The CPU mapping code walks a padded power-of-two Hilbert square, filters coordinates outside the rectangular output grid, and stores only valid `(tile_m, tile_n)` pairs. Each of the `ceil(M/64) × ceil(N/128)` CTAs loads exactly one pair from a persistent int32 GPU lookup table. There are no empty padded CTAs. The one-time CPU construction and host-to-device copy are measured separately; timed Hilbert launches include the two lookup loads. This assumes the table is reused across launches. If a caller builds it for every GEMM, the recorded setup cost must be included in end-to-end latency. Program-ID assignment does not guarantee hardware CTA execution order.

The compute loop, FP32 accumulation, FP16 store, masking, and tile sizes are shared by all three specializations. Verification tests tile coverage independently, then checks every strategy against the scoped FP16 `torch.mm` reference for two deterministic seeds on all four shapes, including boundary rows and columns. Any failure prevents timing. It captures source hashes, TTIR/TTGIR/LLIR/PTX/cubin/SASS, executable section hashes, registers, shared memory, spills, and whole-kernel instruction counts. Whole-kernel count differences are descriptive and cannot isolate mapping instructions or establish speed. No memory sanitizer is included; reviewed masks, mapping coverage, and correctness checks are the available memory-safety evidence.

## Explicit stages

From the repository root, first run correctness and compiler verification:

```bash
.venv/bin/python experiment_005_hilbert.py --verify
```

The command prints a UTC `RUN_ID`. Review its `verify/result.json` and manifest. Initial paired CUDA-event screening is a separate command:

```bash
.venv/bin/python experiment_005_hilbert.py --screen --run-id RUN_ID
```

After reviewing screening and explicitly approving the next stage, run independent confirmation:

```bash
.venv/bin/python experiment_005_hilbert.py --confirm --run-id RUN_ID --human-approved
```

Both timing stages use preallocated outputs, warm all candidates, interleave balanced pairs, retain per-round raw samples, and report the median paired ratio, median absolute deviation, and a seeded bootstrap interval. Confirmation uses fresh inputs and timing order. These intervals are descriptive uncertainty estimates; a numerical ratio alone is not a statistical or causal proof. Screening remains labeled unconfirmed.

Optional profiling is available only after validated screening and explicit approval. For one selected workload and strategy, create an output directory and run Nsight Compute separately, for example:

```bash
mkdir -p evaluations/experiment-005-hilbert-tile-traversal/RUN_ID/profile
ncu --set full --kernel-name regex:matmul_kernel --launch-count 1 -o evaluations/experiment-005-hilbert-tile-traversal/RUN_ID/profile/hilbert_2048x4096x1024 .venv/bin/python experiment_005_hilbert.py --profile-kernel --run-id RUN_ID --shape 2048 4096 1024 --schedule hilbert --human-approved
```

Select only comparisons worth investigating. L2 counters, DRAM traffic, stalls, occupancy, and instruction overhead can help explain behavior, but improved L2 hit rate alone does not prove a latency benefit. Hilbert remaps CTAs; it does not change the intra-warp memory access pattern, so the proposal's claim about improved DRAM coalescing needs particular scrutiny. Profiler overhead and cache state differ from ordinary screening. Save profiler outputs under the run's `profile/` directory; profiling is not part of the verified timing manifest.

## Artifacts and interpretation

Verification writes `evaluations/experiment-005-hilbert-tile-traversal/RUN_ID/verify/`, with `result.json`, source snapshots, the accepted Muse proposal snapshot, per-shape Hilbert order and compiler artifacts, and a SHA-256 manifest. Screening and confirmation each write an independently manifested timestamped directory under the same run. Failed stages retain `result.json` but do not get a complete manifest. The dashboard shows only validated stages.

The tested hypothesis receives preliminary latency support only if Hilbert's paired bootstrap lower bound exceeds one against **both** alternatives for the same shape in both screening and confirmation. No cache-locality support is claimed without profiling. Similar or worse results, or mapping overhead that offsets any locality gain, weaken this direction for the tested hardware and workloads. Human review is needed before any broader conclusion.

Run CPU-only deterministic tests with `.venv/bin/python -m pytest -q`. No GPU stage runs during import or testing.
