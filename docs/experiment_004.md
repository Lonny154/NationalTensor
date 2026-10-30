# Experiment 004 — partial M boundary tiles

This is a new compiler and correctness diagnostic. It does not run a benchmark or call Muse. The control is FP16 GEMM 128×3072×768; the new workload is 130×3072×768. Both use 64×128×32 tiles, four warps, three stages, input seeds 42 and 43, and the RTX 4060 Ti. The grids are 2×24 and 3×24 respectively. The last M tile of the new workload contains two valid rows.

The compared sources are the original `benchmark_proposal.matmul_kernel` baseline and the unchanged, independently reviewed `trusted_compute_kernel.matmul_kernel` mask-hoisting implementation. The trusted variant includes its existing explicit `tl.range` staging hint. This experiment does not substitute or execute the original Muse-generated source. Source hashes and the validated Experiment 003 identity are bound into the new result.

After reviewing the implementation, explicitly run one GPU diagnostic from the repository root:

```bash
.venv/bin/python experiment_004_partial_boundary.py --run
```

Each invocation creates a fresh `evaluations/experiment-004-partial-boundary-tiles/<UTC timestamp>/` directory. It never writes to Experiment 003 or either accepted debrief. For each shape it saves `diagnostic.json`, `correctness.json`, baseline and trusted TTIR, TTGIR, LLIR, PTX, cubin, decoded SASS, and ordered PTX/SASS diffs. The root result records environment, source identity, compiler comparisons, outcome, and a screening recommendation. `manifest.json` hashes all artifacts and source snapshots. Failed runs retain `result.json` with an error but have no complete manifest and do not enter the dashboard record.

Correctness uses the established scoped FP16 `torch.mm` reference, `rtol=0.02`, `atol=0.02`, and four checks per shape. Outputs begin as NaNs to reveal missed stores. The full output and final M tile are checked separately; CUDA synchronization surfaces kernel errors. No memory sanitizer is run, so absence of out-of-bounds access is supported by the reviewed masks and these checks rather than by an instrumented memory trace.

The comparison uses complete `.text.*` bytes, relevant constant and `.nv.info` sections, register/shared-memory/spill metrics, decoded SASS streams, ordered PTX, and predicate/branch counts. Instruction counts are descriptive; they do not prove why code differs or whether it is faster. If the M=128 control no longer matches, the outcome flags control drift and defers benchmarking. If the control matches but M=130 differs, a separate paired CUDA-event benchmark may be considered after human review. If both match, no timing is warranted for this unchanged pair. No runtime outcome is inferred in any case.

The read-only artifact loader adds only complete, hash-consistent Experiment 004 records to the existing Timeline/compiler filter and evidence detail page. The approved planner memory is unchanged, preserving its source binding. Deterministic tests use CPU tensors and copied historical binaries; they do not execute CUDA code.
