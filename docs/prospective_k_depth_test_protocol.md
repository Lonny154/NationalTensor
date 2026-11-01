# Prospective K-depth prediction test — prepared, GPU work pending

## Frozen forecast and scoring

Prepared measurement run: `20261101T020025480962Z` in `evaluations/prospective-k-depth/v1/`. The independent Muse forecast is run `20261101T015247349750Z`. Its original audit SHA-256 is `c384ef6508cec7994155477b80d0cea667093149712c8c26438f773afafaebb5`; its audit manifest SHA-256 is `14c0cb9263ab08b6a43b8eecbbf9b8903cf9ef1662dfe975a1d351746224469b`. Read-only forecast validation passes and the immutable prediction says **lower** for the stage-3/stage-2 latency ratio at `512×3072×1536`. It offers no defensible point estimate or quantitative range.

The frozen reference is `0.953120870845799`, the median of the three confirmed K=768 session medians. The preregistered absolute deadband is `0.04`: target median `< 0.913120870846` means **lower**; `0.913120870846–0.993120870846` means **near**; `> 0.993120870846` means **higher**. Score the first preregistered correctness-passing K=1536 session only. A category is conclusive only if its descriptive bootstrap interval lies wholly within that category and the apparent movement is separated from A/A variation. Otherwise report **inconclusive**. Preserve every attempted result. The fresh K=768 control informs mechanism interpretation; it cannot replace or select the scored target result. This rule and the Muse audit predate target timing and remain unchanged.

The model did not see the numerical ±0.04 deadband in its prompt. Its word “lower” is therefore evaluated against a stricter frozen threshold than its own prose established. Report that limitation with any score.

## Measurement plan

The new runner, [`prospective_k_depth_test.py`](../prospective_k_depth_test.py), reuses the reviewed `trusted_two_slab_kernel.launch`, `trusted_two_slab_experiment._inputs`, `pipeline_stage_discovery.timed_phase`, and its phase validator. It pins the source files, three discovery-session manifests, original prediction audit, model decision, and evaluation rule in a manifested setup. Both workloads are FP16 `512×3072×K` with `64×128×32` tiles, four warps, direct `8×24` output grid, preallocated outputs, and only `num_stages=3` versus `2` within each comparison. The target K is 1536; the control K is 768. No prior result is copied into this namespace.

Run target `target_k1536` first, in a fresh process. It uses fresh correctness seeds 45 and 46, the original `torch.mm` reference policy, `rtol=atol=0.02` for both variants, and a direct stage-to-stage agreement check. Any correctness failure stops timing and is manifested. Timing uses seed 45 inputs, 14 balanced paired rounds, 20 A/A rounds for each stage, 200 eager launches per timed block, 10 initial and per-block warmups, and 2,000 seeded bootstrap draws. Stage 3 is the numerator. All CUDA-event block values, host submission diagnostics, orders, telemetry snapshots, compiler artifacts, summaries, and errors are retained; no outlier filter is applied.

Run `control_k768` only by a separate approval and command after the target has been attempted. It uses new seeds 47 and 48, the same method and workload dimensions except K, and the reverse A/A phase order. The fixed target-before-control order leaves a time/state confounder. If a K-depth mechanism claim matters, a separately authorized reversed shape-order campaign would be needed; it would not replace this first target for forecast scoring. Different K values also change input footprint, compiler specialization, and cache behavior, so a ratio difference alone does not prove deeper copy/compute overlap.

Each session has 21,600 timed launches plus correctness, priming, and warmups. Each session is a one-attempt path; a failed or interrupted attempt remains in place and cannot be silently retried. An interrupted process may leave a session directory without a final manifest, which validation treats as an integrity failure requiring operator inspection. No profiler or Muse call is reachable from this runner.

## Offline verification and commands for separate authorization

The manifested setup SHA-256 is `ca4565e1bf76ecf19af566b53d569387123fab07c910611a94e5a57e15df75c4`; its manifest SHA-256 is `1cd6307be81780696e5e5bfa43aca6ffbdb50d153aea9f7d8d3ed1cb38529e11`. Read-only validation shows both sessions `not_run`. The original discovery's three sessions and forecast audit also validate. CPU-only targeted tests: **19 passed**. The public-evidence verifier passed.

Commands below are prepared for future, separate human authorization. Each `--run-session` command executes GPU kernels and is **not authorized by this offline preparation**:

```bash
.venv/bin/python prospective_k_depth_test.py --validate --run-id 20261101T020025480962Z
.venv/bin/python prospective_k_depth_test.py --run-session --run-id 20261101T020025480962Z --session target_k1536 --human-approved
.venv/bin/python prospective_k_depth_test.py --validate --run-id 20261101T020025480962Z
.venv/bin/python prospective_k_depth_test.py --run-session --run-id 20261101T020025480962Z --session control_k768 --human-approved
.venv/bin/python prospective_k_depth_test.py --validate --run-id 20261101T020025480962Z
```

The setup records preparation without GPU authorization. No GPU kernel, Muse request, Nsight capture, historical edit, commit, or push was performed while preparing this protocol.
