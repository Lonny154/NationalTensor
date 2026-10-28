# Controlled research workflow: first slice

The dashboard's **Research workflow** tab connects a saved Muse advisory plan to
an explicit human decision, a separate execution action, validated artifacts,
and persistent experiment memory. The generated kernel-source proposal action
remains separate. Planning never approves or starts execution.

Exactly one trusted runner is registered: `mask_hoisting_m128_diagnostic_v1`.
It accepts only the saved Experiment 003 plan SHA-256
`5a5ec80842b9a544f02471bd0517d77c7ce598d82a88355d1980ed418e5fd8b1`.
Other advisory plans can be viewed or rejected, but have no execution mapping.
This deliberately avoids interpreting model prose as a program or command.

The fixed workload is FP16 128×3072×768 with 64×128×32 tiles, four warps,
three stages, and correctness seeds 42/43. Approval permits compiler diagnosis
and the four baseline/trusted correctness checks at rtol=0.02, atol=0.02.
Timing screening is excluded. Because this replays completed Experiment 003,
every job is labeled **workflow validation**, not new independent research
evidence; these replays are inspectable in memory but excluded from ordinary
planner evidence retrieval.

## Manual demonstration, after implementation review

1. Start the existing dashboard from the project root in the GPU-enabled
   environment:

   ```bash
   .venv/bin/python dashboard_app.py
   ```

2. Open `http://127.0.0.1:8501/research`. Select saved plan
   `20261028T145028833360Z` if a newer unsupported advisory is displayed.
   Read its question, mechanism, citations, and the exact approval identity.
3. Click **Approve fixed diagnostic**, or **Reject execution** to exercise
   the rejection gate. Approval alone creates no execution job. Reviews are
   immutable; a rejection cannot later be changed to approval in this slice.
4. After approval, click **Execute approved diagnostic**. This is the separate
   manual GPU trigger. The page polls job status through queued, running,
   validating, and completed/failed states.
5. On completion, open **Inspect validated evidence in persistent memory**.
   Expect `compiler_equivalent_screening_skipped` if the compiler still emits
   equivalent code; otherwise expect `compiler_difference_screening_skipped`.
   Either outcome is a completed diagnostic, with no latency or optimization
   success claim. Inspect the job record and stdout/stderr links.
6. Reload or restart the dashboard and verify that the job and memory evidence
   persist. Another execute click for this approval is rejected.

The separate **Request one advisory plan** button makes exactly one explicit
Muse request using `experiment_planner.py` and verified memory. It requires
`MODEL_API_KEY` in the server environment or repository `.env`, a 30-second
timeout, and no SDK retries. It is not needed to demonstrate the saved
Experiment 003 path. A new response never starts a diagnostic.

If a dashboard started before the native-form fix returns 403 on approval,
restart it and reload `/research` using the same hostname throughout. The page
must send `Referrer-Policy: same-origin`: Chromium sends `Origin: null` for
native form POSTs under `no-referrer`, which the strict Origin check rejects.
Reloading also obtains a fresh session token after a server restart. Keep the
Origin and CSRF checks enabled; approval alone still creates no execution job.

## Artifacts and controls

`evaluations/research-workflow/reviews/<plan-id>/review.json` binds the human
decision to the saved plan hash, registered runner, all local runner dependency
source hashes, plan/audit and prior-evidence input hashes, fixed configuration,
allowed phases, and timeout. The binding
is rechecked before queueing, immediately before spawning, and in the trusted
worker before it imports the GPU runner. Source or plan edits invalidate
approval and require a new implementation review.

Jobs live under `evaluations/research-workflow/jobs/<job-id>/`. Each retains
`job.json`, stdout/stderr logs, and an `artifacts/` directory with raw compiler
outputs, correctness evidence, source snapshots, and a hash manifest. Validation
rechecks approval, saved plan audit/hash, archived source identities, exact
correctness cases and tolerances, compiler artifact hashes, and complete ELF
executable/constant/launch-metadata comparisons. Only completed, validated jobs
enter persistent memory; the dashboard revalidates evidence when it is read.

The parent starts a fixed argv list with a 180-second timeout and no shell.
The child environment omits API credentials. A workspace file lock permits
only one experiment job across dashboard processes. The worker inherits the
lock descriptor so a server crash does not permit overlapping orphan GPU work.
Timeout kills the process group; the worker also has an OS wall-clock alarm
that remains active if the server exits. Interrupted owners are recorded without an
automatic retry; partial artifacts and logs remain available. There is one
attempt per approval, including failed/interrupted attempts.

This slice has no automatic retries, cancellation UI, approval replacement,
remote authentication, or additional registered runners. It assumes the existing
single-user loopback dashboard and trusted repository. It does not import or
execute model-generated Python/Triton or Python saved inside artifact folders.
The existing generated-source evaluator is not connected to this workflow.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_research_workflow.py tests/test_mask_hoisting_memory.py tests/test_dashboard.py tests/test_dashboard_proposals.py tests/test_experiment_planner.py
```

Workflow execution tests use saved artifacts as mock output. They never compile
or launch a GPU kernel. The manual GPU demonstration above remains a separate
action after reviewing the implementation.

An optional real Chromium integration test clicks the rendered approval form
against an isolated temporary dashboard, with experiment execution blocked:

```bash
NT_APPROVAL_BROWSER=chromium .venv/bin/python -m pytest -q tests/test_research_workflow.py -k real_browser_native_approval
```
