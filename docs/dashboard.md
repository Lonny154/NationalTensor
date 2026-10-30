# National Tensor research dashboard

The Day 4 reading layout keeps the same routes and verified records. Each page leads with one question and a clear next step. Totals, timing tables, citations, hashes, and raw artifacts remain available through labeled expandable sections and evidence links. Research Mode and Hollywood Mode share the same five navigation labels. The cinematic Muse debrief renders each character turn on its own line while keeping verified observations, Muse interpretation, and fiction distinct. Keyboard focus and reduced-motion preferences are supported by the shared styles.

The dashboard reads the checkpointed files under `evaluations/` and
`docs/day2_checkpoint.md`. An explicit **Request one hypothesis** action can
call Muse and write a new proposal under `evaluations/kernel-evolution/`.
The Overview page also displays the latest saved advisory research plan from
`evaluations/experiment-planner/`, if one exists. The separate
[Research workflow](research_workflow.md) tab exposes advisory planning, human
review, and one registered trusted compiler-diagnostic runner.
The server binds to `127.0.0.1` with debug mode disabled. Generated source and
model text are displayed through Flask's escaped templates; they are never
imported, compiled, approved, or executed.

Install the small web dependency in the repository environment, then launch:

```bash
.venv/bin/python -m pip install -r requirements-dashboard.txt
.venv/bin/python dashboard_app.py
```

Open `http://127.0.0.1:8501/`. Stop with Ctrl-C.

The shared **Research Mode / Hollywood Mode** switch changes presentation in
place. Hollywood Mode v2 uses dedicated layouts, not the Research Mode panel
arrangement: Mission Control has a briefing/action split and recorded workflow
sequence; Mission Archive uses dossiers with all five filters; the Treasure
Vault uses paired-ratio receipts with latencies and recorded MAD/A/A variability;
Signal Integrity uses calibration investigation files and settling condition
reports; the Heist uses a target/evidence board and authorization/dispatch desks.
The theatrical copy is presentation, and scientific classifications remain
explicit. Both modes use the same protected forms and validated evidence.

### Operation Stolen Cycles

Start the dashboard with `.venv/bin/python dashboard_app.py`, open
`http://127.0.0.1:8501/research`, switch to **Hollywood Mode**, and select
**Replay Experiment 003**. The replay reads the hash-validated historical
diagnostic, creates no job or approval, and advances only when the viewer clicks.
Its vault reports four passing checks, 6,912 identical executable bytes per
kernel, and intentionally skipped runtime benchmarking for the tested
configuration. If historical validation fails, replay is unavailable.

For a live mission, select saved plan `20261028T145028833360Z`, click
**Open Live Mission**, and review the briefing and clearance. Close the overlay
to use the existing **Authorize the Operation** form. After approval, use the
separate **Initiate the Heist** form. That protected submission starts the fixed
trusted diagnostic and opens the mission at live dispatch after navigation.
Closing the overlay never cancels the job. The vault opens only after a completed
job has a validated memory entry; failed jobs and missing evidence get accurate
reports. Live runs are workflow validation replays, not new independent research
evidence. This manual live path can use the GPU; development and automated tests
do not.

Mission sound is off by default, with an opt-in mute control. Escape closes the
overlay. Reduced-motion settings remove decorative animation. Cinematic terminal
copy is labeled separately from the linked job record and execution logs.
The vault dial, warning lamps, and scanning beam are decorative. The beam runs
only for queued, running, or validating live jobs; replay labels its scan as an
archive effect. **Skip effects** stops animation independently of the sound
control. The briefing explains the proposed loop-check change, the clearance
and dispatch explain the compiler/correctness checks, and the vault summarizes
what validated evidence actually established. Compiler equivalence, confirmed
improvement, measured screening slowdown, inconclusive evidence, and failed
execution have separate endings. A completed job without validated evidence
keeps the vault sealed.

CPU checks: `.venv/bin/python -m pytest -q tests/test_mission.py`.
Chromium scene and synthetic live-state checks: `NT_MISSION_BROWSER=chromium
.venv/bin/python -m pytest -q tests/test_mission.py -k chromium`. The tests
use saved or fixture evidence and launch no GPU code. Locked, scanning, and
revealed screenshots are saved as `locked.png`, `scanning.png`, and
`revealed.png` under `evaluations/dashboard-previews/operation-stolen-cycles/`.
Approval and execution remain separate. No percentages or performance gains
are simulated. Active job animations reflect only saved queued/running/validating
states and stop under reduced-motion preferences.

The preference is saved in browser local storage for this origin; switching
does not navigate, submit forms, change selections, or reset jobs or approvals.
If storage is unavailable the current page still switches, but reloads default
to Research Mode. JavaScript is required for Hollywood Mode. Research Mode
remains the default. Use the Timeline filters in either mode; workflow replays
remain explicitly distinct from independent research evidence.

The Heist moves the original workflow controls and polling node into dedicated
slots, then restores them when returning to Research Mode. It never clones
forms, changes their fields, or adds an execution endpoint. Mission Control's
next-action link navigates to the appropriate review/dispatch desk. Its phase
sequence describes persisted records, not a progress estimate. A completed job
is described as verified only when its validated memory entry is present.
Detailed scientific evidence pages continue to use the existing evidence summary.

CPU tests: `.venv/bin/python -m pytest -q tests/test_hollywood_mode.py tests/test_hollywood_layouts.py`.
Optional Chromium visual smoke (isolated fixtures, GPU executor blocked):
`NT_HOLLYWOOD_BROWSER=chromium .venv/bin/python -m pytest -q tests/test_hollywood_mode.py tests/test_hollywood_layouts.py -k browser`.
V2 screenshots, including side-by-side Overview, Reliability, and Workflow
comparisons, are saved under `evaluations/dashboard-previews/hollywood-v2/`
(override with `NT_HOLLYWOOD_V2_SCREENSHOTS`). V1 preference smoke screenshots
use `hollywood-v1/` (override `NT_HOLLYWOOD_SCREENSHOTS`). Browser layout tests
check switching, original form-node identity, selected plan and job-snapshot
preservation, persistent preference, and zero POST requests. The separate
`NT_APPROVAL_BROWSER=chromium` test in `test_research_workflow.py` submits the
native form from Hollywood Mode against isolated fixtures and verifies CSRF,
Origin, approval-only behavior, and no GPU execution.

For a live proposal, place `MODEL_API_KEY` in the server environment or the
repository `.env` file, launch the dashboard, open the Overview page, and
click **Request one hypothesis** once. The fixed request uses
`muse-spark-1.3`, FC1 `512 × 3072 × 768` FP16, and the latest verified FC1
kernel proposal as prior context. No workload, model, path, or command is
accepted from the browser. The dashboard uses the existing proposal-only
validation and artifact writer with one model attempt, a 30-second API
timeout, and no SDK retries. The CLI retains its existing retry behavior.
The page polls for the request state and links to the saved proposal; a fresh
timeline page reads the artifact index without restarting the server.
Experiment links in Overview, Timeline, and Performance open a dedicated
`/experiments/<experiment-id>` page with the full saved record at the top.
Its **Back to Timeline** link returns to the chronological list. Older
`/?view=timeline&experiment=...` links redirect to the detail URL when the
experiment ID is valid and present.

The Timeline now projects all verified research activity from reconstructed
experiment memory, saved advisory plans, and existing review/job records.
`dashboard_timeline.py` creates display events on each read; it stores no second
history or cache. Each event preserves its recorded outcome and links to the
existing experiment, memory-evidence, plan-review, or job-record page.

Filters are All activity, Scientific experiments, Compiler diagnostics,
Research decisions, and Workflow validation. The scientific filter excludes
workflow replays. The compiler filter includes genuine compiler diagnostics
and diagnostic replays, with the latter visibly labeled as workflow validation.
Proposals, human reviews, and completed deferrals belong to Research decisions.
Date-only review records explicitly state that the time was not recorded.

The Performance table uses only recorded latency medians and paired ratios,
with each comparison scoped to its own run. Compiler diagnostics and replays
without timings appear as **Not measured** in the Timeline. Existing screening,
inconclusive, negative, and confirmation classifications are preserved.
Hash-invalid artifacts and inconsistent timing records are omitted with a
validation warning. Legacy structural files retain their existing source-hash
and internal timing-consistency checks; no retrospective signed manifests are
created for those files.

`GET /api/research/timeline?filter=all` exposes the same renderer-neutral event
data and evidence links. The other filter keys are `scientific`, `compiler`,
`decisions`, and `workflow`. This projection can support a future Mission
History view; no Hollywood Mode is implemented here.

Only one request runs at a time in the server process. The five states are
Ready, Requesting proposal, Validating response, Saved for review, and Failed.
If a response fails static review or an API call fails, the server reports a
generic failure without exception text; any received response and rejection
remain in the proposal audit file. The action requires a session CSRF token,
an exact loopback Host and matching HTTP Origin. Requests have a 1 KiB body
limit. The app has no CORS allowance, uses a strict content security policy,
and sends no credentials to the browser. The separate controlled research
workflow has one explicit approval and execution path for trusted code; the
generated-source proposal action never launches GPU work.

## Experiment memory

`experiment_memory.py` rebuilds a bounded index from the saved structural,
generated-proposal, and trusted-evaluation artifacts on each read. No new
database or persistent cache is created. Generated proposals retain their
own unevaluated status; correctness, compiler evidence, and screening remain
attached to the independently reviewed implementations. Prompt retrieval
includes at most six summaries, prioritizing matching workload and GPU,
evaluated results, compiler evidence, and related mechanisms. It records
timing method, numeric configuration, and baseline/launcher source-version
compatibility, and labels incompatible measurements as context only.

Both the CLI and dashboard supply this memory to the next Muse prompt. The
dashboard's proposal details show earlier experiments in the same known
optimization families. Classification is conservative: unknown ideas remain
reviewable, and matching source structure is only a hint. A full-slab loop
with a separate K tail is primarily tagged K-bound specialization; mask
hoisting can remain a secondary tag. After static
validation, an exact SHA-256 repeat is saved in the raw audit as
`duplicate_source`, linked to the original proposal, and not saved as a new
review candidate. A different source in a known family can still be saved
for human review with its relationship recorded. No generated source is run.

The overview counts the saved structural and accepted kernel proposals, and
separately counts rejected responses. The timeline links an original Muse
source to a trusted experiment only when saved hashes match. A later proposal
is linked to trusted feedback only when its saved prompt contains the matching
feedback and source hashes. The generated source itself remains unevaluated.
The detail page distinguishes this direct feedback from historical trusted
summaries retrieved into the saved prompt. It shows the latter with local
source-hash verification status. Earlier generated proposals in the prompt
remain labelled unevaluated. The related-experiments list is rebuilt for
viewing and does not imply that every listed record reached Muse.

The live memory-informed proposal in
`evaluations/kernel-evolution/20261027T224653132929Z/` remains **awaiting
review** (source SHA-256
`1a41b6c475acd16ef382127004cb1c2fcaf7280c3269f58e74c4a6cacd98e52a`).
Its static AST review accepted the source for review only. Inspection of the
installed Triton 3.6.0 implementation found a likely compilation blocker:
the proposal passes scalar `BN` and `BK` values to `tl.max_contiguous` for 2D
pointer tensors. Triton converts each scalar hint to a one-element list, then
requires one value per tensor dimension. This is an inference from the
installed API, not a confirmed compilation failure. The generated source has
not been imported, compiled, checked for correctness, or benchmarked.

Performance plots show every saved paired screening sample on a logarithmic
time axis, with the measurement method and A/A variability beside it. The
reliability page keeps eager Python launch and CUDA Graph A/A captures
separate. Matmul Inspector's large comparison JSON files are read only after
selecting the detailed compiler view for the two-slab experiment.

Display labels keep source review, correctness, and performance separate. The
saved `optimization_status` remains visible. Known ambiguous Day 2 screenings,
including small-M Grouped-M4 and the mask/pipeline trusted test, are labelled
**Inconclusive** in the UI. A raw timing warning appears when a side's maximum
exceeds four times its minimum or median; an A/A warning appears at 3% median
absolute ratio deviation or above. These are descriptive display thresholds:
they do not remove samples, adjust timings, or test statistical significance.
The hypothesis/evidence comparison is available only for the two trusted
studies with matching saved Muse and trusted-source hashes. It distinguishes
the generated proposal from the independently written kernel.

This MVP understands the committed Day 2 structural-loop JSONL, accepted
kernel-evolution directories, and the two trusted experiment layouts. It
ignores prepared-only requests and older Day 1 search formats. Missing optional
files are omitted or reported as artifact warnings. No screening ratio is
treated as confirmation or statistical significance. No current confirmation
artifact is loaded; a future confirmation format would need an explicit
verified-artifact loader before the UI can show **Confirmed**.
