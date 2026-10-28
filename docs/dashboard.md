# Day 2 research console

The dashboard reads the checkpointed files under `evaluations/` and
`docs/day2_checkpoint.md`. An explicit **Request one hypothesis** action can
call Muse and write a new proposal under `evaluations/kernel-evolution/`.
The Overview page also displays the latest saved advisory research plan from
`evaluations/experiment-planner/`, if one exists; [the planner CLI](experiment_planner.md)
is a separate explicit action.
The server binds to `127.0.0.1` with debug mode disabled. Generated source and
model text are displayed through Flask's escaped templates; they are never
imported, compiled, approved, or executed.

Install the small web dependency in the repository environment, then launch:

```bash
.venv/bin/python -m pip install -r requirements-dashboard.txt
.venv/bin/python dashboard_app.py
```

Open `http://127.0.0.1:8501/`. Stop with Ctrl-C.

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

Only one request runs at a time in the server process. The five states are
Ready, Requesting proposal, Validating response, Saved for review, and Failed.
If a response fails static review or an API call fails, the server reports a
generic failure without exception text; any received response and rejection
remain in the proposal audit file. The action requires a session CSRF token,
an exact loopback Host and matching HTTP Origin. Requests have a 1 KiB body
limit. The app has no CORS allowance, uses a strict content security policy,
and sends no credentials to the browser. It still has no GPU evaluation path;
human approval and isolated execution remain separate workflows.

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
