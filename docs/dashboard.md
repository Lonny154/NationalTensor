# Day 2 research console

The dashboard reads the checkpointed files under `evaluations/` and
`docs/day2_checkpoint.md`. An explicit **Request one hypothesis** action can
call Muse and write a new proposal under `evaluations/kernel-evolution/`.
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

The overview counts the saved structural and accepted kernel proposals, and
separately counts rejected responses. The timeline links an original Muse
source to a trusted experiment only when saved hashes match. A later proposal
is linked to trusted feedback only when its saved prompt contains the matching
feedback and source hashes. The generated source itself remains unevaluated.

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
