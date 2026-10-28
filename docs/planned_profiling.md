# Planned profiling experiment 001

`planned_profile.py --prepare-only` verifies the accepted advisory plan, its
original prompt and audit, the two cited trusted experiments, their six saved
correctness cases, and the three current reviewed kernel hashes. The manifest
also binds the profiling runner's own source hash. It queries the
installed Nsight Compute metric catalog for Ada AD106 without launching a GPU
kernel. It writes a new immutable `manifest.json` and `COMMANDS.txt` under
`evaluations/planned-profiling/<run-id>/`. The plan itself grants no execution
permission. No generated `candidate.py` is loaded.

Prepare with `.venv/bin/python planned_profile.py --prepare-only`. The command
prints the frozen manifest path and digest; `COMMANDS.txt` in that directory
contains the exact review command and six separate capture commands.

The fixed FC1 experiment uses FP16 inputs and output, FP32 accumulation,
`512×3072×768`, seed 42, 64×128×32 tiles, four warps, and three stages. The
capture order is baseline, reviewed mask hoisting, reviewed two-slab, then the
same order for independent second captures. Each `--run-one` invocation starts
a separate worker process, rechecks the manifest hash and source identities,
generates its own deterministic inputs, verifies the selected implementation
against the scoped FP16 `torch.mm` reference, compiles and warms up ten times,
then opens a CUDA profiler start/stop range around exactly one GEMM launch.
Nsight Compute starts with profiling disabled and collects at most one matching
`matmul_kernel` launch. The wrapper has a 170-second subprocess timeout; generated review
commands also apply an external 180-second timeout. Its raw CSV and stderr
command log have unique names and cannot overwrite existing capture files.

The manifest records selected metric definitions and units, plus null values
for uncollected measurements. Profiler observations are separate from prior
CUDA-event screening. The launch occupancy-limit counters indicate resource
constraints; they do not establish measured achieved occupancy. The offline
catalog does not prove collection permission or that all counters will return
values. Nsight Compute replay, uncontrolled cache state, and unmodified clocks
can change execution conditions. Static compiler resources and historical
correctness apply only to the saved reviewed sources; a capture rechecks the
source hashes and FC1 correctness before collecting counters. No observed
stall cause or performance improvement follows from preparation.

Review `manifest.json`, its SHA-256, and `COMMANDS.txt` before explicitly
running one of the bounded commands. An unsuccessful capture keeps its raw
output and log for diagnosis; it is not valid evidence until the expected
kernel marker, one profiled launch, metric units, and profiler warnings are
reviewed. The six completed captures were validated by `profile_report.py` and
summarized in `evaluations/planned-profiling/20261028T004249599584Z/report.json`
and `summary.md`. The validator reads counters by metric name, requires exactly
one expected GEMM row per capture, checks the correctness marker and profiler
command, and hashes every raw CSV and log. It keeps observations,
interpretation, and speculation separate. `experiment_memory.py` loads this
report only when its metadata hashes and raw captures still validate;
`experiment_planner.py` exposes it as `profiling_evidence` in the next advisory
prompt. This is a profiler study, not a kernel optimization or confirmed
latency improvement.
