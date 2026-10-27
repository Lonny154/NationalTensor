# Model-generated Triton GEMM, proposal milestone

This experiment is separate from `optimize.py` and `structural_loop.py`. It
targets only the 512×3072×768 FP16 workload with the existing 64×128×32,
four-warp, three-stage launcher. A proposal replaces only the decorated
`matmul_kernel` function in `benchmark_proposal.py`; the launcher interface,
signature, and one-kernel launch count remain fixed. The model may change
Triton indexing and loop organization. It cannot submit host Python, imports,
additional kernels, or a new launcher.

## Proposal format and review

Muse returns one JSON object with exactly these fields:

```json
{
  "kernel_source": "@triton.jit\ndef matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):\n    ...complete Triton body...\n",
  "hypothesis": "Recompute K offsets inside the reduction loop instead of advancing two pointer tensors.",
  "expected_mechanism": "A different instruction schedule may reduce pointer dependency chains; it may also be slower.",
  "reasoning_from_feedback": "The reviewed schedule experiments produced no confirmed gain, so this proposal changes the kernel body rather than CTA ordering."
}
```

The `...complete Triton body...` text illustrates the schema and must be
replaced by a full kernel. A complete, hand-written example is in
`examples/kernel_evolution_proposal.json`; it changes pointer-index calculation
inside the K loop and is supplied only to show the accepted format. It has not
been executed or measured. `kernel_evolution.py` gives Muse the exact baseline
kernel and launcher source, target GPU, workload, fixed numeric configuration,
and recent FC1 structural results. It parses the JSON and AST, requires the
fixed signature, rejects imports and unsupported calls, and rejects a kernel
whose AST is identical to the baseline. These checks are a review filter,
**not a security sandbox or proof of numerical correctness**.

After implementation review, the first live proposal-only command is:

```bash
.venv/bin/python kernel_evolution.py propose --request-from-muse \
  --gpu 'NVIDIA GeForce RTX 4060 Ti'
```

This makes one accepted proposal at most, with up to three API attempts for
invalid responses. It writes a timestamped directory under
`evaluations/kernel-evolution/` containing `candidate.py`, `metadata.json`,
and `proposal_audit.jsonl`. The source is never imported, compiled, or run by
that command. Rejected raw responses and reasons remain in the audit. No Muse
API call was made while implementing this milestone.

## Independent compute-focused request after the first review

The first accepted proposal at `20261027T144337580507Z` changes only the tile
ordering. CPU arithmetic shows full, unique tile coverage for the fixed
64×128 tiles on all three structural workloads. Its flattening expression
`pid_m0 * num_pid_n + pid_n0` does not match the 2D grid's dimension-0-fastest
linear order, so the proposed Grouped-M4 locality does not follow from the
launcher. GPU block scheduling itself is not guaranteed. The first rejected
response used `GROUP_M: tl.constexpr = 8`, an annotated local assignment.
That construct was unsupported by the static review filter; replacing only
the annotation with a plain assignment in memory passes the filter. The
filter remains unchanged. No compilation or measurement artifact is present
for either generated response.

Prepare a separate request artifact without a model call:

```bash
.venv/bin/python kernel_evolution.py propose --prepare-only \
  --review-from evaluations/kernel-evolution/20261027T144337580507Z
```

This saves the exact prompt and its hash under a new timestamped directory in
`evaluations/kernel-evolution/`. It checks the first candidate's SHA-256 and
includes its review findings, the prior rejected attempt, baseline source,
launcher, hardware, and structural screening history. The prompt requires a
change to computation, K-loop organization, loads, or pipelining, and excludes
Grouped-M4, Split-K, and program-ID-only swizzles. The compute-focused parser
requires the two direct program IDs and rejects a candidate whose loop and
operand-load logic match the baseline; this is a review check, not a proof
that a change is beneficial or safe.

After reviewing the prepared prompt, request the next proposal with:

```bash
.venv/bin/python kernel_evolution.py propose --request-from-muse \
  --review-from evaluations/kernel-evolution/20261027T144337580507Z
```

This creates a new proposal-only artifact. It does not import, compile,
execute, or benchmark any generated source. The earlier proposal and audit
remain intact. This independent request does not claim evaluation feedback;
the measured-feedback path below still requires an evaluation artifact.

## Feedback from a separate trusted implementation

The manually reviewed implementation of Muse candidate
`38b23fad21dc8bb6d194239c7ac0d0a20b1c8566de4055442e3e927d8f97e59a`
has its own source SHA-256 and experiment record. The generated candidate itself
was not evaluated. Supply the external `feedback.json` alongside the original
unevaluated proposal; the CLI checks both identities against `experiment.json`
and retains this provenance in the next proposal metadata and audit. This path
does not use `--feedback-from`, which is reserved for evaluations of the exact
generated source.

Prepare the next prompt without a Muse call:

```bash
.venv/bin/python kernel_evolution.py propose --prepare-only \
  --review-from evaluations/kernel-evolution/20261027T150316362990Z \
  --trusted-feedback evaluations/trusted-compute/20261027T154233319712Z/feedback.json
```

After reviewing the prepared prompt, request one new candidate in proposal-only
mode:

```bash
.venv/bin/python kernel_evolution.py propose --request-from-muse \
  --review-from evaluations/kernel-evolution/20261027T150316362990Z \
  --trusted-feedback evaluations/trusted-compute/20261027T154233319712Z/feedback.json
```

The prompt includes correctness, screening and A/A variability, PTX and
resource observations, and asks for a different compute mechanism that could
survive compilation. Static validation and the separate approved evaluation
gate remain unchanged.

### Two-slab compiler feedback

The independently reviewed two-slab kernel has separate `check.json` and
`screening.json` records. Its `feedback.json` links those records to existing
Matmul Inspector PTX/SASS comparisons and the one-launch NCU profile. The
proposal prompt labels measured results separately from possible occupancy
effects, which were not measured. The original Muse source remains unevaluated.

Inspect the next request without making an API call:

```bash
.venv/bin/python kernel_evolution.py propose --prepare-only \
  --review-from evaluations/kernel-evolution/20261027T160014125560Z \
  --trusted-feedback evaluations/trusted-two-slab/20261027T161358957319Z/feedback.json
```

After reviewing that request, generate one candidate in proposal-only mode:

```bash
.venv/bin/python kernel_evolution.py propose --request-from-muse \
  --review-from evaluations/kernel-evolution/20261027T160014125560Z \
  --trusted-feedback evaluations/trusted-two-slab/20261027T161358957319Z/feedback.json
```

## Approved evaluation gate

The separate `kernel_evolution_eval.py` worker requires a reviewer-created
approval file naming the **exact** SHA-256 from `metadata.json`:

```json
{
  "scope": "kernel_evolution_v1",
  "candidate_sha256": "<64-character hash from candidate metadata>",
  "approved_by": "<human reviewer>",
  "approved_at": "2026-10-27T15:00:00+00:00"
}
```

Build a **sanitized trusted-code bundle** containing only these reviewed files;
do not mount the project directory, `.env`, credentials, `.git`, or the Muse
proposal audit into the worker:

```bash
mkdir -p /tmp/national-tensor-eval-bundle
cp benchmark_core.py benchmark_proposal.py reference_precision.py \
   optimize.py grouped_matmul.py split_k_matmul.py \
   kernel_evolution.py kernel_evolution_eval.py \
   /tmp/national-tensor-eval-bundle/
```

For a prebuilt, locally reviewed GPU image named `national-tensor-gpu:local`
with Python 3.13, PyTorch, Triton, `openai`, and `python-dotenv` installed, the host-side
screening launch is:

```bash
CANDIDATE_DIR="$PWD/evaluations/kernel-evolution/<candidate-id>"
APPROVAL_DIR="$PWD/approvals/kernel-evolution/<candidate-id>"
OUTPUT_DIR="$PWD/evaluations/kernel-evolution-evals/<candidate-id>"
mkdir -p "$OUTPUT_DIR"
docker run --rm --gpus all --network none --read-only \
  --user "$(id -u):$(id -g)" --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 8g --pids-limit 128 --cpus 4 \
  --ulimit cpu=600:600 --ulimit nofile=256:256 \
  --env TRITON_CACHE_DIR=/tmp/triton-cache \
  --env CUDA_CACHE_PATH=/tmp/cuda-cache \
  --env XDG_CACHE_HOME=/tmp \
  --tmpfs /tmp:rw,nosuid,size=1g \
  --mount type=bind,src=/tmp/national-tensor-eval-bundle,dst=/work,readonly \
  --mount type=bind,src="$CANDIDATE_DIR",dst=/candidate,readonly \
  --mount type=bind,src="$APPROVAL_DIR",dst=/approval,readonly \
  --mount type=bind,src="$OUTPUT_DIR",dst=/out \
  national-tensor-gpu:local python /work/kernel_evolution_eval.py \
    --candidate-dir /candidate --approval-file /approval/approved.json \
    --output /out/screening.json --mode screen
```

The approval directory should contain only `approved.json`. The image must
already exist locally; this command does not grant the worker network access.
The worker checks the resulting environment before candidate import. Docker
and Podman are not installed in the current workspace, so this launch has not
been exercised here.

Use the following command **inside a GPU-capable isolated container**, with
the sanitized bundle mounted at `/work` read-only, only the reviewed candidate
mounted at `/candidate` read-only, the approval mounted read-only, and `/out` a
separate writable artifact mount:

```bash
python /work/kernel_evolution_eval.py \
  --candidate-dir /candidate \
  --approval-file /approval/approved.json \
  --output /out/screening.json --mode screen
```

The worker refuses to import candidate source unless it finds a non-root user,
only loopback in its network namespace, a read-only root/candidate/approval
mount, a writable separate output mount, seccomp filtering, a CUDA GPU,
finite cgroup limits (memory ≤8 GiB, PIDs ≤256, CPU ≤8), and CPU-time and
open-file rlimits. It also rejects known API-key environment variables and a
mounted Docker control socket. The container operator must independently verify that the
container was launched with disabled networking, dropped capabilities, no
privilege escalation, a read-only sanitized code bundle, and those resource limits.
These runtime checks do **not** turn AST inspection, Python import, or a
subprocess into a security sandbox. This workspace currently has no Docker or
Podman and no finite cgroup resource limits, so approved evaluation cannot be
performed here; the worker fails closed.

Once isolated, screening checks the candidate against the scoped FP16
`torch.mm` reference at the main shape and two boundary shapes
(509×3071×769 and 65×129×33), each with seeds 42 and 43. It saves compilation,
correctness, A/A, full raw paired CUDA-event timing, source hashes, and
isolation evidence. A screening speedup is provisional. A separate container
process may confirm a promising result:

```bash
python /work/kernel_evolution_eval.py \
  --candidate-dir /candidate \
  --approval-file /approval/approved.json \
  --output /out/confirmation.json --mode confirm \
  --prior-evaluation /prior/screening.json
```

Confirmation uses the existing 14-round, 200-launch heuristic. It is not a
statistical significance test, and no timing sample is filtered or corrected.
The `/out` paths must be new files. Preserve those JSON files alongside the
proposal artifacts.

## Revision from measured feedback

After an approved evaluation, request one new proposal with the exact prior
source and a compact summary of its compilation, correctness, failure,
performance, and uncertainty results:

```bash
.venv/bin/python kernel_evolution.py propose --request-from-muse \
  --gpu 'NVIDIA GeForce RTX 4060 Ti' \
  --feedback-from evaluations/kernel-evolution/<candidate-id> \
  --feedback-evaluation /path/to/screening.json
```

The next proposal receives the prior source text, hash, hypothesis, and
evaluation summary. Each version has its own directory and audit. This shows
feedback being supplied to Muse; it does not establish model learning or a
confirmed optimization without a reviewed, isolated evaluation and separate
confirmation.
