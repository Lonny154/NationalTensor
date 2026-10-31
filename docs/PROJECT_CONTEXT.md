# National Tensor — Project Context

**Last Updated:** October 30, 2026  
**Purpose:** Read this at the start of a new ChatGPT or Codex session, then follow the linked protocols and current files. This is a handoff, not a substitute for validating artifacts before making new claims.

## Vision and principles

National Tensor asks: **Can an AI research system generate hypotheses, test them against real GPU hardware, learn from negative results, and improve future research decisions?** It is a Meta Global AI Developer Hackathon 2026 project using Meta Muse, Triton, and reproducible GPU experiments. Keep it ambitious, scientifically honest, and enjoyable.

- Treat model proposals, model interpretations, measured observations, and human decisions as different kinds of evidence.
- Prefer a useful negative result to an unsupported speedup claim. Screening is not independent confirmation; a compiler match is not a runtime measurement.
- Preserve source identity, raw local audits, rejected attempts, and immutable hashes. Use the existing validators; do not silently retry or rewrite historical records.
- Human review gates kernel implementation and GPU execution. No model recommendation automatically authorizes either.
- Prior Matmul Inspector work supplies compiler and profiler context; distinguish it from new hackathon research. See the [root README](../README.md).

## Architecture and current capabilities

The implemented path is **Muse advisory proposal → structured/citation validation → human review → selected Triton or trusted GPU diagnostic → hashed artifacts → evidence loaders → dashboard, reflection, and criticism**. The local Flask dashboard has Research Mode and Hollywood Mode. The controlled [research workflow](research_workflow.md) registers one fixed trusted diagnostic runner; it does not execute generated kernels. The [dashboard guide](dashboard.md) describes its approval, replay, and presentation behavior.

Implemented study tools include Experiment 007's matched proposal requests and blind manual review, Experiment 008's independent critic and revision audit, and Experiment 009's paired decision-request and blind-scoring protocol. These are workflows, not proof of improved research quality. See [Experiments 007](experiment_007.md), [008](experiment_008.md), and [009](experiment_009.md).

## Findings and limits: Experiments 003–009

| Experiment | Validated state and boundary |
| --- | --- |
| **003** | Reviewed mask-hoisting variant passed scoped correctness; the compared compiled executable and relevant metadata matched at M=128, so timing was skipped. Original Muse-generated source was not executed. [Debrief](muse_research_debrief.md). |
| **004** | At M=130 with partial boundary tiles, the same reviewed source pair again compiled equivalently within the tested workload. Correctness passed; runtime was not measured. [Protocol](experiment_004.md). |
| **005** | Hilbert tile traversal passed 24 scoped correctness checks over four shapes and three schedules. Eight paired screening comparisons showed no bootstrap lower bound above 1 against either alternative. Independent timing confirmation is missing, and cache-locality causation is not validated by profiling. [Protocol](experiment_005.md); [portable summary](../public_evidence/summary.md). |
| **006** | Accepted Muse reflection chose **abandon** as an advisory judgment about the Hilbert direction. It relied on screening and acknowledged missing confirmation and profiling; it is not a new hardware result. [Protocol](experiment_006.md). |
| **007** | Six original proposals from matched conditions were manually scored after a blind review and condition reveal. The scores are subjective and do not establish statistical superiority. [Protocol](experiment_007.md). |
| **008** | P01 has one accepted evidence-grounded criticism recommending **deprioritize**. Historical revision attempts remained rejected. Other criticism requests exposed citation and numerical-validation failures; rejected text must not be treated as accepted scientific assessment. [Protocol](experiment_008.md). |
| **009** | Six proposal-level criticism outcomes: **1 accepted, 5 rejected**. P01 has accepted A/B decisions, but the six-proposal paired decision-quality evaluation, blind scoring, and reveal are incomplete. There is no measured GPU compute saving. [Protocol](experiment_009.md). |

## GitHub and public evidence

The project owner reports that the PR containing the README and verified [public evidence bundle](../public_evidence/README.md) was merged into GitHub `main` on October 30. The local checkout inspected for this handoff remains on `day4-muse-cinematic-debrief` at `1aef461` (`Add verified public research evidence bundle`); its local `main` ref is older (`33781d5`). **Check/fetch remote refs before starting new branch or merge work.** This document is not yet committed.

The tracked public bundle covers Experiments 005–009 with allowlisted JSON, a generated summary, source-artifact SHA-256 identifiers, and a standalone verifier. Run `python verify_public_evidence.py public_evidence` from a source-only clone. That verifies bundle integrity and internal arithmetic, **not** independent GPU reproduction. Original Experiment 003–009 run directories under `evaluations/` are ignored by Git and remain local; the public export excludes credentials, raw prompts/responses, local paths, and GPU UUIDs. Regeneration requires those original directories and `.venv/bin/python export_public_evidence.py --output public_evidence`.

## Weaknesses and open questions

- **Evidence grounding and critic reliability are the next engineering priority.** Five Experiment 009 criticism requests were rejected: four for unsupported numerical claims under the current validator and one for a historical claim without a scoped citation. Examine the preserved audits and validated observation keys before changing prompts or validation rules. Avoid broad prose regexes that imply scientific truth; keep deterministic checks to structure, references, recorded measurements, and explicit arithmetic.
- Experiment 009 cannot answer whether criticism improves decisions until valid criticism and independent A/B decisions exist for all six proposals, followed by complete blind scoring. Preserve condition isolation and record possible wording-based unblinding.
- No repeatable Hilbert latency advantage or causal locality mechanism has been established. Tested FP16 GEMMs are representative, not production workloads; results may change across GPUs and toolchains. The environment lacks a complete pinned lockfile.
- The dashboard's Hollywood mission replay centers on Experiment 003. A cohesive Hilbert → screening → reflection → criticism presentation is a **demo plan**, not an implemented end-to-end scene. Its data should come from validated local records or the public bundle.
- The human-gated workflow has one registered diagnostic runner and a local loopback UI. Broader autonomous experiment execution and recommendation adoption are not implemented.

## October 31–November 4 priorities

| Date | Intended work |
| --- | --- |
| **Oct 31** | Audit the five rejected criticism cases; identify which numerical or citation claims lack valid support. Design a tighter evidence presentation and narrowly scoped validator fixes with CPU fixtures. |
| **Nov 1** | Implement and test critic-reliability improvements. Keep old accepted/rejected audits immutable; any new Muse attempts must be explicit and separately versioned. |
| **Nov 2** | Decide whether enough valid criticism exists to finish Experiment 009. If not, report the pilot as incomplete. Draft the demo script and verify every on-screen scientific label against the public bundle. |
| **Nov 3** | Record the final demo around this date, after a dry run of the actual UI and artifact-backed story. Keep a source-only verification path available for judges. |
| **Nov 4** | Verify links, bundle integrity, demo playback, submission text, and repository state; submit before **9:00 PM Eastern**. Do not leave the first verification pass to the deadline hour. |

These are priorities, not claims that the work has happened. The submission deadline is **November 4, 2026, at 9:00 PM Eastern**.

## Demo and presentation plan

Build the story around a genuine research cycle: **Muse's Hilbert hypothesis → human-reviewed Triton implementation → correctness and GPU screening → Muse reflection → independent research criticism**. Show the public evidence receipt alongside the cinematic UI. A short recorded sequence can use the existing Hollywood interface and narration style, then display the validated Experiment 005–009 evidence in a clearly labeled view or edit. The Experiment 003 mission replay is implemented; the integrated Hilbert story needs design and verification before it can be described as a live feature.

Hollywood narration may be playful, but **presentation must never misrepresent scientific evidence**. Label observed measurements, speculative mechanisms, model judgments, rejected responses, missing stages, and fictional dialogue separately. Do not animate an unrecorded benchmark as a real job, turn screening into confirmation, imply that criticism was accepted for all six proposals, or claim speedup or compute savings. Keep the exact artifact and limitation accessible from the scene. See [Hollywood Mode and Operation Stolen Cycles](dashboard.md#operation-stolen-cycles).

## Next Session

- **Last completed milestone:** README and portable public evidence bundle were added in `1aef461`; the project owner reports their PR merged into `main` on October 30. This handoff documents that state without altering experiments.
- **Next recommended task:** Improve evidence grounding and critic reliability using the five Experiment 009 rejected criticism audits as fixtures. First map each rejected claim to its available evidence key and validator failure; then make the smallest prompt or deterministic-validation change and run CPU-only tests. Keep scientific judgments for human review.
- **Start here:** `public_evidence/README.md`, `public_evidence/evidence.json`, `docs/experiment_008.md`, `docs/experiment_009.md`, `muse_research_critic.py`, and `tests/test_muse_research_critic.py`.
- **Useful commands:** `python verify_public_evidence.py public_evidence`; `.venv/bin/python -m pytest -q tests/test_muse_research_critic.py tests/test_muse_critic_value_study.py tests/test_public_evidence.py`; `.venv/bin/python dashboard_app.py`. The local-only pilot is `20261030T205425148094Z`; `.venv/bin/python muse_critic_value_study.py --validate --run-id 20261030T205425148094Z` requires ignored original artifacts. These commands make no Muse request or GPU benchmark.
