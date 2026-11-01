# Experiment 012 — value of imperfect AI criticism

## Question and status

Exploratory question: **Can an AI research decision-maker benefit from rejected AI-generated criticism when its validation failure is disclosed?** This is a new four-proposal paired study, separate from frozen Experiment 009. It has no Muse decision responses yet. No GPU work is part of this protocol. The local prepared run is `20261101T002309458666Z` (UTC case-style ID; seed `1212`) under `evaluations/muse-imperfect-critic-study/v1/`. Preparation and the P03 preview made no model request.

## Frozen sources, selection, and treatment

The units are original Experiment 007 proposals P03–P06, with the validated evidence packages and proposal hashes from Experiment 009 run `20261030T205425148094Z`. The source setup is hash-bound and revalidated by its loader; Experiment 009 itself is not modified. For each proposal, the selection rule is the **first chronological manifested rejected Experiment 008 v1 criticism request**, ordered by timestamp case ID, regardless of its persuasiveness or later attempts. Each selected case and its audit are reloaded with Experiment 008's original manifest and source checks. A newer rejected case, a valid v3.1 case, or human analysis cannot silently replace it.

| Proposal | Selected first rejected case | Original rejection audit SHA-256 | Original proposal SHA-256 |
| --- | --- | --- | --- |
| P03 | `20261030T211309477752Z` | `e0043e8661ba6217df686cddc96d336267e439b8ef72b023c205f87cc0739c6b` | `15cd9a573ba66d0cefa697c9741716dcb8de77500383727a0eb56a729ac128dc` |
| P04 | `20261030T211546504949Z` | `d2d1cfda537b5e53946ac70b56762618f8e2fc568707d2bca3ecd277bca5ccd0` | `45d089a85259ae4f2c36fad96a8864432d0f51ee44ba089a29c0b75e93436961` |
| P05 | `20261030T211855242529Z` | `1108ea00b7b38881e1576973e661b5b826775cb0734914d04f70f47e7bb84ef8` | `07a4124a2c4f46199376731dcd2e07b4450da082733fa208d6efddd073ca9392` |
| P06 | `20261030T212025073515Z` | `29c287c77f20e51752cf951445c9daa801365808a08912b999df949b42ff923e` | `e07624e3c6e51120475e651ce085185d18b86bf378269d913bc08221eb93cc07` |

Every selected audit's SHA-256, manifest SHA-256, and raw-response SHA-256 are frozen in `setup/study.json`. The raw rejected criticism remains byte-for-byte unchanged in its original Experiment 008 audit; Experiment 012 reads and embeds that exact string. The selected cases' original first errors are the only validation feedback supplied to the decision model. No human diagnosis from `docs/rejected_criticism_analysis.md` enters either prompt.

Each proposal gets two **independent** decision requests. Condition A is exactly Experiment 009's `decision_prompt(source)` core: original proposal, validated historical evidence, fixed decision instructions and schema; no criticism. Condition B begins with the **identical A prompt bytes** and appends, in this order:

1. `REJECTED_INDEPENDENT_CRITICISM_RAW` and the selected audit's exact `raw_response` string.
2. `VALIDATION_FAILURE_REPORT` JSON with exactly `schema`, `case_id`, `original_status: rejected`, `rejection_kind`, `first_validation_error` (the audit's stored first rejection), and `original_audit_sha256`.
3. `TREATMENT_INSTRUCTION` saying the rejected criticism is unverified advisory text, not validated evidence; technical suggestions may be considered, but factual claims must be independently grounded in the validated `SOURCE`, and unsupported historical or numerical claims must not be copied.

The entire appended block is the sole **intended treatment difference**. It contains both criticism and failure disclosure, so this study estimates their *joint* effect; it cannot isolate the value of criticism from the warning. Response wording could still reveal condition during scoring. The decision model never sees the other condition's response. This study does not alter the Experiment 008 v1 criticism validator or bind a rejected criticism to Experiment 009.

## Frozen model, prompts, and validator

The decision model is `muse-spark-1.3` at `https://api.meta.ai/v1` via the Responses API. Both conditions use `max_output_tokens: 8000`, `reasoning.effort: low`, API-default temperature, 30-second timeout, and `max_retries=0`. The CLI requires a separate `--human-authorized` flag for **each** live request; no such request is authorized by preparing this protocol. No code path invokes a GPU kernel.

The **core prompt version** is Experiment 009 `decision_v1`; its per-proposal SHA-256 is frozen in setup. The appended treatment prompt is separately versioned `decision_v1_plus_rejected_criticism_feedback_v1`. The exact core hashes for P03–P06 are, respectively, `e4b9a981432c6e8235fa91edcc210ab26484a49c0d3efe1872775b394fc6984b`, `02d5081850727158a495b5ff02e797b5b6ab692039dccfbe5ddee2cc069c7940`, `11edd0b41d5db3381c24c210db302a3e019e8967dfed698a62f3a22c9e5d86bb`, and `24afa5d04d212ac6ff67b84e1b119d74fd6d076ae9c4cfda34525791296e14be`. The setup also stores the complete proposal/source hashes, model, endpoint, settings, seed, rubric, and treatment selection.

The existing Experiment 009 **decision response schema and `parse_decision` validator can be reused unchanged**. It already checks proposal identity, structured fields, field-scoped evidence references, unsupported numerical and stage claims, and source values. Experiment 012 introduces its own request, audit, preview, blind, score, and reveal artifact schemas; it does not change the decision response contract or validator. Deterministic acceptance remains separate from human scientific scoring.

## Preregistered scoring and blind review

Before any decision request, the setup freezes `experiment_012_rubric_v1`. A human scores each eligible card from **1 to 5**, with a written justification, on evidence fidelity, error resistance, discriminating tests, discovery preservation, and resource awareness. The rubric does not reward `proceed`, `revise`, or `deprioritize` as a category. Each card also records `repeats_rejected_claim` (boolean), `separates_observations_inferences` (boolean), `recommendation_differs_from_pair` (boolean), and `possible_unblinding` (text; empty if none). The recommendation-difference flag is checked against the two visible categories within a proposal; it is not a quality score.

After **all eight condition requests have been attempted once**, the workflow forms a blind packet from only proposals with **both accepted decision outputs**. The packet has opaque `Pxx-1`/`Pxx-2` cards, shuffled within each proposal with seed `1212`; it removes condition labels and proposal hashes. It supplies the original proposal, validated evidence, and selected audit's first validation error equally to both cards as reference material. The operator keeps raw audits, request specifications, and `operator/assignment.json` sealed until scores are recorded. A neutral operator may verify manifests without rewriting responses. The reviewer records wording-based suspected unblinding on each card. Scores and the blind packet are hash-bound before reveal.

The reveal reports the status of **all eight attempts**, accepted-decision count, eligible-pair count, excluded pairs and reasons, condition means, and paired B-minus-A differences for every criterion. Four pairs at most make these descriptive subjective judgments; no significance or compute-savings claim follows. A pair with an incomplete, malformed, or validator-rejected decision is **not scored as a pair**. Its attempt(s) and missingness remain in feasibility accounting. There are no silent retries, replacement decisions, or post hoc changes to the rubric. If no pairs are eligible, scoring an empty packet still records feasibility and produces no quality estimate.

## Offline workflow and future authorized requests

Run from the repository root. `--prepare`, `--validate`, and `--preview` make no Muse request. They write only under the new Experiment 012 namespace; preview stores raw local prompts in a SHA-256 manifested file. The current run has already been prepared and P03 previewed:

```bash
.venv/bin/python muse_imperfect_critic_study.py --validate --run-id 20261101T002309458666Z
.venv/bin/python muse_imperfect_critic_study.py --preview --run-id 20261101T002309458666Z --proposal-id P03
```

The preview is `evaluations/muse-imperfect-critic-study/v1/20261101T002309458666Z/preview/P03/preview.json`. P03 A and B prompts are **11,389 and 20,522 UTF-8 bytes**, with SHA-256 `e4b9a981432c6e8235fa91edcc210ab26484a49c0d3efe1872775b394fc6984b` and `1e3534f8cf1fa0441cf47cff54a35990b88da789a8ad295795fd3b93996630bb`. Inspection confirmed B starts with the exact A bytes and A has no criticism section. The run currently has no decision attempts.

For a **future separately authorized** P03 pair, inspect the preview, obtain authorization for each call, then execute each independent request once (in either order):

```bash
.venv/bin/python muse_imperfect_critic_study.py --request-decision --run-id 20261101T002309458666Z --proposal-id P03 --condition A --human-authorized
.venv/bin/python muse_imperfect_critic_study.py --request-decision --run-id 20261101T002309458666Z --proposal-id P03 --condition B --human-authorized
```

Repeat only with new explicit authorization for P04–P06. Each condition has one run-local attempt. Requests preserve raw response, validation errors, response metadata/token usage, timestamps, request spec, and SHA-256 manifest. After all eight attempts, `--prepare-blind` writes separate blind and operator artifacts; give the reviewer **only** `blind/review.json`. Submit a complete score JSON via `--score --scores-file PATH`, then `--reveal` and `--validate`. Do not share an A response with B or reveal assignments before scoring.

Local setup, previews, raw prompts/responses, credentials, response IDs, and environment paths must remain out of public exports. Any later public summary should use an explicit allowlist of aggregate, non-sensitive fields and verify source hashes; this task creates no public export. The frozen Experiment 009 run, its bindings and decisions, and all historical Experiment 008 audits remain unchanged.

## Offline verification at preparation

The prepared run and manifested P03 preview passed the Experiment 012 read-only validator. CPU-only targeted tests for Experiment 012, the existing Experiment 008/009 workflows, and the public evidence code passed **42 tests**. `python verify_public_evidence.py public_evidence` passed and reported **20 source-artifact identifiers**. These checks establish local workflow behavior and artifact integrity; they do not validate a future Muse decision or any GPU performance claim. Scientific value can be evaluated only after separately authorized decision requests and blinded human scoring.
