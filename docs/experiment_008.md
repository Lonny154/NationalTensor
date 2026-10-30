# Experiment 008 — Evidence-Grounded Research Critic

This workflow critiques and optionally revises one of Experiment 007's six validated proposals at a time. It validates the Experiment 007 study, both request audits, blind packet, scores, and reveal, and revalidates the underlying Experiment 003–005 evidence through the existing loaders. The accepted Experiment 006 reflection is supplied as model interpretation, separate from measured evidence. The original proposal is copied into a new case with its hash and remains unchanged in Experiment 007.

From the repository root, prepare one case. This is read-only with respect to Experiments 005–007 and makes no Muse request:

```bash
.venv/bin/python muse_research_critic.py --prepare \
  --study-id 20261030T142150402688Z \
  --review-id 20261030T142419308682Z \
  --proposal-id P04
```

The command prints a `CASE_ID`. Inspect `evaluations/muse-research-critic/v1/CASE_ID/setup/case.json`, including the exact critique prompt. To request one independent criticism explicitly:

```bash
.venv/bin/python muse_research_critic.py --criticize --case-id CASE_ID
```

After reviewing the accepted criticism, prepare a separate revision prompt, inspect `revision_setup/setup.json`, then request one response:

```bash
.venv/bin/python muse_research_critic.py --prepare-revision --case-id CASE_ID
.venv/bin/python muse_research_critic.py --revise --case-id CASE_ID
```

New revision preparations use prompt version `revision_provenance_v2`. The prompt requires an exact `evidence_id` and `observation_key` for each historical claim and numerical workload reference in `revised_proposal.evidence_references`. Response-level `evidence_citations` cover `what_changed`, `why`, and `response_to_criticism` separately; citations in one scope do not cover the other. Existing revision setups without `prompt_version` retain and validate their original prompt bytes. A case with an existing revision setup or request is not silently re-prepared or retried.

For a historical **rejected** criticism, run an explicit offline revalidation instead of retrying Muse. This checks the original manifested audit, its source proposal, response completeness, and current evidence validator. It prints a `revalidation_id` and writes a new artifact under `revalidations/v1/REVALIDATION_ID/`:

```bash
.venv/bin/python muse_research_critic.py --revalidate --case-id 20261030T145230747865Z
.venv/bin/python muse_research_critic.py --validate-revalidation \
  --case-id 20261030T145230747865Z --revalidation-id REVALIDATION_ID
```

The same commands work for case `20261030T150228098163Z`. A valid revalidation can be named explicitly when preparing revision:

```bash
.venv/bin/python muse_research_critic.py --prepare-revision \
  --case-id 20261030T145230747865Z --revalidation-id REVALIDATION_ID
.venv/bin/python muse_research_critic.py --validate --case-id 20261030T145230747865Z
```

An invalid revalidation retains its reason and cannot authorize revision. The revision setup and any later revision audit bind the revalidation hash to the original response and audit hashes. The historical criticism audit keeps its original `rejected` status and bytes. Revalidation and both validation commands are offline; only `--criticize` and `--revise` call Muse.

`--criticize` and `--revise` require `MODEL_API_KEY` in the environment or repository `.env`. Each stage permits one attempt per case. Rejected and incomplete responses retain their raw output and safe response metadata with a SHA-256 manifest; they are not silently retried. Prepare a new case if a fresh Muse attempt is needed.

The critic returns structured assessments of prior overlap, evidence consistency, hardware mechanism, falsifiability, numerical correctness, and research value. It cites validated observation keys and chooses `proceed`, `revise`, or `deprioritize`. A separate revision response chooses `revise`, `defend`, or `withdraw`. A revision must return a complete proposal in Experiment 007's schema and explain what changed; a defense retains the original proposal and cites evidence. The critic's recommendation and any revision remain advisory. GPU execution requires separate human approval.

Validate the case and its available accepted or rejected stage artifacts without making a model request:

```bash
.venv/bin/python muse_research_critic.py --validate --case-id CASE_ID
```

The case, criticism, revision preparation, and revision each have separate SHA-256 manifests. Deterministic checks catch malformed schemas, invalid evidence references, unsupported numerical claims, missing-stage claims, and source or artifact tampering where mechanically detectable. Human review remains necessary to judge whether citations actually support the critic's prose and whether the proposed tests isolate the mechanism.
