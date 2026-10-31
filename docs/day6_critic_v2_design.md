# Day 6 — evidence-grounded critic v2 design

## Scope and version boundary

`muse_research_critic_v2.py` introduces `experiment_008_critic_v2` as an **offline preview and response validator**. It does not issue Muse requests, write cases, revalidate old audits, or bind criticism into Experiment 009. The original `muse_research_critic.py` prompt, parser, CLI behavior, existing tests, and local historical records are unchanged. Experiment 009 still names `experiment_008_v1` in its frozen setup and accepts only its existing binding path. A future live v2 experiment needs its own versioned request, manifest, and binding rules.

The v2 prompt adds a claim ledger and compact instructions to: cite exact evidence IDs and observation keys for historical and quantitative claims; distinguish observation, derivation, and inference; prefer exact numbers; declare operands, operation, precision, and result for rounding; cite every input to aggregate claims; label screening and confirmation separately; avoid transferring prior-kernel measurements to an untested proposal; and say when evidence is insufficient. The preview uses the same validated source package as v1. Its 17,000-character scale for P06 includes the source evidence, while the new instruction block is compact. A live model's ability to fit a complete ledger within the existing 8,000-output-token setting remains untested.

## Claim-ledger schema

The response keeps the v1 assessment fields and adds `protocol_version: "experiment_008_critic_v2"` and `claim_ledger` (1–16 entries). Each entry has:

| Field | Meaning |
| --- | --- |
| `claim_id` | Unique `C1`, `C2`, … identifier. |
| `claim_type` | `observation`, `derivation`, or `inference`. |
| `field`, `statement` | The named response field and an exact substring of its prose. |
| `references` | 1–12 exact source references. Each has `evidence_id`, `observation_key`, `value_path` into a nested observation, exact JSON `recorded_value`, and `unit`. |
| `aggregate_scope` | The declared evidence ID/key set for an aggregate. It must equal the set used in `references`; an empty list means no declared aggregate. |
| `arithmetic` | `null` outside derivations. A derivation gives `operation` (`add`, `subtract`, `multiply`, `divide`, or `rounded_range`), `operand_indices` into `references`, `precision_decimal_places`, and a decimal-string `result` (two endpoints for `rounded_range`). Decimal `ROUND_HALF_UP` is used. |
| `measurement_stage` | `historical`, `screening`, `confirmation`, or `not_applicable`. Inferences use `not_applicable`; screening measurements cannot be labeled confirmation. |
| `limitations` | Nonempty description of what the referenced evidence does not establish. |

Units are deterministic for known paths: `*_ms` is `ms`, ratio fields are `ratio`, hardware memory is `bytes`, workload/shape is `elements`, otherwise `none`. Ratios are unitless; `ratio` labels their metric type. For a P06-style range, the four exact `median_paired_ratio` values from the 512 and 2048 shapes are four references and four `aggregate_scope` entries. A `rounded_range` derivation with precision 3 produces `0.997` and `1.002`. Those rounded endpoints are a descriptive summary of **Hilbert screening**, not evidence that P06's new dispatch has been tested.

## What validation establishes

The v2 parser checks identity, response shape, evidence ID and key existence, value path, exact recorded value and unit, distinct citations, claim-to-field substring, declared aggregate coverage, arithmetic operands and result, and stage metadata. A direct observation statement must include the exact observation key and value. A derivation must use every cited numerical operand exactly once and include its reported result in the statement. Inference cannot be labeled as measured evidence. A screening key cannot support a `confirmation` stage, and missing E005 confirmation blocks it. The parser returns **structural validation pending human review**; it does not mark criticism scientifically accepted.

These checks do **not** prove that all claims in the prose have ledger entries, that a model chose the complete real-world aggregate scope, or that a cited observation supports a causal interpretation. A model could omit a required source from both `references` and `aggregate_scope`, or phrase an unsupported inference around valid context references. Human review must compare prose, ledger, and source, assess mechanism and confounds, and decide whether the criticism is useful. We did not add general prose rounding tolerance or weaken v1 numerical checks.

## Offline preview

From the repository root:

```bash
.venv/bin/python muse_research_critic_v2.py --preview \
  --study-id 20261030T142150402688Z \
  --review-id 20261030T142419308682Z \
  --proposal-id P06
```

This prints the full `SCHEMA` and validated `SOURCE` request locally without contacting Muse or writing an audit. The tested preview identified proposal P06 and showed `claim_ledger`, `aggregate_scope`, exact values, arithmetic metadata, screening/confirmation instructions, and the source evidence. Treat the full preview as local research material; do not add it to the public bundle.

## CPU-only regression results

Executed after implementation:

| Command | Result |
| --- | --- |
| `python verify_public_evidence.py public_evidence` | Passed; bundle reports 20 source-artifact identifiers. |
| `.venv/bin/python -m pytest -q tests/test_muse_research_critic.py tests/test_muse_research_critic_v2.py tests/test_muse_critic_value_study.py tests/test_public_evidence.py` | **45 passed, 0 failed**; the eight new tests are synthetic, and existing v1 tests were unchanged. |
| The v2 `--preview` command above | Exit 0; offline request generated. |

Synthetic cases reject missing screening citations, an unrelated historical key for a direct observation, altered values or units, omitted declared aggregate members, incorrect arithmetic or precision, screening represented as confirmation, and inference labeled measured. The P06 four-ratio range validates only with the exact four source values and declared arithmetic metadata.

## Proposed controlled evaluation

Create a new study version with a preregistered set of proposal/source packages and matched independent v1/v2 critic requests. Freeze model, settings, request count, evidence, randomization, and response budget; do not reuse or relabel P02–P06 historical attempts as v2 outputs. Record all raw attempts and first validation failures without retries. Have blinded human reviewers score evidence fidelity, complete citation coverage, observation/inference separation, scientific correctness, and useful discriminating tests, with possible protocol unblinding recorded. Compare paired human scores and error types alongside structural acceptance, reporting uncertainty for the small sample. Only after that study could one claim a reliability improvement; a higher parse rate alone would not establish better scientific reasoning or compute savings.
