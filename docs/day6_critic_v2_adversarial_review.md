# Day 6 — adversarial review of critic v2

## Decision

**Do not freeze the current v2 validator as the admission gate for a controlled Muse v1/v2 comparison.** It validates recorded values and declared arithmetic, but the twelve CPU-only adversarial tests show deterministic gaps in prose coverage, stage claims, and mixed-unit arithmetic. Some other attacks depend on scientific interpretation and need human review rather than a broad prose rule. No production validator, v1 protocol, Experiment 009 request, or historical artifact was changed during this review.

The test file is `tests/test_muse_research_critic_v2_adversarial.py`. It uses a synthetic P06 source with two shapes, two opponents, four screening ratios, a missing confirmation status, and separate historical observations. The tests intentionally assert the **current** behavior, including successful parsing of unsafe examples. “Pass” below means the parser returned a structurally valid response pending human review; it does not mean the scientific claim is true.

## Findings and negative controls

All behaviors below reproduced deterministically in one local run. Test names identify the exact synthetic input construction. “Expected” describes the desired scientific gate, not an assertion that the current parser promises to provide it.

| Test / input | Expected behavior | Actual v2 behavior | Classification; severity | Smallest defensible next step |
| --- | --- | --- | --- | --- |
| `test_negative_controls_exact_screening_and_complete_range`: exact cited screening ratio; four cited ratios with complete declared scope and three-decimal `rounded_range`. | Accept structural evidence and arithmetic. | Both parse. | Correct control; no issue. | Retain as regression controls. |
| `test_omitted_claim_in_unledgered_prose_passes`: `assessments.evidence_consistency` says “P06 achieved a confirmed 30% speedup,” while the only ledger entry concerns missing confirmation. | Reject or require review before any acceptance decision. | Parses; validator only visits existing ledger statements. | **False negative relative to the v2 prompt contract; high.** | Make coverage an explicit separately versioned requirement, such as claim-only structured response fields with a reviewed rendering step. Do not claim regex can prove full coverage. |
| `test_citation_laundering_in_observation_statement_passes`: a valid prior screening ratio sentence appends “Therefore P06 is faster” within the same `observation` statement. | Keep measured result separate from inference about the untested P06 kernel. | Parses because exact key and value appear somewhere in the substring. | **Scientific limitation with a structural loophole; high review risk.** | Require atomic observation entries, with separate inference entries and human review of the conclusion. A valid citation is not proof of a causal or transfer claim. |
| `test_numerical_misattribution_in_mixed_statement_passes`: cite ratio from screening key A, then state key B has the same ratio although B's recorded ratio differs. | Reject the wrong key/value association. | Parses because A's exact key and value appear in the statement. | **False negative for a checkable key/value mismatch; medium–high.** | Limit a direct observation entry to one atomic key/path/value assertion or validate separately structured assertions; avoid treating a free-text substring as an entire verified fact. |
| `test_wrong_recorded_value_is_rejected`: change a ledger `recorded_value` while keeping its key. | Reject. | Rejects with `claim recorded value differs from evidence`. | Correct control; no issue. | Retain. |
| `test_arithmetic_on_incompatible_units_passes`: add a unitless screening ratio to a baseline latency in milliseconds; `0.9971394589037794 + 2.0` correctly rounds to `2.997`. | Reject dimensionally meaningless arithmetic. | Parses; `_arithmetic` checks numeric types and decimal result, not unit compatibility. | **Deterministic false negative; high.** | Add operation-specific unit rules and reject mixed dimensions for addition/subtraction/range. Preserve valid dimensionless ratios and same-unit operations. |
| `test_aggregate_can_omit_relevant_middle_observation`: prose says “All four” comparisons, but references, scope, and arithmetic contain only three; omitted middle ratio does not change min/max. | Detect incomplete coverage of the stated comparison set. | Parses; declared scope equals the incomplete references. | **False negative against stated coverage; medium.** The true intended set cannot be inferred safely from arbitrary prose. | Make comparison set an independently defined source selector or preregistered set; otherwise require a human to check aggregate completeness. |
| `test_declared_aggregate_omission_is_rejected`: declare all four keys, then remove one reference. | Reject. | Rejects with `aggregate claim omits a declared source observation`. | Correct control; no issue. | Retain. |
| `test_screening_can_be_described_as_confirmed_in_prose`: ledger says `screening`, but the same prose says it “independently confirmed P06's performance gain.” | Reject a confirmed-performance assertion grounded only in screening. | Parses; stage check sees only metadata, not contradictory prose. | **False negative for the protocol's screening boundary; high.** | Represent stage and target kernel in structured claim content rendered into prose, or add a narrow contradiction check as a defense in depth with human review. Do not infer scientific confirmation from wording alone. |
| `test_confirmation_metadata_mismatch_is_rejected`: change that screening entry's metadata to `confirmation`. | Reject. | Rejects with `screening evidence cannot support confirmed performance`. | Correct control; no issue. | Retain. |
| `test_inference_can_claim_established_mechanism`: an `inference` entry says P06's cache-locality mechanism was “experimentally established,” despite missing profiling/confirmation. | Flag as unsupported mechanism conclusion. | Parses because `inference` and `not_applicable` metadata are internally consistent. | **Scientific limitation; high review risk.** | Require explicit inference language in a structured conclusion and mandatory human review of mechanism claims. No generic citation check can establish causality. |
| `test_valid_plain_language_observation_is_rejected`: exact, correctly cited ratio is written as “The recorded screening median paired ratio was …” without the machine observation-key token in prose. | Accept as a valid scientific summary with the key in the ledger. | Rejects with `direct observation must state its exact key and value`. | **Format-driven false positive; low–medium.** | In a new version, keep exact key/value in the structured ledger and allow ordinary prose, while preserving atomic linkage and human review. |

The citation-laundering and inference-laundering examples should not be converted into a claim that the parser can judge scientific truth. Their important failure mode is **overtrust**: a caller could treat structural parse success as evidence-grounded scientific acceptance. The source itself says the return is pending human review, and any pilot must enforce that boundary operationally.

## Root causes and ranked recommendations

1. **High: coverage and stage are not bound to all prose.** `validate_ledger` iterates submitted ledger entries and checks `statement in fields[field]`; it never checks the converse. The stage check inspects `measurement_stage` and key prefixes, so contradictory prose survives. Before freeze, define a versioned response form that makes each factual assertion an atomic structured claim, with a clear human-review gate for any free text. Keep the existing v2 behavior as an unfrozen prototype.
2. **High: arithmetic is dimensionally blind.** `_arithmetic` converts cited numbers to `Decimal` and verifies the result. It does not inspect `unit` when deciding whether an operation is meaningful. A small operation/unit compatibility table is a defensible deterministic fix. Also test that correct arithmetic on irrelevant but same-unit measurements remains subject to human interpretation.
3. **Medium: aggregate scope is self-declared.** The equality check proves only that `aggregate_scope` and references agree with each other. It cannot prove that “all four,” “all shapes,” or “all opponents” includes every relevant source record. An externally defined set or selector is needed for machine-enforced completeness; otherwise the reviewer must verify it.
4. **Medium: key/value association is substring based.** Direct observation validation searches for an exact key and value anywhere in the statement. A second, false attribution can sit beside the valid pair. Atomic structured assertions are preferable to a wider regex vocabulary.
5. **Low–medium: ordinary scientific prose can be rejected for formatting.** Requiring the internal observation-key token in the human sentence is a presentation constraint, not evidence validation. Relax it only within a new design that keeps structured identity and value checks.

## Reproducible checks

Run from the repository root:

```bash
.venv/bin/python -m pytest -q \
  tests/test_muse_research_critic.py \
  tests/test_muse_research_critic_v2.py \
  tests/test_muse_research_critic_v2_adversarial.py \
  tests/test_muse_critic_value_study.py \
  tests/test_public_evidence.py
python verify_public_evidence.py public_evidence
```

Observed results: **57 passed, 0 failed** across the targeted suite; **12** of those are the new adversarial characterization tests. The public-evidence verifier exited 0 and reported 20 source-artifact identifiers. The tests are CPU-only and make no model or GPU request. No production behavior changed, so the observed unsafe parses are documented rather than fixed.

## Freeze criterion for a controlled v1/v2 pilot

Freeze only a separately versioned v2 protocol after: (1) every high-severity **deterministic** gap above is closed or explicitly made a mandatory human-review exclusion; (2) the new version passes adversarial tests for omitted claims, screening/confirmation contradictions, mixed-unit arithmetic, wrong key/value association, and aggregate-set completeness, plus valid plain-language controls; (3) a source-only fixture path and response-size test demonstrate that the schema fits the chosen model budget; (4) model/settings/evidence/request count, acceptance gate, human rubric, and failure handling are preregistered; and (5) historical v1/Experiment 009 records remain immutable. The blind review must score scientific fidelity and useful discrimination separately from JSON parse rate, with possible condition unblinding recorded.

The current v2 is useful as an offline evidence-ledger prototype. It is **not ready to freeze as the controlled Muse pilot's reliability gate**. A pilot that deliberately studies these weaknesses could be run under a different, explicitly exploratory protocol, but its parser success must not be reported as validated scientific criticism.
