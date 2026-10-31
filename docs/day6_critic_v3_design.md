# Day 6 — atomic evidence claims v3

## Boundary and implementation

`muse_research_critic_v3.py` is an independent, offline `experiment_008_critic_v3` preview, parser, and renderer. It makes no Muse request, writes no case, and has no Experiment 009 binding path. The v1/v2 implementations, historical audits, and frozen Experiment 009 comparison remain unchanged.

The model response has exactly five top-level fields: `protocol_version`, `proposal_id`, `proposal_sha256`, `claims`, and advisory `recommendation`. There are no free-form assessment or observed-evidence prose fields. A response may contain 1–16 ordered claims with unique `C1`-style IDs. Validation returns `structural_status: valid` separately from `scientific_status: pending_human_review`. No parser result means scientific acceptance.

| Claim | Required fields | Deterministic rule |
| --- | --- | --- |
| Observation | `claim_id`, `type`, `evidence_id`, `observation_key`, `value_path`, `recorded_value`, `unit` | ID, key, nested path, exact JSON value and unit match the validated source. `original_hypothesis` is excluded because it is a prior model proposal, not a measured observation. Screening/confirmation stage comes from the source key, not model text. |
| Derivation | `claim_id`, `type`, `operand_claim_ids`, `operation`, `precision_decimal_places`, `result`, `unit`, `aggregate_set` | Operands are earlier observations or derivations. Decimal arithmetic uses `ROUND_HALF_UP`. Addition, subtraction, and ranges require matching units; multiplication/division have limited explicit unit rules. A rounded range requires a source-derived aggregate set and exactly all its members. |
| Inference | `claim_id`, `type`, `support_claim_ids`, `interpretation` | Supporting IDs must exist earlier. `interpretation` is **unverified model commentary** even if it cites valid facts. Rendering always labels it as such and requires human review. |

`aggregate_sets(source)` builds screening comparison membership from E005's `screening:<shape>:<opponent>` observations. It exposes each shape, each pair of shapes, and the all-screening set. The model chooses a set name but cannot define its members. The P06 two-shape set includes four median-ratio observations; dropping a middle value fails even when the rounded min and max would stay the same. A model could choose a narrower available set and use it correctly; the renderer displays the chosen set, and human review must judge whether that set answers the research question.

The renderer constructs readable text from validated fields. Observation keys remain in metadata; a direct observation appears as, for example, `Screening observation (median paired ratio): 0.9971394589037794.` A derivation is labeled as a calculation, with its source claim IDs and aggregate set. Inferences are prefixed `Unverified model inference, human review required`; control characters and line breaks in model interpretation are rejected to keep that label attached to the text. The renderer does not put unrestricted model text into a verified-observation sentence. Screening facts and calculations keep the screening label. The recommendation is advisory and marked for human review.

## Adversarial coverage and limits

The new tests turn the v2 unsafe parses into v3 rejection or explicit unverified rendering checks. They cover an extra factual prose field, wrong key/value attribution, an appended observation statement, mixed ratio/millisecond arithmetic, incomplete independently defined aggregates, and attempted screening-to-confirmation metadata promotion. Negative controls accept exact observations, same-unit subtraction, ordinary human-readable summaries generated from metadata, and the four-ratio three-decimal P06 range. A speculative mechanism interpretation may still contain overconfident words, but the renderer labels it **unverified**; deterministic validation cannot prove or disprove causal scientific claims. A correct latency subtraction likewise does not establish a speedup for a newly proposed kernel.

Unit rules are intentionally narrow. Known latency fields are `ms`, ratio fields are `ratio` (a unitless metric), known memory fields are `bytes`, and shape/workload fields are `elements`; other values are `none` and cannot enter arithmetic. Addition/subtraction/ranges require identical units. Multiplication is supported when at least one operand is a ratio; division supports same-unit operands or a ratio denominator. These rules reject the demonstrated dimensional error, but do not establish that a mathematically valid metric is useful. Human review remains mandatory for comparisons, relevance, mechanism, and extrapolation.

## Offline preview and response budget

From the repository root:

```bash
.venv/bin/python muse_research_critic_v3.py --preview \
  --study-id 20261030T142150402688Z \
  --review-id 20261030T142419308682Z \
  --proposal-id P06
```

This prints the v3 instructions, response `SCHEMA`, source-derived `AGGREGATE_SETS`, and validated `SOURCE` locally. It contacted no model and wrote no audit. In the inspected P06 preview, the full request was **22,715 characters**, of which the instructions were **684 characters**, and the schema had five top-level fields. The original Muse request setting is 8,000 maximum output tokens. No compatible tokenizer was available locally, and no model response was requested, so actual token use, complete 16-claim response fit, and model compliance remain **unverified**. Before a live controlled study, use a separately versioned dry-run protocol and record truncation, output length, and response completeness rather than silently changing the limit.

## Verification and freeze assessment

Executed CPU-only from the repository root:

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/test_muse_research_critic.py tests/test_muse_research_critic_v2.py tests/test_muse_research_critic_v2_adversarial.py tests/test_muse_research_critic_v3.py tests/test_muse_critic_value_study.py tests/test_public_evidence.py` | **71 passed, 0 failed**. The v2 adversarial tests remain unchanged as a record of v2 behavior. |
| `python verify_public_evidence.py public_evidence` | Exit 0; 20 source-artifact identifiers. |
| v3 P06 `--preview` | Exit 0; schema and independent aggregate sets displayed. |

V3 closes the **demonstrated high-severity deterministic v2 gaps** in the synthetic suite and is ready for protocol review. It is **not yet frozen for a controlled Muse v1/v3 pilot**: response-budget fit and model compliance are unresolved, and the human scoring/acceptance plan must be preregistered. A valid v3 parse should never be called an accepted scientific criticism without that review. A future comparison should hold source evidence, model, settings, request counts, and blind human rubric fixed, preserve all failed attempts without retry, and score evidence fidelity and research usefulness separately from structural validity. It must be a new study, not a change to the incomplete Experiment 009 run.
