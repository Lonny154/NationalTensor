# Experiment 010 — diagnosis of the rejected Muse v3 response

## Finding

The live response for run `20261031T205719056741Z` was complete JSON and within the 8,000-output-token budget, but the v3 parser correctly retained it as **rejected**. C6 and C7 are both `observation` claims that omit `value_path`, a field required by the v3 observation schema. For these two scalar status observations, the meaningful path is the empty list `[]`: the value is located directly at the observation key, with no nested traversal. The omission violates the validator's exact schema, while the request example leaves the empty-path convention insufficiently clear. This is a prompt/schema presentation inconsistency, not evidence that the scalar values were fabricated.

This report is a read-only analysis of the original `result/audit.json`, the manifested request prompt, v3 source and tests, and the validated P06 evidence package. It does not edit, revalidate into acceptance, or replace the historical response. The original audit SHA-256 remains `82ada61335a2e6423e90819aa153cbc25f8f9297696aec047d797dfe3c6655f1`.

## C6 and C7: contract versus meaning

| Claim | Type and cited source | Recorded value | Required path | Diagnosis |
| --- | --- | --- | --- | --- |
| **C6**, `claims[5]` | `observation`; E005 `confirmation_status` | `"missing"` | `value_path: []` | First parser failure: `observation schema mismatch`. The ID, key, value, and unit `none` are otherwise the appropriate fields for a source status, but the response omitted the required path. |
| **C7**, `claims[6]` | `observation`; E005 `profiling_status` | `"not validated by Experiment 005 evidence pipeline"` | `value_path: []` | Same missing-field failure after C6. The value is a statement about this evidence pipeline's validation status, not proof that no profiling was ever performed anywhere. |

`muse_research_critic_v3.py` defines `OBS_FIELDS` with `value_path` and rejects a claim whose key set differs before it looks up the evidence. `_observation` starts at `observations[observation_key]`, then traverses each element of `value_path`. An empty list therefore directly selects the scalar. The field is **contractually required for every observation**, though a nonempty path is **not semantically needed** for scalar values. Keeping it explicit gives observations one uniform representation and avoids an implicit default.

The exact original request says to return JSON matching `SCHEMA` and its observation example includes `value_path`, so the model broke an explicit field-list contract. Yet that example gives `value_path: ["nested key if needed"]`, and the short instruction says only to copy the “nested value”; neither states “use `[]` for a scalar observation.” A reader could reasonably understand “if needed” as making the field optional. The source itself contains both nested timing records and top-level scalar status keys, but the prompt provides no scalar example. The most precise conclusion is **a genuine strict-schema violation encouraged by an ambiguous example**, rather than a numerical-grounding failure.

The current tests establish exact nested paths and include an offline fake-client scalar with `value_path: []`. They do not test that the generated v3 prompt clearly teaches the empty-list convention for scalar observations. The original API response was 11 claims long; 3,166 output tokens were used, including 2,224 reasoning tokens. Truncation is not a plausible explanation for two selectively omitted fields in otherwise complete C6/C7 objects.

## Remaining deterministic checks

Read-only inspection verified the manifested request, audit, and subsequent field-review manifests. The audit's first stored error remains the generic `observation schema mismatch`; the separate review artifact identifies the first path as `claims[5].value_path` and C7 as `claims[6].value_path`.

| Claims | What was checked | Result and limit |
| --- | --- | --- |
| C1–C4 | Four E005 `median_paired_ratio` values on the 512×3072×768 and 2048×3072×768 shapes, against row-major and grouped-M4 opponents; exact IDs, keys, nested paths, values, and ratio units. | Each independently passes `_observation`. These are **Hilbert screening** comparisons, not P06 measurements. |
| C5 | `rounded_range` using C1–C4, the source-derived four-member aggregate set, and three-decimal rounding. | Independently passes `_derivation`; result `0.997` to `1.002` is arithmetically correct. Arithmetic does not establish a performance gain or mechanism. |
| C6–C7 | Required observation fields. | Both fail because `value_path` is absent. Their status strings match the source, but the historical JSON remains invalid. |
| C8–C9 | `screening_scope.stage_decision` and `verification.limitations`. | Both independently pass `_observation`. C9 is recorded limitation text, not a fresh hardware measurement. |
| C10–C11 | Inference schema and earlier support IDs. | In a temporary **in-memory copy only**, adding `value_path: []` to C6 and C7 allows the complete response to pass v3 structural parsing, including these inference records. That hypothetical check establishes no additional deterministic failure; it does not change the recorded rejection or constitute scientific acceptance. |

## Scientific reading

The response separates four prior screening ratios from interpretations and explicitly calls P06 untested. C10 says the descriptive range does not transfer to P06; C11 labels the L2/DRAM mechanism speculative and cites the recorded program-ID scheduling limitation. There is no visible claim that P06 achieved a measured speedup, a completed confirmation stage, or a demonstrated locality mechanism.

Two issues still need human judgment. First, C11's phrase “no profiling evidence” should be read within the validated E005 evidence pipeline; the source records `profiling_status` as **not validated by that pipeline**, which is narrower than a universal absence claim. Second, advisory `deprioritize` may or may not be the best research choice for a distinct grouping design: the four medians are from Hilbert-versus-control screening, and no P06 candidate was run. Correct citations and arithmetic do not settle whether a different dispatch deserves a discriminating test. The response cannot be promoted to accepted scientific criticism on this analysis.

## Smallest justified next-protocol change

Keep the strict observation schema and value checks. In a **separately versioned prompt**, state plainly: “Every observation includes `value_path`. Use `[]` when `observation_key` directly contains the recorded scalar; use a nonempty list only to traverse a nested value.” Give one scalar-status example and one nested-ratio example with real schema shapes. This changes instruction clarity without accepting omitted fields, relaxing provenance checks, or altering Experiment 010. A field-specific error such as `claims[5].value_path is required` would improve future diagnosis, but is secondary to the prompt correction.

If later evidence shows that models repeatedly omit empty paths despite an unambiguous prompt, consider a **new** schema where a missing path explicitly defaults to `[]` for scalar observations only. That would change validation behavior and must be versioned and evaluated separately. It should not be applied retroactively to this response.

### CPU-only regression-test plan

1. Add a source-only scalar-status fixture: `confirmation_status` with `value_path: []` validates; the same observation with `value_path` omitted rejects under v3, and a nonempty incorrect path rejects. Include a `profiling_status` scalar with the exact pipeline-scoped string.
2. Add a nested timing fixture: `screening:<shape>:<opponent>` with `["median_paired_ratio"]` validates; `[]` with the scalar ratio fails because the source value at the key is an object. Keep exact value and unit checks.
3. Snapshot the **new** prompt instructions and schema examples to assert that they explicitly distinguish scalar `[]` from nested paths. Check the prompt/version hash independently of the old Experiment 010 request.
4. In memory, replay the rejected raw response and assert its original v3 failure. On a disposable copy with only the two empty paths added, assert structural parsing passes while `scientific_status` remains `pending_human_review`. Never write that copy to an audit or reclassify the original.
5. Retain the aggregate, screening/confirmation, and inference controls from the v3 suite. Test any future field-specific error path without changing the acceptance boundary.

This diagnosis supports a small prompt clarification and better diagnostics, **not** a retry of Experiment 010. A later live attempt would require a distinct versioned protocol and explicit authorization.
