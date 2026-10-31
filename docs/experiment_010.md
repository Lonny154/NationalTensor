# Experiment 010 — one live Muse v3 feasibility request

## Question and boundary

Can `muse-spark-1.3` produce a complete, structurally valid criticism of Experiment 007 proposal P06 under the atomic-claim v3 protocol and an 8,000-output-token cap? This is **one feasibility attempt**, not a v1/v3 comparison and not a continuation of the frozen Experiment 009 pilot. No kernel was compiled or executed, and no GPU benchmark was run.

The original P06 source package was revalidated through the Experiment 007 chain. Its proposal is **Platooned Tile Dispatch for Locality** (`proposal_sha256: e07624e3c6e51120475e651ce085185d18b86bf378269d913bc08221eb93cc07`). The source package hash is `9ea3884510dc6498324f687476d158b7f1d5208db1ea8a4e1df5da8ac73c5f54`. It includes E003, E004, and E005 evidence IDs and their observation keys; the exact key inventory and source-artifact hashes are in the local request record. E005 screening is unconfirmed and profiling is not validated by its evidence pipeline. Prior measurements concern Hilbert traversal, not the proposed P06 kernel.

## Protocol and preflight

The separately versioned local entry point is `muse_critic_feasibility_010.py`. `--prepare` writes a manifested request record under `evaluations/muse-critic-feasibility/v1/<run-id>/setup/`; `--request` permits one attempt only and writes a manifested result audit; `--validate` checks the stored request and result without an API call. An exclusive attempt marker is created before the API call, so a failure or process interruption cannot silently trigger a second call. The client is configured with `max_retries=0`. Raw response text, safe response metadata, timing, errors, and hashes are retained locally; credentials are never written to the artifacts. The path imports no GPU runner or generated-kernel executor.

Preflight completed before the live call:

| Check | Result |
| --- | --- |
| Targeted CPU suite, including three offline Experiment 010 fake-client tests | **74 passed, 0 failed**. |
| `python verify_public_evidence.py public_evidence` | Passed; bundle reports 20 source-artifact identifiers. This checks integrity and internal consistency, not GPU reproduction. |
| P06 source identity and v3 preview | Validated P06, its proposal hash, source hash, evidence IDs, and source-derived aggregate sets. |
| Model input limit | [Meta's Muse Spark model documentation](https://dev.meta.ai/docs/models) states a 1,048,576-token context window for Muse Spark 1.3. The exact request prompt was 22,714 UTF-8 bytes; even a conservative one-token-per-byte upper bound plus 8,000 requested output tokens is far below that limit. Meta's [long-context guide](https://dev.meta.ai/docs/cookbook/long-context) says input and output share the window. No separate token-counting API request was made. |
| Output cap and retries | `max_output_tokens: 8000`, `reasoning.effort: low`, `max_retries: 0`, 30-second client timeout. `temperature` was omitted and recorded as API default; no temperature value was invented. |

Run ID: `20261031T205719056741Z`. The model was `muse-spark-1.3` at `https://api.meta.ai/v1/responses`, using prompt version `experiment_008_critic_v3`. Prompt SHA-256: `3b1f459c96d32d5e75b6280d95c898a15ce48ba16c6f7584f371f7dabfcc0b0f`. The expected response schema has exactly `protocol_version`, `proposal_id`, `proposal_sha256`, `claims`, and `recommendation`; observations, derivations, and inferences have separate atomic schemas. The request record preserves the full prompt locally and is not part of the public evidence bundle.

## Actual execution

**Exactly one** live request was made. The API returned response ID `resp_6ae6564e8696e354779c4836` with matching model ID and `status: completed`. The safe metadata reported no incomplete reason, refusal, unsupported output, or unfinished item. The service did not provide a finish reason (`null`), so completion is supported by the status and item statuses, not a reported stop code. The call took **11,023.779 ms**. Token usage was **7,171 input**, **3,166 output** (including **2,224 reasoning**), and **10,337 total**. The 8,000-output-token cap was not reached; visible response text was 3,146 characters. The API call succeeded, and the output was **not reported truncated**.

The v3 parser **rejected** the complete JSON response. Its first recorded error is `observation schema mismatch`. Read-only field review located the first failure at `claims[5].value_path` (C6): the required field was omitted for the scalar `confirmation_status` observation. C7 similarly omitted `value_path` at `claims[6].value_path` for `profiling_status`. The original rejection remains unchanged. The audit's `error_field` is null because the current parser returns a generic first error; the separate manifested field review records these paths without rewriting the audit. A missing `value_path` should have been an empty array in both entries. No automatic retry or response rewrite occurred.

The response had 11 claims. Independent local checks found that C1–C4 used exact E005 screening ratios, C5's source-defined four-member rounded range of **0.997–1.002** at three decimals was arithmetically correct, C8 matched `screening_scope.stage_decision`, and C9 matched the recorded `verification.limitations`. In a **temporary in-memory copy only**, adding `value_path: []` to C6 and C7 made the whole JSON pass v3 structural parsing. This establishes no further deterministic failure under that hypothetical correction; it does **not** change the rejected historical response, its audit, or the need for human review.

## Scientific reading and limits

The response chose advisory `deprioritize`. Its two inference entries explicitly describe P06 as untested, call the prior screening range descriptive, note missing confirmation and profiling, and treat the L2/DRAM mechanism as speculative. It does not present the four Hilbert screening ratios as measured P06 performance. That is a favorable **qualitative observation about this one rejected response**, not an accepted scientific judgment or proof of reliable reasoning. A human reviewer should still assess the recommendation, whether P06 differs enough from the Hilbert controls to warrant a discriminating test, and whether “no profiling evidence” is appropriately scoped to the validated E005 pipeline. The C9 limitation list is recorded text, not a new hardware measurement.

The experiment answers the API-feasibility part positively: a complete response fit within the budget. It answers the full success question negatively: that response did **not** satisfy the strict v3 structural contract. One attempt provides no acceptance-rate estimate, no evidence of improved decision quality, no kernel performance result, and no compute-saving estimate.

## Artifacts and read-only verification

Local, ignored artifact directory: `evaluations/muse-critic-feasibility/v1/20261031T205719056741Z/`.

| File or stage | Contents |
| --- | --- |
| `setup/request.json`, `setup/manifest.json` | Full request specification, validated source hashes, evidence keys, prompt, prompt hash, model and parameter record. |
| `attempt_started.json` | One-attempt marker created before the request. |
| `result/audit.json`, `result/manifest.json` | Immutable original result, raw response, response metadata and usage, duration, first validation error. Audit SHA-256: `82ada61335a2e6423e90819aa153cbc25f8f9297696aec047d797dfe3c6655f1`. |
| `review/review.json`, `review/manifest.json` | Subsequent read-only forensic classification of C6/C7 and locally checkable claims, bound to the original audit SHA-256. Review manifest SHA-256: `786986bc04f78855b2c3315081c2c070b31b4923df8a90bf114629c9bbd80f4b`. |

Read-only command `.venv/bin/python muse_critic_feasibility_010.py --validate --run-id 20261031T205719056741Z` exited 0 and reproduced `structural_status: rejected`, `scientific_status: pending_human_review`, and the stored first error. The review-stage manifest also verified. The original audit and hashes were not altered after the live attempt.

## Recommended next experiment

First test offline whether a revised, separately versioned schema example that shows `value_path: []` for scalar observations improves completeness in synthetic and mock-response tests. Also make the validator report claim index and field path on schema failure, preserving strict rejection. If a new live feasibility attempt is later authorized, give it a **new run and prompt version**, preserve its first result without retry, and compare field-level failure modes. A controlled v1/v3 scientific-quality study still needs preregistered matched conditions and blinded human review; this Experiment 010 attempt does not supply that comparison.
