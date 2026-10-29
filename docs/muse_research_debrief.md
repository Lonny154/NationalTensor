# Muse Research Debrief

The debrief interprets the historical, hash-validated Experiment 003 compiler diagnostic. It reads the fixed artifact through `mask_hoisting_memory.load` and packages correctness, compiler resources, executable and metadata comparison, source identities, the screening skip reason, and explicit `latency_measured: false` and `speedup_measured: false` fields. It does not use generated prose as experimental evidence.

Inspect the exact prompt without a network call:

```bash
.venv/bin/python muse_debrief.py --prepare-only
```

Request the first live debrief explicitly:

```bash
.venv/bin/python muse_debrief.py --request-from-muse
```

Or open the dashboard's **Research workflow** tab and select **Request one live Muse Research Debrief**. The command or button requires `MODEL_API_KEY` (loaded from `.env` if present). It makes at most one API call with a 30-second timeout and no SDK retries. A hash-verified accepted debrief prevents repeat calls. Rejected attempts can be retried by an explicit request.

Requests use `reasoning.effort: low` and an 8,000-token output limit, which covers private reasoning plus visible JSON. These fixed settings are saved in the audit. The limit does not guarantee a complete response; incomplete results are rejected without retrying automatically. See [Meta's reasoning documentation](https://dev.meta.ai/docs/reasoning).

Each attempt is saved under `evaluations/muse-research-debrief/<UTC timestamp>/audit.json`, including the exact prompt, raw response, reported model identity, and rejection reason or accepted interpretation. An accepted record is also saved as `accepted.json`. The response must contain six bounded text fields with citations to the supplied evidence ID and observation keys. Invalid references, malformed responses, unsupported numerals, and claims of measured runtime benefit are rejected. Citation validation is structural; prose remains Muse interpretation and should be read as such.

Research Mode shows the interpretation, citations, limitations, and recommendation. Hollywood Mode adds clearly labeled theatrical dialogue beside the same verified findings. Neither route invokes an experiment runner, changes trusted kernels, grants approval, or starts GPU execution.

## Response diagnostics

The response handler supports SDK objects and decoded Responses API mappings. It extracts `message.content[].output_text` blocks in order, with a fallback for convenience-only `output_text` adapters. Reasoning items, tool arguments, and refusal text are never interpreted as the debrief. Missing completion status, incomplete output, API errors, refusals, unsupported output, empty text, and malformed JSON have distinct rejection paths. Scientific claim and citation validation still runs after response handling.

New audit records include safe response metadata: completion status, output/content item types, finish or incomplete reasons, error code, refusal presence, text length, and available token usage. Request headers, credentials, raw reasoning, and arbitrary API error messages are excluded.

The first request (`resp_6ae2a691ef034f7fb3454b9d`) was inspected through a read-only stored-response GET. It had `status: incomplete`, `incomplete_details.reason: max_output_tokens`, and only an incomplete reasoning item. It consumed 2,500 output tokens, including 2,498 reasoning tokens, and produced no visible text. No API error or refusal was reported. The former 2,500-token budget was exhausted before a message answer was available; the SDK did not miss text in another output field. The old JSON rejection was therefore misleading.

The original audit is unchanged. Its SHA-256 and retrieved safe metadata are recorded beside it in `evaluations/muse-research-debrief/20261029T015104101751Z/response_diagnostics.json`. A manual retry is appropriate with the revised settings and citation example; no new generation was performed during diagnosis. Completion and scientific acceptance of a future response remain unverified.
