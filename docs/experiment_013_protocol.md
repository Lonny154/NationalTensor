# Experiment 013 — model configuration and evidence fidelity

## Question and scope

Can model configuration affect the ability to produce evidence-grounded research decisions under the existing strict decision validator? This is a **two-request P03 feasibility comparison**, separately versioned from Experiments 009 and 012. It tests Spark High against Wonder High with identical prompt bytes, source, output budget, and validator. No request has been made. The offline run is `20261101T004838506734Z` under `evaluations/muse-model-capability-study/v1/`.

The comparison can isolate the **model choice at high reasoning effort** within this prompt. It cannot isolate the effect of increasing reasoning effort: both arms are high, and prior low-effort P03 requests used a different citation prompt. Nor is `muse-wonder-1.0` independently established as more scientifically capable by its name. Two attempts cannot support a general model ranking.

## Frozen P03 source and matched settings

Both configurations use the original P03 proposal and validated historical evidence from Experiment 009 run `20261030T205425148094Z`. The Experiment 009 loader checks its source and upstream evidence; Experiment 013 stores a copy and verifies it again on load. There is no rejected-criticism treatment and no response from one configuration enters the other's request.

| Setting | A — Spark High | B — Wonder High |
| --- | --- | --- |
| Model | `muse-spark-1.3` | `muse-wonder-1.0` |
| Reasoning effort | `high` | `high` |
| `max_output_tokens` | 16,000 | 16,000 |
| Temperature | API default | API default |
| Client timeout | 120 seconds | 120 seconds |
| Automatic retries | 0 | 0 |
| Prompt, source, validator | Identical | Identical |

The 16,000-token limit covers **reasoning and visible output together**. It gives more room than the previous 8,000-token low-effort requests, but sufficient capacity and server acceptance are unverified offline. The client records total output tokens and reported reasoning tokens separately when returned. A response that exhausts the budget remains a rejected, counted attempt. No fallback model or automatic retry is allowed.

Proposal SHA-256: `15cd9a573ba66d0cefa697c9741716dcb8de77500383727a0eb56a729ac128dc`. Canonical source-package SHA-256: `f2ad6a6781ff47d1ddbd36a901594614e620d22d3c006e7456db3455515bc445`. Original Experiment 009 setup file SHA-256: `742af6daf3eb120d139e05ae96a70edf6bacc2523ea8c494a887f3a7eb6fc9cb`.

## Separately versioned citation clarification

Prompt version `decision_v1_citation_objects_v1` retains the Experiment 009 decision instructions and source JSON. It replaces the original schema's empty citation-array example with a complete JSON response example containing this actual P03 observation reference:

```json
{
  "observed_evidence": "Experiment 005 independent confirmation is missing.",
  "evidence_references": {
    "observed_evidence": [
      {
        "evidence_id": "experiment_005:20261030T013245142673Z",
        "observation_key": "confirmation_status"
      }
    ]
  }
}
```

The shown fragment is excerpted from the full schema example; the actual prompt includes every required decision field. It says each field's reference list must contain two-key objects with exact supplied IDs and observation keys. Free-text citations and colon-joined citation strings do not count. Empty arrays are for fields without historical claims; `observed_evidence` needs at least one reference. The example is illustrative and is not a measured P03 result. The **unchanged** Experiment 009 `parse_decision` validator remains the acceptance gate.

The A and B prompt bytes are identical: 12,261 UTF-8 bytes, SHA-256 `effc2e85f7a850ea3f9a528366bbd744e9e3d6ba410fe74f4d4a10b043b43644` for each. Model and request settings live in the request specification, outside the prompt. This clarification addresses the Experiment 012 operator audit's shared citation-format failure without changing any historical response, validation outcome, or Experiment 012 treatment.

## One-attempt workflow and provenance

The prepared setup and preview are immutable SHA-256 manifested artifacts. The setup records source/proposal and prompt hashes, both configurations, request options, endpoint, timeout, retry count, validator identity, and `no_live_request_authorized_at_preparation`. The preview contains two full request specifications with `not_authorized_in_offline_preview`. Separate future request directories, `decisions/A` and `decisions/B`, each permit one attempt. An explicit `--human-authorized` flag is required for each live command, and the request function also enforces authorization. Neither a failed request nor a validator rejection can be retried in the same run. A future audit would preserve the exact request specification, raw response if any, response ID/model, safe response metadata and token usage, creation timestamp, parsed response only on acceptance, rejection kind and first error otherwise, explicit authorization state, and a SHA-256 manifest.

Offline commands, which make no Muse request:

```bash
.venv/bin/python muse_model_capability_study.py --validate --run-id 20261101T004838506734Z
.venv/bin/python muse_model_capability_study.py --preview --run-id 20261101T004838506734Z
```

The preview is `evaluations/muse-model-capability-study/v1/20261101T004838506734Z/preview/preview.json`. The live CLI exists for a **future, separately authorized** attempt; this preparation does not authorize either call. Before any such call, verify account access to both exact models, acceptance of `reasoning.effort: high` and the 16,000-token cap, credentials, and current manifests. Offline local repository and SDK inspection cannot establish server-side availability or supported settings. The local project has prior Spark requests but no verified Wonder response. A server error must be recorded as that configuration's failed attempt, with no substitution.

| Prepared artifact | SHA-256 |
| --- | --- |
| `setup/study.json` | `c5f14994b64e4a7ef708121bf71d504ff789d95200ba0acd4036145fd3581302` |
| `setup/manifest.json` | `f100f9108e12ee65db743fc77b5c2d4b86ca4a40953f75cdf72adea50fd14278` |
| `preview/preview.json` | `922eb871d3c51af608c3e45ec26e936f2c03ccb6691cbb7c83ba3367065763fa` |
| `preview/manifest.json` | `e6d926ca2b945609f2ff21d73a4c4c24ee6845ff1c7c6948d35dbd65b289ca7b` |

## Comparison and human assessment

Report both attempts, including request errors and incomplete outputs. For each, record API completion, JSON/root-schema validity, citation-object and source-key validity, the first unsupported numerical or experimental-stage claim reported by the validator, and input/output/reasoning token usage. A rejection is a feasibility outcome; it is not an estimate of the model's scientific quality. If a validator stops at a first error, label later diagnostics as operator-only and hypothetical, without changing the historical result.

Assess scientific usefulness **separately after structural validation**, using a human reviewer who sees the original P03 source and each decision without model labels or request metadata. Before reading either decision, retain this rubric: evidence fidelity; distinction between screening observations and inference; correct limits on confirmation and profiling; a discriminating grouped-traversal test; and proportional resource cost and discovery value. Record field-specific reasons, suspected wording-based unblinding, and any unsupported claim. Preserve the model-to-card mapping with the operator until both assessments are recorded. Do not change the rubric after seeing outputs. With one response per model, report a descriptive comparison only; fewer validator errors do not prove better reasoning.

The previous Experiment 012 P03 A/B audits remain rejected and unchanged. Their restricted operator report stays outside the reviewer packet. Experiment 009's frozen protocol and all source evidence remain unchanged. No GPU work is part of Experiment 013.

## Offline verification and remaining risks

The real Experiment 013 run and both previewed specifications passed read-only validation: both conditions are `not_requested`. The targeted CPU-only suite passed **26 tests** across Experiment 013, the reused decision validator, Experiment 012, and public evidence. `python verify_public_evidence.py public_evidence` passed and reported **20 source-artifact identifiers**. These checks establish local consistency and artifact integrity, not model availability or scientific performance.

Remaining risks are server-side model/settings availability, high-effort truncation despite the larger shared budget, strict numeric checks after citations are corrected, and one-sample subjective scientific review. If a configuration is unavailable or rejected, preserve that failure; do not silently replace the model, prompt, validator, or attempt.
