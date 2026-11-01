# Experiment 013 — scientific and validator failure analysis

## Scope and preserved result

This is a read-only audit of run `20261101T004838506734Z`, using the frozen P03 proposal and validated Experiments 003–005 source package. Both high-effort requests returned complete JSON, used the same 12,261-byte prompt (`effc2e85f7a850ea3f9a528366bbd744e9e3d6ba410fe74f4d4a10b043b43644`), and remain **rejected** by the unchanged Experiment 009 decision validator. The two responses are observations of model behavior, not GPU experiments. Nothing here accepts, repairs, or retrospectively scores either response.

The Experiment 013 loader and each audit loader passed read-only manifest and source checks. The P03 proposal SHA-256 is `15cd9a573ba66d0cefa697c9741716dcb8de77500383727a0eb56a729ac128dc`; the canonical validated source SHA-256 is `f2ad6a6781ff47d1ddbd36a901594614e620d22d3c006e7456db3455515bc445`.

| Preserved artifact | SHA-256 |
| --- | --- |
| Spark High audit / manifest | `3339831b10a36a551183b95af7187355c5568e605dc78d745c0cb54bf9624c0c` / `9da2f8b56a2e0d4a349020b3b75d32eea57ce30e38d6eff4b71162e99b40ed59` |
| Wonder High audit / manifest | `3f8c354072493adc1c5807568f3eb533a2bb7454aaceac5e4f9a2830f79f1dd5` / `ea0e56c12fd73e9cac5bb60e31d45254bbd42b8a0f72bd35ee36074c7729506d` |
| Spark High request / raw response | `7980c875394762583a4dff7f988c4ccef120260172cddcfe3a3553a53ae9af87` / `c832cd1766bf805c4c515e083fa69fd1333217131a5759ae0a2db17fb551deb4` |
| Wonder High request / raw response | `7519d3eeaba654250c392568f7a25c44d3f8d6e605b6f0bbe51f3d7097b13951` / `d5b2a1284d485e3a64a2a717ced8ecc0ecc31f6a6a38493c118f6c967188454d` |

Both responses have the expected root schema, P03 identity, all eleven evidence-reference fields, and well-formed two-key citation objects that resolve to supplied evidence IDs and observation keys. The first failures are **prose-validation failures**, unlike Experiment 012's citation-object failures. Structural source-key validity alone does not establish that a citation supports each sentence.

## Wonder High: confirmed stage-check false positive

The first error is `uncertainties: unsupported confirmation or profiling claim`. The exact triggering sentence is:

> Independent confirmation and validated profiling remain missing.

This sentence asserts **absence**, not completion. The cited Experiment 005 observations say `confirmation_status: missing` and `profiling_status: not validated by Experiment 005 evidence pipeline`. That is the correct scope: the supplied Experiment 005 pipeline has no validated profiling; the sentence does not establish anything about every possible external profiler. The cited `verification` record also cautions that compiler differences do not establish cache locality. No supplied observation contradicts the sentence.

`_check_text` searches for an affirmative-looking phrase such as `validated profiling`. It then looks only in the **25 characters before** that match for `no`, `not`, `without`, `missing`, `absent`, or `unconfirmed`. In the response, `missing` comes **after** the match. The guard therefore misses the sentence's negation, and the code rejects any matched profiling phrase. This is a **confirmed validator false positive for this sentence**, not evidence that Wonder asserted completed profiling. A field-by-field read-only diagnostic found no other deterministic error in Wonder's response. It did not alter the original rejection.

Minimal, standalone CPU-only reproduction with synthetic source data matching the two validated status values:

```python
from muse_research_critic import _check_text, _references

source = {
    "proposal": {},
    "validated_evidence": {
        "005": {
            "evidence_id": "e:005",
            "observations": {
                "confirmation_status": "missing",
                "profiling_status": "not validated by Experiment 005 evidence pipeline",
            },
        },
    },
}
refs = [
    {"evidence_id": "e:005", "observation_key": "confirmation_status"},
    {"evidence_id": "e:005", "observation_key": "profiling_status"},
]
values = _references(refs, source, required=False)
_check_text("Independent confirmation and validated profiling remain missing.",
            source, values, refs, field="synthetic")  # raises false positive
_check_text("Independent confirmation remains missing; profiling is not validated.",
            source, values, refs, field="synthetic")  # passes
```

The same fixture correctly rejects `Validated profiling was completed.` and passes `No validated profiling was completed.` It also incorrectly rejects `Validated profiling remains missing.` The failure depends on wording position, not a change in evidence. This demonstrates a validator limitation; it does not retroactively make the complete Wonder decision accepted or scientifically approved.

## Spark High: numerical and citation findings

Spark's stored first error is `observed_evidence: unsupported numerical measurement: '1.0'`. Its statement that screening medians are “near 1.0” is a plausible **rounded interpretation** of the Experiment 005 Hilbert screening ratios, whose eight exact medians span `0.9807409315299132` to `1.0019068437051466`. All eight recorded 95% bootstrap intervals include 1. The value `1.0` is not an exact cited observation, so rejection follows the frozen exact-number rule. This is model noncompliance with that rule, while the qualitative “near one” interpretation is arithmetically reasonable. The `observed_evidence` field cites only the two M=128 screening comparisons for its broader “reported shapes” statement; support for all shapes is not field-complete.

Spark's `evidence_assessment` gives the exact `0.9807409315299132` ratio **for M=128 against `row_major`** but cites `screening:2048x3072x768:row_major` in that field. The cited 2048-shape ratio is `0.9991797675787395`. The number itself is accurate for the 128-shape observation, but the field-scoped citation is wrong. This is a genuine **provenance error**, not a bad calculation. A minimal fixture with the 2048 key rejects the exact 128 ratio; replacing that reference with `screening:128x3072x768:row_major` passes. This fixture is a diagnostic of the rule, not a correction to the audit.

The corresponding minimal CPU-only fixture, with synthetic records containing the actual reported ratios, reproduces the scope and rounding failures:

```python
from muse_research_critic import _check_text, _references

source = {"proposal": {}, "validated_evidence": {"005": {
    "evidence_id": "e:005", "observations": {
        "confirmation_status": "missing",
        "screening:128x3072x768:row_major": {
            "median_paired_ratio": 0.9807409315299132,
            "bootstrap_median_ratio_95pct": [0.8893462207441152, 1.0473196685026485]},
        "screening:2048x3072x768:row_major": {
            "median_paired_ratio": 0.9991797675787395},
    }}}}

def check(text, key):
    refs = [{"evidence_id": "e:005", "observation_key": key}]
    _check_text(text, source, _references(refs, source, required=False),
                refs, field="synthetic")

check("Screening medians are near 1.0 with uncertainty.",
      "screening:128x3072x768:row_major")  # rejects rounded 1.0
check("The ratio 0.9807409315299132 is reported.",
      "screening:2048x3072x768:row_major")  # rejects wrong-shape citation
check("The ratio 0.9807409315299132 is reported.",
      "screening:128x3072x768:row_major")   # passes
check("The bootstrap 95% interval includes 1.",
      "screening:128x3072x768:row_major")   # rejects 95 label
```

Spark also writes an approximate `~0.02–0.03 ms` range for M=128 kernel medians. The cited 128-versus-row-major comparison records `0.03098096013069153` and `0.030825600624084473` ms; the separate grouped-M4 comparison records `0.024791040420532227` and `0.02635759949684143` ms. The approximation is broadly in the right scale, but neither endpoint is an exact cited value, one observed median exceeds `0.03`, and the field cites only the row-major comparison. The frozen validator's additional field check rejects `0.02`. This is imprecise model reporting and incomplete attribution, not a newly measured cost for P03.

A later, independent field check rejects `95` in Spark's `bootstrap 95% intervals` phrase. The cited screening observation's key `bootstrap_median_ratio_95pct` specifies the 95% interval, and the interval endpoints quoted by Spark match the two cited 128-shape records. The validator scans numbers in prose but does not admit the `95` embedded in an observation **key** as an allowed value. A synthetic `bootstrap 95% interval` phrase rejects while `bootstrap interval` passes with the same reference. This is a **second, narrower validator false positive**, encountered only after bypassing earlier Spark failures in an in-memory diagnostic; it is not Spark's historical first error.

Finally, Spark describes P03 as failing latency and mechanism thresholds. P03's proposed grouped mapping has **not** been timed or profiled. The historical screening tested Hilbert against `row_major` and `grouped_m4`; it did not establish a P03 performance failure. The defensible interpretation is that P03 lacks evidence to pass such thresholds. This is an unsupported conclusion about an unmeasured proposal, even though Spark elsewhere acknowledges that the mechanism remains unconfirmed.

## Scientific assessment on the same criteria

| Criterion | Spark High | Wonder High |
| --- | --- | --- |
| Evidence fidelity | Correctly recognizes missing confirmation and prior Hilbert screening; weakens field-level fidelity through one wrong-shape citation and shape-wide wording supported by only M=128 citations. | Uses exact M=128 screening values with matching keys and separates the tested Hilbert candidate from untested P03. No comparable wrong-key claim was identified in this review. |
| Experimental-stage correctness | Acknowledges screening is unconfirmed and profiling is not validated. The claim that P03 “fails” performance thresholds overstates what the untested proposal can show. | Explicitly says confirmation and validated profiling remain missing. The validator rejects that true negative statement because of its word order. |
| Numerical accuracy | Exact quoted 128 interval endpoints and ratio match the source, but `1.0` and `~0.02–0.03 ms` are unsupported approximations under the frozen rule; the cited row-major kernel medians both exceed the range's `0.03 ms` upper endpoint. The `95%` label is supported by the observation key despite a later validator rejection. | The quoted 128 ratio `0.9807409315299132` and interval `[0.8893462207441152, 1.0473196685026485]` match Experiment 005. This review identified no unsupported numerical claim. |
| Hardware-mechanism reasoning | Treats L2 reuse, DRAM traffic, and physical CTA order as hypotheses; correctly notes program-ID order does not guarantee execution order. Its claim that P03's mapping differs from prior orderings needs an explicit tile-sequence definition. | Recognizes possible overlap with `grouped_m4`, distinguishes logical tile order from physical CTA order, and does not claim measured locality. Its overlap assessment is an interpretation until P03's exact mapping is specified. |
| Discriminating experiment | Proposes holding tile sizes and depth fixed, comparing grouped versus raster mapping, and measuring latency plus L2/traffic counters. A prior mapping audit and codegen controls are needed so the candidate really differs and the mechanism can be isolated. | Proposes a low-cost tile-sequence audit before a later controlled latency/profiling test. That audit can establish logical mapping differences, but cannot by itself establish physical CTA co-residency or L2 behavior. |

These are **unblinded, qualitative scientific interpretations**, separate from Experiment 005's measured GPU observations and from any future blind human scoring. Neither response demonstrates a P03 speedup, cache-locality improvement, or actual physical scheduling change. One response per model cannot rank their reasoning ability.

## Implications and verification

The two-request feasibility result is **0/2 accepted decisions** under the frozen validator. The prompt clarification solved the earlier citation-object format problem in both outputs, but model compliance and validator wording sensitivity remain distinct issues. The confirmed Wonder false positive means acceptance counts alone would misrepresent scientific fidelity in a future model comparison. Spark has a genuine field-scoped citation error and an overstatement about unmeasured P03 performance, alongside a later `95%` validator false positive. Any future comparison should predeclare how to report first errors, latent diagnostic errors, and human scientific review without rewriting old outcomes. The next project priority is **GPU performance anomaly discovery**, as directed; this audit does not propose another schema-development cycle.

Read-only verification used `.venv/bin/python muse_model_capability_study.py --validate --run-id 20261101T004838506734Z`, `load_decision` for both manifested audits, the frozen `_references` and `_check_text` functions on synthetic CPU-only sentences, and direct checks of the validated Experiment 003–005 observations. The run validator still reports A and B as `rejected`, and the four decision audit/manifest hashes above remain unchanged. The targeted CPU-only regression suite passed **42 tests** (`test_muse_model_capability_study.py`, `test_muse_critic_value_study.py`, `test_muse_research_critic.py`, and `test_public_evidence.py`). `python verify_public_evidence.py public_evidence` passed with **20 source-artifact identifiers**. No Muse request, GPU benchmark, validator change, historical-audit edit, commit, or push was made.
