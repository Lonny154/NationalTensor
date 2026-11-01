# Experiment 011 — human scientific review for P06

## Review boundary and recommended disposition

Review target: original, manifested run `20261031T210750782912Z`, `evaluations/muse-critic-feasibility/v2/20261031T210750782912Z/result/audit.json` (SHA-256 `c022a29ffe14af06353f2f72b529613fd629620836a5016c25706310ed9df69c`). The request uses the v3.1 prompt and unchanged strict v3 response validator. Read-only `--validate` reports `structural_status: valid`, `scientific_status: pending_human_review`, and no first error. This is a single schema-feasibility response, not evidence of improved scientific reasoning.

**Recommended human-review disposition:** accept the structural validity and the narrow evidence-grounded conclusion that P06 is untested; qualify C10's claim-level support and the scope of its profiling phrase; retain `revise` as a defensible *advisory* next step, conditional on a controlled mechanism test. Do not endorse a P06 speedup, cache benefit, physical scheduling benefit, or confirmed performance result. The project owner's qualified sign-off is recorded below. This report does not change the audit's `scientific_status`.

## Claim-by-claim assessment

The exact source is the validated P06 proposal and Experiment 005 evidence in the frozen source package. E005 denotes `experiment_005:20261030T013245142673Z`. Ratios below are E005 **Hilbert candidate versus the named control** paired screening ratios, on representative FP16 GEMMs. None measures P06's platooned dispatch. The four-member range is deliberately limited to P06's two named shapes; it is not the range of all eight E005 screening comparisons.

| Claim | Actual support and assessment |
| --- | --- |
| C1 | E005 `confirmation_status`, scalar `"missing"` at `value_path: []`. Direct record of missing confirmation **in this E005 evidence package**, not a claim about all possible later work or P06. |
| C2 | E005 `profiling_status`, scalar `"not validated by Experiment 005 evidence pipeline"` at `[]`. Direct pipeline status. It does not establish that profiling never happened elsewhere. |
| C3 | E005 `screening:512x3072x768:grouped_m4` → `median_paired_ratio = 0.9992195058576494`. Correct exact attribution to prior Hilbert screening against grouped-M4. |
| C4 | E005 `screening:512x3072x768:row_major` → `0.9971394589037794`. Correct exact attribution to prior Hilbert screening against row-major. |
| C5 | E005 `screening:2048x3072x768:grouped_m4` → `1.0019068437051466`. Correct exact attribution to prior Hilbert screening against grouped-M4. |
| C6 | E005 `screening:2048x3072x768:row_major` → `0.9991797675787395`. Correct exact attribution to prior Hilbert screening against row-major. |
| C7 | `rounded_range` names source-derived set `screening:2048x3072x768+512x3072x768` and includes exactly C3–C6. Minimum `0.9971394589037794` rounds to `0.997`; maximum `1.0019068437051466` rounds to `1.002` at three decimal places. Complete and reproducible for that four-member set. It is a descriptive range across distinct shapes and opponents, not a pooled effect estimate or evidence for P06. |
| C8 | E005 `screening_scope.stage_decision = "screening_unconfirmed"`. Direct status of prior Hilbert screening. It does not by itself evaluate P06. |
| C9 | Supports C1, C7, C8. The claim that prior Hilbert ratios do not confirm P06's platooned-dispatch latency benefit is a scientifically sound inference: the cited ratios are prior-kernel screening and E005 lacks confirmation. The stronger statement that P06 itself is untested also relies on the proposal and absence of a P06 experiment in this source package, rather than an explicit cited observation claim. Treat it as a qualified inference, not a measured null result or proof that no test exists anywhere. |
| C10 | Supports **only C2**. C2 warrants saying that the validated E005 pipeline supplies no profiling support for a locality mechanism. It does not directly support the distinct statement that program-ID grouping cannot establish a physical execution-order benefit. The source's `verification.limitations` explicitly says program-ID assignment does not guarantee physical CTA execution order, and P06's `biggest_obstacle` acknowledges this; neither is cited as an atomic support claim in C10. The scheduling clause is plausible and consistent with that source, but **claim-level support is incomplete**. “Without profiling evidence” must be restricted to this validated pipeline. No L2 reuse, DRAM turnover, or scheduling benefit has been observed for P06. |

The strict parser checks exact observation values, units, paths, source-defined range membership, arithmetic, and that inference support IDs point backward. It cannot establish that an inference's wording follows from those support IDs. C10 illustrates that remaining scientific-review duty. C9 and C10 are interpretations, not direct measurements.

## Recommendation and discriminating test

`revise` is reasonable as an advisory research decision if it means specifying an implementable grouped mapping, a same-kernel row-major control, and a test that could reject the mechanism before committing larger resources. The four Hilbert ratios do not select `revise` mechanically: they neither test P06 nor establish its likely effect. `proceed` or `deprioritize` would require a separately argued judgment of information value and cost. The present evidence supports withholding a performance endorsement while preserving a testable idea.

A discriminating P06 experiment would first verify exact tile coverage and numerical equivalence for grouped and row-major mappings. Run both on P06's named `512x3072x768` and `2048x3072x768` workloads with the same GEMM body, tile configuration, inputs, environment, and timing procedure; alter only the dispatch mapping. Record independent paired latency screening and confirmation, including uncertainty intervals and mapping overhead. Measure relevant L2 hit/miss or traffic and DRAM traffic with a separately validated profiling stage. Where feasible, measure or trace actual CTA execution order/overlap rather than inferring it from program IDs. A latency difference without corresponding locality or execution-order evidence would not establish the proposed cause; unchanged latency and locality indicators versus the control would meet P06's stated disproof observation for these workloads. Keep these future GPU steps separate from this review; none was run here.

## Provenance and limits

The review used `docs/experiment_011.md`, `docs/experiment_010_failure_analysis.md`, `muse_research_critic_v3.py`, the original manifested Experiment 011 setup and result, and P06's validated source package as frozen in the Experiment 009 setup. The original Experiment 010 response remains rejected for its missing scalar paths; its diagnosis helps explain why v3.1 achieved structural compliance, not why this response should receive scientific approval.

## Human Review Sign-Off

| Field | Recorded review |
| --- | --- |
| Reviewer | Project owner |
| Review decision | **Qualified acceptance** |
| Review date | October 31, 2026 |
| Original Experiment 011 run ID | `20261031T210750782912Z` |
| Original audit SHA-256 | `c022a29ffe14af06353f2f72b529613fd629620836a5016c25706310ed9df69c` |
| Basis | The project owner's explicit approval of the documented scientific assessment. |

The project owner accepts that the four historical Hilbert screening ratios are correctly attributed, the rounded range is arithmetically correct, the model distinguishes prior screening from the untested P06 proposal, and `revise` is a defensible advisory next step.

Acceptance is qualified: C10's scheduling assertion lacks complete claim-level evidence support; the profiling statement applies only to the validated Experiment 005 evidence pipeline; P06's proposed mechanism has not been experimentally tested; and a discriminating experiment is needed before any performance conclusion.

This sign-off does **not** establish a P06 latency improvement, improved cache locality or physical scheduling, a confirmed GPU performance result, or a general improvement in Muse's scientific reasoning. Automatic structural validation and this separate human scientific review have different roles. Qualified acceptance retains every limitation identified above. The historical audit's `scientific_status` remains `pending_human_review` and its bytes are unchanged; this document records the owner's review without rewriting that audit. This is the project owner's approval, not an independent external peer review.
