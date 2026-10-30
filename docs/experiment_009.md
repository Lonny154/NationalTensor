# Experiment 009 — Value of Research Criticism

This controlled pilot compares advisory decisions for the six original validated Experiment 007 proposals. Each proposal gets two independent decision requests. A sees the original proposal and the same validated historical evidence as B; B additionally sees an independent evidence-grounded criticism. The decision model (`muse-spark-1.3`), inference settings, schema, and core instruction bytes are fixed. The only added prompt section in B is `INDEPENDENT_CRITICISM`. Requests are separate and never include the other condition's decision.

The frozen five-part rubric is stored in the setup before any decision request. It scores evidence fidelity, redundancy avoidance, discriminating tests, discovery preservation, and resource awareness from 1 to 5. Each score needs a written explanation. Choosing a particular decision category earns no points. Scores are subjective manual judgments; they are not model confidence or measured compute savings.

Run from the repository root. Offline preparation has already created run `20261030T205425148094Z` with seed `9009`, and bound P01 to accepted Experiment 008 case `20261030T152949236420Z`. The two earlier rejected P01 criticism attempts remain preserved and unused. For a new run:

```bash
.venv/bin/python muse_critic_value_study.py --prepare --seed 9009
```

The command prints `RUN_ID`. Review `evaluations/muse-critic-value-study/v1/RUN_ID/setup/study.json`. It contains the original six source proposals, source hashes, validated evidence, frozen rubric, model, settings, and seed. The source loader revalidates the Experiment 007 study and reveal and its upstream evidence. Experiment 007 subjective blind scores are used by that loader solely to validate reveal provenance; **they are not included in either decision prompt or the critic prompt**.

## Criticism

P01's accepted case uses the same Experiment 008 critique prompt and validator as cases prepared for P02–P06. Its provenance is recorded as `accepted_008`. For any run, bind it explicitly:

```bash
.venv/bin/python muse_critic_value_study.py --attach-criticism --run-id RUN_ID --proposal-id P01 --case-id 20261030T152949236420Z
```

For each of P02–P06, inspect the expected prompt offline, prepare an Experiment 008 case, inspect its setup, then make an explicit criticism request:

```bash
.venv/bin/python muse_critic_value_study.py --critic-request --run-id RUN_ID --proposal-id P02
.venv/bin/python muse_research_critic.py --prepare --study-id 20261030T142150402688Z --review-id 20261030T142419308682Z --proposal-id P02
.venv/bin/python muse_research_critic.py --criticize --case-id CASE_ID
.venv/bin/python muse_critic_value_study.py --attach-criticism --run-id RUN_ID --proposal-id P02 --case-id CASE_ID
```

Repeat with P03, P04, P05, and P06. The bind step requires the exact original source and Experiment 008 prompt. It records case ID, prompt version and hash, accepted audit hash, and criticism hash. A historical rejected criticism may be used only after an explicit valid Experiment 008 `--revalidate` result, passed with `--revalidation-id`; the binding then records `revalidated_rejected_008`. Rejected or incomplete criticism requests remain in Experiment 008 audit directories and are never silently retried.

## Decisions

Inspect both request specifications for each proposal. The A prompt contains no critic assessment. Both prompts have the identical original proposal and validated evidence. `--decision-request` is offline; `--request-decision` is the only Experiment 009 command that calls Muse. B requires a bound criticism. Run A and B in either order; use separate calls and do not share one response with the other.

```bash
.venv/bin/python muse_critic_value_study.py --decision-request --run-id RUN_ID --proposal-id P01 --condition A
.venv/bin/python muse_critic_value_study.py --decision-request --run-id RUN_ID --proposal-id P01 --condition B
.venv/bin/python muse_critic_value_study.py --request-decision --run-id RUN_ID --proposal-id P01 --condition A
.venv/bin/python muse_critic_value_study.py --request-decision --run-id RUN_ID --proposal-id P01 --condition B
```

Repeat for P02–P06. Live commands require `MODEL_API_KEY` in the environment or `.env`. They use no client retries. Each condition permits one attempt per run; accepted and rejected responses retain raw text, response ID, model, settings, timestamp, token usage metadata, validation outcome, and a SHA-256 manifest. If a request fails, inspect its audit and start a new run for another attempt. A `proceed` decision remains advisory and does not authorize GPU execution or kernel implementation.

Decision JSON requires proposal identity, `proceed`/`revise`/`deprioritize`, distinct observation and inference fields, evidence assessment, prior overlap, expected information gain, potential discovery value, false rejection risk, next validation step, resource cost, uncertainties, justification, and field-scoped evidence references. Each reference must name an existing validated evidence ID and observation key. Invalid citations and unsupported numerical claims are rejected with field names. Deterministic validation checks structure, references, source hashes, artifact hashes, actual recorded measurements, and explicit arithmetic; human scoring judges scientific reasoning.

## Blind review and reveal

After all twelve decisions are accepted:

```bash
.venv/bin/python muse_critic_value_study.py --prepare-blind --run-id RUN_ID
```

Give the reviewer **only** `evaluations/muse-critic-value-study/v1/RUN_ID/blind/review.json`. Keep `operator/assignment.json`, source setup, criticism, and audits sealed with the operator until scoring is complete. Within each proposal, the two decisions are shuffled using the setup seed. The blind packet omits condition labels, source hashes, and critic text, but wording may suggest the condition. Record possible unblinding for every card.

Create a local `scores.json` with all `P01-1`, `P01-2`, through `P06-2` entries. Each entry has `possible_unblinding` (empty string if none) and all five rubric criteria, each with a 1–5 integer score and a written justification of at least ten characters. Example of one entry:

```json
{
  "P01-1": {
    "possible_unblinding": "Decision wording may refer to an external assessment.",
    "criteria": {
      "evidence_fidelity": {"score": 3, "justification": "Accurate but misses one limitation."},
      "redundancy_avoidance": {"score": 3, "justification": "Identifies the closest prior test."},
      "discriminating_tests": {"score": 3, "justification": "Comparison has one unresolved confound."},
      "discovery_preservation": {"score": 3, "justification": "Weighs upside against false rejection."},
      "resource_awareness": {"score": 3, "justification": "Stages work before expensive checks."}
    }
  }
}
```

The actual file must contain all twelve complete entries. Then record scores and reveal assignments:

```bash
.venv/bin/python muse_critic_value_study.py --score --run-id RUN_ID --scores-file scores.json
.venv/bin/python muse_critic_value_study.py --reveal --run-id RUN_ID
.venv/bin/python muse_critic_value_study.py --validate --run-id RUN_ID
```

The reveal includes condition means and six paired B-minus-A score differences per criterion. Interpret these as a small, subjective pilot. Decision categories themselves are not scored, recommendations are never automatically adopted, and no compute savings claim is made without resource measurements. All Experiment 005–008 artifacts remain unchanged.
