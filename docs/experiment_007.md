# Experiment 007 — Does Reflection Improve Discovery?

This study compares three advisory Muse hypotheses made from the standard validated research context with three made from that same context plus validated Experiment 005 observations and the accepted Experiment 006 reflection. Both requests use `muse-spark-1.3`, the same inference settings, instructions, and output schema. The schema extends the Wild Research Lab idea format with a specific `testable_prediction`; its existing `disproof_observation` supplies the falsification criterion. No GPU work or kernel implementation is part of this study.

Run these commands from the repository root. The first command validates the specified accepted reflection and its upstream evidence, freezes both prompts, and prints a `STUDY_ID`. It makes no Muse request:

```bash
.venv/bin/python muse_discovery_study.py --prepare-study --reflection-id 20261030T141047613123Z
```

Review `evaluations/muse-discovery-study/v1/STUDY_ID/setup/study.json`, especially both prompts. The baseline prompt has no Experiment 005 result or Experiment 006 reflection. Then make each independent request explicitly, in either order:

```bash
.venv/bin/python muse_discovery_study.py --request-condition A --study-id STUDY_ID
.venv/bin/python muse_discovery_study.py --request-condition B --study-id STUDY_ID
```

Set `MODEL_API_KEY` in the environment or repository `.env` for those two commands. Each condition is requested at most once per study; accepted and rejected attempts are retained in separate manifested audit directories under `requests/A/` and `requests/B/`. An incomplete or invalid response is rejected without replacing it. Start a new study if another attempt is needed.

After both requests validate, create a reproducible blind packet using a recorded seed:

```bash
.venv/bin/python muse_discovery_study.py --prepare-review --study-id STUDY_ID --seed 1729
```

The command prints a `REVIEW_ID`. Give the reviewer **only** `reviews/REVIEW_ID/blind/review.json`. It contains six shuffled, opaque proposal cards without condition labels or evidence reference IDs. Keep the setup, request audits, and `operator/assignment.json` with the study operator. Proposal wording itself could still suggest provenance; reviewers should record any suspected unblinding in their written justifications.

Create `scores.json` as a JSON object keyed by `P01` through `P06`. Each proposal must have all five criteria, each with an integer score from 1 to 5 and a written justification:

```json
{
  "P01": {
    "novelty": {"score": 3, "justification": "Explain the novelty judgment in at least ten characters."},
    "mechanism_specificity": {"score": 3, "justification": "Explain how specifically the mechanism is stated."},
    "falsifiability": {"score": 3, "justification": "Explain the prediction and disproof quality."},
    "evidence_alignment": {"score": 3, "justification": "Explain alignment with the available evidence."},
    "implementation_cost": {"score": 3, "justification": "Explain the estimated effort; five means lower cost."}
  },
  "P02": {"...": "Repeat the same five criteria"}
}
```

The example abbreviates `P02`; the submitted file must contain complete entries for all six IDs. Record the blind scores, then reveal the assignment:

```bash
.venv/bin/python muse_discovery_study.py --score-review --study-id STUDY_ID --review-id REVIEW_ID --scores-file scores.json
.venv/bin/python muse_discovery_study.py --reveal-review --study-id STUDY_ID --review-id REVIEW_ID
```

All setup, request, blind packet, score, and reveal artifacts have SHA-256 manifests. The reveal binds opaque IDs to their original full proposals and condition labels after complete scoring, and reports mean scores by criterion and condition. Scores are **subjective manual judgments**, not objective proof that one condition is a better researcher. The workflow makes no automatic winner selection and uses no model confidence score.
