"""Experiment 011 prompt-only clarification of the unchanged v3 response contract."""
from __future__ import annotations

import argparse
import json

from dashboard_data import ROOT
from muse_research_critic import source_package
from muse_research_critic_v3 import critique_prompt_v3

PROMPT_VERSION = "experiment_008_critic_prompt_v3_1"


def critique_prompt_v31(source):
    original = critique_prompt_v3(source)
    instructions, rest = original.split("\nSCHEMA\n", 1)
    schema_raw, suffix = rest.split("\nAGGREGATE_SETS\n", 1)
    schema = json.loads(schema_raw)
    schema["claims"][0]["value_path"] = []
    scalar_key = "confirmation_status"
    evidence = source["validated_evidence"]["005"]
    scalar = {"claim_id": "C1", "type": "observation", "evidence_id": evidence["evidence_id"],
              "observation_key": scalar_key, "value_path": [],
              "recorded_value": evidence["observations"][scalar_key], "unit": "none"}
    timing_key = next(key for key in sorted(evidence["observations"])
                      if key.startswith("screening:") and "median_paired_ratio" in evidence["observations"][key])
    nested = {"claim_id": "C2", "type": "observation", "evidence_id": evidence["evidence_id"],
              "observation_key": timing_key, "value_path": ["median_paired_ratio"],
              "recorded_value": evidence["observations"][timing_key]["median_paired_ratio"], "unit": "ratio"}
    clarification = (" Every observation MUST include value_path. Use [] when observation_key directly "
                     "contains the recorded scalar; use a nonempty path only to select a nested value. "
                     "The two complete EXAMPLES below illustrate these cases; copy the schema fields, "
                     "not their values unless supported by SOURCE.")
    instructions += clarification
    examples = {"scalar_status_observation": scalar, "nested_screening_ratio_observation": nested}
    return (instructions + "\nSCHEMA\n" + json.dumps(schema, sort_keys=True)
            + "\nEXAMPLES\n" + json.dumps(examples, sort_keys=True)
            + "\nAGGREGATE_SETS\n" + suffix)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline v3.1 critic prompt preview")
    parser.add_argument("--preview", action="store_true", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--proposal-id", required=True)
    args = parser.parse_args(argv)
    source = source_package(ROOT, args.study_id, args.review_id, args.proposal_id)
    print(critique_prompt_v31(source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
