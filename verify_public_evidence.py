"""Verify the portable National Tensor evidence export, using only Python stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import random
from statistics import median

SCHEMA = 'national_tensor_public_evidence_v1'
FILES = ('evidence.json', 'summary.md', 'README.md')
CRITERIA = ('novelty', 'mechanism_specificity', 'falsifiability', 'evidence_alignment', 'implementation_cost')
PROPOSALS = tuple(f'P{i:02d}' for i in range(1, 7))
SHAPES = ('128x3072x768', '512x3072x768', '2048x3072x768', '2048x4096x1024')

README = """# Public research evidence

This portable bundle summarizes validated local records for National Tensor Experiments 005–009. `evidence.json` contains allowlisted observations, model judgments, source identifiers, and reported review scores. `summary.md` is generated from those fields. `manifest.json` hashes every published file and separately records hashes of the original local artifacts used for export.

## Inspect and verify

From the repository root, use only Python's standard library:

```bash
python verify_public_evidence.py public_evidence
python -m json.tool public_evidence/evidence.json
```

The verifier checks file hashes, schemas, citation/reference shape, screening arithmetic and intervals, score means, missing confirmation and profiling, and the incomplete Experiment 009 state. It needs no GPU, API key, or original `evaluations/` directory. Its success establishes **bundle integrity and internal consistency**, not independent reproduction of the GPU measurements or scientific correctness of model prose.

`experiment_005` contains hardware observations; `experiment_006` and `experiment_008` contain advisory Muse judgments; `experiment_007` contains human subjective scores; `experiment_009` records attempts and a partial decision pilot. A rejected response remains rejected even when its validation lesson is summarized. No confirmed speedup or GPU compute saving is claimed.

Original run artifacts under `evaluations/` are ignored by Git. Reproducing measurements requires compatible GPU hardware, toolchain, source snapshots, and explicit human authorization for each GPU stage. The relevant protocols are in `docs/experiment_005.md` through `docs/experiment_009.md` at the repository root. Regenerating this bundle additionally requires the preserved original artifact directories and a compatible environment:

```bash
.venv/bin/python export_public_evidence.py --output public_evidence
```

The exporter validates its local sources and fails if a required record is missing or changed. It never calls Muse or launches a GPU experiment. The exported manifest preserves source hashes so reviewers can compare a separately supplied original artifact without requiring it for routine verification. Raw prompts, raw model responses, request headers, credentials, local paths, GPU UUIDs, and unnecessary machine metadata are excluded.
"""


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def means(scores, assignments):
    result = {}
    for condition in ('A', 'B'):
        ids = [row['evaluation_id'] for row in assignments if row['condition'] == condition]
        require(len(ids) == 3, f'Experiment 007 {condition} needs three proposals')
        result[condition] = {criterion: sum(scores[pid][criterion]['score'] for pid in ids) / 3
                             for criterion in CRITERIA}
    return result


def render_summary(data):
    e5, e6, e7, e8, e9 = (data[f'experiment_00{i}'] for i in range(5, 10))
    lines = [
        '# National Tensor: public evidence summary', '',
        'This summary is generated from `evidence.json`. Model judgments and manual scores are labeled separately from hardware observations.', '',
        '## Experiment 005 — Hilbert traversal', '',
        f"Original hypothesis: **{e5['hypothesis']['name']}** — {e5['hypothesis']['core_mechanism']}", '',
        f"Hardware: {e5['environment']['gpu']}; Python {e5['environment']['python']}, PyTorch {e5['environment']['torch']}, CUDA {e5['environment']['cuda']}, Triton {e5['environment']['triton']}.", '',
        f"All {len(e5['correctness'])} scoped correctness checks passed across four shapes and three schedules. Eight paired timing comparisons were recorded as **screening only**; independent confirmation and validated profiling are missing.", '',
        '| Shape | Opponent | Median paired ratio | Descriptive 95% bootstrap interval |',
        '| --- | --- | ---: | --- |',
    ]
    for row in e5['screening']:
        low, high = row['bootstrap_median_ratio_95pct']
        lines.append(f"| {row['shape']} | {row['opponent']} | {row['median_paired_ratio']:.4f}× | [{low:.4f}, {high:.4f}] |")
    lines += ['', 'No comparison has a bootstrap lower bound above 1. These observations do not establish a repeatable speedup or the proposed cache-locality mechanism.', '',
              '## Experiment 006 — Muse reflection', '',
              f"Model {e6['model']} ({e6['prompt_version']}) chose **{e6['decision']}** as an advisory decision. {e6['evidence_based_justification']}", '',
              '## Experiment 007 — discovery comparison', '',
              'Six original hypotheses were blindly scored, then their conditions were revealed. The preserved scores and assignments in `evidence.json` reproduce the recorded means. These are subjective assessments, not statistical evidence of model superiority.', '',
              '| Condition | ' + ' | '.join(CRITERIA) + ' |',
              '| --- | ' + ' | '.join('---:' for _ in CRITERIA) + ' |']
    for condition in ('A', 'B'):
        lines.append('| ' + condition + ' | ' + ' | '.join(f"{e7['means'][condition][c]:.2f}" for c in CRITERIA) + ' |')
    lines += ['', '## Experiment 008 — independent critic', '',
              f"P01 received an **{e8['accepted_criticism']['status']}** criticism recommending **{e8['accepted_criticism']['recommendation']}**. {e8['accepted_criticism']['reasoning']}",
              'Revision attempts listed in `evidence.json` retain their historical rejection status; no rejected revision is treated as accepted.', '',
              '## Experiment 009 — critic value pilot', '',
              f"Six proposal-level criticism outcomes are recorded: {sum(x['status'] == 'accepted' for x in e9['criticism_attempts'])} accepted and {sum(x['status'] == 'rejected' for x in e9['criticism_attempts'])} rejected. P01 has accepted A and B decision outputs. The six-proposal paired scoring and reveal are **incomplete**. No GPU compute savings were measured.", '']
    return '\n'.join(lines)


def privacy_check(value):
    text = json.dumps(value, ensure_ascii=False)
    for pattern in (r'/home/[^\s"\\]+', r'/Users/[^\s"\\]+', r'GPU-[0-9a-fA-F-]{16,}',
                    r'\bBearer\s+\S+', r'\bsk-[A-Za-z0-9]{16,}', r'(?i)MODEL_API_KEY\s*[=:]'):
        require(not re.search(pattern, text), f'public evidence contains disallowed sensitive text: {pattern}')


def validate_data(data):
    require(isinstance(data, dict) and set(data) == {'schema', *(f'experiment_00{i}' for i in range(5, 10))}, 'evidence top-level schema mismatch')
    require(data['schema'] == SCHEMA, 'evidence schema version mismatch')
    e5, e6, e7, e8, e9 = (data[f'experiment_00{i}'] for i in range(5, 10))
    require(set(e5) == {'run_id', 'hypothesis', 'environment', 'config', 'shapes', 'correctness', 'screening', 'confirmation_status', 'profiling_status', 'observation', 'inference'}, 'Experiment 005 schema mismatch')
    require(e5['confirmation_status'] == 'missing' and e5['profiling_status'] == 'not_validated', 'Experiment 005 missing stage misrepresented')
    require(len(e5['shapes']) == 4 and set(map(lambda x: 'x'.join(map(str, x)), e5['shapes'])) == set(SHAPES), 'Experiment 005 shape set mismatch')
    require(len(e5['correctness']) == 24 and all(set(row) == {'shape', 'schedule', 'seed', 'status'} and row['status'] == 'PASS' and row['shape'] in SHAPES and row['schedule'] in ('row_major', 'grouped_m4', 'hilbert') and row['seed'] in (42, 43) for row in e5['correctness']), 'Experiment 005 correctness mismatch')
    require(len({(row['shape'], row['schedule'], row['seed']) for row in e5['correctness']}) == 24, 'Experiment 005 duplicate correctness check')
    require(len(e5['screening']) == 8 and {(x['shape'], x['opponent']) for x in e5['screening']} == {(s, o) for s in SHAPES for o in ('row_major', 'grouped_m4')}, 'Experiment 005 screening comparison set mismatch')
    for row in e5['screening']:
        require(set(row) == {'shape', 'opponent', 'rounds', 'median_paired_ratio', 'bootstrap_median_ratio_95pct', 'paired_ratios'}, 'screening row schema mismatch')
        ratios = row['paired_ratios']
        require(len(ratios) == row['rounds'] == 8 and all(type(x) in (int, float) and x > 0 for x in ratios), 'screening raw ratio set mismatch')
        require(abs(median(ratios) - row['median_paired_ratio']) < 1e-10, 'screening median arithmetic mismatch')
        boot_rng = random.Random(5005)
        boot = sorted(median(boot_rng.choices(ratios, k=len(ratios))) for _ in range(2000))
        lo, hi = row['bootstrap_median_ratio_95pct']
        require(abs(boot[49] - lo) < 1e-10 and abs(boot[1949] - hi) < 1e-10, 'screening bootstrap interval mismatch')
        require(0 < lo <= hi and lo <= 1, 'screening interval or support status mismatch')
    require(set(e6) == {'audit_id', 'model', 'prompt_version', 'decision', 'evidence_based_justification', 'citations', 'source_evidence_id'}, 'Experiment 006 schema mismatch')
    require(e6['decision'] in ('abandon', 'revise', 'investigate_further') and e6['citations'] and all(set(x) == {'evidence_id', 'observation_key'} and x['evidence_id'] == e6['source_evidence_id'] for x in e6['citations']), 'Experiment 006 decision or citations mismatch')
    require(set(e7) == {'study_id', 'review_id', 'proposals', 'scores', 'assignments', 'means', 'limitations'}, 'Experiment 007 schema mismatch')
    require([x['evaluation_id'] for x in e7['proposals']] == list(PROPOSALS) and set(e7['scores']) == set(PROPOSALS), 'Experiment 007 proposal set mismatch')
    require(len(e7['assignments']) == 6 and {x['evaluation_id'] for x in e7['assignments']} == set(PROPOSALS) and {x['condition'] for x in e7['assignments']} == {'A', 'B'}, 'Experiment 007 assignment mismatch')
    for pid, entry in e7['scores'].items():
        require(set(entry) == set(CRITERIA), f'{pid}: score criteria mismatch')
        for item in entry.values():
            require(set(item) == {'score', 'justification'} and type(item['score']) is int and 1 <= item['score'] <= 5 and isinstance(item['justification'], str) and len(item['justification']) >= 10, f'{pid}: invalid score')
    computed = means(e7['scores'], e7['assignments'])
    require(all(abs(computed[c][k] - e7['means'][c][k]) < 1e-12 for c in ('A', 'B') for k in CRITERIA), 'Experiment 007 means mismatch')
    require(set(e8) == {'proposal_id', 'original_proposal', 'accepted_criticism', 'revision_attempts', 'validator_lessons'}, 'Experiment 008 schema mismatch')
    require(e8['proposal_id'] == 'P01' and e8['original_proposal'] == e7['proposals'][0]['proposal'], 'P01 source proposal mismatch')
    require(e8['accepted_criticism']['status'] == 'accepted' and e8['accepted_criticism']['recommendation'] in ('proceed', 'revise', 'deprioritize'), 'Experiment 008 accepted criticism mismatch')
    require(e8['revision_attempts'] and all(x['status'] == 'rejected' for x in e8['revision_attempts']), 'Experiment 008 revision status mismatch')
    require(set(e9) == {'run_id', 'criticism_attempts', 'p01_decisions', 'paired_evaluation_status', 'compute_savings_status'}, 'Experiment 009 schema mismatch')
    require(len(e9['criticism_attempts']) == 6 and {x['proposal_id'] for x in e9['criticism_attempts']} == set(PROPOSALS), 'Experiment 009 criticism set mismatch')
    require(sum(x['status'] == 'accepted' for x in e9['criticism_attempts']) == 1 and sum(x['status'] == 'rejected' for x in e9['criticism_attempts']) == 5, 'Experiment 009 criticism statuses mismatch')
    require(all(set(x) == {'proposal_id', 'case_id', 'status', 'rejection_kind', 'explanation'} and isinstance(x['explanation'], str) and x['explanation'] and (x['status'] == 'accepted' and x['rejection_kind'] is None if x['proposal_id'] == 'P01' else x['status'] == 'rejected' and x['rejection_kind'] == 'invalid_response') for x in e9['criticism_attempts']), 'Experiment 009 per-proposal criticism status mismatch')
    require(next(x['case_id'] for x in e9['criticism_attempts'] if x['proposal_id'] == 'P01') == e8['accepted_criticism']['case_id'], 'P01 criticism source case mismatch')
    require(set(e9['p01_decisions']) == {'A', 'B'} and all(v['status'] == 'accepted' and v['decision'] in ('proceed', 'revise', 'deprioritize') for v in e9['p01_decisions'].values()), 'Experiment 009 P01 decision status mismatch')
    require(e9['paired_evaluation_status'] == 'incomplete' and e9['compute_savings_status'] == 'not_measured', 'Experiment 009 completion or savings claim mismatch')
    privacy_check(data)


def verify(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'bundle directory missing or linked')
    require({p.name for p in directory.iterdir()} == set(FILES) | {'manifest.json'}, 'public bundle missing or contains unexpected files')
    for name in FILES + ('manifest.json',):
        p = directory / name
        require(p.is_file() and not p.is_symlink(), f'missing or linked {name}')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    require(set(manifest) == {'schema', 'files', 'source_artifacts'} and manifest['schema'] == SCHEMA, 'public manifest schema mismatch')
    require(set(manifest['files']) == set(FILES), 'public manifest file set mismatch')
    require(isinstance(manifest['source_artifacts'], dict) and manifest['source_artifacts'], 'source artifact hashes missing')
    for key, value in manifest['source_artifacts'].items():
        require(isinstance(key, str) and not key.startswith('/') and '..' not in Path(key).parts and re.fullmatch(r'[0-9a-f]{64}', value), 'invalid source artifact identifier or hash')
    for name in FILES:
        require(sha((directory / name).read_bytes()) == manifest['files'][name], f'public file hash mismatch: {name}')
    require((directory / 'README.md').read_text() == README, 'public README mismatch')
    data = json.loads((directory / 'evidence.json').read_bytes())
    validate_data(data)
    require((directory / 'summary.md').read_text() == render_summary(data), 'public summary mismatch')
    return {'schema': SCHEMA, 'files': list(FILES), 'source_artifacts': len(manifest['source_artifacts'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='public_evidence')
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.directory), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
