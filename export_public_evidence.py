"""Export a fixed, allowlisted public view of validated Experiments 005–009."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dashboard_data import ROOT
from experiment_005_evidence import latest as latest_005, sha
from muse_research_reflection import validate_audit
from muse_discovery_study import _read_artifact, SECTION as STUDY_SECTION
from muse_research_critic import load_case, load_criticism, validate_case, SECTION as CRITIC_SECTION
from muse_critic_value_study import load as load_009, load_criticism_binding, load_decision, SECTION as PILOT_SECTION
from verify_public_evidence import README, SCHEMA, canonical, render_summary, validate_data, verify

RUN_005 = '20261030T013245142673Z'
SCREEN_005 = '20261030T013442437146Z'
AUDIT_006 = '20261030T141047613123Z'
STUDY_007 = '20261030T142150402688Z'
REVIEW_007 = '20261030T142419308682Z'
RUN_009 = '20261030T205425148094Z'
CRITIC_CASES = {'P01': '20261030T152949236420Z', 'P02': '20261030T210928808538Z',
                'P03': '20261030T211309477752Z', 'P04': '20261030T211546504949Z',
                'P05': '20261030T211855242529Z', 'P06': '20261030T212025073515Z'}
REVISION_CASES = ('20261030T145230747865Z', '20261030T152949236420Z')
PROPOSAL_FIELDS = ('category', 'name', 'core_mechanism', 'why_it_might_work',
                   'what_makes_it_unconventional', 'necessary_assumptions', 'biggest_obstacle',
                   'disproof_observation', 'smallest_proof_of_concept', 'numerical_equivalence',
                   'evidence_references', 'testable_prediction')


def selected(record, fields, label):
    if not isinstance(record, dict) or any(name not in record for name in fields):
        raise ValueError(f'{label}: missing required allowlisted field')
    return {name: record[name] for name in fields}


def source_hashes(root, paths):
    result = {}
    for relative in paths:
        p = Path(root) / relative
        if p.is_symlink() or not p.is_file():
            raise ValueError(f'missing or linked source artifact: {relative}')
        result[relative] = sha(p.read_bytes())
    return dict(sorted(result.items()))


def rejection_reason(audit, label):
    if audit.get('status') != 'rejected' or audit.get('rejection_kind') != 'invalid_response':
        raise ValueError(f'{label}: expected rejected invalid response')
    reason = audit.get('rejection', '')
    if 'unsupported numerical measurement' in reason:
        return 'Unsupported numerical measurement under the recorded citation and numeric validator.'
    if 'historical experiment identifier lacks citation' in reason:
        return 'Historical experiment identifier lacked a citation in the required scope.'
    if 'invalid evidence reference' in reason:
        return 'The response contained an evidence reference outside the validated source scope.'
    raise ValueError(f'{label}: unreviewed rejection category; update export allowlist explicitly')


def collect(root=ROOT):
    """Validate first, project fixed fields second; no live requests or GPU runners."""
    root = Path(root).resolve()
    evidence = latest_005(root)
    if not evidence or evidence['run_id'] != RUN_005 or not evidence['screening'] or evidence['confirmation'] is not None:
        raise ValueError('required validated Experiment 005 verification/screening missing or confirmation status changed')
    verify005, screen005 = evidence['verify'], evidence['screening']
    if screen005['decision'] != 'screening_unconfirmed':
        raise ValueError('Experiment 005 screening status changed')
    reflection = validate_audit(root, AUDIT_006)
    if (reflection['evidence']['observations'].get('confirmation_status') != 'missing'
            or reflection['evidence']['observations'].get('profiling_status') != 'not validated by Experiment 005 evidence pipeline'):
        raise ValueError('Experiment 005 confirmation or profiling provenance changed')
    setup009 = load_009(root, RUN_009)  # Revalidates all six Experiment 007 sources and upstream evidence.
    review_dir = root / STUDY_SECTION / STUDY_007 / 'reviews' / REVIEW_007
    reveal = _read_artifact(review_dir / 'reveal', 'result.json', 'experiment_007_reveal_v1')
    scores = _read_artifact(review_dir / 'scores', 'scores.json', 'experiment_007_scores_v1')
    if reveal['scores'] != scores['scores'] or set(setup009['sources']) != {f'P{i:02d}' for i in range(1, 7)}:
        raise ValueError('Experiment 007 scores or source identity mismatch')

    hypothesis = reflection['evidence']['observations']['original_hypothesis']
    e5 = {
        'run_id': RUN_005,
        'hypothesis': selected(hypothesis, ('name', 'core_mechanism', 'why_it_might_work', 'necessary_assumptions', 'disproof_observation', 'evidence_references'), '005 hypothesis'),
        'environment': selected(verify005['environment'], (), '005 environment'),
        'config': selected(verify005['config'], ('block_m', 'block_n', 'block_k', 'num_warps', 'num_stages'), '005 config'),
        'shapes': verify005['shapes'],
        'correctness': [selected(row, ('shape', 'schedule', 'seed', 'status'), '005 correctness') for row in verify005['correctness']],
        'screening': [], 'confirmation_status': 'missing', 'profiling_status': 'not_validated',
        'observation': 'Correctness passed in the recorded scope; eight paired CUDA-event comparisons are screening only.',
        'inference': 'A cache-locality explanation is unconfirmed without independent timing confirmation and validated profiling.'}
    gpu = verify005['environment']
    e5['environment'] = {'gpu': gpu['gpu']['name'], 'python': gpu['python'],
                         'torch': gpu['torch'], 'cuda': gpu['torch_cuda'], 'triton': gpu['triton']}
    for item in screen005['comparisons']:
        timing = item['timing']
        e5['screening'].append({'shape': item['shape'], 'opponent': item['opponent'],
            'rounds': len(timing['rounds']),
            'paired_ratios': [row['baseline_over_candidate'] for row in timing['rounds']],
            'median_paired_ratio': item['summary']['median_paired_ratio'],
            'bootstrap_median_ratio_95pct': item['summary']['bootstrap_median_ratio_95pct']})
    e5['screening'].sort(key=lambda x: (x['shape'], x['opponent']))

    parsed006 = reflection['parsed_reflection']
    e6 = {'audit_id': AUDIT_006, 'model': reflection['model'],
          'prompt_version': reflection['prompt_version'], 'decision': parsed006['decision'],
          'evidence_based_justification': parsed006['mechanism_assessment']['text'],
          'citations': parsed006['mechanism_assessment']['citations'],
          'source_evidence_id': reflection['evidence']['evidence_id']}
    e7 = {'study_id': STUDY_007, 'review_id': REVIEW_007,
          'proposals': [{'evaluation_id': row['evaluation_id'],
                         'proposal': selected(row['proposal'], PROPOSAL_FIELDS, '007 proposal')}
                        for row in reveal['proposals']],
          'scores': scores['scores'], 'assignments': [selected(row,
              ('evaluation_id', 'condition', 'proposal_index', 'proposal_sha256'), '007 assignment')
              for row in reveal['assignments']],
          'means': reveal['mean_subjective_scores_by_condition'],
          'limitations': 'Six proposals and manual 1–5 scores are subjective; wording may reveal provenance. No statistical superiority is established.'}

    accepted_case = CRITIC_CASES['P01']
    criticism = load_criticism(root, accepted_case)
    parsed008 = criticism['parsed_response']
    if criticism['status'] != 'accepted' or parsed008['proposal_id'] != 'P01':
        raise ValueError('required accepted P01 criticism unavailable')
    revisions = []
    for case_id in REVISION_CASES:
        state = validate_case(root, case_id)
        if state['revision'] != 'rejected':
            raise ValueError(f'{case_id}: historical revision acceptance status changed')
        path = root / CRITIC_SECTION / case_id / 'revision/audit.json'
        audit = json.loads(path.read_bytes())
        revisions.append({'case_id': case_id, 'status': 'rejected',
                          'rejection_category': rejection_reason(audit, f'{case_id} revision')})
    e8 = {'proposal_id': 'P01', 'original_proposal': e7['proposals'][0]['proposal'],
          'accepted_criticism': {'case_id': accepted_case, 'status': 'accepted',
              'model': criticism['model'], 'recommendation': parsed008['recommendation'],
              'reasoning': parsed008['recommendation_explanation'],
              'observed_evidence': parsed008['observed_evidence'],
              'inferred_explanations': parsed008['inferred_explanations'],
              'evidence_citations': parsed008['evidence_citations']},
          'revision_attempts': revisions,
          'validator_lessons': 'Rejected responses remain rejected. Numeric claims and historical references must meet the validator citation rules; scientific judgments require human review.'}

    attempts = []
    for pid, case_id in CRITIC_CASES.items():
        case = load_case(root, case_id)
        if case['source'] != setup009['sources'][pid]:
            raise ValueError(f'{pid}: criticism source differs from Experiment 009')
        state = validate_case(root, case_id)
        if pid == 'P01':
            load_criticism_binding(root, RUN_009, pid, setup009['sources'][pid])
            if state['criticism'] != 'accepted':
                raise ValueError('P01 criticism binding is not accepted')
            attempts.append({'proposal_id': pid, 'case_id': case_id, 'status': 'accepted',
                             'rejection_kind': None, 'explanation': 'Validated independent criticism bound to the pilot.'})
        else:
            if state['criticism'] != 'rejected':
                raise ValueError(f'{pid}: expected rejected criticism attempt')
            audit = json.loads((root / CRITIC_SECTION / case_id / 'criticism/audit.json').read_bytes())
            attempts.append({'proposal_id': pid, 'case_id': case_id, 'status': 'rejected',
                             'rejection_kind': audit['rejection_kind'],
                             'explanation': rejection_reason(audit, pid)})
    decisions = {}
    for condition in ('A', 'B'):
        audit = load_decision(root, RUN_009, 'P01', condition, setup009)
        if audit['status'] != 'accepted':
            raise ValueError(f'P01 {condition}: required accepted decision missing')
        decisions[condition] = {'status': 'accepted', 'decision': audit['parsed_response']['decision'],
                                'proposal_sha256': audit['parsed_response']['proposal_sha256']}
    if (root / PILOT_SECTION / RUN_009 / 'scores').exists() or (root / PILOT_SECTION / RUN_009 / 'reveal').exists():
        raise ValueError('Experiment 009 scoring state changed; update public export protocol explicitly')
    e9 = {'run_id': RUN_009, 'criticism_attempts': attempts, 'p01_decisions': decisions,
          'paired_evaluation_status': 'incomplete', 'compute_savings_status': 'not_measured'}
    data = {'schema': SCHEMA, 'experiment_005': e5, 'experiment_006': e6,
            'experiment_007': e7, 'experiment_008': e8, 'experiment_009': e9}
    validate_data(data)
    paths = [
        f'evaluations/experiment-005-hilbert-tile-traversal/{RUN_005}/verify/result.json',
        f'evaluations/experiment-005-hilbert-tile-traversal/{RUN_005}/verify/manifest.json',
        f'evaluations/experiment-005-hilbert-tile-traversal/{RUN_005}/screening/{SCREEN_005}/result.json',
        f'evaluations/experiment-005-hilbert-tile-traversal/{RUN_005}/screening/{SCREEN_005}/manifest.json',
        'evaluations/muse-wild-research-lab/v1/accepted.json',
        f'evaluations/muse-research-reflection/v1/{AUDIT_006}/audit.json',
        f'evaluations/muse-discovery-study/v1/{STUDY_007}/reviews/{REVIEW_007}/reveal/result.json',
        f'evaluations/muse-discovery-study/v1/{STUDY_007}/reviews/{REVIEW_007}/scores/scores.json',
        f'evaluations/muse-critic-value-study/v1/{RUN_009}/setup/study.json',
        f'evaluations/muse-critic-value-study/v1/{RUN_009}/criticism/P01/binding.json',
    ]
    paths += [f'evaluations/muse-research-critic/v1/{case_id}/criticism/audit.json'
              for case_id in CRITIC_CASES.values()]
    paths += [f'evaluations/muse-research-critic/v1/{case_id}/revision/audit.json'
              for case_id in REVISION_CASES]
    paths += [f'evaluations/muse-critic-value-study/v1/{RUN_009}/decisions/P01/{c}/audit.json'
              for c in ('A', 'B')]
    return data, source_hashes(root, paths)


def export(root=ROOT, output='public_evidence'):
    data, source_artifacts = collect(root)
    output = Path(output)
    payloads = {'evidence.json': canonical(data), 'summary.md': render_summary(data).encode(),
                'README.md': README.encode()}
    from verify_public_evidence import sha as public_sha
    public_manifest = {'schema': SCHEMA, 'files': {name: public_sha(raw) for name, raw in payloads.items()},
                       'source_artifacts': source_artifacts}
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or any(p.is_symlink() for p in output.iterdir()):
        raise ValueError('linked public bundle output')
    if {p.name for p in output.iterdir()} - {'manifest.json', *payloads}:
        raise ValueError('public bundle output contains unexpected files')
    for name, raw in payloads.items():
        (output / name).write_bytes(raw)
    (output / 'manifest.json').write_bytes(canonical(public_manifest))
    verify(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='public_evidence')
    args = parser.parse_args(argv)
    print(export(ROOT, args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
