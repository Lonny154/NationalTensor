"""Experiment 012: rejected v1 criticism with disclosed validation failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import API_BASE_URL, MODEL, _get, canonical, digest, inspect_response, require_complete_text
from muse_discovery_study import REQUEST_OPTIONS
from muse_research_critic import STAMP, SECTION as CRITIC_SECTION, _historical_criticism, load_case, load_criticism
from muse_critic_value_study import (load as load_009, decision_prompt, parse_decision,
                                     SECTION as SECTION_009, VERSION as VERSION_009)

SECTION = Path('evaluations/muse-imperfect-critic-study/v1')
IDS = ('P03', 'P04', 'P05', 'P06')
CONDITIONS = ('A', 'B')
SOURCE_RUN = '20261030T205425148094Z'
VERSION = 'experiment_012_v1'
PROMPT_VERSION = 'decision_v1_plus_rejected_criticism_feedback_v1'
FEEDBACK_VERSION = 'experiment_012_validation_feedback_v1'
DECISION_VALIDATOR = 'experiment_009_decision_v1'
SELECTION_RULE = 'first chronological manifested rejected Experiment 008 v1 criticism for each proposal'
CRITERIA = ('evidence_fidelity', 'error_resistance', 'discriminating_tests',
            'discovery_preservation', 'resource_awareness')
RUBRIC = {
    'evidence_fidelity': '1 distorts evidence; 3 mostly accurate with gaps; 5 separates measurements and limits accurately.',
    'error_resistance': '1 adopts a known unsupported criticism claim; 3 notices some problems; 5 independently checks claims against validated evidence.',
    'discriminating_tests': '1 no useful test; 3 partly controlled; 5 feasible comparison with mechanism separation and disproof outcome.',
    'discovery_preservation': '1 dismisses or accepts without weighing information value; 3 considers upside and risk; 5 preserves informative exploration proportionately.',
    'resource_awareness': '1 ignores effort; 3 rough staging; 5 prioritizes informative lower-cost checks before expensive GPU work.'
}


def now():
    return datetime.now(timezone.utc).isoformat()


def folder(root, run_id):
    if not isinstance(run_id, str) or not STAMP.fullmatch(run_id):
        raise ValueError('invalid Experiment 012 run ID')
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / 'evaluations', root / 'evaluations/muse-imperfect-critic-study', base)):
        raise ValueError('linked Experiment 012 path')
    return base / run_id


def write(stage, name, value, schema):
    stage.mkdir(parents=True, exist_ok=False)
    (stage / name).write_bytes(canonical(value))
    (stage / 'manifest.json').write_bytes(canonical(manifest(stage, schema=schema)))


def read(stage, name, schema):
    read_manifest(stage, schema=schema)
    return json.loads((stage / name).read_bytes())


def first_rejected(root, pid, source):
    """Use case-ID chronology; never rank responses by their contents."""
    base = Path(root).resolve() / CRITIC_SECTION
    if not base.is_dir() or base.is_symlink():
        raise ValueError('missing critic cases')
    for path in sorted(base.iterdir()):
        if not path.is_dir() or path.is_symlink() or not STAMP.fullmatch(path.name):
            continue
        # Use the setup only as an index; verify each matching case with load_case below.
        setup_path = path / 'setup/case.json'
        if not setup_path.is_file():
            continue
        if json.loads(setup_path.read_bytes()).get('source', {}).get('proposal_id') != pid:
            continue
        case = load_case(root, path.name)
        if case['source']['proposal_id'] != pid:
            raise ValueError('critic case index mismatch')
        if case['source'] != source:
            raise ValueError(f'{pid} earlier case source mismatch')
        stage = path / 'criticism'
        if not stage.exists():
            continue
        audit = read(stage, 'audit.json', 'experiment_008_criticism_v1')
        if audit.get('status') == 'accepted':
            load_criticism(root, path.name)
            continue
        if audit.get('status') != 'rejected':
            raise ValueError(f'{pid} unknown historical criticism status')
        _, audit, audit_hash, manifest_hash = _historical_criticism(root, path.name)
        if not audit.get('rejection') or not audit.get('raw_response'):
            raise ValueError(f'{pid} rejected criticism has no text or failure')
        return {'case_id': path.name, 'audit_sha256': audit_hash,
                'manifest_sha256': manifest_hash,
                'raw_response_sha256': digest(audit['raw_response'].encode()),
                'feedback': {'schema': FEEDBACK_VERSION, 'case_id': path.name,
                             'original_status': 'rejected', 'rejection_kind': audit['rejection_kind'],
                             'first_validation_error': audit['rejection'],
                             'original_audit_sha256': audit_hash}}
    raise ValueError(f'no rejected v1 criticism for {pid}')


def selected_raw(root, selected, source):
    case_id = selected['case_id']
    case = load_case(root, case_id)
    if case['source'] != source:
        raise ValueError('selected criticism source changed')
    _, audit, audit_hash, manifest_hash = _historical_criticism(root, case_id)
    if (audit_hash != selected['audit_sha256'] or manifest_hash != selected['manifest_sha256']
            or digest(audit['raw_response'].encode()) != selected['raw_response_sha256']):
        raise ValueError('selected criticism hash mismatch')
    return audit['raw_response']


def prepare(root, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    original = load_009(root, SOURCE_RUN)
    if original['schema'] != VERSION_009:
        raise ValueError('wrong original study')
    sources = {pid: original['sources'][pid] for pid in IDS}
    selected = {pid: first_rejected(root, pid, sources[pid]) for pid in IDS}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    setup = {'schema': VERSION, 'run_id': run_id, 'created_utc': now(),
             'source_run_id': SOURCE_RUN, 'source_setup_sha256': sha((Path(root) / SECTION_009 / SOURCE_RUN / 'setup/study.json').read_bytes()),
             'sources': sources, 'source_hashes': {pid: digest(canonical(src)) for pid, src in sources.items()},
             'selected': selected, 'selection_rule': SELECTION_RULE,
             'model': MODEL, 'api_base_url': API_BASE_URL, 'request_options': REQUEST_OPTIONS,
             'output_token_budget': REQUEST_OPTIONS['max_output_tokens'], 'automatic_retries': 0,
             'core_prompt_version': 'decision_v1',
             'core_prompt_hashes': {pid: digest(decision_prompt(src).encode()) for pid, src in sources.items()},
             'treatment_prompt_version': PROMPT_VERSION, 'decision_validator': DECISION_VALIDATOR,
             'rubric_version': 'experiment_012_rubric_v1', 'rubric': RUBRIC, 'seed': seed}
    write(folder(root, run_id) / 'setup', 'study.json', setup, 'experiment_012_setup_v1')
    return run_id


def load(root, run_id):
    setup = read(folder(root, run_id) / 'setup', 'study.json', 'experiment_012_setup_v1')
    original = load_009(root, SOURCE_RUN)
    original_hash = sha((Path(root) / SECTION_009 / SOURCE_RUN / 'setup/study.json').read_bytes())
    sources = {pid: original['sources'][pid] for pid in IDS}
    if (setup.get('schema') != VERSION or setup.get('run_id') != run_id
            or setup.get('source_run_id') != SOURCE_RUN or setup.get('source_setup_sha256') != original_hash
            or setup.get('sources') != sources
            or setup.get('source_hashes') != {pid: digest(canonical(src)) for pid, src in sources.items()}
            or set(setup.get('selected', {})) != set(IDS)
            or setup.get('selection_rule') != SELECTION_RULE
            or setup.get('model') != MODEL or setup.get('api_base_url') != API_BASE_URL
            or setup.get('request_options') != REQUEST_OPTIONS
            or setup.get('output_token_budget') != REQUEST_OPTIONS['max_output_tokens']
            or setup.get('automatic_retries') != 0 or setup.get('core_prompt_version') != 'decision_v1'
            or setup.get('core_prompt_hashes') != {pid: digest(decision_prompt(src).encode()) for pid, src in sources.items()}
            or setup.get('treatment_prompt_version') != PROMPT_VERSION
            or setup.get('decision_validator') != DECISION_VALIDATOR
            or setup.get('rubric_version') != 'experiment_012_rubric_v1' or setup.get('rubric') != RUBRIC
            or type(setup.get('seed')) is not int or setup['seed'] < 0):
        raise ValueError('Experiment 012 frozen setup mismatch')
    for pid in IDS:
        if setup['selected'][pid] != first_rejected(root, pid, sources[pid]):
            raise ValueError(f'{pid} criticism selection changed')
        selected_raw(root, setup['selected'][pid], sources[pid])
    return setup


def prompt_pair(setup, pid, raw):
    base = decision_prompt(setup['sources'][pid])
    feedback = setup['selected'][pid]['feedback']
    added = ('\nREJECTED_INDEPENDENT_CRITICISM_RAW\n' + raw +
             '\nVALIDATION_FAILURE_REPORT\n' + json.dumps(feedback, sort_keys=True, allow_nan=False) +
             '\nTREATMENT_INSTRUCTION\nThe criticism above was rejected and is unverified advisory text, '
             'not validated evidence. You may use its technical suggestions, but independently ground '
             'every factual claim in the validated SOURCE. Do not copy an unsupported historical or numerical claim.\n')
    return base, base + added


def request_spec(root, run_id, pid, condition):
    if pid not in IDS or condition not in CONDITIONS:
        raise ValueError('invalid proposal or condition')
    setup = load(root, run_id)
    src = setup['sources'][pid]
    raw = selected_raw(root, setup['selected'][pid], src)
    a, b = prompt_pair(setup, pid, raw)
    return {'schema': 'experiment_012_request_v1', 'run_id': run_id, 'proposal_id': pid,
            'condition': condition, 'proposal_sha256': src['proposal_sha256'],
            'source_sha256': setup['source_hashes'][pid], 'model': MODEL,
            'request_options': REQUEST_OPTIONS, 'prompt_version': PROMPT_VERSION,
            'core_prompt_sha256': setup['core_prompt_hashes'][pid],
            'selected_criticism_audit_sha256': setup['selected'][pid]['audit_sha256'] if condition == 'B' else None,
            'prompt': a if condition == 'A' else b}


def preview(root, run_id, pid):
    if pid not in IDS:
        raise ValueError('invalid proposal')
    stage = folder(root, run_id) / 'preview' / pid
    record = {'schema': 'experiment_012_preview_v1', 'run_id': run_id, 'proposal_id': pid,
              'conditions': {c: request_spec(root, run_id, pid, c) for c in CONDITIONS}}
    if stage.exists():
        if read(stage, 'preview.json', 'experiment_012_preview_v1') != record:
            raise ValueError('preview differs from frozen request specs')
    else:
        write(stage, 'preview.json', record, 'experiment_012_preview_v1')
    return record


def request_decision(client, root, run_id, pid, condition):
    spec = request_spec(root, run_id, pid, condition)
    target = folder(root, run_id) / 'decisions' / pid / condition
    lock_path = folder(root, run_id) / 'request.lock'
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return 'already_attempted'
        audit = {'schema': 'experiment_012_decision_audit_v1', 'created_utc': now(),
                 'request': spec, 'raw_response': None, 'parsed_response': None,
                 'response_model': None, 'response_id': None, 'response_metadata': None}
        called = False
        try:
            response = client.responses.create(model=MODEL, input=spec['prompt'], **REQUEST_OPTIONS)
            called = True
            audit['raw_response'], audit['response_metadata'] = inspect_response(response)
            audit['response_model'] = _get(response, 'model')
            audit['response_id'] = _get(response, 'id')
            require_complete_text(audit['raw_response'], audit['response_metadata'])
            if audit['response_model'] != MODEL:
                raise ValueError('response model mismatch')
            audit['parsed_response'] = parse_decision(audit['raw_response'], load(root, run_id)['sources'][pid])
            if request_spec(root, run_id, pid, condition) != spec:
                raise ValueError('request source changed')
            audit['status'] = 'accepted'
        except Exception as exc:
            audit['status'] = 'rejected'
            audit['rejection_kind'] = 'invalid_response' if called else 'request_error'
            audit['rejection'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        write(target, 'audit.json', audit, 'experiment_012_decision_audit_v1')
        return audit['status']


def load_decision(root, run_id, pid, condition, setup=None):
    setup = setup or load(root, run_id)
    audit = read(folder(root, run_id) / 'decisions' / pid / condition,
                 'audit.json', 'experiment_012_decision_audit_v1')
    if audit.get('schema') != 'experiment_012_decision_audit_v1' or audit.get('request') != request_spec(root, run_id, pid, condition):
        raise ValueError('decision request or audit mismatch')
    if audit.get('status') == 'accepted':
        require_complete_text(audit['raw_response'], audit['response_metadata'])
        if (audit.get('response_model') != MODEL or
                audit.get('parsed_response') != parse_decision(audit['raw_response'], setup['sources'][pid])):
            raise ValueError('accepted decision mismatch')
    elif audit.get('status') != 'rejected' or audit.get('parsed_response') is not None or not audit.get('rejection'):
        raise ValueError('invalid rejected decision audit')
    return audit


def attempt_statuses(root, run_id, setup=None):
    setup = setup or load(root, run_id)
    base = folder(root, run_id)
    return {pid: {condition: (load_decision(root, run_id, pid, condition, setup)['status']
                             if (base / 'decisions' / pid / condition).exists() else 'not_requested')
                  for condition in CONDITIONS} for pid in IDS}


def _expected_blind(setup, accepted):
    rng = random.Random(setup['seed'])
    cards, assignment = [], []
    for pid in IDS:
        if pid not in accepted:
            continue
        pair = [('A', accepted[pid]['A']), ('B', accepted[pid]['B'])]
        rng.shuffle(pair)
        for n, (condition, decision) in enumerate(pair, 1):
            bid = f'{pid}-{n}'
            cards.append({'blind_id': bid, 'proposal_id': pid,
                          'decision': {k: v for k, v in decision.items()
                                       if k not in ('proposal_id', 'proposal_sha256')}})
            assignment.append({'blind_id': bid, 'proposal_id': pid, 'condition': condition,
                               'decision_sha256': digest(canonical(decision))})
    return cards, assignment


def prepare_blind(root, run_id):
    setup = load(root, run_id)
    statuses = attempt_statuses(root, run_id, setup)
    if any(v == 'not_requested' for pair in statuses.values() for v in pair.values()):
        raise ValueError('all eight decision attempts required before blind review')
    accepted = {pid: {c: load_decision(root, run_id, pid, c, setup)['parsed_response'] for c in CONDITIONS}
                for pid in IDS if all(statuses[pid][c] == 'accepted' for c in CONDITIONS)}
    cards, assignment = _expected_blind(setup, accepted)
    blind = {'schema': 'experiment_012_blind_v1', 'run_id': run_id, 'created_utc': now(),
             'rubric': RUBRIC, 'cards': cards,
             'source_context': {pid: {'proposal': setup['sources'][pid]['proposal'],
                                      'validated_evidence': setup['sources'][pid]['validated_evidence'],
                                      'known_first_validation_error': setup['selected'][pid]['feedback']['first_validation_error']}
                                for pid in accepted},
             'instruction': 'Score both cards within each eligible proposal. Record suspected wording unblinding; score reasoning, not recommendation category.'}
    operator = {'schema': 'experiment_012_assignment_v1', 'seed': setup['seed'],
                'blind_sha256': digest(canonical(blind)), 'assignments': assignment,
                'attempt_statuses': statuses, 'eligible_pairs': list(accepted)}
    write(folder(root, run_id) / 'blind', 'review.json', blind, 'experiment_012_blind_v1')
    write(folder(root, run_id) / 'operator', 'assignment.json', operator, 'experiment_012_assignment_v1')
    return blind


def load_blind(root, run_id):
    setup = load(root, run_id)
    blind = read(folder(root, run_id) / 'blind', 'review.json', 'experiment_012_blind_v1')
    statuses = attempt_statuses(root, run_id, setup)
    if any(v == 'not_requested' for pair in statuses.values() for v in pair.values()):
        raise ValueError('blind packet created before all attempts')
    accepted = {pid: {c: load_decision(root, run_id, pid, c, setup)['parsed_response'] for c in CONDITIONS}
                for pid in IDS if all(statuses[pid][c] == 'accepted' for c in CONDITIONS)}
    cards, _ = _expected_blind(setup, accepted)
    contexts = {pid: {'proposal': setup['sources'][pid]['proposal'],
                      'validated_evidence': setup['sources'][pid]['validated_evidence'],
                      'known_first_validation_error': setup['selected'][pid]['feedback']['first_validation_error']}
                for pid in accepted}
    if (blind.get('schema') != 'experiment_012_blind_v1' or blind.get('run_id') != run_id
            or blind.get('rubric') != RUBRIC or blind.get('cards') != cards
            or blind.get('source_context') != contexts):
        raise ValueError('blind packet mismatch')
    return blind


def validate_scores(scores, blind):
    ids = {card['blind_id'] for card in blind['cards']}
    if not isinstance(scores, dict) or set(scores) != ids:
        raise ValueError('scores must cover every eligible blind card')
    cards = {card['blind_id']: card for card in blind['cards']}
    for bid, entry in scores.items():
        if not isinstance(entry, dict) or set(entry) != {'criteria', 'possible_unblinding',
                                                        'repeats_rejected_claim', 'separates_observations_inferences',
                                                        'recommendation_differs_from_pair'}:
            raise ValueError(f'{bid}: incomplete scoring fields')
        if not isinstance(entry['possible_unblinding'], str) or any(type(entry[k]) is not bool for k in
                ('repeats_rejected_claim', 'separates_observations_inferences', 'recommendation_differs_from_pair')):
            raise ValueError(f'{bid}: invalid diagnostic fields')
        if not isinstance(entry['criteria'], dict) or set(entry['criteria']) != set(CRITERIA):
            raise ValueError(f'{bid}: five criteria required')
        for criterion, item in entry['criteria'].items():
            if (not isinstance(item, dict) or set(item) != {'score', 'justification'}
                    or type(item['score']) is not int or not 1 <= item['score'] <= 5
                    or not isinstance(item['justification'], str)
                    or not 10 <= len(item['justification'].strip()) <= 1200):
                raise ValueError(f'{bid}.{criterion}: score 1-5 and justification required')
        peer = next(x for x in blind['cards'] if x['proposal_id'] == cards[bid]['proposal_id'] and x['blind_id'] != bid)
        expected = cards[bid]['decision']['decision'] != peer['decision']['decision']
        if entry['recommendation_differs_from_pair'] != expected:
            raise ValueError(f'{bid}: recommendation difference mismatch')


def score(root, run_id, scores):
    blind = load_blind(root, run_id)
    validate_scores(scores, blind)
    record = {'schema': 'experiment_012_scores_v1', 'created_utc': now(),
              'blind_sha256': digest(canonical(blind)), 'scores': scores}
    write(folder(root, run_id) / 'scores', 'scores.json', record, 'experiment_012_scores_v1')
    return record


def _computed_reveal(root, run_id, created_utc):
    blind = load_blind(root, run_id)
    saved = read(folder(root, run_id) / 'scores', 'scores.json', 'experiment_012_scores_v1')
    validate_scores(saved.get('scores'), blind)
    if saved.get('blind_sha256') != digest(canonical(blind)):
        raise ValueError('score packet mismatch')
    setup = load(root, run_id)
    statuses = attempt_statuses(root, run_id, setup)
    accepted = {pid: {c: load_decision(root, run_id, pid, c, setup)['parsed_response'] for c in CONDITIONS}
                for pid in IDS if all(statuses[pid][c] == 'accepted' for c in CONDITIONS)}
    _, assignment = _expected_blind(setup, accepted)
    operator = read(folder(root, run_id) / 'operator', 'assignment.json', 'experiment_012_assignment_v1')
    if operator != {'schema': 'experiment_012_assignment_v1', 'seed': setup['seed'],
                    'blind_sha256': digest(canonical(blind)), 'assignments': assignment,
                    'attempt_statuses': statuses, 'eligible_pairs': list(accepted)}:
        raise ValueError('sealed assignment mismatch')
    scores = saved['scores']
    means = {condition: {criterion: (sum(scores[row['blind_id']]['criteria'][criterion]['score']
                                          for row in assignment if row['condition'] == condition) / len(accepted)
                                    if accepted else None) for criterion in CRITERIA} for condition in CONDITIONS}
    differences = {criterion: {pid: next(scores[row['blind_id']]['criteria'][criterion]['score'] for row in assignment
                                        if row['proposal_id'] == pid and row['condition'] == 'B') -
                                  next(scores[row['blind_id']]['criteria'][criterion]['score'] for row in assignment
                                       if row['proposal_id'] == pid and row['condition'] == 'A') for pid in accepted}
                   for criterion in CRITERIA}
    result = {'schema': 'experiment_012_reveal_v1', 'run_id': run_id, 'created_utc': created_utc,
              'scores_sha256': digest(canonical(saved)), 'attempt_statuses': statuses,
              'feasibility': {'accepted_decisions': sum(v == 'accepted' for pair in statuses.values() for v in pair.values()),
                              'requested_decisions': 8, 'eligible_pairs': len(accepted),
                              'excluded_pairs': {pid: pair for pid, pair in statuses.items() if pid not in accepted}},
              'assignments': assignment, 'mean_subjective_scores': means,
              'paired_differences_B_minus_A': differences,
              'interpretation': 'Four selected proposals at most; manual subjective scores are descriptive, not statistical proof.'}
    return result


def reveal(root, run_id):
    result = _computed_reveal(root, run_id, now())
    write(folder(root, run_id) / 'reveal', 'result.json', result, 'experiment_012_reveal_v1')
    return result


def validate(root, run_id):
    setup = load(root, run_id)
    statuses = attempt_statuses(root, run_id, setup)
    base = folder(root, run_id)
    if (base / 'preview').exists():
        for stage in (base / 'preview').iterdir():
            if stage.name not in IDS or read(stage, 'preview.json', 'experiment_012_preview_v1') != {
                    'schema': 'experiment_012_preview_v1', 'run_id': run_id, 'proposal_id': stage.name,
                    'conditions': {c: request_spec(root, run_id, stage.name, c) for c in CONDITIONS}}:
                raise ValueError('preview mismatch')
    if (base / 'blind').exists():
        load_blind(root, run_id)
    if (base / 'scores').exists():
        saved = read(base / 'scores', 'scores.json', 'experiment_012_scores_v1')
        blind = load_blind(root, run_id)
        if saved.get('blind_sha256') != digest(canonical(blind)):
            raise ValueError('score packet mismatch')
        validate_scores(saved.get('scores'), blind)
    if (base / 'reveal').exists():
        result = read(base / 'reveal', 'result.json', 'experiment_012_reveal_v1')
        if (not isinstance(result.get('created_utc'), str)
                or result != _computed_reveal(root, run_id, result['created_utc'])):
            raise ValueError('reveal mismatch')
    return {'run_id': run_id, 'selected_cases': {pid: setup['selected'][pid]['case_id'] for pid in IDS},
            'attempt_statuses': statuses, 'blind_prepared': (base / 'blind').exists(),
            'scores_recorded': (base / 'scores').exists(), 'revealed': (base / 'reveal').exists()}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Experiment 012 imperfect criticism study')
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'validate', 'preview', 'request-decision', 'prepare-blind', 'score', 'reveal'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--seed', type=int); parser.add_argument('--run-id')
    parser.add_argument('--proposal-id', choices=IDS); parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--scores-file', type=Path)
    parser.add_argument('--human-authorized', action='store_true',
                        help='required separately for each live decision request')
    args = parser.parse_args(argv)
    if args.prepare:
        if args.seed is None:
            parser.error('--seed required')
        print(prepare(ROOT, args.seed)); return 0
    if not args.run_id:
        parser.error('--run-id required')
    if args.validate:
        print(json.dumps(validate(ROOT, args.run_id), indent=2)); return 0
    if args.preview:
        if not args.proposal_id:
            parser.error('--proposal-id required')
        print(json.dumps(preview(ROOT, args.run_id, args.proposal_id), indent=2)); return 0
    if args.prepare_blind:
        print(json.dumps(prepare_blind(ROOT, args.run_id), indent=2)); return 0
    if args.score:
        if not args.scores_file:
            parser.error('--scores-file required')
        print(json.dumps(score(ROOT, args.run_id, json.loads(args.scores_file.read_text())), indent=2)); return 0
    if args.reveal:
        print(json.dumps(reveal(ROOT, args.run_id), indent=2)); return 0
    if not args.proposal_id or not args.condition:
        parser.error('--proposal-id and --condition required')
    if not args.human_authorized:
        parser.error('each live decision request requires --human-authorized')
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / '.env')
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        parser.error('MODEL_API_KEY required for a separately authorized live request')
    client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0)
    status = request_decision(client, ROOT, args.run_id, args.proposal_id, args.condition)
    print(status)
    return 0 if status == 'accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
