"""Experiment 009: paired, advisory decisions with sealed manual evaluation."""
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
from muse_research_critic import (STAMP, SECTION as CRITIC_SECTION, _check_text, _references,
                                  critique_prompt, load_case, load_criticism, load_revalidation,
                                  source_package)

SECTION = Path('evaluations/muse-critic-value-study/v1')
VERSION = 'experiment_009_v1'
STUDY_ID = '20261030T142150402688Z'
REVIEW_ID = '20261030T142419308682Z'
IDS = tuple(f'P{i:02d}' for i in range(1, 7))
CRITERIA = ('evidence_fidelity', 'redundancy_avoidance', 'discriminating_tests',
            'discovery_preservation', 'resource_awareness')
FIELDS = ('observed_evidence', 'inferred_explanations', 'evidence_assessment',
          'prior_research_overlap', 'expected_information_gain', 'potential_discovery_value',
          'risk_of_rejecting_useful_hypothesis', 'proposed_next_validation_step',
          'resource_cost_assessment', 'uncertainties', 'decision_justification')
DECISIONS = {'proceed', 'revise', 'deprioritize'}
RUBRIC = {
    'evidence_fidelity': '1 invents or distorts evidence; 3 mostly accurate with material gaps; 5 accurately separates observations, inference, and missing evidence.',
    'redundancy_avoidance': '1 repeats prior work without purpose; 3 recognizes some overlap; 5 identifies tested scope and a justified new question or avoids redundant work.',
    'discriminating_tests': '1 no useful test; 3 a partially controlled test; 5 a feasible comparison that separates plausible mechanisms with a disproof outcome.',
    'discovery_preservation': '1 dismisses or accepts a useful hypothesis without weighing information value; 3 considers upside and rejection risk; 5 preserves informative exploration proportionately.',
    'resource_awareness': '1 ignores effort or claims unmeasured savings; 3 gives rough effort; 5 ties a staged validation to likely cost without claiming measured savings.'
}


def now():
    return datetime.now(timezone.utc).isoformat()


def folder(root, run_id):
    if not isinstance(run_id, str) or not STAMP.fullmatch(run_id):
        raise ValueError('invalid Experiment 009 run ID')
    root = Path(root).resolve()
    path = root / SECTION
    if any(p.is_symlink() for p in (root / 'evaluations', root / 'evaluations/muse-critic-value-study', path)):
        raise ValueError('linked Experiment 009 path')
    return path / run_id


def write(stage, name, value, schema):
    stage.mkdir(parents=True, exist_ok=False)
    (stage / name).write_bytes(canonical(value))
    (stage / 'manifest.json').write_bytes(canonical(manifest(stage, schema=schema)))


def read(stage, name, schema):
    read_manifest(stage, schema=schema)
    return json.loads((stage / name).read_bytes())


def prepare(root, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    sources = {pid: source_package(root, STUDY_ID, REVIEW_ID, pid) for pid in IDS}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    setup = {'schema': VERSION, 'run_id': run_id, 'created_utc': now(),
             'study_id': STUDY_ID, 'review_id': REVIEW_ID, 'sources': sources,
             'rubric': RUBRIC, 'rubric_version': 'rubric_v1', 'seed': seed,
             'model': MODEL, 'request_options': REQUEST_OPTIONS,
             'decision_prompt_version': 'decision_v1',
             'critic_prompt_version': 'experiment_008_v1'}
    write(folder(root, run_id) / 'setup', 'study.json', setup, 'experiment_009_setup_v1')
    return run_id


def load(root, run_id):
    setup = read(folder(root, run_id) / 'setup', 'study.json', 'experiment_009_setup_v1')
    if (setup.get('schema') != VERSION or setup.get('run_id') != run_id
            or setup.get('study_id') != STUDY_ID or setup.get('review_id') != REVIEW_ID
            or setup.get('rubric') != RUBRIC or setup.get('rubric_version') != 'rubric_v1'
            or type(setup.get('seed')) is not int or setup['seed'] < 0
            or setup.get('model') != MODEL or setup.get('request_options') != REQUEST_OPTIONS
            or setup.get('decision_prompt_version') != 'decision_v1'
            or setup.get('critic_prompt_version') != 'experiment_008_v1'
            or set(setup.get('sources', {})) != set(IDS)):
        raise ValueError('Experiment 009 setup mismatch')
    for pid in IDS:
        if setup['sources'][pid] != source_package(root, STUDY_ID, REVIEW_ID, pid):
            raise ValueError(f'{pid} source or upstream evidence changed')
    return setup


def critic_request(root, run_id, pid):
    source = load(root, run_id)['sources'][pid]
    return {'proposal_id': pid, 'prompt_version': 'experiment_008_v1', 'model': MODEL,
            'request_options': REQUEST_OPTIONS, 'prompt': critique_prompt(source)}


def attach_criticism(root, run_id, pid, case_id, revalidation_id=None):
    source = load(root, run_id)['sources'][pid]
    case = load_case(root, case_id)
    if case['source'] != source or case['critique_prompt'] != critique_prompt(source):
        raise ValueError(f'{pid} critic case uses a different source or protocol')
    if revalidation_id:
        record = load_revalidation(root, case_id, revalidation_id)
        if record['validation_status'] != 'valid':
            raise ValueError('critic revalidation is invalid')
        criticism = record['parsed_criticism']
        evidence_path = Path(root) / CRITIC_SECTION / case_id / 'revalidations/v1' / revalidation_id / 'result.json'
        provenance = 'revalidated_rejected_008'
    else:
        criticism = load_criticism(root, case_id)['parsed_response']
        evidence_path = Path(root) / CRITIC_SECTION / case_id / 'criticism/audit.json'
        provenance = 'accepted_008'
    binding = {'schema': 'experiment_009_critic_binding_v1', 'proposal_id': pid,
               'proposal_sha256': source['proposal_sha256'], 'case_id': case_id,
               'revalidation_id': revalidation_id, 'provenance': provenance,
               'critic_prompt_version': 'experiment_008_v1',
               'critic_prompt_sha256': digest(case['critique_prompt'].encode()),
               'critic_artifact_sha256': sha(evidence_path.read_bytes()),
               'criticism_sha256': digest(canonical(criticism)), 'criticism': criticism}
    write(folder(root, run_id) / 'criticism' / pid, 'binding.json', binding, 'experiment_009_critic_binding_v1')
    return binding


def load_criticism_binding(root, run_id, pid, source=None):
    source = source or load(root, run_id)['sources'][pid]
    binding = read(folder(root, run_id) / 'criticism' / pid, 'binding.json', 'experiment_009_critic_binding_v1')
    case_id = binding['case_id']
    case = load_case(root, case_id)
    if case['source'] != source or case['critique_prompt'] != critique_prompt(source):
        raise ValueError(f'{pid} critic provenance mismatch')
    rev = binding['revalidation_id']
    if rev:
        record = load_revalidation(root, case_id, rev)
        if record['validation_status'] != 'valid':
            raise ValueError('invalid revalidated criticism')
        criticism = record['parsed_criticism']
        path = Path(root) / CRITIC_SECTION / case_id / 'revalidations/v1' / rev / 'result.json'
        provenance = 'revalidated_rejected_008'
    else:
        criticism = load_criticism(root, case_id)['parsed_response']
        path = Path(root) / CRITIC_SECTION / case_id / 'criticism/audit.json'
        provenance = 'accepted_008'
    if binding != {'schema': 'experiment_009_critic_binding_v1', 'proposal_id': pid,
                   'proposal_sha256': source['proposal_sha256'], 'case_id': case_id,
                   'revalidation_id': rev, 'provenance': provenance,
                   'critic_prompt_version': 'experiment_008_v1',
                   'critic_prompt_sha256': digest(case['critique_prompt'].encode()),
                   'critic_artifact_sha256': sha(path.read_bytes()),
                   'criticism_sha256': digest(canonical(criticism)), 'criticism': criticism}:
        raise ValueError(f'{pid} criticism binding mismatch')
    return binding


def decision_prompt(source, criticism=None):
    example = {'proposal_id': source['proposal_id'], 'proposal_sha256': source['proposal_sha256'],
               'decision': '<proceed|revise|deprioritize>',
               'evidence_references': {field: [] for field in FIELDS},
               **{field: 'Specific assessment; distinguish observation from inference.' for field in FIELDS}}
    context = {'proposal_id': source['proposal_id'], 'proposal_sha256': source['proposal_sha256'],
               'proposal': source['proposal'], 'validated_evidence': source['validated_evidence'],
               'experiment_005_criteria': source['experiment_005_criteria']}
    common = ('Choose one advisory research decision for this original proposal. Return JSON matching SCHEMA exactly. '
              'Use proceed, revise, or deprioritize without favoring a category. Cite supplied evidence IDs and '
              'observation keys in the evidence_references map for each field that makes historical claims. '
              'Keep observations and inferences separate. Assess overlap, information gain, discovery potential, '
              'false rejection risk, next discriminating step, cost, uncertainties, and reasons. No invented '
              'measurements, compute savings, code, GPU execution, or automatic adoption. A proceed decision '
              'does not authorize GPU execution.\nSCHEMA\n' + json.dumps(example, sort_keys=True) +
              '\nSOURCE\n' + json.dumps(context, sort_keys=True, allow_nan=False))
    if criticism is not None:
        common += '\nINDEPENDENT_CRITICISM\n' + json.dumps(criticism, sort_keys=True, allow_nan=False)
    return common


def request_spec(root, run_id, pid, condition):
    if pid not in IDS or condition not in ('A', 'B'):
        raise ValueError('invalid proposal ID or decision condition')
    setup = load(root, run_id)
    source = setup['sources'][pid]
    criticism = load_criticism_binding(root, run_id, pid, source)['criticism'] if condition == 'B' else None
    return {'schema': 'experiment_009_request_v1', 'run_id': run_id, 'proposal_id': pid,
            'condition': condition, 'proposal_sha256': source['proposal_sha256'],
            'model': MODEL, 'request_options': REQUEST_OPTIONS,
            'prompt_version': 'decision_v1', 'prompt': decision_prompt(source, criticism),
            'source_sha256': digest(canonical(source)),
            'criticism_sha256': digest(canonical(criticism)) if criticism is not None else None}


def parse_decision(raw, source):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 32000:
        raise ValueError('missing or oversized decision response')
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError('decision response is not JSON') from exc
    if (not isinstance(value, dict) or set(value) != set(FIELDS) | {'proposal_id', 'proposal_sha256', 'decision', 'evidence_references'}
            or value['proposal_id'] != source['proposal_id']
            or value['proposal_sha256'] != source['proposal_sha256']
            or value['decision'] not in DECISIONS):
        raise ValueError('decision identity, category, or schema mismatch')
    refs = value['evidence_references']
    if not isinstance(refs, dict) or set(refs) != set(FIELDS):
        raise ValueError('evidence_references must contain every decision field')
    for field in FIELDS:
        citations = refs[field]
        cited_values = _references(citations, source, required=False)
        try:
            _check_text(value[field], source, cited_values, citations, field=field)
        except ValueError as exc:
            raise ValueError(f'{field}: {exc}') from exc
    if not refs['observed_evidence']:
        raise ValueError('observed_evidence requires at least one citation')
    return value


def request_decision(client, root, run_id, pid, condition):
    spec = request_spec(root, run_id, pid, condition)
    target = folder(root, run_id) / 'decisions' / pid / condition
    lock_path = folder(root, run_id) / 'request.lock'
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return 'already_attempted'
        audit = {'schema': 'experiment_009_decision_audit_v1', 'created_utc': now(),
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
                raise ValueError('decision request source changed')
            audit['status'] = 'accepted'
        except Exception as exc:
            audit['status'] = 'rejected'
            audit['rejection_kind'] = 'invalid_response' if called else 'request_error'
            audit['rejection'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        write(target, 'audit.json', audit, 'experiment_009_decision_audit_v1')
        return audit['status']


def load_decision(root, run_id, pid, condition, setup=None):
    setup = setup or load(root, run_id)
    audit = read(folder(root, run_id) / 'decisions' / pid / condition,
                 'audit.json', 'experiment_009_decision_audit_v1')
    spec = request_spec(root, run_id, pid, condition)
    if audit.get('schema') != 'experiment_009_decision_audit_v1' or audit.get('request') != spec:
        raise ValueError('decision request or audit mismatch')
    if audit.get('status') == 'accepted':
        require_complete_text(audit['raw_response'], audit['response_metadata'])
        if (audit.get('response_model') != MODEL or
                audit.get('parsed_response') != parse_decision(audit['raw_response'], setup['sources'][pid])):
            raise ValueError('accepted decision response mismatch')
    elif audit.get('status') != 'rejected' or audit.get('parsed_response') is not None or not audit.get('rejection'):
        raise ValueError('invalid rejected decision audit')
    return audit


def prepare_blind(root, run_id):
    setup = load(root, run_id)
    entries = [(pid, condition, load_decision(root, run_id, pid, condition, setup)['parsed_response'])
               for pid in IDS for condition in ('A', 'B')]
    if any(value is None for _, _, value in entries):
        raise ValueError('all twelve decisions must be accepted before blind review')
    rng = random.Random(setup['seed'])
    # Shuffle within each matched proposal; proposal order remains fixed.
    pairs = [entries[i:i + 2] for i in range(0, 12, 2)]
    for pair in pairs:
        rng.shuffle(pair)
    assignment = []
    cards = []
    for pid, pair in zip(IDS, pairs):
        for number, (_, condition, decision) in enumerate(pair, 1):
            blind_id = f'{pid}-{number}'
            cards.append({'blind_id': blind_id, 'proposal_id': pid,
                          'decision': {key: value for key, value in decision.items()
                                       if key not in ('proposal_id', 'proposal_sha256')}})
            assignment.append({'blind_id': blind_id, 'proposal_id': pid, 'condition': condition,
                               'decision_sha256': digest(canonical(decision))})
    blind = {'schema': 'experiment_009_blind_v1', 'run_id': run_id, 'created_utc': now(),
             'rubric': setup['rubric'], 'cards': cards,
             'instruction': 'Score decision quality, not decision category. Record any wording that may reveal condition.'}
    key = {'schema': 'experiment_009_assignment_v1', 'seed': setup['seed'],
           'blind_sha256': digest(canonical(blind)), 'assignments': assignment}
    write(folder(root, run_id) / 'blind', 'review.json', blind, 'experiment_009_blind_v1')
    write(folder(root, run_id) / 'operator', 'assignment.json', key, 'experiment_009_assignment_v1')
    return blind


def load_blind(root, run_id):
    blind = read(folder(root, run_id) / 'blind', 'review.json', 'experiment_009_blind_v1')
    if (blind.get('schema') != 'experiment_009_blind_v1' or blind.get('run_id') != run_id
            or blind.get('rubric') != RUBRIC or len(blind.get('cards', [])) != 12
            or [c.get('blind_id') for c in blind['cards']] != [f'{pid}-{n}' for pid in IDS for n in (1, 2)]):
        raise ValueError('blind review packet mismatch')
    return blind


def validate_scores(scores, blind):
    ids = {card['blind_id'] for card in blind['cards']}
    if not isinstance(scores, dict) or set(scores) != ids:
        raise ValueError('scores must cover all twelve blind decisions')
    for bid, entry in scores.items():
        if not isinstance(entry, dict) or set(entry) != {'criteria', 'possible_unblinding'}:
            raise ValueError(f'{bid}: criteria and possible_unblinding are required')
        if not isinstance(entry['possible_unblinding'], str):
            raise ValueError(f'{bid}: possible_unblinding must be text; use an empty string if none')
        values = entry['criteria']
        if not isinstance(values, dict) or set(values) != set(CRITERIA):
            raise ValueError(f'{bid}: all five criteria are required')
        for criterion, item in values.items():
            if (not isinstance(item, dict) or set(item) != {'score', 'justification'}
                    or type(item['score']) is not int or not 1 <= item['score'] <= 5
                    or not isinstance(item['justification'], str)
                    or not 10 <= len(item['justification'].strip()) <= 1200):
                raise ValueError(f'{bid}.{criterion}: score 1-5 and written justification required')


def score(root, run_id, scores):
    blind = load_blind(root, run_id)
    validate_scores(scores, blind)
    record = {'schema': 'experiment_009_scores_v1', 'created_utc': now(),
              'blind_sha256': digest(canonical(blind)), 'scores': scores}
    write(folder(root, run_id) / 'scores', 'scores.json', record, 'experiment_009_scores_v1')
    return record


def reveal(root, run_id):
    blind = load_blind(root, run_id)
    saved = read(folder(root, run_id) / 'scores', 'scores.json', 'experiment_009_scores_v1')
    if saved.get('blind_sha256') != digest(canonical(blind)):
        raise ValueError('scores are for another blind packet')
    validate_scores(saved.get('scores'), blind)
    setup = load(root, run_id)
    key = read(folder(root, run_id) / 'operator', 'assignment.json', 'experiment_009_assignment_v1')
    entries = [(pid, condition, load_decision(root, run_id, pid, condition, setup)['parsed_response'])
               for pid in IDS for condition in ('A', 'B')]
    rng = random.Random(setup['seed'])
    pairs = [entries[i:i + 2] for i in range(0, 12, 2)]
    for pair in pairs:
        rng.shuffle(pair)
    expected = []
    for pid, pair in zip(IDS, pairs):
        for n, (_, condition, decision) in enumerate(pair, 1):
            bid = f'{pid}-{n}'
            expected.append({'blind_id': bid, 'proposal_id': pid, 'condition': condition,
                             'decision_sha256': digest(canonical(decision))})
            card = next(c for c in blind['cards'] if c['blind_id'] == bid)
            if card != {'blind_id': bid, 'proposal_id': pid, 'decision':
                        {k: v for k, v in decision.items() if k not in ('proposal_id', 'proposal_sha256')}}:
                raise ValueError('blind card differs from recorded decision')
    if key != {'schema': 'experiment_009_assignment_v1', 'seed': setup['seed'],
               'blind_sha256': digest(canonical(blind)), 'assignments': expected}:
        raise ValueError('sealed assignment mismatch')
    means = {condition: {criterion: sum(saved['scores'][row['blind_id']]['criteria'][criterion]['score']
                                       for row in expected if row['condition'] == condition) / 6
                         for criterion in CRITERIA} for condition in ('A', 'B')}
    result = {'schema': 'experiment_009_reveal_v1', 'created_utc': now(), 'run_id': run_id,
              'scores_sha256': digest(canonical(saved)), 'assignments': expected,
              'mean_subjective_scores': means,
              'paired_differences_B_minus_A': {criterion: [
                  next(saved['scores'][row['blind_id']]['criteria'][criterion]['score'] for row in expected
                       if row['proposal_id'] == pid and row['condition'] == 'B') -
                  next(saved['scores'][row['blind_id']]['criteria'][criterion]['score'] for row in expected
                       if row['proposal_id'] == pid and row['condition'] == 'A') for pid in IDS]
                  for criterion in CRITERIA},
              'interpretation': 'Manual subjective scores; no measured compute savings or automatic recommendation adoption.'}
    write(folder(root, run_id) / 'reveal', 'result.json', result, 'experiment_009_reveal_v1')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Experiment 009 critic value study')
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'critic-request', 'attach-criticism', 'decision-request',
                 'request-decision', 'prepare-blind', 'score', 'reveal', 'validate'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--run-id'); parser.add_argument('--proposal-id')
    parser.add_argument('--condition', choices=['A', 'B']); parser.add_argument('--case-id')
    parser.add_argument('--revalidation-id'); parser.add_argument('--seed', type=int)
    parser.add_argument('--scores-file', type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        if args.seed is None:
            parser.error('--seed is required')
        print(prepare(ROOT, args.seed)); return 0
    if not args.run_id:
        parser.error('--run-id is required')
    if args.critic_request:
        if not args.proposal_id:
            parser.error('--proposal-id is required')
        print(json.dumps(critic_request(ROOT, args.run_id, args.proposal_id), indent=2)); return 0
    if args.attach_criticism:
        if not args.proposal_id or not args.case_id:
            parser.error('--proposal-id and --case-id are required')
        print(json.dumps(attach_criticism(ROOT, args.run_id, args.proposal_id,
                                          args.case_id, args.revalidation_id), indent=2)); return 0
    if args.decision_request:
        if not args.proposal_id or not args.condition:
            parser.error('--proposal-id and --condition are required')
        print(json.dumps(request_spec(ROOT, args.run_id, args.proposal_id, args.condition), indent=2)); return 0
    if args.prepare_blind:
        print(json.dumps(prepare_blind(ROOT, args.run_id), indent=2)); return 0
    if args.score:
        if not args.scores_file:
            parser.error('--scores-file is required')
        print(json.dumps(score(ROOT, args.run_id, json.loads(args.scores_file.read_text())), indent=2)); return 0
    if args.reveal:
        print(json.dumps(reveal(ROOT, args.run_id), indent=2)); return 0
    if args.validate:
        setup = load(ROOT, args.run_id)
        print(json.dumps({'run_id': args.run_id, 'sources': list(setup['sources']),
                          'criticisms': [pid for pid in IDS if (folder(ROOT, args.run_id) / 'criticism' / pid).exists()
                                         and load_criticism_binding(ROOT, args.run_id, pid)],
                          'decisions': [f'{pid}-{c}' for pid in IDS for c in ('A', 'B')
                                        if (folder(ROOT, args.run_id) / 'decisions' / pid / c).exists()
                                        and load_decision(ROOT, args.run_id, pid, c)]}, indent=2)); return 0
    if not args.proposal_id or not args.condition:
        parser.error('--proposal-id and --condition are required')
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / '.env')
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        parser.error('MODEL_API_KEY is required for a live request')
    client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=30.0, max_retries=0)
    result = request_decision(client, ROOT, args.run_id, args.proposal_id, args.condition)
    print(result)
    return 0 if result == 'accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
