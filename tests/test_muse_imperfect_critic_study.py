"""CPU-only tests for Experiment 012's frozen selection and paired review."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_imperfect_critic_study as study
from muse_critic_value_study import FIELDS
from muse_debrief import canonical, digest


def source(pid):
    return {'proposal_id': pid, 'proposal_sha256': pid.lower().ljust(64, 'a'),
            'proposal': {'name': f'Idea {pid}'},
            'validated_evidence': {'003': {'evidence_id': 'e:003',
                                           'observations': {'result': 'PASS'}},
                                   '005': {'evidence_id': 'e:005',
                                           'observations': {'confirmation_status': 'missing'}}},
            'experiment_005_criteria': {}}


def decision(pid, category='revise'):
    value = {'proposal_id': pid, 'proposal_sha256': source(pid)['proposal_sha256'],
             'decision': category, 'evidence_references': {field: [] for field in FIELDS}}
    value.update({field: 'Specific advisory assessment with no historical claim.' for field in FIELDS})
    value['observed_evidence'] = 'The validated result is PASS.'
    value['evidence_references']['observed_evidence'] = [{'evidence_id': 'e:003', 'observation_key': 'result'}]
    return value


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    sources = {pid: source(pid) for pid in study.IDS}
    setup_path = tmp_path / study.SECTION_009 / study.SOURCE_RUN / 'setup/study.json'
    setup_path.parent.mkdir(parents=True)
    setup_path.write_text('{}')
    monkeypatch.setattr(study, 'load_009', lambda root, run: {'schema': study.VERSION_009, 'sources': sources})
    monkeypatch.setattr(study, 'first_rejected', lambda root, pid, src: {
        'case_id': f'20261030T00000{study.IDS.index(pid):06d}Z',
        'audit_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64,
        'raw_response_sha256': digest(b'{"recommendation":"revise"}'),
        'feedback': {'schema': study.FEEDBACK_VERSION, 'case_id': pid,
                     'original_status': 'rejected', 'rejection_kind': 'invalid_response',
                     'first_validation_error': 'ValueError: unsupported number',
                     'original_audit_sha256': 'a' * 64}})
    monkeypatch.setattr(study, 'selected_raw', lambda root, selected, src: '{"recommendation":"revise"}')
    return tmp_path, study.prepare(tmp_path, 1212)


def test_first_chronological_rejected_and_manifest(monkeypatch, tmp_path):
    src = source('P03')
    base = tmp_path / study.CRITIC_SECTION
    cases = [('20261030T000000000001Z', 'accepted'),
             ('20261030T000000000002Z', 'rejected'),
             ('20261030T000000000003Z', 'rejected')]
    for cid, status in cases:
        path = base / cid
        (path / 'setup').mkdir(parents=True)
        (path / 'setup/case.json').write_text(json.dumps({'source': {'proposal_id': 'P03'}}))
        study.write(path / 'criticism', 'audit.json', {'status': status, 'raw_response': 'raw',
                                                     'rejection': 'first error', 'rejection_kind': 'invalid_response'},
                    'experiment_008_criticism_v1')
    monkeypatch.setattr(study, 'load_case', lambda root, cid: {'source': src})
    monkeypatch.setattr(study, 'load_criticism', lambda root, cid: {'status': 'accepted'})
    monkeypatch.setattr(study, '_historical_criticism', lambda root, cid: (
        {'source': src}, study.read(base / cid / 'criticism', 'audit.json', 'experiment_008_criticism_v1'),
        study.sha((base / cid / 'criticism/audit.json').read_bytes()),
        study.sha((base / cid / 'criticism/manifest.json').read_bytes())))
    assert study.first_rejected(tmp_path, 'P03', src)['case_id'] == cases[1][0]
    (base / cases[1][0] / 'criticism/audit.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='hashes'):
        study.first_rejected(tmp_path, 'P03', src)


def test_preparation_condition_isolation_and_hashes(prepared):
    root, run = prepared
    setup = study.load(root, run)
    assert list(setup['selected']) == list(study.IDS)
    a = study.request_spec(root, run, 'P03', 'A')
    b = study.request_spec(root, run, 'P03', 'B')
    assert b['prompt'].startswith(a['prompt'])
    assert 'REJECTED_INDEPENDENT_CRITICISM_RAW' not in a['prompt']
    assert '{"recommendation":"revise"}' in b['prompt']
    assert 'VALIDATION_FAILURE_REPORT' in b['prompt']
    assert 'unverified advisory text' in b['prompt']
    assert a['model'] == b['model'] and a['request_options'] == b['request_options']
    assert a['source_sha256'] == b['source_sha256']
    assert a['core_prompt_sha256'] == b['core_prompt_sha256']
    assert a['selected_criticism_audit_sha256'] is None
    assert b['selected_criticism_audit_sha256'] == 'a' * 64
    assert study.preview(root, run, 'P03')['conditions'] == {'A': a, 'B': b}
    assert study.validate(root, run)['attempt_statuses']['P03'] == {'A': 'not_requested', 'B': 'not_requested'}
    path = study.folder(root, run) / 'preview/P03/preview.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hashes'):
        study.validate(root, run)


def test_rejected_request_preserved_without_retry(monkeypatch, prepared):
    root, run = prepared
    calls = []
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: calls.append(kw) or
                             SimpleNamespace(model=study.MODEL, id='response-1')))
    monkeypatch.setattr(study, 'inspect_response', lambda response: ('{"bad":true}',
        {'status': 'completed', 'text_characters': 12}))
    assert study.request_decision(client, root, run, 'P03', 'A') == 'rejected'
    assert study.request_decision(client, root, run, 'P03', 'A') == 'already_attempted'
    assert len(calls) == 1
    audit = study.load_decision(root, run, 'P03', 'A')
    assert audit['raw_response'] == '{"bad":true}' and audit['status'] == 'rejected'


def test_live_cli_requires_separate_authorization(monkeypatch, prepared):
    root, run = prepared
    monkeypatch.setattr(study, 'ROOT', root)
    with pytest.raises(SystemExit, match='2'):
        study.main(['--request-decision', '--run-id', run, '--proposal-id', 'P03', '--condition', 'A'])
    assert not (study.folder(root, run) / 'decisions').exists()


def _save_audit(root, run, pid, condition, accepted):
    spec = study.request_spec(root, run, pid, condition)
    parsed = decision(pid, 'revise' if condition == 'A' else 'proceed') if accepted else None
    raw = json.dumps(parsed) if accepted else '{"bad":true}'
    audit = {'schema': 'experiment_012_decision_audit_v1', 'created_utc': study.now(),
             'request': spec, 'raw_response': raw, 'parsed_response': parsed,
             'response_model': study.MODEL, 'response_id': 'fake',
             'response_metadata': {'status': 'completed', 'text_characters': len(raw),
                                   'error_present': False, 'refusal_present': False,
                                   'incomplete_reason': None, 'unfinished_items': False,
                                   'finish_reason': None, 'output_items': [],
                                   'unsupported_output': False},
             'status': 'accepted' if accepted else 'rejected'}
    if not accepted:
        audit.update(rejection_kind='invalid_response', rejection='ValueError: invalid JSON')
    study.write(study.folder(root, run) / 'decisions' / pid / condition,
                'audit.json', audit, 'experiment_012_decision_audit_v1')


def test_blind_partial_scoring_and_failure_accounting(prepared):
    root, run = prepared
    _save_audit(root, run, 'P03', 'A', True)
    with pytest.raises(ValueError, match='all eight'):
        study.prepare_blind(root, run)
    for pid in study.IDS:
        for condition in study.CONDITIONS:
            if pid == 'P03' and condition == 'A':
                continue
            _save_audit(root, run, pid, condition, pid == 'P03')
    blind = study.prepare_blind(root, run)
    assert len(blind['cards']) == 2
    assert all('condition' not in card for card in blind['cards'])
    assert all('P03' in card['blind_id'] for card in blind['cards'])
    assert all('criticism' not in card for card in blind['cards'])
    assert study.validate(root, run)['blind_prepared']
    scores = {card['blind_id']: {'possible_unblinding': '', 'repeats_rejected_claim': False,
              'separates_observations_inferences': True, 'recommendation_differs_from_pair': True,
              'criteria': {criterion: {'score': 3, 'justification': 'Specific reason here.'}
                           for criterion in study.CRITERIA}} for card in blind['cards']}
    with pytest.raises(ValueError, match='every eligible'):
        study.validate_scores({}, blind)
    study.score(root, run, scores)
    result = study.reveal(root, run)
    assert result['feasibility'] == {'accepted_decisions': 2, 'requested_decisions': 8,
                                     'eligible_pairs': 1,
                                     'excluded_pairs': {pid: {'A': 'rejected', 'B': 'rejected'}
                                                        for pid in ('P04', 'P05', 'P06')}}
    assert result['paired_differences_B_minus_A']['error_resistance'] == {'P03': 0}
    assert study.validate(root, run)['revealed']
    path = study.folder(root, run) / 'operator/assignment.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hashes'):
        study.reveal(root, run)
