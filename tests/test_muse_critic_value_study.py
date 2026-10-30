"""CPU-only checks for Experiment 009 boundaries and offline validation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_critic_value_study as study
from muse_debrief import canonical, digest


def source(pid='P01'):
    return {'proposal_id': pid, 'proposal_sha256': 'a' * 64,
            'proposal': {'name': 'test idea'},
            'validated_evidence': {'003': {'evidence_id': 'e:003', 'observations': {'result': 'PASS', 'count': 4}},
                                   '005': {'evidence_id': 'e:005', 'observations': {'confirmation_status': 'missing'}}},
            'experiment_005_criteria': {}}


def decision(src=None):
    src = src or source()
    result = {'proposal_id': src['proposal_id'], 'proposal_sha256': src['proposal_sha256'],
              'decision': 'revise', 'evidence_references': {field: [] for field in study.FIELDS}}
    result.update({field: 'A specific advisory assessment without a numerical claim.' for field in study.FIELDS})
    result['observed_evidence'] = 'The validated result is PASS.'
    result['evidence_references']['observed_evidence'] = [{'evidence_id': 'e:003', 'observation_key': 'result'}]
    return result


def test_import_and_preparation_make_no_api_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'source_package', lambda root, sid, rid, pid: source(pid))
    run = study.prepare(tmp_path, 17)
    saved = study.load(tmp_path, run)
    assert saved['rubric'] == study.RUBRIC
    assert len(saved['sources']) == 6
    assert not (study.folder(tmp_path, run) / 'decisions').exists()


def test_condition_isolation_and_matching(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'source_package', lambda root, sid, rid, pid: source(pid))
    run = study.prepare(tmp_path, 7)
    monkeypatch.setattr(study, 'load_criticism_binding', lambda *args: {'criticism': {'recommendation': 'revise'}})
    a = study.request_spec(tmp_path, run, 'P01', 'A')
    b = study.request_spec(tmp_path, run, 'P01', 'B')
    assert a['model'] == b['model'] and a['request_options'] == b['request_options']
    assert a['proposal_sha256'] == b['proposal_sha256'] and a['source_sha256'] == b['source_sha256']
    assert 'INDEPENDENT_CRITICISM' not in a['prompt']
    assert b['prompt'].startswith(a['prompt'])
    assert a['criticism_sha256'] is None and b['criticism_sha256']


def test_source_and_manifest_tampering(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'source_package', lambda root, sid, rid, pid: source(pid))
    run = study.prepare(tmp_path, 1)
    monkeypatch.setattr(study, 'source_package', lambda root, sid, rid, pid: source('P06'))
    with pytest.raises(ValueError, match='source or upstream evidence'):
        study.load(tmp_path, run)
    path = study.folder(tmp_path, run) / 'setup/study.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hashes'):
        study.load(tmp_path, run)


def test_citation_scope_and_identity():
    src = source()
    value = decision(src)
    assert study.parse_decision(json.dumps(value), src) == value
    value['evidence_references']['observed_evidence'] = []
    with pytest.raises(ValueError, match='requires at least one citation'):
        study.parse_decision(json.dumps(value), src)
    value['evidence_references']['observed_evidence'] = [{'evidence_id': 'wrong', 'observation_key': 'result'}]
    with pytest.raises(ValueError, match='invalid or duplicate evidence citation'):
        study.parse_decision(json.dumps(value), src)
    value = decision(src)
    value['evidence_assessment'] = 'Experiment 003 recorded PASS.'
    with pytest.raises(ValueError, match='evidence_assessment'):
        study.parse_decision(json.dumps(value), src)
    value = decision(src)
    value['proposal_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='identity'):
        study.parse_decision(json.dumps(value), src)


def test_criticism_provenance_rejects_wrong_source(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'source_package', lambda root, sid, rid, pid: source(pid))
    run = study.prepare(tmp_path, 1)
    monkeypatch.setattr(study, 'load_case', lambda *args: {'source': source('P02'), 'critique_prompt': 'other'})
    with pytest.raises(ValueError, match='different source or protocol'):
        study.attach_criticism(tmp_path, run, 'P01', '20261030T152949236420Z')


def test_scores_complete_and_assignment_sealed(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'source_package', lambda root, sid, rid, pid: source(pid))
    run = study.prepare(tmp_path, 17)
    monkeypatch.setattr(study, 'load_decision', lambda root, run, pid, condition, setup=None:
                        {'parsed_response': decision(source(pid))})
    blind = study.prepare_blind(tmp_path, run)
    assert all('condition' not in card and 'criticism' not in card for card in blind['cards'])
    assert len(blind['cards']) == 12
    scores = {card['blind_id']: {'possible_unblinding': '', 'criteria': {
        criterion: {'score': 3, 'justification': 'Specific reason here.'} for criterion in study.CRITERIA}}
        for card in blind['cards']}
    with pytest.raises(ValueError, match='all twelve'):
        study.validate_scores({k: v for k, v in scores.items() if k != 'P01-1'}, blind)
    study.score(tmp_path, run, scores)
    result = study.reveal(tmp_path, run)
    assert set(result['mean_subjective_scores']) == {'A', 'B'}
    assert all(len(v) == 6 for v in result['paired_differences_B_minus_A'].values())
    path = study.folder(tmp_path, run) / 'operator/assignment.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hashes'):
        study.reveal(tmp_path, run)


def test_rejected_response_preserved_without_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'source_package', lambda root, sid, rid, pid: source(pid))
    run = study.prepare(tmp_path, 9)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        model=study.MODEL, id='response-1')))
    monkeypatch.setattr(study, 'inspect_response', lambda response: ('{"bad":true}',
        {'status': 'completed', 'text_characters': 12}))
    assert study.request_decision(client, tmp_path, run, 'P01', 'A') == 'rejected'
    audit = study.load_decision(tmp_path, run, 'P01', 'A')
    assert audit['raw_response'] == '{"bad":true}' and audit['status'] == 'rejected'
    assert study.request_decision(client, tmp_path, run, 'P01', 'A') == 'already_attempted'
