"""CPU-only checks for Experiment 013's matched, offline P03 protocol."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_model_capability_study as study
import muse_critic_value_study as original
from muse_debrief import digest


def source():
    return {'proposal_id': 'P03', 'proposal_sha256': 'a' * 64,
            'proposal': {'name': 'grouped traversal'},
            'validated_evidence': {
                '003': {'evidence_id': 'diagnostic:003', 'observations': {'result': 'PASS'}},
                '005': {'evidence_id': 'experiment_005:verified',
                        'observations': {'confirmation_status': 'missing',
                                         'screening_scope': 'screening only'}}},
            'experiment_005_criteria': {}}


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    src = source()
    path = tmp_path / study.SECTION_009 / study.SOURCE_RUN / 'setup/study.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    monkeypatch.setattr(study, 'load_009', lambda root, run: {
        'schema': study.VERSION_009, 'sources': {'P03': src}})
    return tmp_path, study.prepare(tmp_path)


def test_matched_prompt_and_citation_objects(prepared):
    root, run = prepared
    setup = study.load(root, run)
    a = study.request_spec(root, run, 'A')
    b = study.request_spec(root, run, 'B')
    assert a['prompt'].encode() == b['prompt'].encode()
    assert a['prompt_sha256'] == b['prompt_sha256'] == digest(a['prompt'].encode())
    assert a['model'] == 'muse-spark-1.3'
    assert b['model'] == 'muse-wonder-1.0'
    assert a['request_options'] == b['request_options'] == {
        'max_output_tokens': 16000, 'reasoning': {'effort': 'high'}}
    assert a['max_retries'] == b['max_retries'] == 0
    assert a['timeout_seconds'] == b['timeout_seconds'] == 120.0
    assert a['proposal_sha256'] == b['proposal_sha256'] == source()['proposal_sha256']
    assert a['source_sha256'] == b['source_sha256'] == setup['source_sha256']
    assert a['prompt_version'] == b['prompt_version'] == study.PROMPT_VERSION
    assert a['authorization_state'] == 'not_authorized_in_offline_preview'
    assert 'REJECTED_INDEPENDENT_CRITICISM_RAW' not in a['prompt']
    assert 'VALIDATION_FAILURE_REPORT' not in a['prompt']
    schema = json.loads(a['prompt'].split('\nSCHEMA\n')[1].split('\nSOURCE\n')[0])
    ref = schema['evidence_references']['observed_evidence'][0]
    assert ref == {'evidence_id': 'experiment_005:verified',
                   'observation_key': 'confirmation_status'}
    assert set(schema['evidence_references']) == set(original.FIELDS)
    assert 'free-text mentions or colon-joined' in a['prompt']
    assert study.parse_decision is original.parse_decision
    assert setup['validator'] == 'experiment_009_decision_v1'
    assert setup['model_availability'] == 'unverified_offline'


def test_preview_manifest_and_separate_namespace(prepared):
    root, run = prepared
    record = study.preview(root, run)
    assert record['requests']['A']['prompt'] == record['requests']['B']['prompt']
    assert study.validate(root, run)['request_statuses'] == {'A': 'not_requested', 'B': 'not_requested'}
    assert study.SECTION != study.SECTION_009
    assert study.SECTION != Path('evaluations/muse-imperfect-critic-study/v1')
    assert not (study.folder(root, run) / 'decisions').exists()
    path = study.folder(root, run) / 'preview/preview.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hashes'):
        study.validate(root, run)


def test_one_rejected_attempt_preserved_without_retry(monkeypatch, prepared):
    root, run = prepared
    study.preview(root, run)
    calls = []
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: calls.append(kw) or
        SimpleNamespace(model='muse-spark-1.3', id='fake-response')))
    monkeypatch.setattr(study, 'inspect_response', lambda response: ('{"bad":true}', {
        'status': 'completed', 'text_characters': 12, 'error_present': False,
        'refusal_present': False, 'incomplete_reason': None, 'unfinished_items': False,
        'finish_reason': None, 'output_items': [], 'unsupported_output': False}))
    with pytest.raises(ValueError, match='authorization'):
        study.request_decision(client, root, run, 'A')
    assert study.request_decision(client, root, run, 'A', human_authorized=True) == 'rejected'
    assert study.request_decision(client, root, run, 'A', human_authorized=True) == 'already_attempted'
    assert len(calls) == 1 and calls[0]['model'] == 'muse-spark-1.3'
    assert calls[0]['reasoning'] == {'effort': 'high'}
    assert study.load_decision(root, run, 'A')['raw_response'] == '{"bad":true}'
    assert study.validate(root, run)['request_statuses'] == {'A': 'rejected', 'B': 'not_requested'}


def test_wonder_no_fallback_and_failed_request_recorded(prepared):
    root, run = prepared
    study.preview(root, run)
    calls = []
    def fail(**kw):
        calls.append(kw)
        raise RuntimeError('model unavailable')
    client = SimpleNamespace(responses=SimpleNamespace(create=fail))
    assert study.request_decision(client, root, run, 'B', human_authorized=True) == 'rejected'
    assert len(calls) == 1 and calls[0]['model'] == 'muse-wonder-1.0'
    assert study.load_decision(root, run, 'B')['rejection_kind'] == 'request_error'
    assert study.request_decision(client, root, run, 'B', human_authorized=True) == 'already_attempted'


def test_live_cli_gate_and_no_gpu_imports(monkeypatch, prepared):
    root, run = prepared
    monkeypatch.setattr(study, 'ROOT', root)
    with pytest.raises(SystemExit, match='2'):
        study.main(['--request-decision', '--run-id', run, '--configuration', 'A'])
    assert not (study.folder(root, run) / 'decisions').exists()
    text = Path(study.__file__).read_text()
    for forbidden in ('import torch', 'import triton', 'trusted_compute_experiment',
                      'benchmark_proposal', 'optimize.py'):
        assert forbidden not in text
