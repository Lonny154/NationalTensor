"""CPU-only prospective hypothesis request and provenance checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_pipeline_hypothesis_study as study


def tiny_package():
    return {'schema': 'muse_pipeline_observations_v1',
            'target_status': 'not_measured_in_this_experiment',
            'reference_ratio': {'value': 0.953120870845799},
            'observations': [{'id': 'TIMING_SCREENING_1'}, {'id': 'PROFILE_1'}]}


def valid_response():
    return {'chosen_observation_id': 'TIMING_SCREENING_1',
            'why_interesting': 'The recorded direction is unexpected.',
            'measured_facts': [{'claim': 'The timing direction was recorded.',
                                'observation_ids': ['TIMING_SCREENING_1']}],
            'explanations': [
                {'label': 'copy path', 'mechanism': 'One stage changes copy scheduling.',
                 'support_ids': ['TIMING_SCREENING_1'], 'weakening_ids': ['PROFILE_1'],
                 'falsifier': 'A controlled longer-K test does not show the proposed interaction.'},
                {'label': 'state', 'mechanism': 'Device state changes block timing.',
                 'support_ids': ['PROFILE_1'], 'weakening_ids': ['TIMING_SCREENING_1'],
                 'falsifier': 'The effect persists under matched state.'}],
            'preferred_label': 'copy path',
            'prediction': {'target_shape_mnk': study.TARGET_SHAPE,
                           'ratio_orientation': 'stage3_latency_over_stage2_latency',
                           'direction': 'near', 'point_ratio': 0.95,
                           'plausible_range': [0.90, 1.00],
                           'range_rationale': 'This is a broad prospective range.'},
            'discriminating_experiment': 'Use matched paired timing at both K depths.',
            'falsifying_outcome': 'The stage direction reverses in controlled timing.',
            'uncertainties': ['Profiler replay may affect cache state.']}


def test_real_package_is_neutral_and_manifested():
    if (not (study.ROOT / 'evaluations/pipeline-stage-discovery/v1' / study.DISCOVERY_RUN).exists()
            or not (study.ROOT / 'evaluations/pipeline-stage-mechanism/v1' / study.PROFILE_RUN).exists()):
        pytest.skip('local ignored GPU artifacts unavailable')
    package = study.evidence_package(study.ROOT)
    prompt = study.prompt_for(package)
    assert len(package['observations']) == 8
    assert package['target_status'] == 'not_measured_in_this_experiment'
    assert package['target_shape_mnk'] == [512, 3072, 1536]
    assert package['reference_ratio']['value'] == 0.953120870845799
    assert len(package['observations'][0]['measurement']['comparison_pairs']) == 20
    assert len(package['observations'][2]['measurement']['comparison_pairs']) == 14
    assert 'EVIDENCE\n' in prompt and 'SCHEMA\n' in prompt
    for forbidden in ('H1:', 'H2:', 'H3:', 'H4:',
                      'stage-3 relative advantage persists or grows',
                      'preferred hardware explanation'):
        assert forbidden not in prompt
    assert 'No timing result exists for target_shape_mnk' in prompt


def test_schema_accepts_prediction_and_rejects_bad_ids_and_numbers():
    package = tiny_package()
    value = valid_response()
    assert study.parse_response(json.dumps(value), package) == value
    value['explanations'][0]['support_ids'] = ['NOT_AN_ID']
    with pytest.raises(ValueError, match='observation IDs'):
        study.parse_response(json.dumps(value), package)
    value = valid_response()
    value['prediction']['plausible_range'] = [1.1, 0.9]
    with pytest.raises(ValueError, match='ratio range'):
        study.parse_response(json.dumps(value), package)
    value = valid_response()
    value['prediction']['point_ratio'] = float('nan')
    with pytest.raises(ValueError, match='point ratio'):
        study.parse_response(json.dumps(value), package)


def test_leakage_guard_and_preregistered_direction_deadband():
    package = tiny_package()
    study.check_no_leak(package)
    with pytest.raises(ValueError, match='leaked'):
        study.check_no_leak({'human_text': 'H1: pipeline overlap hypothesis'})
    with pytest.raises(ValueError, match='target measurement'):
        study.check_no_leak({'target_status': 'measured'})
    rule = study.evaluation_rule(package['reference_ratio']['value'])
    assert rule['deadband_absolute_ratio'] == 0.04
    assert '0.913120870846' in rule['lower_if']
    assert '0.993120870846' in rule['higher_if']


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'evidence_package', lambda root: tiny_package())
    monkeypatch.setattr(study, 'hashes', lambda root, names: {name: 'a' * 64 for name in names})
    return tmp_path, study.prepare(tmp_path)


def test_offline_prepare_manifest_and_one_attempt_rejection(prepared):
    root, identifier = prepared
    assert study.validate(root, identifier)['request_status'] == 'not_requested'
    setup, package, spec = study.load(root, identifier)
    assert spec['prompt_sha256'] == study.digest(spec['prompt'].encode())
    assert spec['model'] == study.MODEL
    assert spec['request_options'] == study.REQUEST_OPTIONS
    assert spec['max_retries'] == 0
    assert not (study.folder(root, identifier) / 'attempt').exists()
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('synthetic transport failure')
    client = SimpleNamespace(responses=SimpleNamespace(create=fail))
    with pytest.raises(ValueError, match='authorization'):
        study.request_once(client, root, identifier)
    assert study.request_once(client, root, identifier, human_authorized=True) == 'rejected'
    assert study.request_once(client, root, identifier, human_authorized=True) == 'already_attempted'
    assert len(calls) == 1 and calls[0]['model'] == study.MODEL
    assert study.validate(root, identifier)['request_status'] == 'rejected'
    path = study.folder(root, identifier) / 'attempt/result/audit.json'
    audit = json.loads(path.read_text())
    assert audit['rejection_kind'] == 'request_error'


def test_source_and_manifest_tampering_fail_closed(prepared, monkeypatch):
    root, identifier = prepared
    setup_path = study.folder(root, identifier) / 'setup/study.json'
    original = setup_path.read_bytes()
    setup_path.write_bytes(setup_path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hashes'):
        study.validate(root, identifier)
    setup_path.write_bytes(original)
    monkeypatch.setattr(study, 'hashes', lambda root, names: {name: 'b' * 64 for name in names})
    with pytest.raises(ValueError, match='source'):
        study.validate(root, identifier)


def test_no_gpu_import_or_api_call_on_offline_preview(prepared, monkeypatch):
    root, identifier = prepared
    monkeypatch.setattr(study, 'ROOT', root)
    assert study.main(['--validate', '--run-id', identifier]) == 0
    with pytest.raises(SystemExit):
        study.main(['--request-once', '--run-id', identifier])
    assert not (study.folder(root, identifier) / 'attempt').exists()
    source = Path(study.__file__).read_text()
    assert 'import torch' not in source and 'import triton' not in source
