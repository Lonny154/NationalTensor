"""CPU-only checks for the held-out K-depth measurement workflow."""
import json
import sys

import pytest

import prospective_k_depth_test as study


def test_prepared_run_is_frozen_and_gpu_free():
    before = set(sys.modules)
    run = '20261101T020025480962Z'
    setup = study.load(study.ROOT, run)
    state = study.validate(study.ROOT, run)
    assert state['sessions'] == {'target_k1536': 'complete', 'control_k768': 'complete'}
    assert setup['evaluation_rule']['reference_ratio'] == 0.953120870845799
    assert setup['evaluation_rule']['deadband_absolute_ratio'] == 0.04
    assert setup['evaluation_rule']['lower_if'] == 'ratio < 0.913120870846'
    assert setup['prediction_direction'] == 'lower'
    assert setup['method']['comparison_rounds'] == 14
    assert setup['method']['aa_rounds_per_variant'] == 20
    assert setup['method']['iterations_per_side_per_round'] == 200
    assert setup['configurations']['stages_3']['num_stages'] == 3
    assert setup['configurations']['stages_2']['num_stages'] == 2
    assert 'torch' not in set(sys.modules) - before
    assert 'triton' not in set(sys.modules) - before


def test_completed_sessions_need_approval_and_cannot_be_retried():
    run = '20261101T020025480962Z'
    with pytest.raises(ValueError, match='approval'):
        study.execute_session(study.ROOT, run, 'target_k1536')
    with pytest.raises(ValueError, match='already attempted'):
        study.execute_session(study.ROOT, run, 'control_k768', human_approved=True)
    assert study.validate(study.ROOT, run)['sessions']['target_k1536'] == 'complete'


def test_setup_manifest_and_forecast_hash_fail_closed(tmp_path):
    run = '20261101T020025480962Z'
    source = study.base(study.ROOT, run) / 'setup'
    copy = study.base(tmp_path, run) / 'setup'
    copy.mkdir(parents=True)
    for name in ('study.json', 'manifest.json'):
        (copy / name).write_bytes((source / name).read_bytes())
    data = json.loads((copy / 'study.json').read_bytes())
    data['evaluation_rule']['deadband_absolute_ratio'] = 0.05
    (copy / 'study.json').write_bytes(study.canonical(data))
    with pytest.raises(ValueError, match='hashes'):
        study.read(copy, 'study.json', study.VERSION)


def test_balanced_orders_bootstrap_and_session_shape():
    for spec in study.SESSIONS.values():
        orders = study.prior.balanced_orders(study.ROUNDS, spec['order_seed'])
        assert orders.count(['baseline', 'candidate']) == 7
        assert orders.count(['candidate', 'baseline']) == 7
        assert spec['timing_seed'] in spec['seeds']
        assert spec['shape_mnk'][:2] == [512, 3072]
    assert study.SESSIONS['target_k1536']['shape_mnk'][2] == 1536
    assert study.SESSIONS['control_k768']['shape_mnk'][2] == 768
    assert study.prior.bootstrap_interval([0.8, 0.9, 1.0, 1.1], 2545) == \
        study.prior.bootstrap_interval([0.8, 0.9, 1.0, 1.1], 2545)
