"""Synthetic CPU-only validation of the pipeline-stage confirmation harness."""
import json
import math
from pathlib import Path
import sys

import pytest

import benchmark_core
import pipeline_stage_discovery as study
from smoke_aa import attach_diagnostics


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    monkeypatch.setattr(study, 'source_hashes', lambda root: {'reviewed.py': 'a' * 64})
    monkeypatch.setattr(study, 'original_evidence', lambda root: ({}, {}))
    return tmp_path, study.prepare(tmp_path)


def synthetic_phase(rounds, seed, iterations, aa, bootstrap_seed, warmups=10):
    outputs = []
    samples = {'baseline': iter([0.06] * rounds), 'candidate': iter([0.065] * rounds)}
    blocks = []

    def output_factory(_a, _b):
        out = object()
        outputs.append(out)
        return out

    def launch(_a, _b, _out, _config):
        pass

    def timer(function, count):
        function()
        side = 'baseline' if len(blocks) % 2 == 0 else 'candidate'
        # Recover actual side by timer closure's most recent launch through
        # the phase order below; constant values keep tests independent of it.
        value = next(samples[side])
        blocks.append({'cuda_event_ms_per_block': value * count,
                       'cuda_event_ms_per_launch': value,
                       'host_submission_ms_per_block': 1.0,
                       'host_wall_ms_per_block': 2.0})
        return value

    config = {'num_stages': 3}
    runner = benchmark_core.run_aa_calibration if aa else benchmark_core.run_paired
    kwargs = {'rounds': rounds, 'seed': seed, 'warmups': warmups,
              'iterations': iterations, 'timer': timer,
              'output_factory': output_factory, 'synchronize': lambda: None}
    phase = runner(launch, object(), object(), config, **kwargs) if aa else runner(
        launch, object(), object(), config, {'num_stages': 2}, **kwargs)
    attach_diagnostics(phase, blocks)
    return phase, study.phase_summary(phase, bootstrap_seed)


def synthetic_session(root, run, session, *, status='complete'):
    setup = study.load(root, run)
    spec = setup['session_specs'][session]
    checks = [{'configuration': stage, 'shape_mnk': list(shape), 'seed': seed,
               'config': study.CONFIG_3 if stage == 'stages_3' else study.CONFIG_2,
               'rtol': 0.02, 'atol': 0.02, 'status': 'PASS'}
              for shape in study.SHAPES for seed in study.SEEDS
              for stage in ('stages_3', 'stages_2')]
    if session == 'confirmation':
        checks += [{'configuration': stage, 'shape_mnk': list(study.SHAPES[0]),
                    'seed': spec['input_seed'],
                    'config': study.CONFIG_3 if stage == 'stages_3' else study.CONFIG_2,
                    'rtol': 0.02, 'atol': 0.02, 'status': 'PASS'}
                   for stage in ('stages_3', 'stages_2')]
    order = [f'aa_{spec["aa_first"]}',
             f'aa_{"stages_2" if spec["aa_first"] == "stages_3" else "stages_3"}',
             'comparison']
    result = {'schema': 'pipeline_stage_discovery_session_v1', 'run_id': run,
              'session': session, 'spec': spec, 'source_hashes': setup['source_hashes'],
              'ratio_orientation': setup['ratio_orientation'], 'status': status,
              'correctness': checks, 'timing': {}, 'summaries': {},
              'phase_order': order, 'compiler': {'stages_3': {'artifacts': {}},
                                                'stages_2': {'artifacts': {}}}}
    if status == 'complete':
        for name in order:
            offset = {'aa_stages_3': 101, 'aa_stages_2': 102, 'comparison': 0}[name]
            phase, summary = synthetic_phase(
                spec['aa_rounds'] if name != 'comparison' else spec['rounds'],
                spec['order_seed'] + offset, spec['iterations'], name != 'comparison',
                spec['bootstrap_seed'] + offset)
            result['timing'][name] = phase
            result['summaries'][name] = summary
    study.write(study.folder(root, run) / 'sessions' / session, 'result.json',
                result, 'pipeline_stage_discovery_session_v1')
    return result


def test_offline_setup_and_session_freeze(prepared):
    root, run = prepared
    setup = study.load(root, run)
    assert setup['gpu_execution_authorized_at_preparation'] is False
    assert setup['ratio_orientation'] == 'stage_3_latency_over_stage_2_latency'
    assert setup['configurations']['stages_3']['num_stages'] == 3
    assert setup['configurations']['stages_2']['num_stages'] == 2
    assert setup['session_specs']['screening_2']['aa_first'] == 'stages_2'
    assert setup['session_specs']['confirmation']['stage'] == 'independent_confirmation'
    assert study.validate(root, run)['sessions'] == {name: 'not_run' for name in study.SESSION_SPECS}
    assert study.dry_run(root, run)['gpu_requests_made'] == 0
    assert not (study.folder(root, run) / 'sessions').exists()
    with pytest.raises(FileExistsError):
        # The UTC ID is immutable; a second setup write cannot overwrite it.
        study.write(study.folder(root, run) / 'setup', 'study.json', setup,
                    'pipeline_stage_discovery_setup_v1')


def test_balanced_order_and_bootstrap_reproducibility():
    one = study.balanced_orders(20, 43)
    two = study.balanced_orders(20, 44)
    assert one.count(['baseline', 'candidate']) == one.count(['candidate', 'baseline']) == 10
    assert one[0] != two[0]
    assert study.bootstrap_interval([0.9, 1.0, 1.1, 0.95], 7) == study.bootstrap_interval(
        [0.9, 1.0, 1.1, 0.95], 7)
    with pytest.raises(ValueError):
        study.balanced_orders(19, 43)
    with pytest.raises(ValueError):
        study.bootstrap_interval([0.9, math.nan], 7)


def test_cpu_pair_orientation_aa_and_readonly_validation(prepared):
    root, run = prepared
    result = synthetic_session(root, run, 'screening_1')
    pair = result['timing']['comparison']
    assert result['summaries']['comparison']['median_paired_ratio'] == pair['paired_ratios']['median']
    assert result['summaries']['comparison']['pair_count'] == 20
    assert result['timing']['aa_stages_3']['aa_variability']['median_absolute_ratio_deviation'] >= 0
    assert study.validate_session(root, run, 'screening_1')['status'] == 'complete'
    assert study.validate(root, run)['sessions']['screening_1'] == 'complete'
    assert study.validate(root, run)['sessions']['confirmation'] == 'not_run'


def test_malformed_samples_ratio_and_manifest(prepared):
    root, run = prepared
    result = synthetic_session(root, run, 'screening_1')
    spec = study.SESSION_SPECS['screening_1']
    phase = result['timing']['comparison']
    phase['rounds'][0]['baseline_over_candidate'] = 1 / phase['rounds'][0]['baseline_over_candidate']
    with pytest.raises(ValueError, match='orientation'):
        study.validate_phase(phase, 20, 100, spec['order_seed'], aa=False,
                             bootstrap_seed=spec['bootstrap_seed'])
    phase['rounds'][0]['baseline_over_candidate'] = (
        phase['rounds'][0]['baseline_ms'] / phase['rounds'][0]['candidate_ms'])
    del phase['rounds'][0]['blocks']['baseline']
    with pytest.raises(ValueError, match='diagnostic blocks'):
        study.validate_phase(phase, 20, 100, spec['order_seed'], aa=False,
                             bootstrap_seed=spec['bootstrap_seed'])
    path = study.folder(root, run) / 'sessions/screening_1/result.json'
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hashes'):
        study.validate(root, run)


def test_correctness_failure_blocks_timing_and_confirmation_gate(prepared):
    root, run = prepared
    result = synthetic_session(root, run, 'screening_1', status='correctness_failed')
    result['correctness'][0]['status'] = 'FAIL'
    result['timing'] = {}
    # Write a separate synthetic attempt in a fresh run; the original stage is immutable.
    fresh = study.prepare(root)
    study.write(study.folder(root, fresh) / 'sessions/screening_1', 'result.json', result |
                {'run_id': fresh}, 'pipeline_stage_discovery_session_v1')
    assert study.validate_session(root, fresh, 'screening_1')['status'] == 'correctness_failed'
    with pytest.raises(ValueError, match='approval'):
        study.execute_session(None, fresh, 'confirmation')
    with pytest.raises(ValueError, match='screenings'):
        study.execute_session(root, fresh, 'confirmation', human_approved=True,
                              confirmation_approved=True)


def test_no_cuda_import_during_offline_modes(monkeypatch, prepared):
    root, run = prepared
    before = set(sys.modules)
    study.dry_run(root, run)
    study.validate(root, run)
    assert 'torch' not in (set(sys.modules) - before)
    assert 'triton' not in (set(sys.modules) - before)
    with pytest.raises(ValueError, match='approval'):
        study.execute_session(root, run, 'screening_1')
    assert not (study.folder(root, run) / 'sessions').exists()


def test_missing_sample_and_frozen_timing_settings(prepared):
    root, run = prepared
    result = synthetic_session(root, run, 'screening_1')
    phase = result['timing']['comparison']
    spec = study.SESSION_SPECS['screening_1']
    phase['rounds'].pop()
    with pytest.raises(ValueError, match='phase mode, count'):
        study.validate_phase(phase, 20, 100, spec['order_seed'], aa=False,
                             bootstrap_seed=spec['bootstrap_seed'], warmups=10)
    phase['rounds'].append({'round': 20})
    phase['method']['warmups_per_side_per_round'] = 0
    with pytest.raises(ValueError, match='warmup'):
        study.validate_phase(phase, 20, 100, spec['order_seed'], aa=False,
                             bootstrap_seed=spec['bootstrap_seed'], warmups=10)


def test_wrong_correctness_config_and_session_label_fail_closed(prepared):
    root, run = prepared
    result = synthetic_session(root, run, 'screening_1')
    result['correctness'][0]['config'] = study.CONFIG_2
    fresh = study.prepare(root)
    study.write(study.folder(root, fresh) / 'sessions/screening_1', 'result.json',
                result | {'run_id': fresh}, 'pipeline_stage_discovery_session_v1')
    with pytest.raises(ValueError, match='correctness configuration'):
        study.validate(root, fresh)
    assert study.SESSION_SPECS['screening_1']['stage'] == 'screening'
    assert study.SESSION_SPECS['confirmation']['stage'] == 'independent_confirmation'


def test_source_change_breaks_readonly_load(prepared, monkeypatch):
    root, run = prepared
    monkeypatch.setattr(study, 'source_hashes', lambda _root: {'reviewed.py': 'b' * 64})
    with pytest.raises(ValueError, match='source mismatch'):
        study.validate(root, run)
