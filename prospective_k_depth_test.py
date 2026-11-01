"""Offline preparation and separately authorized GPU sessions for the frozen Muse forecast.

Offline commands do not import Torch or launch kernels. Each session is a single
fresh-process attempt. The K=1536 target is the first timing attempt; the K=768
control is a separate later command and never replaces the target for scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import canonical
import muse_pipeline_hypothesis_study as forecast
import pipeline_stage_discovery as prior

VERSION = 'prospective_k_depth_test_v1'
SECTION = Path('evaluations/prospective-k-depth/v1')
FORECAST_ID = '20261101T015247349750Z'
FORECAST_AUDIT_SHA = 'c384ef6508cec7994155477b80d0cea667093149712c8c26438f773afafaebb5'
FORECAST_MANIFEST_SHA = '14c0cb9263ab08b6a43b8eecbbf9b8903cf9ef1662dfe975a1d351746224469b'
DISCOVERY_ID = '20261101T012329698902Z'
SOURCE_FILES = ('prospective_k_depth_test.py', 'pipeline_stage_discovery.py',
                'benchmark_core.py', 'smoke_aa.py', 'trusted_two_slab_kernel.py',
                'trusted_two_slab_experiment.py', 'stage_reduction_experiment.py',
                'reference_precision.py')
SESSIONS = {
    'target_k1536': {'shape_mnk': [512, 3072, 1536], 'seeds': [45, 46],
                     'timing_seed': 45, 'order_seed': 1545, 'bootstrap_seed': 2545,
                     'aa_first': 'stages_3'},
    'control_k768': {'shape_mnk': [512, 3072, 768], 'seeds': [47, 48],
                     'timing_seed': 47, 'order_seed': 1546, 'bootstrap_seed': 2546,
                     'aa_first': 'stages_2'},
}
ROUNDS = 14
AA_ROUNDS = 20
ITERATIONS = 200
WARMUPS = 10


def base(root, run_id):
    prior.folder(root, run_id)  # Enforce the established run-ID format.
    return Path(root).resolve() / SECTION / run_id


def write(folder, name, value, schema):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / name).write_bytes(canonical(value))
    (folder / 'manifest.json').write_bytes(canonical(manifest(folder, schema=schema)))


def read(folder, name, schema):
    read_manifest(folder, schema=schema)
    return json.loads((folder / name).read_bytes())


def frozen_forecast(root):
    record = forecast.validate(root, FORECAST_ID)
    audit = Path(root) / forecast.SECTION / FORECAST_ID / 'attempt/result/audit.json'
    audit_manifest = audit.with_name('manifest.json')
    if (sha(audit.read_bytes()) != FORECAST_AUDIT_SHA or
            sha(audit_manifest.read_bytes()) != FORECAST_MANIFEST_SHA or
            record['request_status'] != 'accepted_structurally' or
            record['reference_ratio'] != 0.953120870845799 or
            record['evaluation_rule'] != forecast.evaluation_rule(record['reference_ratio'])):
        raise ValueError('frozen prediction or evaluation rule changed')
    prediction = json.loads(audit.read_bytes())['parsed_response']['prediction']
    if (prediction['direction'] != 'lower' or prediction['target_shape_mnk'] !=
            SESSIONS['target_k1536']['shape_mnk']):
        raise ValueError('unexpected frozen prediction')
    return record, prediction


def expected_setup(root, run_id, created_utc):
    forecast_record, prediction = frozen_forecast(root)
    discovery = prior.validate(root, DISCOVERY_ID)
    if any(status != 'complete' for status in discovery['sessions'].values()):
        raise ValueError('discovery sessions are incomplete')
    sources = {name: sha((Path(root) / name).read_bytes()) for name in SOURCE_FILES}
    if sources['trusted_two_slab_kernel.py'] != prior.KERNEL_SHA:
        raise ValueError('reviewed kernel changed')
    discovery_manifests = {name: sha((prior.folder(root, DISCOVERY_ID) /
                                     'sessions' / name / 'manifest.json').read_bytes())
                           for name in prior.SESSION_SPECS}
    return {'schema': VERSION, 'run_id': run_id, 'created_utc': created_utc,
            'forecast_run_id': FORECAST_ID, 'forecast_audit_sha256': FORECAST_AUDIT_SHA,
            'forecast_audit_manifest_sha256': FORECAST_MANIFEST_SHA,
            'prediction_direction': prediction['direction'],
            'evaluation_rule': forecast_record['evaluation_rule'],
            'discovery_run_id': DISCOVERY_ID, 'discovery_manifest_sha256': discovery_manifests,
            'source_hashes': sources, 'gpu': prior.GPU, 'dtype': 'float16',
            'configurations': {'stages_3': prior.CONFIG_3, 'stages_2': prior.CONFIG_2},
            'tolerance': prior.TOLERANCE, 'sessions': SESSIONS,
            'method': {'comparison_rounds': ROUNDS, 'aa_rounds_per_variant': AA_ROUNDS,
                       'iterations_per_side_per_round': ITERATIONS, 'warmups': WARMUPS,
                       'bootstrap_draws': prior.BOOTSTRAP_DRAWS,
                       'timing': 'eager Python launches within CUDA-event blocks',
                       'ratio_orientation': 'stage3_latency_over_stage2_latency'},
            'gpu_authorized_at_preparation': False}


def prepare(root=ROOT):
    run_id = prior.new_run_id()
    write(base(root, run_id) / 'setup', 'study.json',
          expected_setup(root, run_id, prior.now()), VERSION)
    return run_id


def load(root, run_id):
    setup = read(base(root, run_id) / 'setup', 'study.json', VERSION)
    if (not isinstance(setup.get('created_utc'), str) or
            setup != expected_setup(root, run_id, setup['created_utc'])):
        raise ValueError('prepared study or source changed')
    return setup


def validate_session(root, run_id, session, setup=None):
    setup = setup or load(root, run_id)
    spec = setup['sessions'][session]
    folder = base(root, run_id) / 'sessions' / session
    result = read(folder, 'result.json', 'prospective_k_depth_session_v1')
    if (result.get('schema') != 'prospective_k_depth_session_v1' or
            result.get('run_id') != run_id or result.get('session') != session or
            result.get('spec') != spec or result.get('source_hashes') != setup['source_hashes'] or
            result.get('status') not in ('complete', 'correctness_failed', 'failed')):
        raise ValueError('session identity mismatch')
    expected = {(name, seed) for name in ('stages_3', 'stages_2') for seed in spec['seeds']}
    seen = set()
    for row in result.get('correctness', []):
        key = (row.get('configuration'), row.get('seed'))
        if (key not in expected or key in seen or row.get('shape_mnk') != spec['shape_mnk'] or
                row.get('config') != setup['configurations'][key[0]] or
                row.get('rtol') != 0.02 or row.get('atol') != 0.02 or
                row.get('status') not in ('PASS', 'FAIL')):
            raise ValueError('correctness identity mismatch')
        seen.add(key)
    agreement = result.get('stage_agreement', [])
    if result['status'] == 'complete':
        if (seen != expected or any(row['status'] != 'PASS' for row in result['correctness']) or
                {row.get('seed') for row in agreement} != set(spec['seeds']) or
                any(row.get('status') != 'PASS' for row in agreement)):
            raise ValueError('timing without complete correctness gate')
        order = [f"aa_{spec['aa_first']}",
                 f"aa_{'stages_2' if spec['aa_first'] == 'stages_3' else 'stages_3'}",
                 'comparison']
        if result.get('phase_order') != order or set(result.get('timing', {})) != set(order):
            raise ValueError('timing phase mismatch')
        for name in order:
            offset = {'comparison': 0, 'aa_stages_3': 101, 'aa_stages_2': 102}[name]
            summary = prior.validate_phase(result['timing'][name],
                ROUNDS if name == 'comparison' else AA_ROUNDS, ITERATIONS,
                spec['order_seed'] + offset, aa=name != 'comparison',
                bootstrap_seed=spec['bootstrap_seed'] + offset, warmups=WARMUPS)
            if result['summaries'].get(name) != summary:
                raise ValueError('timing summary mismatch')
        for capture in result.get('compiler', {}).values():
            for artifact in capture.get('artifacts', {}).values():
                path = folder / artifact['path']
                if (Path(artifact['path']).name != artifact['path'] or path.is_symlink() or
                        sha(path.read_bytes()) != artifact['sha256'] or
                        len(path.read_bytes()) != artifact['size_bytes']):
                    raise ValueError('compiler artifact mismatch')
        if set(result.get('compiler', {})) != {'stages_3', 'stages_2'}:
            raise ValueError('compiler capture missing')
    elif result['status'] == 'correctness_failed':
        if (not any(row.get('status') == 'FAIL' for row in result['correctness'] + agreement)
                or result.get('timing')):
            raise ValueError('correctness failure did not block timing')
    elif not result.get('error'):
        raise ValueError('failed session lacks error')
    return result


def validate(root, run_id):
    setup = load(root, run_id)
    states = {}
    for session in SESSIONS:
        folder = base(root, run_id) / 'sessions' / session
        states[session] = validate_session(root, run_id, session, setup)['status'] if folder.exists() else 'not_run'
    return {'run_id': run_id, 'sessions': states,
            'forecast_audit_sha256': FORECAST_AUDIT_SHA,
            'reference_ratio': setup['evaluation_rule']['reference_ratio'],
            'deadband_absolute_ratio': setup['evaluation_rule']['deadband_absolute_ratio'],
            'gpu_authorized_at_preparation': False}


def execute_session(root, run_id, session, *, human_approved=False):
    if not human_approved:
        raise ValueError('separate human approval required')
    setup = load(root, run_id)
    if session not in SESSIONS or validate(root, run_id)['sessions'][session] != 'not_run':
        raise ValueError('session already attempted or invalid')
    if session == 'control_k768' and validate(root, run_id)['sessions']['target_k1536'] == 'not_run':
        raise ValueError('target session must be attempted first')
    spec = setup['sessions'][session]
    folder = base(root, run_id) / 'sessions' / session
    folder.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'prospective_k_depth_session_v1', 'run_id': run_id,
              'session': session, 'created_utc': prior.now(), 'status': 'incomplete',
              'spec': spec, 'source_hashes': setup['source_hashes'],
              'human_approved': True, 'correctness': [], 'stage_agreement': [],
              'timing': {}, 'summaries': {}, 'phase_order': [], 'telemetry': {}}
    try:
        import torch
        import stage_reduction_experiment as original
        import trusted_two_slab_experiment as trusted
        import trusted_two_slab_kernel as kernel
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != prior.GPU:
            raise ValueError('expected RTX 4060 Ti unavailable')
        identity = trusted._identity()
        if (identity['trusted_module_sha256'] != prior.KERNEL_SHA or
                identity['trusted_implementation_sha256'] !=
                prior.original_evidence(root)[0]['source_identity']['trusted_implementation_sha256']):
            raise ValueError('reviewed kernel identity mismatch')
        result['environment'] = original.environment()
        result['input_policy'] = 'trusted_two_slab_experiment._inputs; torch.randn FP16, contiguous'
        primary = None
        handles = {}
        for seed in spec['seeds']:
            a, b, reference, policy = trusted._inputs(tuple(spec['shape_mnk']), seed)
            outputs = {}
            for name, config in (('stages_3', prior.CONFIG_3), ('stages_2', prior.CONFIG_2)):
                row = {'configuration': name, 'shape_mnk': spec['shape_mnk'],
                       'seed': seed, 'config': config, 'reference_precision': policy,
                       'rtol': 0.02, 'atol': 0.02,
                       'input_strides': {'a': list(a.stride()), 'b': list(b.stride())}}
                try:
                    out = torch.empty_like(reference)
                    handle = kernel.launch(a, b, out, config)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
                    row['status'] = 'PASS'
                    outputs[name] = out
                    if seed == spec['timing_seed']:
                        handles[name] = handle
                        primary = a, b
                except Exception as exc:
                    row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:1000])
                    result['correctness'].append(row)
                    result['status'] = 'correctness_failed'
                    return result
                result['correctness'].append(row)
            agreement = {'seed': seed, 'rtol': 0.02, 'atol': 0.02}
            try:
                torch.testing.assert_close(outputs['stages_3'], outputs['stages_2'],
                                           rtol=0.02, atol=0.02)
                agreement['status'] = 'PASS'
            except Exception as exc:
                agreement.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:1000])
                result['stage_agreement'].append(agreement)
                result['status'] = 'correctness_failed'
                return result
            result['stage_agreement'].append(agreement)
        result['compiler'] = {name: trusted._capture(folder, name, handles[name])
                              for name in ('stages_3', 'stages_2')}
        a, b = primary
        order = [f"aa_{spec['aa_first']}",
                 f"aa_{'stages_2' if spec['aa_first'] == 'stages_3' else 'stages_3'}",
                 'comparison']
        result['phase_order'] = order
        result['telemetry']['before'] = prior.telemetry()
        for name in order:
            offset = {'comparison': 0, 'aa_stages_3': 101, 'aa_stages_2': 102}[name]
            config = prior.CONFIG_3 if name in ('comparison', 'aa_stages_3') else prior.CONFIG_2
            result['telemetry'][f'before_{name}'] = prior.telemetry()
            start = time.perf_counter_ns()
            phase, summary = prior.timed_phase(kernel.launch, a, b, config,
                prior.CONFIG_2 if name == 'comparison' else config,
                rounds=ROUNDS if name == 'comparison' else AA_ROUNDS,
                iterations=ITERATIONS, warmups=WARMUPS,
                order_seed=spec['order_seed'] + offset,
                bootstrap_seed=spec['bootstrap_seed'] + offset, aa=name != 'comparison')
            phase['session_host_interval'] = {'start_perf_counter_ns': start,
                                              'end_perf_counter_ns': time.perf_counter_ns()}
            result['timing'][name] = phase
            result['summaries'][name] = summary
            result['telemetry'][f'after_{name}'] = prior.telemetry()
        result['telemetry']['after'] = prior.telemetry()
        result['status'] = 'complete'
    except Exception as exc:
        if result['status'] != 'correctness_failed':
            result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
        if hasattr(exc, 'partial_timing_blocks'):
            result['partial_timing_blocks'] = exc.partial_timing_blocks
    finally:
        (folder / 'result.json').write_bytes(canonical(result))
        (folder / 'manifest.json').write_bytes(canonical(manifest(folder, schema='prospective_k_depth_session_v1')))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'validate', 'run-session'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--session', choices=SESSIONS)
    parser.add_argument('--human-approved', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare()); return 0
    if not args.run_id:
        parser.error('--run-id required')
    if args.validate:
        print(json.dumps(validate(ROOT, args.run_id), indent=2)); return 0
    if not args.session or not args.human_approved:
        parser.error('--run-session requires --session and --human-approved')
    result = execute_session(ROOT, args.run_id, args.session, human_approved=True)
    print(json.dumps({'run_id': args.run_id, 'session': args.session,
                      'status': result['status']}, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
