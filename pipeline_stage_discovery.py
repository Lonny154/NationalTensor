"""Offline preparation and human-gated confirmation of the two-slab stage inversion.

Importing this module, --prepare, --dry-run, --compiler-inventory, and --validate
do not import Torch or execute a GPU kernel. Each --run-session is one fresh
process invocation and requires explicit approval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import canonical

SECTION = Path('evaluations/pipeline-stage-discovery/v1')
ORIGINAL = Path('evaluations/stage-reduction/20261028T140056186673Z')
ORIGINAL_RESULT_SHA = '98ff37ba5f1a62cc4de6b5823bd39b44878c5641175c777867bc2a2b375c3305'
ORIGINAL_PROFILE_SHA = 'fa339f6e8aa25ab0af994f88c2d1c06817677bf77eae04e67841b0ed6b8f6dff'
KERNEL_SHA = '98188d25c726c2301cd81e36e6dfd22f02db5f3acbc3d05d385d63ce7cffd06d'
GPU = 'NVIDIA GeForce RTX 4060 Ti'
SHAPES = ((512, 3072, 768), (509, 3071, 769), (65, 129, 33))
SEEDS = (42, 43)
CONFIG_3 = {'block_m': 64, 'block_n': 128, 'block_k': 32, 'num_warps': 4, 'num_stages': 3}
CONFIG_2 = {**CONFIG_3, 'num_stages': 2}
TOLERANCE = {'rtol': 0.02, 'atol': 0.02}
SOURCE_FILES = ('pipeline_stage_discovery.py', 'benchmark_core.py', 'smoke_aa.py',
                'trusted_two_slab_kernel.py', 'trusted_two_slab_experiment.py',
                'reference_precision.py', 'stage_reduction_experiment.py')
BOOTSTRAP_DRAWS = 2000
SESSION_SPECS = {
    'screening_1': {'stage': 'screening', 'input_seed': 42, 'rounds': 20,
                    'aa_rounds': 20, 'iterations': 100, 'warmups': 10,
                    'order_seed': 43, 'aa_first': 'stages_3', 'bootstrap_seed': 1301},
    'screening_2': {'stage': 'screening', 'input_seed': 43, 'rounds': 20,
                    'aa_rounds': 20, 'iterations': 100, 'warmups': 10,
                    'order_seed': 44, 'aa_first': 'stages_2', 'bootstrap_seed': 1302},
    'confirmation': {'stage': 'independent_confirmation', 'input_seed': 44,
                     'rounds': 14, 'aa_rounds': 20, 'iterations': 200,
                     'warmups': 10, 'order_seed': 45, 'aa_first': 'stages_2',
                     'bootstrap_seed': 1303},
}
PROFILE_METRICS = (
    'launch__registers_per_thread', 'launch__shared_mem_per_block',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active',
    'smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active',
    'smsp__warps_issue_stalled_barrier.avg.per_cycle_active',
    'dram__bytes.sum.per_second', 'lts__t_bytes.sum.per_second',
    'profiler__replayer_passes',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def new_run_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def folder(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'\d{8}T\d{12}Z', run_id):
        raise ValueError('invalid run ID')
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / 'evaluations', root / 'evaluations/pipeline-stage-discovery', base)):
        raise ValueError('linked experiment path')
    return base / run_id


def write(stage, name, value, schema):
    stage.mkdir(parents=True, exist_ok=False)
    (stage / name).write_bytes(canonical(value))
    (stage / 'manifest.json').write_bytes(canonical(manifest(stage, schema=schema)))


def read(stage, name, schema):
    read_manifest(stage, schema=schema)
    return json.loads((stage / name).read_bytes())


def source_hashes(root):
    root = Path(root)
    files = {}
    for name in SOURCE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'missing or linked source: {name}')
        files[name] = sha(path.read_bytes())
    if files['trusted_two_slab_kernel.py'] != KERNEL_SHA:
        raise ValueError('reviewed two-slab kernel changed')
    return files


def original_evidence(root):
    base = Path(root) / ORIGINAL
    result_path, profile_path = base / 'result.json', base / 'profile_summary.json'
    if (not result_path.is_file() or not profile_path.is_file()
            or result_path.is_symlink() or profile_path.is_symlink()
            or sha(result_path.read_bytes()) != ORIGINAL_RESULT_SHA
            or sha(profile_path.read_bytes()) != ORIGINAL_PROFILE_SHA):
        raise ValueError('original stage-reduction evidence changed')
    result = json.loads(result_path.read_bytes())
    profile = json.loads(profile_path.read_bytes())
    if (result.get('status') != 'screening_complete_unconfirmed'
            or result.get('configurations') != {'stages_3': CONFIG_3, 'stages_2': CONFIG_2}
            or result.get('workload', {}).get('shape_mnk') != list(SHAPES[0])
            or result.get('source_identity', {}).get('trusted_module_sha256') != KERNEL_SHA
            or len(result.get('correctness', [])) != 12
            or any(row.get('status') != 'PASS' for row in result['correctness'])
            or profile.get('status') != 'descriptive_profile_complete'):
        raise ValueError('original evidence content mismatch')
    return result, profile


def expected_setup(root, run_id, created_utc):
    original_evidence(root)
    return {'schema': 'pipeline_stage_discovery_setup_v1', 'run_id': run_id,
            'created_utc': created_utc, 'gpu': GPU, 'shape_mnk': list(SHAPES[0]),
            'dtype': 'float16', 'configurations': {'stages_3': CONFIG_3, 'stages_2': CONFIG_2},
            'correctness_shapes': [list(shape) for shape in SHAPES],
            'correctness_seeds': list(SEEDS), 'tolerance': TOLERANCE,
            'source_hashes': source_hashes(root),
            'original_result_sha256': ORIGINAL_RESULT_SHA,
            'original_profile_sha256': ORIGINAL_PROFILE_SHA,
            'session_specs': SESSION_SPECS,
            'ratio_orientation': 'stage_3_latency_over_stage_2_latency',
            'bootstrap_draws': BOOTSTRAP_DRAWS,
            'profile_plan': {'status': 'conditional_after_independent_confirmation',
                             'capture_order': ['stages_3', 'stages_2', 'stages_2', 'stages_3'],
                             'metrics_if_supported': list(PROFILE_METRICS),
                             'separate_from_cuda_event_timing': True},
            'gpu_execution_authorized_at_preparation': False}


def prepare(root):
    run_id = new_run_id()
    setup = expected_setup(root, run_id, now())
    write(folder(root, run_id) / 'setup', 'study.json', setup,
          'pipeline_stage_discovery_setup_v1')
    return run_id


def load(root, run_id):
    setup = read(folder(root, run_id) / 'setup', 'study.json',
                 'pipeline_stage_discovery_setup_v1')
    if (not isinstance(setup.get('created_utc'), str)
            or setup != expected_setup(root, run_id, setup['created_utc'])):
        raise ValueError('frozen setup or source mismatch')
    return setup


def balanced_orders(rounds, seed):
    if type(rounds) is not int or rounds < 2 or rounds % 2 or type(seed) is not int or seed < 0:
        raise ValueError('invalid balanced-order specification')
    rng = random.Random(seed)
    orders = []
    for _ in range(rounds // 2):
        pair = [['baseline', 'candidate'], ['candidate', 'baseline']]
        rng.shuffle(pair)
        orders.extend(pair)
    return orders


def bootstrap_interval(ratios, seed):
    if (not isinstance(ratios, list) or len(ratios) < 2 or type(seed) is not int
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in ratios)):
        raise ValueError('invalid bootstrap samples')
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                   for _ in range(BOOTSTRAP_DRAWS))
    return [draws[49], draws[1949]]


def phase_summary(result, bootstrap_seed):
    ratios = [row['baseline_over_candidate'] for row in result['rounds']]
    return {'baseline_median_ms': statistics.median(result['samples_ms']['baseline']),
            'candidate_median_ms': statistics.median(result['samples_ms']['candidate']),
            'median_paired_ratio': statistics.median(ratios),
            'bootstrap_median_ratio_95pct': bootstrap_interval(ratios, bootstrap_seed),
            'baseline_faster_pairs': sum(r < 1 for r in ratios),
            'candidate_faster_pairs': sum(r > 1 for r in ratios),
            'ties': sum(r == 1 for r in ratios), 'pair_count': len(ratios)}


def validate_phase(result, rounds, iterations, seed, *, aa, bootstrap_seed, warmups=None):
    from benchmark_core import _statistics
    if (not isinstance(result, dict) or result.get('mode') != ('aa' if aa else 'comparison')
            or not isinstance(result.get('rounds'), list) or len(result['rounds']) != rounds
            or result.get('method', {}).get('iterations_per_side_per_round') != iterations
            or result['method'].get('order_seed') != seed
            or result['method'].get('clock') != 'cuda_event'
            or result['method'].get('preallocated_outputs') is not True
            or result['method'].get('balanced_order') is not True):
        raise ValueError('phase mode, count, or timing method mismatch')
    if warmups is not None and (result['method'].get('initial_warmups_per_side') != warmups
                                or result['method'].get('warmups_per_side_per_round') != warmups
                                or result['method'].get('prime_calls_per_side') != 1):
        raise ValueError('warmup or compilation-prime mismatch')
    expected_orders = balanced_orders(rounds, seed)
    samples = {'baseline': [], 'candidate': []}
    ratios = []
    for index, row in enumerate(result['rounds']):
        if (row.get('round') != index + 1 or row.get('order') != expected_orders[index]
                or set(row.get('blocks', {})) != {'baseline', 'candidate'}):
            raise ValueError('round order or diagnostic blocks mismatch')
        for side in ('baseline', 'candidate'):
            value = row.get(f'{side}_ms')
            block = row['blocks'][side]
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    or not isinstance(block, dict)
                    or any(type(block.get(key)) not in (int, float)
                           or not math.isfinite(block[key]) or block[key] < 0 for key in
                           ('cuda_event_ms_per_block', 'cuda_event_ms_per_launch',
                            'host_submission_ms_per_block', 'host_wall_ms_per_block'))
                    or not math.isclose(block['cuda_event_ms_per_launch'], value, rel_tol=1e-9)
                    or not math.isclose(block['cuda_event_ms_per_block'] / iterations,
                                        value, rel_tol=1e-9)):
                raise ValueError('missing or malformed timing sample')
            samples[side].append(value)
        ratio = row['baseline_ms'] / row['candidate_ms']
        if not math.isclose(row.get('baseline_over_candidate', -1), ratio, rel_tol=1e-9):
            raise ValueError('paired ratio orientation mismatch')
        ratios.append(ratio)
    if result.get('samples_ms') != samples:
        raise ValueError('raw samples mismatch')
    for key, values in [('baseline_ms', samples['baseline']),
                        ('candidate_ms', samples['candidate']), ('paired_ratios', ratios)]:
        if result.get(key) != _statistics(values):
            raise ValueError('reported phase statistics mismatch')
    if not math.isclose(result.get('ratio_of_medians', -1),
                        statistics.median(samples['baseline']) / statistics.median(samples['candidate']),
                        rel_tol=1e-9):
        raise ValueError('ratio of medians mismatch')
    if aa:
        deviations = [abs(r - 1) for r in ratios]
        expected = {'median_absolute_ratio_deviation': statistics.median(deviations),
                    'max_absolute_ratio_deviation': max(deviations)}
        if result.get('aa_variability') != expected:
            raise ValueError('A/A variation mismatch')
    elif 'aa_variability' in result:
        raise ValueError('comparison mislabeled as A/A')
    return phase_summary(result, bootstrap_seed)


def session_spec(setup, session):
    if session not in SESSION_SPECS:
        raise ValueError('invalid session')
    return setup['session_specs'][session]


def validate_session(root, run_id, session, setup=None):
    setup = setup or load(root, run_id)
    result = read(folder(root, run_id) / 'sessions' / session, 'result.json',
                  'pipeline_stage_discovery_session_v1')
    spec = session_spec(setup, session)
    if (result.get('schema') != 'pipeline_stage_discovery_session_v1'
            or result.get('run_id') != run_id or result.get('session') != session
            or result.get('spec') != spec or result.get('source_hashes') != setup['source_hashes']
            or result.get('ratio_orientation') != setup['ratio_orientation']
            or result.get('status') not in ('complete', 'correctness_failed', 'failed')
            or not isinstance(result.get('correctness'), list)):
        raise ValueError('session identity mismatch')
    checks = result['correctness']
    expected = {(stage, shape, seed) for stage in ('stages_3', 'stages_2')
                for shape in map(tuple, setup['correctness_shapes'])
                for seed in setup['correctness_seeds']}
    if session == 'confirmation':
        expected |= {(stage, tuple(setup['shape_mnk']), spec['input_seed'])
                     for stage in ('stages_3', 'stages_2')}
    seen = set()
    for row in checks:
        key = (row.get('configuration'), tuple(row.get('shape_mnk', [])), row.get('seed'))
        if key not in expected or key in seen or row.get('status') not in ('PASS', 'FAIL'):
            raise ValueError('incorrect or repeated correctness case')
        required_config = setup['configurations'][row['configuration']]
        if (row.get('config') != required_config or row.get('rtol') != TOLERANCE['rtol']
                or row.get('atol') != TOLERANCE['atol']):
            raise ValueError('correctness configuration or tolerance mismatch')
        seen.add(key)
    if result['status'] == 'complete':
        if seen != expected or any(row['status'] != 'PASS' for row in checks):
            raise ValueError('timing without complete correctness gate')
        timing = result.get('timing')
        if not isinstance(timing, dict) or set(timing) != {'aa_stages_3', 'aa_stages_2', 'comparison'}:
            raise ValueError('incomplete timing phases')
        for name in ('aa_stages_3', 'aa_stages_2', 'comparison'):
            phase = timing[name]
            aa = name != 'comparison'
            rounds = spec['aa_rounds'] if aa else spec['rounds']
            seed = spec['order_seed'] + {'aa_stages_3': 101, 'aa_stages_2': 102,
                                         'comparison': 0}[name]
            summary = validate_phase(phase, rounds, spec['iterations'], seed,
                                     aa=aa, warmups=spec['warmups'],
                                     bootstrap_seed=spec['bootstrap_seed'] +
                                     {'aa_stages_3': 101, 'aa_stages_2': 102,
                                      'comparison': 0}[name])
            if result.get('summaries', {}).get(name) != summary:
                raise ValueError('phase summary mismatch')
        if result.get('phase_order') != ([f'aa_{spec["aa_first"]}',
                                          f'aa_{"stages_2" if spec["aa_first"] == "stages_3" else "stages_3"}',
                                          'comparison']):
            raise ValueError('session phase order mismatch')
        compiler = result.get('compiler', {})
        stage = folder(root, run_id) / 'sessions' / session
        if set(compiler) != {'stages_3', 'stages_2'}:
            raise ValueError('missing compiler capture')
        for name, capture in compiler.items():
            if not isinstance(capture, dict) or not isinstance(capture.get('artifacts'), dict):
                raise ValueError('malformed compiler capture')
            for artifact in capture.get('artifacts', {}).values():
                path = stage / artifact['path']
                if (Path(artifact['path']).name != artifact['path'] or path.is_symlink()
                        or not path.is_file() or sha(path.read_bytes()) != artifact['sha256']
                        or len(path.read_bytes()) != artifact.get('size_bytes')):
                    raise ValueError('compiler artifact mismatch')
    elif result['status'] == 'correctness_failed':
        if not any(row['status'] == 'FAIL' for row in checks) or result.get('timing'):
            raise ValueError('correctness failure did not block timing')
    elif not result.get('error'):
        raise ValueError('failed session lacks error')
    return result


def validate(root, run_id):
    setup = load(root, run_id)
    base = folder(root, run_id)
    sessions = {}
    ratios = {}
    for session in SESSION_SPECS:
        stage = base / 'sessions' / session
        if stage.exists():
            result = validate_session(root, run_id, session, setup)
            sessions[session] = result['status']
            if result['status'] == 'complete':
                ratios[session] = result['summaries']['comparison']['median_paired_ratio']
        else:
            sessions[session] = 'not_run'
    if sessions['confirmation'] != 'not_run' and any(
            sessions[name] != 'complete' for name in ('screening_1', 'screening_2')):
        raise ValueError('confirmation without both screenings')
    consistency = {'status': 'awaiting_sessions', 'median_paired_ratios': ratios}
    if len(ratios) == len(SESSION_SPECS):
        consistency = {'status': 'descriptive_only', 'median_paired_ratios': ratios,
                       'same_direction_below_one': all(value < 1 for value in ratios.values()),
                       'range': [min(ratios.values()), max(ratios.values())],
                       'confirmation_minus_screening_median': ratios['confirmation'] -
                       statistics.median((ratios['screening_1'], ratios['screening_2']))}
    return {'run_id': run_id, 'sessions': sessions, 'between_session': consistency,
            'ratio_orientation': setup['ratio_orientation'],
            'gpu_execution_authorized_at_preparation': False}


def compiler_inventory(root, run_id):
    """Offline comparison of saved original compiler artifacts; no compilation."""
    load(root, run_id)
    base = Path(root) / ORIGINAL
    result, _ = original_evidence(root)
    comparison = {}
    for kind in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin', 'sass'):
        paths = {stage: base / f'{stage}.{kind}' for stage in ('stages_3', 'stages_2')}
        if all(path.is_file() and not path.is_symlink() for path in paths.values()):
            content = {stage: path.read_bytes() for stage, path in paths.items()}
            comparison[kind] = {'available': True,
                                'sha256': {stage: sha(raw) for stage, raw in content.items()},
                                'identical_bytes': content['stages_3'] == content['stages_2']}
            if kind != 'cubin':
                comparison[kind]['static_markers'] = {stage: {
                    marker: raw.decode('utf-8', errors='replace').lower().count(marker)
                    for marker in ('cp.async', 'mbarrier', 'bar.sync', 'wait')}
                    for stage, raw in content.items()}
        else:
            comparison[kind] = {'available': False}
    return {'schema': 'pipeline_stage_compiler_inventory_v1', 'run_id': run_id,
            'source': str(ORIGINAL), 'artifacts': comparison,
            'resources': {name: {key: result['compiler'][name][key]
                                 for key in ('registers_per_thread', 'shared_memory_bytes', 'spills')}
                          for name in ('stages_3', 'stages_2')},
            'interpretation': 'Static markers are descriptive; compiled differences do not establish latency causality.'}


def dry_run(root, run_id):
    setup = load(root, run_id)
    status = validate(root, run_id)['sessions']
    return {'run_id': run_id, 'status': status, 'gpu_requests_made': 0,
            'sessions': setup['session_specs'],
            'estimated_timed_launches': {name: 2 * (spec['rounds'] + 2 * spec['aa_rounds']) * spec['iterations']
                                         for name, spec in SESSION_SPECS.items()},
            'profiling': setup['profile_plan'],
            'note': 'Warmups, correctness, compilation, and profiler work are additional; no GPU work occurs in dry-run.'}


def telemetry():
    try:
        process = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,pstate',
                                  '--format=csv,noheader,nounits'], capture_output=True,
                                 text=True, timeout=5, check=True)
        return {'status': 'available', 'raw': process.stdout.strip()[:300]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'status': 'unavailable', 'reason': type(exc).__name__}


def timed_phase(launch, a, b, baseline, candidate, *, rounds, iterations, warmups,
                order_seed, bootstrap_seed, aa):
    from benchmark_core import run_aa_calibration, run_paired
    from smoke_aa import DiagnosticTimer, attach_diagnostics
    timer = DiagnosticTimer()
    runner = run_aa_calibration if aa else run_paired
    kwargs = {'rounds': rounds, 'iterations': iterations, 'warmups': warmups,
              'seed': order_seed, 'timer': timer}
    try:
        result = (runner(launch, a, b, baseline, **kwargs) if aa else
                  runner(launch, a, b, baseline, candidate, **kwargs))
    except Exception as exc:
        exc.partial_timing_blocks = timer.blocks
        raise
    attach_diagnostics(result, timer.blocks)
    result['method']['timing_boundary'] = f'CUDA events around {iterations} Python launches; duration divided by {iterations}'
    result['method']['host_submission_boundary'] = 'perf_counter_ns around Python launch loop only'
    return result, phase_summary(result, bootstrap_seed)


def execute_session(root, run_id, session, *, human_approved=False,
                    confirmation_approved=False):
    """One human-approved GPU session; never called by offline modes."""
    if human_approved is not True or (session == 'confirmation' and confirmation_approved is not True):
        raise ValueError('separate human approval required for GPU session')
    setup = load(root, run_id)
    spec = session_spec(setup, session)
    states = validate(root, run_id)['sessions']
    if states[session] != 'not_run':
        raise ValueError('session already attempted; no retry')
    if session == 'confirmation' and any(states[name] != 'complete' for name in ('screening_1', 'screening_2')):
        raise ValueError('both screenings required before confirmation')
    stage = folder(root, run_id) / 'sessions' / session
    stage.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'pipeline_stage_discovery_session_v1', 'run_id': run_id,
              'session': session, 'created_utc': now(), 'status': 'incomplete',
              'spec': spec, 'source_hashes': setup['source_hashes'],
              'ratio_orientation': setup['ratio_orientation'],
              'human_approved': True, 'confirmation_approved': confirmation_approved,
              'correctness': [], 'timing': {}, 'summaries': {}, 'phase_order': [],
              'process_id': __import__('os').getpid(), 'telemetry': {}}
    try:
        import torch
        import stage_reduction_experiment as original
        import trusted_two_slab_experiment as trusted
        import trusted_two_slab_kernel

        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != GPU:
            raise ValueError('expected RTX 4060 Ti unavailable')
        identity = trusted._identity()
        if (identity['trusted_module_sha256'] != KERNEL_SHA
                or identity['trusted_implementation_sha256'] !=
                original_evidence(root)[0]['source_identity']['trusted_implementation_sha256']):
            raise ValueError('reviewed kernel identity mismatch')
        result['environment'] = original.environment()
        result['input_policy'] = 'trusted_two_slab_experiment._inputs; torch.randn FP16, contiguous'
        handles = {}
        primary = None
        cases = [(shape, seed) for shape in SHAPES for seed in SEEDS]
        if session == 'confirmation':
            cases.append((SHAPES[0], spec['input_seed']))
        for shape, seed in cases:
            a, b, reference, policy = trusted._inputs(shape, seed)
            for name, config in (('stages_3', CONFIG_3), ('stages_2', CONFIG_2)):
                row = {'configuration': name, 'shape_mnk': list(shape), 'seed': seed,
                       'config': config, 'reference_precision': policy,
                       'rtol': TOLERANCE['rtol'], 'atol': TOLERANCE['atol'],
                       'input_strides': {'a': list(a.stride()), 'b': list(b.stride())}}
                try:
                    out = torch.empty_like(reference)
                    handle = trusted_two_slab_kernel.launch(a, b, out, config)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out, reference, **TOLERANCE)
                    row['status'] = 'PASS'
                    if shape == SHAPES[0] and seed == spec['input_seed']:
                        handles[name] = handle
                        primary = (a, b)
                except Exception as exc:
                    row.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'[:1000])
                    result['correctness'].append(row)
                    result['status'] = 'correctness_failed'
                    return result
                result['correctness'].append(row)
        result['compiler'] = {name: trusted._capture(stage, name, handles[name])
                              for name in ('stages_3', 'stages_2')}
        a, b = primary
        phase_names = [f'aa_{spec["aa_first"]}',
                       f'aa_{"stages_2" if spec["aa_first"] == "stages_3" else "stages_3"}',
                       'comparison']
        result['phase_order'] = phase_names
        result['telemetry']['before'] = telemetry()
        launch = trusted_two_slab_kernel.launch
        previous_phase_end_ns = None
        for name in phase_names:
            config = CONFIG_3 if name == 'aa_stages_3' else CONFIG_2
            offset = {'aa_stages_3': 101, 'aa_stages_2': 102, 'comparison': 0}[name]
            result['telemetry'][f'before_{name}'] = telemetry()
            phase_start_ns = time.perf_counter_ns()
            phase, summary = timed_phase(launch, a, b,
                                         CONFIG_3 if name == 'comparison' else config,
                                         CONFIG_2 if name == 'comparison' else config,
                                         rounds=spec['rounds'] if name == 'comparison' else spec['aa_rounds'],
                                         iterations=spec['iterations'], warmups=spec['warmups'],
                                         order_seed=spec['order_seed'] + offset,
                                         bootstrap_seed=spec['bootstrap_seed'] + offset,
                                         aa=name != 'comparison')
            phase_end_ns = time.perf_counter_ns()
            phase['session_host_interval'] = {
                'start_perf_counter_ns': phase_start_ns,
                'end_perf_counter_ns': phase_end_ns,
                'gap_since_previous_phase_ms': None if previous_phase_end_ns is None
                else (phase_start_ns - previous_phase_end_ns) / 1e6,
                'intentional_idle_ms': 0}
            previous_phase_end_ns = phase_end_ns
            result['timing'][name] = phase
            result['summaries'][name] = summary
            result['telemetry'][f'after_{name}'] = telemetry()
        result['telemetry']['after'] = telemetry()
        result['status'] = 'complete'
    except Exception as exc:
        if result['status'] != 'correctness_failed':
            result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'[:1000]
        if hasattr(exc, 'partial_timing_blocks'):
            result['partial_timing_blocks'] = exc.partial_timing_blocks
    finally:
        (stage / 'result.json').write_bytes(canonical(result))
        (stage / 'manifest.json').write_bytes(canonical(
            manifest(stage, schema='pipeline_stage_discovery_session_v1')))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'dry-run', 'validate', 'compiler-inventory', 'run-session'):
        modes.add_argument('--' + name, action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--session', choices=SESSION_SPECS)
    parser.add_argument('--human-approved', action='store_true')
    parser.add_argument('--confirmation-approved', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare(ROOT)); return 0
    if not args.run_id:
        parser.error('--run-id required')
    if args.dry_run:
        print(json.dumps(dry_run(ROOT, args.run_id), indent=2)); return 0
    if args.validate:
        print(json.dumps(validate(ROOT, args.run_id), indent=2)); return 0
    if args.compiler_inventory:
        print(json.dumps(compiler_inventory(ROOT, args.run_id), indent=2)); return 0
    if not args.session or not args.human_approved:
        parser.error('--run-session requires --session and --human-approved')
    if args.session == 'confirmation' and not args.confirmation_approved:
        parser.error('confirmation additionally requires --confirmation-approved')
    result = execute_session(ROOT, args.run_id, args.session,
                             human_approved=True,
                             confirmation_approved=args.confirmation_approved)
    print(json.dumps({'run_id': args.run_id, 'session': args.session,
                      'status': result['status']}, indent=2))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
