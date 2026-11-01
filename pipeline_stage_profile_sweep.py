"""One-attempt, manifested Nsight A-B-B-A sweep for the reviewed stage experiment.

Preparation and verification are CPU-only. Execution requires an explicit flag and
uses stage_reduction_experiment.py --profile-one; no benchmark or Muse path exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import canonical
import pipeline_stage_discovery as discovery

DISCOVERY_RUN = '20261101T012329698902Z'
SECTION = Path('evaluations/pipeline-stage-mechanism/v1')
QUERY_FILES = {
    'profiling': Path('/tmp/national_tensor_ncu_metrics_20261101.txt'),
    'launch': Path('/tmp/national_tensor_ncu_launch_20261101.txt'),
    'stats': Path('/tmp/national_tensor_ncu_stats_20261101.txt'),
}
METRICS = (
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'launch__registers_per_thread',
    'launch__registers_per_thread_allocated',
    'launch__shared_mem_per_block',
    'launch__shared_mem_per_block_allocated',
    'smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active',
    'smsp__warps_issue_stalled_barrier.avg.per_cycle_active',
    'smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active',
    'lts__t_bytes.sum.per_second',
    'dram__bytes.sum.per_second',
    'profiler__replayer_passes',
)
ORDER = (3, 2, 2, 3)
KERNEL_SELECTOR = 'regex:.*matmul_kernel.*'


def utc():
    return datetime.now(timezone.utc).isoformat()


def run_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def base(root, identifier):
    if not discovery.re.fullmatch(r'\d{8}T\d{12}Z', identifier):
        raise ValueError('invalid mechanism run ID')
    return Path(root).resolve() / SECTION / identifier


def query_units():
    found = {}
    query_hashes = {}
    for category, path in QUERY_FILES.items():
        raw = path.read_bytes()
        query_hashes[category] = sha(raw)
        for line in raw.decode('utf-8').splitlines():
            for name in METRICS:
                if line.startswith(name + ' '):
                    parts = line[len(name):].split()
                    if len(parts) < 2:
                        raise ValueError(f'missing unit: {name}')
                    if name in found:
                        raise ValueError(f'duplicate metric: {name}')
                    found[name] = {'collection': category, 'type': parts[0], 'unit': parts[1]}
    if set(found) != set(METRICS):
        raise ValueError(f'unavailable metrics: {sorted(set(METRICS) - set(found))}')
    return found, query_hashes


def command(ncu_path, stage):
    return [ncu_path, '--target-processes', 'application-only',
            '--replay-mode', 'kernel', '--launch-count', '1', '--launch-skip', '0',
            '--cache-control', 'none', '--clock-control', 'none',
            '--kernel-name', KERNEL_SELECTOR, '--metrics', ','.join(METRICS),
            '--csv', '--page', 'raw', '--print-units', 'base',
            str(ROOT / '.venv/bin/python'),
            'stage_reduction_experiment.py', '--profile-one', str(stage)]


def software():
    import torch
    import triton
    ncu_path = shutil.which('ncu')
    if not ncu_path:
        raise ValueError('ncu missing')
    version = subprocess.run([ncu_path, '--version'], capture_output=True, text=True,
                             check=True, timeout=15).stdout.strip()
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version',
                          '--format=csv,noheader'], capture_output=True, text=True,
                         check=True, timeout=15).stdout.strip()
    if discovery.GPU not in smi:
        raise ValueError('expected GPU unavailable')
    return {'python': sys.version.split()[0], 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda, 'triton': triton.__version__,
            'ncu': version, 'ncu_path': ncu_path, 'nvidia_smi': smi}


def original_hashes(root):
    folder = discovery.folder(root, DISCOVERY_RUN)
    return {session: {
        name: sha((folder / 'sessions' / session / name).read_bytes())
        for name in ('result.json', 'manifest.json')}
        for session in discovery.SESSION_SPECS}


def prepare(root):
    state = discovery.validate(root, DISCOVERY_RUN)
    if set(state['sessions'].values()) != {'complete'}:
        raise ValueError('discovery confirmation incomplete')
    units, query_hashes = query_units()
    environment = software()
    identifier = run_id()
    folder = base(root, identifier)
    setup = folder / 'setup'
    setup.mkdir(parents=True, exist_ok=False)
    plan = {'schema': 'pipeline_stage_profile_plan_v1', 'run_id': identifier,
            'created_utc': utc(), 'discovery_run': DISCOVERY_RUN,
            'discovery_hashes': original_hashes(root),
            'discovery_source_hashes': discovery.load(root, DISCOVERY_RUN)['source_hashes'],
            'operator_sha256': sha(Path(__file__).read_bytes()),
            'profile_entry_sha256': sha((Path(root) / 'stage_reduction_experiment.py').read_bytes()),
            'metric_query_commands': {
                'profiling': 'ncu --query-metrics-mode all --devices 0',
                'launch': 'ncu --query-metrics-collection launch --query-metrics-mode all --devices 0',
                'stats': 'ncu --query-metrics-collection stats --query-metrics-mode all --devices 0'},
            'metric_query_sha256': query_hashes, 'metric_units': units,
            'metrics': list(METRICS), 'software': environment,
            'capture_order': list(ORDER), 'kernel_selector': KERNEL_SELECTOR,
            'commands': [command(environment['ncu_path'], stage) for stage in ORDER],
            'one_selected_launch_per_capture': True,
            'historical_timing_replaced': False}
    (setup / 'plan.json').write_bytes(canonical(plan))
    (setup / 'manifest.json').write_bytes(canonical(
        manifest(setup, schema='pipeline_stage_profile_plan_v1')))
    return identifier


def load(root, identifier):
    folder = base(root, identifier)
    read_manifest(folder / 'setup', schema='pipeline_stage_profile_plan_v1')
    plan = json.loads((folder / 'setup/plan.json').read_bytes())
    if (plan.get('schema') != 'pipeline_stage_profile_plan_v1'
            or plan.get('run_id') != identifier or plan.get('discovery_run') != DISCOVERY_RUN
            or plan.get('capture_order') != list(ORDER)
            or plan.get('metrics') != list(METRICS)
            or plan.get('kernel_selector') != KERNEL_SELECTOR
            or plan.get('commands') != [command(plan['software']['ncu_path'], s) for s in ORDER]
            or plan.get('operator_sha256') != sha(Path(__file__).read_bytes())
            or plan.get('profile_entry_sha256') != sha((Path(root) / 'stage_reduction_experiment.py').read_bytes())
            or plan.get('discovery_hashes') != original_hashes(root)
            or plan.get('discovery_source_hashes') != discovery.load(root, DISCOVERY_RUN)['source_hashes']):
        raise ValueError('frozen profiling plan or source mismatch')
    if set(discovery.validate(root, DISCOVERY_RUN)['sessions'].values()) != {'complete'}:
        raise ValueError('discovery records invalid')
    return plan


def snapshot():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,pstate',
                            '--format=csv,noheader,nounits'], capture_output=True,
                           text=True, timeout=15, check=True)
        return {'status': 'available', 'raw': p.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {'status': 'unavailable', 'error': type(exc).__name__}


def execute(root, identifier, *, approved=False):
    if approved is not True:
        raise ValueError('human authorization required')
    plan = load(root, identifier)
    captures = base(root, identifier) / 'captures'
    captures.mkdir(exist_ok=False)
    for index, (stage, cmd) in enumerate(zip(ORDER, plan['commands']), 1):
        folder = captures / f'{index:02d}_stage_{stage}'
        folder.mkdir(exist_ok=False)
        record = {'schema': 'pipeline_stage_profile_capture_v1', 'run_id': identifier,
                  'index': index, 'stage': stage, 'command': cmd,
                  'started_utc': utc(), 'before': snapshot(), 'status': 'attempted'}
        try:
            with (folder / 'raw.csv').open('wb') as out, (folder / 'raw.log').open('wb') as err:
                finished = subprocess.run(cmd, cwd=root, stdout=out, stderr=err,
                                          check=False)
            record['exit_code'] = finished.returncode
            record['status'] = 'completed' if finished.returncode == 0 else 'failed'
        except OSError as exc:
            record['status'] = 'request_error'
            record['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            record['ended_utc'] = utc()
            record['after'] = snapshot()
            (folder / 'metadata.json').write_bytes(canonical(record))
            (folder / 'manifest.json').write_bytes(canonical(
                manifest(folder, schema='pipeline_stage_profile_capture_v1')))
    return verify(root, identifier)


def verify(root, identifier):
    plan = load(root, identifier)
    captures = base(root, identifier) / 'captures'
    if not captures.is_dir():
        return {'run_id': identifier, 'status': 'not_run', 'captures': []}
    results = []
    expected = [f'{i:02d}_stage_{s}' for i, s in enumerate(ORDER, 1)]
    if sorted(p.name for p in captures.iterdir()) != expected:
        raise ValueError('capture set mismatch')
    for index, stage in enumerate(ORDER, 1):
        folder = captures / expected[index - 1]
        read_manifest(folder, schema='pipeline_stage_profile_capture_v1')
        record = json.loads((folder / 'metadata.json').read_bytes())
        if (record.get('run_id') != identifier or record.get('index') != index
                or record.get('stage') != stage or record.get('command') != plan['commands'][index - 1]
                or record.get('status') not in ('completed', 'failed', 'request_error')):
            raise ValueError('capture identity mismatch')
        results.append({'index': index, 'stage': stage, 'status': record['status'],
                        'exit_code': record.get('exit_code')})
    return {'run_id': identifier, 'status': 'verified', 'captures': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--execute', action='store_true')
    group.add_argument('--verify', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--human-approved', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare(ROOT)); return 0
    if not args.run_id:
        parser.error('--run-id required')
    if args.execute:
        if not args.human_approved:
            parser.error('--execute requires --human-approved')
        result = execute(ROOT, args.run_id, approved=True)
    else:
        result = verify(ROOT, args.run_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
