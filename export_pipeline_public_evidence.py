"""Export manifested pipeline observations into a portable, allowlisted bundle.

This is CPU-only. It reads existing artifacts, validates them, and does not
modify any historical run. The output directory must not already exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

from dashboard_data import ROOT
import pipeline_stage_discovery as discovery
import pipeline_stage_profile_sweep as profile_study
import muse_pipeline_hypothesis_study as muse
import prospective_k_depth_test as heldout

DISCOVERY = '20261101T012329698902Z'
PROFILE = '20261101T014354664000Z'
MUSE = '20261101T015247349750Z'
HELDOUT = '20261101T020025480962Z'
METRICS = (
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'launch__registers_per_thread', 'launch__registers_per_thread_allocated',
    'launch__shared_mem_per_block', 'launch__shared_mem_per_block_allocated',
    'smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active',
    'smsp__warps_issue_stalled_barrier.avg.per_cycle_active',
    'smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active',
    'lts__t_bytes.sum.per_second', 'dram__bytes.sum.per_second',
    'profiler__replayer_passes',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def profile_row(path):
    lines = path.read_text().splitlines()
    pos = next(i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"'))
    rows = list(csv.reader(lines[pos:pos + 3]))
    if len(rows) != 3 or len(rows[0]) != len(rows[1]) or len(rows[0]) != len(rows[2]):
        raise ValueError('Nsight CSV table malformed')
    keys, units, values = rows
    columns = dict(zip(keys, values))
    unit_map = dict(zip(keys, units))
    if columns['Kernel Name'] != 'matmul_kernel':
        raise ValueError('unexpected selected kernel')
    return {key: {'value': float(columns[key]), 'unit': unit_map[key]}
            for key in METRICS}


def timing_session(result, bootstrap_seed, source, sources):
    if result['status'] != 'complete':
        raise ValueError('incomplete timing session')
    sources[str(source.relative_to(ROOT))] = sha(source.read_bytes())
    manifest_path = source.with_name('manifest.json')
    sources[str(manifest_path.relative_to(ROOT))] = sha(manifest_path.read_bytes())
    phases = {}
    for name, phase in result['timing'].items():
        offset = {'comparison': 0, 'aa_stages_3': 101, 'aa_stages_2': 102}[name]
        summary = result['summaries'][name]
        phases[name] = {'method': phase['method'],
                        'bootstrap_seed': bootstrap_seed + offset,
                        'rounds': [{'order': row['order'], 'baseline_ms': row['baseline_ms'],
                                    'candidate_ms': row['candidate_ms'],
                                    'ratio': row['baseline_over_candidate'],
                                    'blocks': row['blocks']} for row in phase['rounds']],
                        'median_paired_ratio': summary['median_paired_ratio'],
                        'bootstrap_interval': summary['bootstrap_median_ratio_95pct']}
        if name == 'comparison':
            phases[name].update(baseline_median_ms=summary['baseline_median_ms'],
                                candidate_median_ms=summary['candidate_median_ms'],
                                stage3_favorable_pairs=summary['baseline_faster_pairs'])
        else:
            phases[name]['aa_median_absolute_deviation'] = phase['aa_variability'][
                'median_absolute_ratio_deviation']
            phases[name]['aa_max_absolute_deviation'] = phase['aa_variability'][
                'max_absolute_ratio_deviation']
    return {'status': result['status'], 'shape_mnk': result.get('spec', {}).get('shape_mnk',
            [512, 3072, 768]), 'correctness': [
            {'stage': row['configuration'], 'seed': row['seed'], 'status': row['status'],
             'rtol': row['rtol'], 'atol': row['atol']} for row in result['correctness']],
            'stage_agreement': result.get('stage_agreement', []),
            'phase_order': result['phase_order'], 'telemetry': result['telemetry'],
            'phases': phases}


def export(root=ROOT, output=Path('public_pipeline_evidence')):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(x != 'complete' for x in discovery.validate(root, DISCOVERY)['sessions'].values()):
        raise ValueError('discovery incomplete')
    profiles = profile_study.verify(root, PROFILE)
    if profiles['status'] != 'verified' or any(x['status'] != 'completed' for x in profiles['captures']):
        raise ValueError('profile sweep incomplete')
    if muse.validate(root, MUSE)['request_status'] != 'accepted_structurally':
        raise ValueError('Muse forecast invalid')
    if any(x != 'complete' for x in heldout.validate(root, HELDOUT)['sessions'].values()):
        raise ValueError('held-out campaign incomplete')
    source_hashes = {}
    data = {'schema': 'pipeline_public_evidence_v1',
            'run_ids': {'discovery': DISCOVERY, 'profile': PROFILE,
                        'muse': MUSE, 'heldout': HELDOUT},
            'discovery': {}, 'profiles': [], 'compiler': {}, 'heldout': {}}
    for name, spec in discovery.SESSION_SPECS.items():
        path = discovery.folder(root, DISCOVERY) / 'sessions' / name / 'result.json'
        result = discovery.validate_session(root, DISCOVERY, name)
        data['discovery'][name] = timing_session(result, spec['bootstrap_seed'], path, source_hashes)
    for name, spec in heldout.SESSIONS.items():
        path = heldout.base(root, HELDOUT) / 'sessions' / name / 'result.json'
        result = heldout.validate_session(root, HELDOUT, name)
        data['heldout'][name] = timing_session(result, spec['bootstrap_seed'], path, source_hashes)
    profile_base = profile_study.base(root, PROFILE)
    for index, stage in enumerate((3, 2, 2, 3), 1):
        folder = profile_base / 'captures' / f'{index:02d}_stage_{stage}'
        meta = json.loads((folder / 'metadata.json').read_bytes())
        raw = folder / 'raw.csv'
        for path in (folder / 'metadata.json', folder / 'manifest.json', raw):
            source_hashes[str(path.relative_to(root))] = sha(path.read_bytes())
        if meta['status'] != 'completed' or meta['exit_code'] != 0:
            raise ValueError('failed profile')
        metrics = profile_row(raw)
        data['profiles'].append({'capture': index, 'stage': stage, 'status': meta['status'],
            'correctness': 'PASS' if 'correctness PASS' in raw.read_text() else 'UNKNOWN',
            'replay_passes': int(metrics['profiler__replayer_passes']['value']),
            'metrics': metrics, 'before': meta['before'], 'after': meta['after'],
            'started_utc': meta['started_utc'], 'ended_utc': meta['ended_utc'],
            'cache_control': 'none', 'clock_control': 'none',
            'warnings': [line for line in raw.read_text().splitlines() if line.startswith('==WARNING==')]})
    setup = muse.folder(root, MUSE) / 'setup/study.json'
    audit = muse.folder(root, MUSE) / 'attempt/result/audit.json'
    for path in (setup, audit, audit.with_name('manifest.json')):
        source_hashes[str(path.relative_to(root))] = sha(path.read_bytes())
    frozen = json.loads(setup.read_bytes())
    response = json.loads(audit.read_bytes())
    pred = response['parsed_response']['prediction']
    data['prediction'] = {'direction': pred['direction'], 'point_ratio': pred['point_ratio'],
        'plausible_range': pred['plausible_range'], 'target_shape_mnk': pred['target_shape_mnk'],
        'preferred_explanation': response['parsed_response']['preferred_label'],
        'alternatives': [x['label'] for x in response['parsed_response']['explanations']],
        'chosen_observation_id': response['parsed_response']['chosen_observation_id'],
        'reference_ratio': frozen['evaluation_rule']['reference_ratio'],
        'deadband': frozen['evaluation_rule']['deadband_absolute_ratio'],
        'audit_sha256': sha(audit.read_bytes()),
        'prompt_sha256': frozen['prompt_sha256'], 'evidence_sha256': frozen['evidence_sha256'],
        'status': response['status'], 'model': frozen['model'],
        'reasoning_effort': frozen['request_options']['reasoning']['effort']}
    compiler = discovery.validate_session(root, DISCOVERY, 'confirmation')['compiler']
    compiler_folder = discovery.folder(root, DISCOVERY) / 'sessions/confirmation'
    output.mkdir(parents=True)
    for stage in ('stages_3', 'stages_2'):
        record = compiler[stage]
        data['compiler'][stage] = {'registers_per_thread': record['registers_per_thread'],
            'shared_memory_bytes': record['shared_memory_bytes'], 'spills': record['spills'],
            'ttir_sha256': record['artifacts']['ttir']['sha256'],
            'ttgir_sha256': record['artifacts']['ttgir']['sha256'],
            'ptx_sha256': record['artifacts']['ptx']['sha256']}
        for kind in ('ptx', 'ttgir'):
            original = compiler_folder / f'{stage}.{kind}'
            source_hashes[str(original.relative_to(root))] = sha(original.read_bytes())
            # Strip source-location prefixes; retain instructions and line numbers.
            clean = re.sub(r'/home/[^"\n]*/trusted_two_slab_kernel\.py',
                           'trusted_two_slab_kernel.py', original.read_text())
            if '/home/' in clean or '/Users/' in clean:
                raise ValueError('compiler text has an unstripped local path')
            (output / f'{stage}.{kind}').write_text(clean)
    (output / 'evidence.json').write_bytes(canonical(data))
    (output / 'verify.py').write_bytes((root / 'verify_pipeline_public_evidence.py').read_bytes())
    (output / 'README.md').write_text(
        '# Portable pipeline-stage evidence\n\n'
        'Run `python verify.py .` from this directory. Python standard library only; '
        'no local `evaluations/` directory, GPU, or API key is needed. '
        '`evidence.json` contains all raw timing pairs and A/A pairs, selected Nsight '
        'metric rows with units, the immutable forecast fields, held-out results, and '
        'original artifact SHA-256 identifiers. The included PTX and TTGIR are '
        'source-path sanitized copies; their hashes differ from the originals. '
        '`manifest.json` verifies the exported files. This verifies internal arithmetic '
        'and bundle integrity, not independent reproduction of GPU measurements or '
        'authenticity of unavailable original artifacts. Nsight counters are replay '
        'measurements, and the forecast outcome is inconclusive.\n')
    files = {p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    (output / 'manifest.json').write_bytes(canonical({
        'schema': 'pipeline_public_evidence_v1', 'files': files,
        'source_artifacts': source_hashes}))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='public_pipeline_evidence')
    args = parser.parse_args()
    print(export(ROOT, Path(args.output)))
