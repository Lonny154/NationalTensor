"""Verify the portable pipeline-stage evidence bundle with Python stdlib only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def interval(values, seed):
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return [draws[49], draws[1949]]


def verify(directory):
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), 'bundle missing or linked')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    need(manifest['schema'] == 'pipeline_public_evidence_v1', 'manifest schema')
    need(set(manifest['files']) == {p.name for p in directory.iterdir()} - {'manifest.json'},
         'bundle file set')
    for name, digest in manifest['files'].items():
        path = directory / name
        need('/' not in name and not path.is_symlink() and path.is_file(), 'missing or linked file')
        need(sha(path.read_bytes()) == digest, f'file hash: {name}')
    for name, digest in manifest['source_artifacts'].items():
        need(isinstance(name, str) and not name.startswith('/') and '..' not in Path(name).parts
             and re.fullmatch(r'[0-9a-f]{64}', digest), 'source identifier or hash')
    data = json.loads((directory / 'evidence.json').read_bytes())
    need(data['schema'] == 'pipeline_public_evidence_v1', 'evidence schema')
    need(data['run_ids'] == {'discovery': '20261101T012329698902Z',
                             'profile': '20261101T014354664000Z',
                             'muse': '20261101T015247349750Z',
                             'heldout': '20261101T020025480962Z'}, 'run IDs')
    need(data['prediction']['direction'] == 'lower' and
         data['prediction']['reference_ratio'] == 0.953120870845799 and
         data['prediction']['deadband'] == 0.04 and
         data['prediction']['point_ratio'] is None and
         data['prediction']['plausible_range'] is None, 'frozen forecast fields')
    for group, expected in (('discovery', {'screening_1', 'screening_2', 'confirmation'}),
                            ('heldout', {'target_k1536', 'control_k768'})):
        need(set(data[group]) == expected, f'{group} sessions')
        for name, session in data[group].items():
            need(session['status'] == 'complete', 'incomplete session')
            need(session['shape_mnk'][:2] == [512, 3072] and
                 session['shape_mnk'][2] == (1536 if name == 'target_k1536' else 768),
                 'shape')
            need(all(row['status'] == 'PASS' for row in session['correctness']),
                 'correctness')
            for phase_name, phase in session['phases'].items():
                rounds = phase['rounds']
                method = phase['method']
                need(method['clock'] == 'cuda_event' and method['preallocated_outputs'] is True
                     and method['balanced_order'] is True and
                     method['iterations_per_side_per_round'] in (100, 200), 'timing method')
                need(len(rounds) == (14 if name in ('confirmation', 'target_k1536', 'control_k768')
                                     and phase_name == 'comparison' else
                                     20), 'round count')
                orders = [r['order'] for r in rounds]
                need(orders.count(['baseline', 'candidate']) == len(rounds) // 2 and
                     orders.count(['candidate', 'baseline']) == len(rounds) // 2,
                     'unbalanced order')
                ratios = []
                for row in rounds:
                    a, b = row['baseline_ms'], row['candidate_ms']
                    need(a > 0 and b > 0 and abs(a / b - row['ratio']) < 1e-12,
                         'raw paired ratio')
                    need(set(row['blocks']) == {'baseline', 'candidate'}, 'raw timing blocks')
                    for side in ('baseline', 'candidate'):
                        block = row['blocks'][side]
                        need(abs(block['cuda_event_ms_per_launch'] - row[f'{side}_ms']) < 1e-12
                             and abs(block['cuda_event_ms_per_block'] /
                                     method['iterations_per_side_per_round'] - row[f'{side}_ms']) < 1e-12
                             and block['host_submission_ms_per_block'] >= 0
                             and block['host_wall_ms_per_block'] >= 0,
                             'timing block arithmetic')
                    ratios.append(row['ratio'])
                need(abs(statistics.median(ratios) - phase['median_paired_ratio']) < 1e-12,
                     'median paired ratio')
                need(interval(ratios, phase['bootstrap_seed']) == phase['bootstrap_interval'],
                     'bootstrap interval')
                if phase_name == 'comparison':
                    need(phase['stage3_favorable_pairs'] == sum(r < 1 for r in ratios),
                         'favorable count')
                    need(abs(statistics.median(r['baseline_ms'] for r in rounds) -
                             phase['baseline_median_ms']) < 1e-12 and
                         abs(statistics.median(r['candidate_ms'] for r in rounds) -
                             phase['candidate_median_ms']) < 1e-12, 'latency medians')
                else:
                    need(abs(statistics.median(abs(r - 1) for r in ratios) -
                             phase['aa_median_absolute_deviation']) < 1e-12,
                         'A/A variation')
    controls = [data['discovery'][name]['phases']['comparison']['median_paired_ratio']
                for name in ('screening_1', 'screening_2', 'confirmation')]
    need(statistics.median(controls) == data['prediction']['reference_ratio'],
         'reference arithmetic')
    target = data['heldout']['target_k1536']['phases']['comparison']
    control = data['heldout']['control_k768']['phases']['comparison']
    need(target['median_paired_ratio'] < control['median_paired_ratio'], 'heldout direction')
    lo = data['prediction']['reference_ratio'] - data['prediction']['deadband']
    hi = data['prediction']['reference_ratio'] + data['prediction']['deadband']
    need(lo < target['median_paired_ratio'] < hi and
         lo < target['bootstrap_interval'][0] <= target['bootstrap_interval'][1] < hi,
         'frozen near category')
    need(abs(target['median_paired_ratio'] - data['prediction']['reference_ratio']) <
         max(data['heldout']['target_k1536']['phases'][p]['aa_median_absolute_deviation']
             for p in ('aa_stages_3', 'aa_stages_2')), 'inconclusive A/A gate')
    need([p['stage'] for p in data['profiles']] == [3, 2, 2, 3], 'profile order')
    for profile in data['profiles']:
        need(profile['status'] == 'completed' and profile['replay_passes'] == 3 and
             profile['correctness'] == 'PASS', 'profile status')
        need(profile['metrics']['sm__warps_active.avg.pct_of_peak_sustained_active']['unit'] == '%',
             'occupancy unit')
        need(all(math.isfinite(v['value']) and v['value'] >= 0 and isinstance(v['unit'], str)
                 for v in profile['metrics'].values()), 'profile metric value or unit')
    compiler = data['compiler']
    need(compiler['stages_3']['ttir_sha256'] == compiler['stages_2']['ttir_sha256'],
         'TTIR identity')
    for stage, count, commits, waits, barriers in [('stages_3', 36, 12, [4, 0], 7),
                                                   ('stages_2', 24, 8, [0, 0], 6)]:
        ptx = (directory / f'{stage}.ptx').read_text()
        need(not re.search(r'/home/|/Users/', ptx), 'local path in PTX')
        need(len(re.findall(r'\bcp\.async\.(?:ca|cg)\.shared\.global\b', ptx)) == count,
             'PTX copy sites')
        need(ptx.count('cp.async.commit_group') == commits, 'PTX commit sites')
        need([int(x) for x in re.findall(r'cp\.async\.wait_group\s+(\d+)', ptx)] == waits,
             'PTX wait operands')
        need(ptx.count('bar.sync') == barriers and ptx.count('mma.sync.aligned') == 64,
             'PTX static barriers or MMA sites')
    raw = json.dumps(data)
    need(not re.search(r'/home/|/Users/|GPU-[0-9a-fA-F-]{16}|MODEL_API_KEY', raw),
         'sensitive value in public data')
    return {'schema': data['schema'], 'timing_sessions': 5, 'profile_captures': 4,
            'prediction_category': 'near', 'prediction_score': 'inconclusive',
            'files_verified': len(manifest['files'])}


if __name__ == '__main__':
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    print(json.dumps(verify(folder), indent=2))
