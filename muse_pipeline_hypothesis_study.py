"""Prospective Muse hardware hypothesis: offline freeze and one-attempt request path.

Offline modes read and hash existing evidence only. No GPU or Muse request occurs
unless --request-once is separately authorized.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import statistics

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import (API_BASE_URL, _get, canonical, digest, inspect_response,
                         require_complete_text)
import pipeline_stage_discovery as discovery
import pipeline_stage_profile_sweep as profiling

SECTION = Path('evaluations/muse-pipeline-hypothesis/v1')
VERSION = 'muse_pipeline_hypothesis_v1'
PROMPT_VERSION = 'prospective_hardware_hypothesis_v1'
MODEL = 'muse-wonder-1.0'
REQUEST_OPTIONS = {'max_output_tokens': 16000, 'reasoning': {'effort': 'high'}}
TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 0
DISCOVERY_RUN = '20261101T012329698902Z'
PROFILE_RUN = '20261101T014354664000Z'
TARGET_SHAPE = [512, 3072, 1536]
DEADBAND = 0.04
PROFILE_METRICS = tuple(profiling.METRICS)
SOURCES = ('muse_pipeline_hypothesis_study.py', 'pipeline_stage_discovery.py',
           'pipeline_stage_profile_sweep.py', 'trusted_two_slab_kernel.py',
           'trusted_two_slab_experiment.py', 'benchmark_core.py',
           'smoke_aa.py', 'reference_precision.py', 'stage_reduction_experiment.py')
DOCS_READ = ('docs/research_discovery_sprint.md',
             'docs/pipeline_stage_discovery_protocol.md',
             'docs/pipeline_stage_mechanism_analysis.md',
             'docs/pipeline_stage_profile_analysis.md')
FORBIDDEN_INPUT = (r'\bH[1-4]\b', r'pipeline[- ]overlap hypothesis',
                   r'preferred hardware explanation', r'persists or grows',
                   r'512\s*[×x]\s*3072\s*[×x]\s*1536\s*(?:measured|result)')


def now():
    return datetime.now(timezone.utc).isoformat()


def run_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def folder(root, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r'\d{8}T\d{12}Z', identifier):
        raise ValueError('invalid prospective-study ID')
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / 'evaluations', root / 'evaluations/muse-pipeline-hypothesis', base)):
        raise ValueError('linked prospective-study path')
    return base / identifier


def write(stage, name, value, schema):
    stage.mkdir(parents=True, exist_ok=False)
    (stage / name).write_bytes(canonical(value))
    (stage / 'manifest.json').write_bytes(canonical(manifest(stage, schema=schema)))


def read(stage, name, schema):
    read_manifest(stage, schema=schema)
    return json.loads((stage / name).read_bytes())


def hashes(root, names):
    result = {}
    for name in names:
        path = Path(root) / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing or linked source: {name}')
        result[name] = sha(path.read_bytes())
    return result


def artifact_ref(root, path, *, manifest_path=None):
    path = Path(path)
    ref = {'path': str(path.relative_to(root)), 'sha256': sha(path.read_bytes())}
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        ref['manifest_path'] = str(manifest_path.relative_to(root))
        ref['manifest_sha256'] = sha(manifest_path.read_bytes())
    return ref


def parse_profile_csv(path, metric_units):
    lines = path.read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')), None)
    if start is None:
        raise ValueError('missing Nsight CSV header')
    rows = list(csv.reader(io.StringIO('\n'.join(lines[start:]))))
    if len(rows) != 3 or len(rows[0]) != len(rows[1]) or len(rows[0]) != len(rows[2]):
        raise ValueError('expected exactly one selected Nsight row')
    values = dict(zip(rows[0], rows[2]))
    units = dict(zip(rows[0], rows[1]))
    if values.get('Kernel Name') != 'matmul_kernel' or values.get('Grid Size') != '(8, 24, 1)':
        raise ValueError('wrong profiled kernel or output grid')
    measured = {}
    for name in PROFILE_METRICS:
        expected_unit = metric_units[name]['unit']
        if units.get(name) != expected_unit:
            raise ValueError(f'Nsight unit mismatch: {name}')
        value = float(values[name])
        if not math.isfinite(value):
            raise ValueError(f'nonfinite Nsight metric: {name}')
        measured[name] = value
    warnings = [line for line in lines[:start] if 'WARNING' in line]
    return measured, warnings


def evidence_package(root=ROOT):
    """Build neutral observations from manifested data, without human hypotheses."""
    root = Path(root).resolve()
    if set(discovery.validate(root, DISCOVERY_RUN)['sessions'].values()) != {'complete'}:
        raise ValueError('discovery run incomplete')
    if profiling.verify(root, PROFILE_RUN)['status'] != 'verified':
        raise ValueError('profiling sweep incomplete')
    discovery_setup = discovery.load(root, DISCOVERY_RUN)
    profile_plan = profiling.load(root, PROFILE_RUN)
    observations = []
    ratios = []
    for session in ('screening_1', 'screening_2', 'confirmation'):
        stage = discovery.folder(root, DISCOVERY_RUN) / 'sessions' / session
        result = discovery.validate_session(root, DISCOVERY_RUN, session, discovery_setup)
        comparison = result['timing']['comparison']
        summary = result['summaries']['comparison']
        ratios.append(summary['median_paired_ratio'])
        observations.append({
            'id': 'TIMING_' + session.upper(), 'kind': 'paired_cuda_event_timing',
            'source': artifact_ref(root, stage / 'result.json', manifest_path=stage / 'manifest.json'),
            'measurement': {
                'stage': result['spec']['stage'], 'shape_mnk': discovery_setup['shape_mnk'],
                'dtype': discovery_setup['dtype'], 'input_seed': result['spec']['input_seed'],
                'correctness_passes': len(result['correctness']),
                'correctness_failures': sum(x['status'] == 'FAIL' for x in result['correctness']),
                'paired_rounds': len(comparison['rounds']),
                'launches_per_timed_block': result['spec']['iterations'],
                'stage3_median_cuda_event_ms': summary['baseline_median_ms'],
                'stage2_median_cuda_event_ms': summary['candidate_median_ms'],
                'stage3_over_stage2_median_paired_ratio': summary['median_paired_ratio'],
                'paired_median_bootstrap_95pct': summary['bootstrap_median_ratio_95pct'],
                'stage3_faster_pairs': summary['baseline_faster_pairs'],
                'stage2_faster_pairs': summary['candidate_faster_pairs'],
                'aa_stage3_median_absolute_ratio_deviation': result['timing']['aa_stages_3']['aa_variability']['median_absolute_ratio_deviation'],
                'aa_stage2_median_absolute_ratio_deviation': result['timing']['aa_stages_2']['aa_variability']['median_absolute_ratio_deviation'],
                'aa_stage3_max_absolute_ratio_deviation': result['timing']['aa_stages_3']['aa_variability']['max_absolute_ratio_deviation'],
                'aa_stage2_max_absolute_ratio_deviation': result['timing']['aa_stages_2']['aa_variability']['max_absolute_ratio_deviation'],
                'comparison_pairs': [{'round': row['round'], 'order': row['order'],
                                      'stage3_cuda_event_ms': row['baseline_ms'],
                                      'stage2_cuda_event_ms': row['candidate_ms'],
                                      'stage3_over_stage2': row['baseline_over_candidate']}
                                     for row in comparison['rounds']],
                'telemetry_before': result['telemetry'].get('before'),
                'telemetry_after': result['telemetry'].get('after')},
            'limits': ['CUDA events bracket eager Python launch blocks, not a pure single-kernel duration.',
                       'Bootstrap interval is descriptive; A/A samples and host timing remain in the source audit.',
                       'Clock and temperature snapshots do not control the GPU state.']})
    for index, stage_num in enumerate(profiling.ORDER, 1):
        stage = profiling.base(root, PROFILE_RUN) / 'captures' / f'{index:02d}_stage_{stage_num}'
        profiling.read_manifest(stage, schema='pipeline_stage_profile_capture_v1')
        record = json.loads((stage / 'metadata.json').read_bytes())
        if record.get('status') != 'completed' or record.get('stage') != stage_num:
            raise ValueError('profile capture not completed')
        metrics, warnings = parse_profile_csv(stage / 'raw.csv', profile_plan['metric_units'])
        observations.append({
            'id': f'PROFILE_{index}', 'kind': 'nsight_compute_replay',
            'source': artifact_ref(root, stage / 'raw.csv', manifest_path=stage / 'manifest.json'),
            'measurement': {'variant_stages': stage_num, 'capture_order': index,
                            'shape_mnk': discovery_setup['shape_mnk'],
                            'metrics': metrics, 'metric_units': {k: profile_plan['metric_units'][k]['unit'] for k in PROFILE_METRICS},
                            'clock_temperature_before': record['before'],
                            'clock_temperature_after': record['after'],
                            'warnings': warnings},
            'limits': ['One selected launch and three replay passes; caches and clocks were not controlled.',
                       'Nsight replay timing is not the independent CUDA-event timing result.',
                       'Stall counters are broad and do not time one PTX wait instruction.']})
    compiler_stage = discovery.folder(root, DISCOVERY_RUN) / 'sessions' / 'confirmation'
    compiler = discovery.validate_session(root, DISCOVERY_RUN, 'confirmation', discovery_setup)['compiler']
    artifacts = {name: {kind: artifact_ref(root, compiler_stage / info['path'],
                                           manifest_path=compiler_stage / 'manifest.json')
                        for kind, info in compiler[name]['artifacts'].items()}
                 for name in ('stages_3', 'stages_2')}
    if not all(set(artifacts[name]) >= {'ttir', 'ttgir', 'llir', 'ptx', 'cubin'} for name in artifacts):
        raise ValueError('missing core compiler artifacts')
    ptx = {name: (compiler_stage / compiler[name]['artifacts']['ptx']['path']).read_text()
           for name in ('stages_3', 'stages_2')}
    observations.append({
        'id': 'COMPILER', 'kind': 'saved_compiler_artifacts', 'source': artifacts,
        'measurement': {
            'ttir_bytes_identical': artifacts['stages_3']['ttir']['sha256'] == artifacts['stages_2']['ttir']['sha256'],
            'ttgir_bytes_identical': artifacts['stages_3']['ttgir']['sha256'] == artifacts['stages_2']['ttgir']['sha256'],
            'llir_bytes_identical': artifacts['stages_3']['llir']['sha256'] == artifacts['stages_2']['llir']['sha256'],
            'ptx_bytes_identical': artifacts['stages_3']['ptx']['sha256'] == artifacts['stages_2']['ptx']['sha256'],
            'cubin_bytes_identical': artifacts['stages_3']['cubin']['sha256'] == artifacts['stages_2']['cubin']['sha256'],
            'resources': {name: {key: compiler[name][key] for key in
                                  ('registers_per_thread', 'shared_memory_bytes', 'spills')}
                          for name in ('stages_3', 'stages_2')},
            'ptx_static_sites': {name: {
                'cp_async_global_to_shared': ptx[name].count('cp.async.cg.shared.global'),
                'cp_async_commit_group': ptx[name].count('cp.async.commit_group'),
                'mma_sync': ptx[name].count('mma.sync.aligned'),
                'bar_sync': ptx[name].count('bar.sync'),
                'cp_async_wait_group_operands': re.findall(r'cp\.async\.wait_group\s+(\d+);', ptx[name])}
                for name in ('stages_3', 'stages_2')}},
        'limits': ['PTX site counts are static, not executed instruction counts.',
                   'Compiler differences are observations, not a causal explanation.',
                   'No manifested SASS artifact is available.']})
    reference = statistics.median(ratios)
    package = {'schema': 'muse_pipeline_observations_v1',
               'device': 'NVIDIA GeForce RTX 4060 Ti',
               'measured_shape_mnk': discovery_setup['shape_mnk'],
               'target_shape_mnk': TARGET_SHAPE,
               'target_status': 'not_measured_in_this_experiment',
               'configuration': {'block_m': 64, 'block_n': 128, 'block_k': 32,
                                 'num_warps': 4, 'compared_num_stages': [3, 2]},
               'reference_ratio': {'definition': 'median of the three session-level stage3/stage2 paired medians',
                                   'value': reference, 'orientation': 'stage3_latency_over_stage2_latency',
                                   'unit': 'dimensionless',
                                   'source_observation_ids': ['TIMING_SCREENING_1', 'TIMING_SCREENING_2', 'TIMING_CONFIRMATION']},
               'source_hashes': {name: discovery_setup['source_hashes'][name] for name in
                                 ('trusted_two_slab_kernel.py', 'benchmark_core.py',
                                  'reference_precision.py', 'stage_reduction_experiment.py')},
               'observations': observations,
               'global_limits': ['The target K=1536 has no result in the supplied evidence.',
                                 'The target changes K, input footprint, and compiler specialization while holding M/N output grid fixed.',
                                 'Profiler counters are replay observations and do not establish causality.',
                                 'Historical timing and profiling use different measurement boundaries.']}
    check_no_leak(package)
    return package


def check_no_leak(value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False)
    for pattern in FORBIDDEN_INPUT:
        if re.search(pattern, raw, re.I):
            raise ValueError(f'human hypothesis or result leaked: {pattern}')
    if 'target_status' in value and value['target_status'] != 'not_measured_in_this_experiment':
        raise ValueError('target measurement leaked')


def response_example():
    return {'chosen_observation_id': '<one supplied observation ID>',
            'why_interesting': '<scientific reason, with observation IDs rather than uncited historical numbers>',
            'measured_facts': [{'claim': '<recorded fact>', 'observation_ids': ['<ID>']}],
            'explanations': [
                {'label': '<short label>', 'mechanism': '<testable explanation>',
                 'support_ids': ['<ID>'], 'weakening_ids': ['<ID>'],
                 'falsifier': '<observation that would weaken or falsify it>'},
                {'label': '<different label>', 'mechanism': '<competing explanation>',
                 'support_ids': ['<ID>'], 'weakening_ids': ['<ID>'],
                 'falsifier': '<observation that would weaken or falsify it>'}],
            'preferred_label': '<one explanation label>',
            'prediction': {'target_shape_mnk': TARGET_SHAPE,
                           'ratio_orientation': 'stage3_latency_over_stage2_latency',
                           'direction': '<lower|near|higher relative to reference_ratio>',
                           'point_ratio': None, 'plausible_range': None,
                           'range_rationale': '<why a range is or is not defensible>'},
            'discriminating_experiment': '<minimum controlled experiment>',
            'falsifying_outcome': '<result that would weaken the preferred explanation>',
            'uncertainties': ['<measurement or inference limit>']}


def prompt_for(package):
    return (
        'You are an independent GPU research analyst. Use only the verified observations in EVIDENCE. '
        'No timing result exists for target_shape_mnk. Before any new GPU measurement, choose the most '
        'interesting unexpected measured performance observation, explain why it matters, and propose '
        'at least two competing hardware or measurement explanations. For each, cite supplied observation '
        'IDs that support and weaken it, and give a falsifier. Distinguish measured facts from inference. '
        'Do not treat profiler counters or compiler differences as causal proof. Predict the stage-3/stage-2 '
        'median paired CUDA-event latency ratio for target_shape_mnk, using the reference_ratio orientation. '
        'State whether it moves lower, stays near, or moves higher than the current approximately 0.95 '
        'reference. Give a point estimate and plausible range if defensible; otherwise use null and explain '
        'why no quantitative range is justified. Choose your prediction now, before the target is measured. '
        'Propose the minimum discriminating follow-up experiment and a result that would falsify your '
        'preferred explanation. Do not run code or request GPU work. Return exactly one JSON object with '
        'the fields shown in SCHEMA. Replace all angle-bracket placeholders. Use exact supplied observation '
        'IDs in every ID array; do not invent IDs. Put recorded numerical facts in measured_facts with '
        'supporting observation IDs rather than introducing uncited historical numbers in prose. '
        'A human will review scientific support independently.\nSCHEMA\n'
        + json.dumps(response_example(), sort_keys=True) + '\nEVIDENCE\n'
        + json.dumps(package, sort_keys=True, allow_nan=False))


def parse_response(raw, package):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON: {exc.msg}') from exc
    fields = {'chosen_observation_id', 'why_interesting', 'measured_facts', 'explanations',
              'preferred_label', 'prediction', 'discriminating_experiment',
              'falsifying_outcome', 'uncertainties'}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError('response fields mismatch')
    ids = {row['id'] for row in package['observations']}
    if not isinstance(data['chosen_observation_id'], str) or data['chosen_observation_id'] not in ids:
        raise ValueError('unknown chosen observation ID')
    for name in ('why_interesting', 'preferred_label', 'discriminating_experiment', 'falsifying_outcome'):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError(f'missing {name}')
    if not isinstance(data['measured_facts'], list) or not data['measured_facts']:
        raise ValueError('missing measured facts')
    for fact in data['measured_facts']:
        if (not isinstance(fact, dict) or set(fact) != {'claim', 'observation_ids'}
                or not isinstance(fact['claim'], str) or not fact['claim'].strip()):
            raise ValueError('malformed measured fact')
        _ids(fact['observation_ids'], ids, required=True)
    if not isinstance(data['explanations'], list) or not 2 <= len(data['explanations']) <= 4:
        raise ValueError('expected two to four explanations')
    labels = []
    for item in data['explanations']:
        if (not isinstance(item, dict) or set(item) != {'label', 'mechanism', 'support_ids',
                                                        'weakening_ids', 'falsifier'}):
            raise ValueError('explanation fields mismatch')
        for key in ('label', 'mechanism', 'falsifier'):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f'missing explanation {key}')
        _ids(item['support_ids'], ids, required=True)
        _ids(item['weakening_ids'], ids, required=False)
        labels.append(item['label'])
    if len(set(labels)) != len(labels) or data['preferred_label'] not in labels:
        raise ValueError('invalid preferred explanation')
    prediction = data['prediction']
    if (not isinstance(prediction, dict) or set(prediction) != {'target_shape_mnk',
            'ratio_orientation', 'direction', 'point_ratio', 'plausible_range', 'range_rationale'}
            or prediction['target_shape_mnk'] != TARGET_SHAPE
            or prediction['ratio_orientation'] != 'stage3_latency_over_stage2_latency'
            or prediction['direction'] not in ('lower', 'near', 'higher')
            or not isinstance(prediction['range_rationale'], str)
            or not prediction['range_rationale'].strip()):
        raise ValueError('prediction fields mismatch')
    point = prediction['point_ratio']
    bounds = prediction['plausible_range']
    if point is not None and (type(point) not in (int, float) or not math.isfinite(point) or point <= 0):
        raise ValueError('invalid prospective point ratio')
    if bounds is not None:
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in bounds)
                or bounds[0] > bounds[1] or (point is not None and not bounds[0] <= point <= bounds[1])):
            raise ValueError('invalid prospective ratio range')
    if not isinstance(data['uncertainties'], list) or not data['uncertainties'] or any(
            not isinstance(x, str) or not x.strip() for x in data['uncertainties']):
        raise ValueError('missing uncertainties')
    return data


def _ids(values, valid, *, required):
    if (not isinstance(values, list) or (required and not values)
            or any(not isinstance(x, str) or x not in valid for x in values)
            or len(set(values)) != len(values)):
        raise ValueError('invalid observation IDs')


def evaluation_rule(reference):
    return {'schema': 'prospective_ratio_evaluation_v1',
            'target_shape_mnk': TARGET_SHAPE,
            'reference_ratio': reference, 'reference_definition': 'median of three frozen K=768 session medians',
            'ratio_orientation': 'stage3_latency_over_stage2_latency',
            'deadband_absolute_ratio': DEADBAND,
            'lower_if': f'ratio < {reference - DEADBAND:.12f}',
            'near_if': f'{reference - DEADBAND:.12f} <= ratio <= {reference + DEADBAND:.12f}',
            'higher_if': f'ratio > {reference + DEADBAND:.12f}',
            'primary_result': 'first preregistered, correctness-passing K=1536 paired session; no favorable-run selection',
            'minimum_method': {'paired_rounds': 14, 'launches_per_block': 200,
                               'aa_rounds_per_variant': 20, 'warmups': 10,
                               'paired_bootstrap_draws': 2000,
                               'correctness_rtol': 0.02, 'correctness_atol': 0.02},
            'scoring': ['Report point absolute error and whether the stated range covers the observed median, when supplied.',
                        'Score the direction only if the target result passes correctness and its bootstrap interval lies entirely in one deadband category.',
                        'If the category is not separated from A/A variation or the interval crosses a boundary, mark direction inconclusive.',
                        'A fresh matched K=768 control is required for mechanism interpretation, not for arithmetic scoring of the prospective K=1536 ratio.',
                        'Keep model response and attempted-run failures fixed before any K=1536 timing; human review remains separate.']}


def expected_setup(root, identifier, created_utc):
    package = evidence_package(root)
    prompt = prompt_for(package)
    check_no_leak({'target_status': package['target_status'], 'prompt': prompt})
    return {'schema': VERSION, 'run_id': identifier, 'created_utc': created_utc,
            'model': MODEL, 'api_base_url': API_BASE_URL,
            'request_options': REQUEST_OPTIONS, 'timeout_seconds': TIMEOUT_SECONDS,
            'max_retries': MAX_RETRIES, 'prompt_version': PROMPT_VERSION,
            'prompt_sha256': digest(prompt.encode()),
            'evidence_sha256': digest(canonical(package)),
            'source_hashes': hashes(root, SOURCES),
            'documentation_sha256': hashes(root, DOCS_READ),
            'discovery_run': DISCOVERY_RUN, 'profile_run': PROFILE_RUN,
            'evaluation_rule': evaluation_rule(package['reference_ratio']['value']),
            'authorization_state': 'not_authorized_at_preparation'}


def request_spec(setup, prompt):
    return {'schema': 'muse_pipeline_request_v1', 'run_id': setup['run_id'],
            'model': MODEL, 'api_base_url': API_BASE_URL,
            'request_options': REQUEST_OPTIONS, 'timeout_seconds': TIMEOUT_SECONDS,
            'max_retries': MAX_RETRIES, 'prompt_version': PROMPT_VERSION,
            'prompt_sha256': setup['prompt_sha256'],
            'evidence_sha256': setup['evidence_sha256'],
            'authorization_state': 'not_authorized_in_offline_preview',
            'prompt': prompt}


def prepare(root=ROOT):
    identifier = run_id()
    package = evidence_package(root)
    prompt = prompt_for(package)
    setup = expected_setup(root, identifier, now())
    base = folder(root, identifier)
    write(base / 'setup', 'study.json', setup, 'muse_pipeline_setup_v1')
    write(base / 'evidence', 'package.json', package, 'muse_pipeline_evidence_v1')
    write(base / 'request', 'spec.json', request_spec(setup, prompt), 'muse_pipeline_request_v1')
    return identifier


def load(root, identifier):
    base = folder(root, identifier)
    setup = read(base / 'setup', 'study.json', 'muse_pipeline_setup_v1')
    package = read(base / 'evidence', 'package.json', 'muse_pipeline_evidence_v1')
    spec = read(base / 'request', 'spec.json', 'muse_pipeline_request_v1')
    if (not isinstance(setup.get('created_utc'), str)
            or setup != expected_setup(root, identifier, setup['created_utc'])
            or package != evidence_package(root)
            or spec != request_spec(setup, prompt_for(package))):
        raise ValueError('prospective study, source, or request changed')
    return setup, package, spec


def request_once(client, root, identifier, *, human_authorized=False):
    if human_authorized is not True:
        raise ValueError('separate human authorization required')
    setup, package, spec = load(root, identifier)
    base = folder(root, identifier)
    with (base / 'request.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = base / 'attempt'
        if target.exists():
            return 'already_attempted'
        write(target / 'request', 'request.json', {'schema': 'muse_pipeline_attempt_request_v1',
              'created_utc': now(), 'request_spec_sha256': digest(canonical(spec)),
              'request_spec': spec, 'explicit_human_authorization': True},
              'muse_pipeline_attempt_request_v1')
        audit = {'schema': 'muse_pipeline_response_audit_v1', 'created_utc': now(),
                 'request_spec_sha256': digest(canonical(spec)),
                 'raw_response': None, 'parsed_response': None,
                 'response_model': None, 'response_id': None, 'response_metadata': None}
        called = False
        try:
            response = client.responses.create(model=MODEL, input=spec['prompt'], **REQUEST_OPTIONS)
            called = True
            audit['raw_response'], audit['response_metadata'] = inspect_response(response)
            audit['response_model'] = _get(response, 'model')
            audit['response_id'] = _get(response, 'id')
            require_complete_text(audit['raw_response'], audit['response_metadata'])
            if audit['response_model'] != MODEL:
                raise ValueError('response model mismatch')
            audit['parsed_response'] = parse_response(audit['raw_response'], package)
            if load(root, identifier)[2] != spec:
                raise ValueError('request source changed')
            audit['status'] = 'accepted_structurally'
        except Exception as exc:
            audit['status'] = 'rejected'
            audit['rejection_kind'] = 'invalid_response' if called else 'request_error'
            audit['rejection'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        write(target / 'result', 'audit.json', audit, 'muse_pipeline_response_audit_v1')
        return audit['status']


def validate(root, identifier):
    setup, package, spec = load(root, identifier)
    target = folder(root, identifier) / 'attempt'
    if not target.exists():
        status = 'not_requested'
    else:
        record = read(target / 'request', 'request.json', 'muse_pipeline_attempt_request_v1')
        if (record.get('request_spec') != spec or record.get('request_spec_sha256') != digest(canonical(spec))
                or record.get('explicit_human_authorization') is not True):
            raise ValueError('attempt request mismatch')
        result = target / 'result'
        if not result.exists():
            status = 'incomplete_attempt'
        else:
            audit = read(result, 'audit.json', 'muse_pipeline_response_audit_v1')
            if audit.get('request_spec_sha256') != digest(canonical(spec)):
                raise ValueError('audit request mismatch')
            if audit.get('status') == 'accepted_structurally':
                require_complete_text(audit['raw_response'], audit['response_metadata'])
                if (audit['response_model'] != MODEL or
                        audit['parsed_response'] != parse_response(audit['raw_response'], package)):
                    raise ValueError('accepted response mismatch')
            elif (audit.get('status') != 'rejected' or audit.get('parsed_response') is not None
                  or audit.get('rejection_kind') not in ('invalid_response', 'request_error')
                  or not audit.get('rejection')):
                raise ValueError('rejected response mismatch')
            status = audit['status']
    return {'run_id': identifier, 'request_status': status,
            'model': setup['model'], 'prompt_sha256': setup['prompt_sha256'],
            'evidence_sha256': setup['evidence_sha256'],
            'reference_ratio': package['reference_ratio']['value'],
            'evaluation_rule': setup['evaluation_rule']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--validate', action='store_true')
    modes.add_argument('--preview', action='store_true')
    modes.add_argument('--request-once', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--human-authorized', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare(ROOT)); return 0
    if not args.run_id:
        parser.error('--run-id required')
    if args.validate:
        print(json.dumps(validate(ROOT, args.run_id), indent=2)); return 0
    if args.preview:
        print(json.dumps(load(ROOT, args.run_id)[2], indent=2)); return 0
    if not args.human_authorized:
        parser.error('--request-once requires --human-authorized')
    from openai import OpenAI
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        parser.error('MODEL_API_KEY required for a separately authorized request')
    client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    status = request_once(client, ROOT, args.run_id, human_authorized=True)
    print(status)
    return 0 if status == 'accepted_structurally' else 1


if __name__ == '__main__':
    raise SystemExit(main())
