"""Experiment 013: offline matched P03 model-capability feasibility study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

from dashboard_data import ROOT
from experiment_005_evidence import manifest, read_manifest, sha
from muse_debrief import (API_BASE_URL, _get, canonical, digest, inspect_response,
                         require_complete_text)
from muse_critic_value_study import (FIELDS, SECTION as SECTION_009, VERSION as VERSION_009,
                                     decision_prompt, load as load_009, parse_decision)

SECTION = Path('evaluations/muse-model-capability-study/v1')
SOURCE_RUN = '20261030T205425148094Z'
PROPOSAL_ID = 'P03'
VERSION = 'experiment_013_v1'
PROMPT_VERSION = 'decision_v1_citation_objects_v1'
VALIDATOR = 'experiment_009_decision_v1'
CONFIGS = {'A': {'name': 'Spark High', 'model': 'muse-spark-1.3'},
           'B': {'name': 'Wonder High', 'model': 'muse-wonder-1.0'}}
REQUEST_OPTIONS = {'max_output_tokens': 16000, 'reasoning': {'effort': 'high'}}
TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 0


def now():
    return datetime.now(timezone.utc).isoformat()


def folder(root, run_id):
    from muse_research_critic import STAMP
    if not isinstance(run_id, str) or not STAMP.fullmatch(run_id):
        raise ValueError('invalid Experiment 013 run ID')
    root = Path(root).resolve()
    base = root / SECTION
    if any(p.is_symlink() for p in (root / 'evaluations', root / 'evaluations/muse-model-capability-study', base)):
        raise ValueError('linked Experiment 013 path')
    return base / run_id


def write(stage, name, value, schema):
    stage.mkdir(parents=True, exist_ok=False)
    (stage / name).write_bytes(canonical(value))
    (stage / 'manifest.json').write_bytes(canonical(manifest(stage, schema=schema)))


def read(stage, name, schema):
    read_manifest(stage, schema=schema)
    return json.loads((stage / name).read_bytes())


def clarified_prompt(source):
    """Keep decision_v1 context/instructions; replace its empty citation example."""
    base = decision_prompt(source)
    before, schema_and_source = base.split('\nSCHEMA\n', 1)
    old_schema, source_part = schema_and_source.split('\nSOURCE\n', 1)
    example = json.loads(old_schema)
    evidence = source['validated_evidence']['005']
    if 'confirmation_status' not in evidence['observations']:
        raise ValueError('P03 confirmation-status observation missing')
    example['observed_evidence'] = 'Experiment 005 independent confirmation is missing.'
    example['evidence_references']['observed_evidence'] = [
        {'evidence_id': evidence['evidence_id'], 'observation_key': 'confirmation_status'}]
    instruction = (
        '\nCITATION OBJECT CLARIFICATION (decision_v1_citation_objects_v1): '
        'Each evidence_references value is an array of objects, never strings. '
        'Each object has exactly two string keys, evidence_id and observation_key. '
        'Copy an exact evidence_id and an exact observation key from the supplied '
        'validated_evidence. Put citations in the array for the particular response '
        'field whose historical claim they support; free-text mentions or colon-joined '
        'IDs do not count as citations. Use [] only for a field with no historical '
        'claim; observed_evidence must have at least one citation. The complete '
        'SCHEMA example below shows one actual P03 observation and its citation '
        'object. Replace example wording and references with your own assessment; '
        'do not cite an observation for claims it does not support.')
    return before + instruction + '\nSCHEMA\n' + json.dumps(example, sort_keys=True) + '\nSOURCE\n' + source_part


def expected_setup(root, run_id, created_utc):
    original = load_009(root, SOURCE_RUN)
    if original['schema'] != VERSION_009:
        raise ValueError('wrong Experiment 009 source study')
    source = original['sources'][PROPOSAL_ID]
    prompt = clarified_prompt(source)
    return {'schema': VERSION, 'run_id': run_id, 'created_utc': created_utc,
            'source_run_id': SOURCE_RUN,
            'source_setup_sha256': sha((Path(root) / SECTION_009 / SOURCE_RUN / 'setup/study.json').read_bytes()),
            'proposal_id': PROPOSAL_ID, 'proposal_sha256': source['proposal_sha256'],
            'source': source, 'source_sha256': digest(canonical(source)),
            'prompt_version': PROMPT_VERSION, 'prompt_sha256': digest(prompt.encode()),
            'validator': VALIDATOR, 'configurations': CONFIGS,
            'api_base_url': API_BASE_URL, 'request_options': REQUEST_OPTIONS,
            'timeout_seconds': TIMEOUT_SECONDS, 'max_retries': MAX_RETRIES,
            'authorization_state': 'no_live_request_authorized_at_preparation',
            'model_availability': 'unverified_offline',
            'request_setting_support': 'unverified_offline'}


def prepare(root):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    setup = expected_setup(root, run_id, now())
    write(folder(root, run_id) / 'setup', 'study.json', setup, 'experiment_013_setup_v1')
    return run_id


def load(root, run_id):
    setup = read(folder(root, run_id) / 'setup', 'study.json', 'experiment_013_setup_v1')
    if not isinstance(setup.get('created_utc'), str) or setup != expected_setup(root, run_id, setup['created_utc']):
        raise ValueError('Experiment 013 setup or upstream source changed')
    return setup


def request_spec(root, run_id, condition, setup=None):
    if condition not in CONFIGS:
        raise ValueError('invalid configuration')
    setup = setup or load(root, run_id)
    prompt = clarified_prompt(setup['source'])
    config = CONFIGS[condition]
    return {'schema': 'experiment_013_request_v1', 'run_id': run_id,
            'configuration': condition, 'configuration_name': config['name'],
            'proposal_id': PROPOSAL_ID, 'proposal_sha256': setup['proposal_sha256'],
            'source_sha256': setup['source_sha256'], 'source_setup_sha256': setup['source_setup_sha256'],
            'model': config['model'], 'request_options': REQUEST_OPTIONS,
            'timeout_seconds': TIMEOUT_SECONDS, 'max_retries': MAX_RETRIES,
            'prompt_version': PROMPT_VERSION, 'prompt_sha256': setup['prompt_sha256'],
            'validator': VALIDATOR, 'prompt': prompt,
            'authorization_state': 'not_authorized_in_offline_preview'}


def preview(root, run_id):
    setup = load(root, run_id)
    record = {'schema': 'experiment_013_preview_v1', 'run_id': run_id,
              'requests': {c: request_spec(root, run_id, c, setup) for c in CONFIGS}}
    if record['requests']['A']['prompt'] != record['requests']['B']['prompt']:
        raise ValueError('configuration prompts differ')
    stage = folder(root, run_id) / 'preview'
    if stage.exists():
        if read(stage, 'preview.json', 'experiment_013_preview_v1') != record:
            raise ValueError('preview differs from frozen request specs')
    else:
        write(stage, 'preview.json', record, 'experiment_013_preview_v1')
    return record


def request_decision(client, root, run_id, condition, *, human_authorized=False):
    """For a future separately authorized call; exactly one attempt per configuration."""
    if human_authorized is not True:
        raise ValueError('explicit authorization required for this live request')
    spec = request_spec(root, run_id, condition)
    target = folder(root, run_id) / 'decisions' / condition
    with (folder(root, run_id) / 'request.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return 'already_attempted'
        audit = {'schema': 'experiment_013_decision_audit_v1', 'created_utc': now(),
                 'request': spec, 'explicit_request_authorization': True,
                 'raw_response': None, 'parsed_response': None,
                 'response_model': None, 'response_id': None, 'response_metadata': None}
        called = False
        try:
            response = client.responses.create(model=spec['model'], input=spec['prompt'], **REQUEST_OPTIONS)
            called = True
            audit['raw_response'], audit['response_metadata'] = inspect_response(response)
            audit['response_model'] = _get(response, 'model')
            audit['response_id'] = _get(response, 'id')
            require_complete_text(audit['raw_response'], audit['response_metadata'])
            if audit['response_model'] != spec['model']:
                raise ValueError('response model mismatch')
            audit['parsed_response'] = parse_decision(audit['raw_response'], load(root, run_id)['source'])
            if request_spec(root, run_id, condition) != spec:
                raise ValueError('request source changed')
            audit['status'] = 'accepted'
        except Exception as exc:
            audit['status'] = 'rejected'
            audit['rejection_kind'] = 'invalid_response' if called else 'request_error'
            audit['rejection'] = f'{type(exc).__name__}: {str(exc)[:300]}'
        write(target, 'audit.json', audit, 'experiment_013_decision_audit_v1')
        return audit['status']


def load_decision(root, run_id, condition, setup=None):
    setup = setup or load(root, run_id)
    audit = read(folder(root, run_id) / 'decisions' / condition,
                 'audit.json', 'experiment_013_decision_audit_v1')
    if (audit.get('schema') != 'experiment_013_decision_audit_v1'
            or audit.get('request') != request_spec(root, run_id, condition, setup)
            or audit.get('explicit_request_authorization') is not True):
        raise ValueError('decision request or authorization mismatch')
    if audit.get('status') == 'accepted':
        require_complete_text(audit['raw_response'], audit['response_metadata'])
        if (audit.get('response_model') != CONFIGS[condition]['model']
                or audit.get('parsed_response') != parse_decision(audit['raw_response'], setup['source'])):
            raise ValueError('accepted decision mismatch')
    elif (audit.get('status') != 'rejected' or audit.get('parsed_response') is not None
          or audit.get('rejection_kind') not in ('invalid_response', 'request_error')
          or not audit.get('rejection')):
        raise ValueError('invalid rejected decision audit')
    return audit


def validate(root, run_id):
    setup = load(root, run_id)
    base = folder(root, run_id)
    if (base / 'preview').exists():
        preview(root, run_id)
    statuses = {c: (load_decision(root, run_id, c, setup)['status']
                    if (base / 'decisions' / c).exists() else 'not_requested') for c in CONFIGS}
    return {'run_id': run_id, 'proposal_id': PROPOSAL_ID,
            'prompt_sha256': setup['prompt_sha256'], 'request_statuses': statuses,
            'model_availability': setup['model_availability'],
            'request_setting_support': setup['request_setting_support']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Experiment 013 model capability study')
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('prepare', 'preview', 'validate', 'request-decision'):
        modes.add_argument('--' + mode, action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--configuration', choices=CONFIGS)
    parser.add_argument('--human-authorized', action='store_true',
                        help='required separately for each future live request')
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare(ROOT)); return 0
    if not args.run_id:
        parser.error('--run-id required')
    if args.preview:
        print(json.dumps(preview(ROOT, args.run_id), indent=2)); return 0
    if args.validate:
        print(json.dumps(validate(ROOT, args.run_id), indent=2)); return 0
    if not args.configuration or not args.human_authorized:
        parser.error('each live request requires --configuration and --human-authorized')
    if not (folder(ROOT, args.run_id) / 'preview').exists():
        parser.error('offline preview required before live request')
    validate(ROOT, args.run_id)
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(dotenv_path=ROOT / '.env')
    key = os.environ.get('MODEL_API_KEY')
    if not key:
        parser.error('MODEL_API_KEY required for a separately authorized live request')
    client = OpenAI(base_url=API_BASE_URL, api_key=key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    status = request_decision(client, ROOT, args.run_id, args.configuration, human_authorized=True)
    print(status)
    return 0 if status == 'accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
