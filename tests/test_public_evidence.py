"""Portable, CPU-only public evidence regression checks."""
import json
from pathlib import Path
import shutil

import pytest

import export_public_evidence as exporter
import verify_public_evidence as public

ROOT = Path(__file__).resolve().parents[1]


def copy_bundle(tmp_path):
    dest = tmp_path / 'public_evidence'
    shutil.copytree(ROOT / 'public_evidence', dest)
    return dest


def rehash(bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['files'] = {name: public.sha((bundle / name).read_bytes()) for name in public.FILES}
    (bundle / 'manifest.json').write_bytes(public.canonical(manifest))


def test_bundle_verifies_without_private_artifacts(tmp_path):
    bundle = copy_bundle(tmp_path)
    assert public.verify(bundle)['source_artifacts'] == 20
    data = json.loads((bundle / 'evidence.json').read_text())
    assert len(data['experiment_005']['screening']) == 8
    assert data['experiment_006']['decision'] == 'abandon'
    assert len(data['experiment_007']['proposals']) == 6
    assert data['experiment_009']['paired_evaluation_status'] == 'incomplete'


def test_export_is_deterministic_with_fixed_validated_projection(monkeypatch, tmp_path):
    data = json.loads((ROOT / 'public_evidence/evidence.json').read_text())
    sources = json.loads((ROOT / 'public_evidence/manifest.json').read_text())['source_artifacts']
    monkeypatch.setattr(exporter, 'collect', lambda root: (data, sources))
    a, b = tmp_path / 'a', tmp_path / 'b'
    exporter.export(tmp_path, a)
    exporter.export(tmp_path, b)
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {p.name: p.read_bytes() for p in b.iterdir()}
    assert public.verify(a)['schema'] == public.SCHEMA


def test_missing_source_fails_before_export(tmp_path):
    with pytest.raises(ValueError, match='required validated Experiment 005'):
        exporter.export(tmp_path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_source_hashing_and_missing_source(tmp_path):
    rel = 'source.json'
    (tmp_path / rel).write_text('{"status":"accepted"}')
    first = exporter.source_hashes(tmp_path, [rel])[rel]
    (tmp_path / rel).write_text('{"status":"rejected"}')
    assert exporter.source_hashes(tmp_path, [rel])[rel] != first
    with pytest.raises(ValueError, match='missing or linked'):
        exporter.source_hashes(tmp_path, ['missing.json'])


def test_manifest_detects_tampering_and_incomplete_bundle(tmp_path):
    bundle = copy_bundle(tmp_path)
    (bundle / 'summary.md').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        public.verify(bundle)
    (bundle / 'summary.md').unlink()
    with pytest.raises(ValueError, match='missing'):
        public.verify(bundle)


def test_numeric_aggregate_rejected_even_if_rehashed(tmp_path):
    bundle = copy_bundle(tmp_path)
    data = json.loads((bundle / 'evidence.json').read_text())
    data['experiment_005']['screening'][0]['median_paired_ratio'] += 0.1
    (bundle / 'evidence.json').write_bytes(public.canonical(data))
    (bundle / 'summary.md').write_text(public.render_summary(data))
    rehash(bundle)
    with pytest.raises(ValueError, match='median arithmetic'):
        public.verify(bundle)
    data = json.loads((ROOT / 'public_evidence/evidence.json').read_text())
    data['experiment_007']['means']['A']['novelty'] += 1
    (bundle / 'evidence.json').write_bytes(public.canonical(data))
    (bundle / 'summary.md').write_text(public.render_summary(data))
    rehash(bundle)
    with pytest.raises(ValueError, match='means mismatch'):
        public.verify(bundle)


def test_missing_confirmation_and_rejection_status_cannot_be_rewritten(tmp_path):
    bundle = copy_bundle(tmp_path)
    data = json.loads((bundle / 'evidence.json').read_text())
    data['experiment_005']['confirmation_status'] = 'confirmed'
    with pytest.raises(ValueError, match='missing stage'):
        public.validate_data(data)
    data = json.loads((ROOT / 'public_evidence/evidence.json').read_text())
    data['experiment_009']['criticism_attempts'][1]['status'] = 'accepted'
    with pytest.raises(ValueError, match='criticism statuses'):
        public.validate_data(data)
    data = json.loads((ROOT / 'public_evidence/evidence.json').read_text())
    data['experiment_008']['revision_attempts'][0]['status'] = 'accepted'
    with pytest.raises(ValueError, match='revision status'):
        public.validate_data(data)


def test_privacy_allowlist_rejects_secrets_and_local_paths():
    for value in ('/home/daniel/private/audit.json', '/Users/alice/file',
                  'GPU-e514b848-c457-f661-2343-cf60d34cc828',
                  'Bearer this-is-a-secret-token', 'MODEL_API_KEY=secret'):
        with pytest.raises(ValueError, match='sensitive text'):
            public.privacy_check({'note': value})


def test_unreviewed_rejection_category_fails_closed():
    with pytest.raises(ValueError, match='unreviewed rejection category'):
        exporter.rejection_reason({'status': 'rejected', 'rejection_kind': 'invalid_response',
                                   'rejection': 'new unknown validator failure'}, 'P02')
