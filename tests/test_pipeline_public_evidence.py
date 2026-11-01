"""CPU-only portability and tamper checks for the pipeline evidence export."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import verify_pipeline_public_evidence as verifier


BUNDLE = Path(__file__).resolve().parents[1] / 'public_pipeline_evidence'


def test_public_bundle_verifies_without_local_evaluations(tmp_path):
    copied = tmp_path / 'bundle'
    shutil.copytree(BUNDLE, copied)
    result = subprocess.run([sys.executable, str(copied / 'verify.py'), str(copied)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['prediction_score'] == 'inconclusive'
    assert verifier.verify(copied)['timing_sessions'] == 5


def test_tampered_pair_or_manifest_is_rejected(tmp_path):
    copied = tmp_path / 'bundle'
    shutil.copytree(BUNDLE, copied)
    evidence = copied / 'evidence.json'
    data = json.loads(evidence.read_bytes())
    row = data['heldout']['target_k1536']['phases']['comparison']['rounds'][0]
    row['ratio'] = 0.1
    evidence.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='file hash'):
        verifier.verify(copied)


def test_export_to_new_directory_is_cpu_only(tmp_path):
    from export_pipeline_public_evidence import export
    output = tmp_path / 'portable'
    assert export(output=output) == output
    assert verifier.verify(output)['profile_captures'] == 4
    with pytest.raises(FileExistsError):
        export(output=output)
