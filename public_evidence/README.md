# Public research evidence

This portable bundle summarizes validated local records for National Tensor Experiments 005–009. `evidence.json` contains allowlisted observations, model judgments, source identifiers, and reported review scores. `summary.md` is generated from those fields. `manifest.json` hashes every published file and separately records hashes of the original local artifacts used for export.

## Inspect and verify

From the repository root, use only Python's standard library:

```bash
python verify_public_evidence.py public_evidence
python -m json.tool public_evidence/evidence.json
```

The verifier checks file hashes, schemas, citation/reference shape, screening arithmetic and intervals, score means, missing confirmation and profiling, and the incomplete Experiment 009 state. It needs no GPU, API key, or original `evaluations/` directory. Its success establishes **bundle integrity and internal consistency**, not independent reproduction of the GPU measurements or scientific correctness of model prose.

`experiment_005` contains hardware observations; `experiment_006` and `experiment_008` contain advisory Muse judgments; `experiment_007` contains human subjective scores; `experiment_009` records attempts and a partial decision pilot. A rejected response remains rejected even when its validation lesson is summarized. No confirmed speedup or GPU compute saving is claimed.

Original run artifacts under `evaluations/` are ignored by Git. Reproducing measurements requires compatible GPU hardware, toolchain, source snapshots, and explicit human authorization for each GPU stage. The relevant protocols are in `docs/experiment_005.md` through `docs/experiment_009.md` at the repository root. Regenerating this bundle additionally requires the preserved original artifact directories and a compatible environment:

```bash
.venv/bin/python export_public_evidence.py --output public_evidence
```

The exporter validates its local sources and fails if a required record is missing or changed. It never calls Muse or launches a GPU experiment. The exported manifest preserves source hashes so reviewers can compare a separately supplied original artifact without requiring it for routine verification. Raw prompts, raw model responses, request headers, credentials, local paths, GPU UUIDs, and unnecessary machine metadata are excluded.
