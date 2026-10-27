# PTX comparison

- Baseline: `evaluations/trusted-two-slab/20261027T161358957319Z/baseline.ptx`
- Candidate: `evaluations/trusted-two-slab/20261027T161358957319Z/trusted.ptx`
- Kernels: `matmul_kernel` → `matmul_kernel`

## Compatibility

| Field | Baseline | Candidate | Match |
|---|---|---|---|
| PTX version | 8.7 | 8.7 | True |
| Target | sm_89 | sm_89 | True |
| Address size | 64 | 64 | True |
| Kernel interface | — | — | True |

## Full-module text comparison

These raw and normalized results compare the complete PTX modules, not only the selected entry point.

| Layer | Result | Changed lines |
|---|---|---:|
| Raw | different | 1306 |
| Normalized | different | 1192 |

## Selected-entry structural comparison

The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| arithmetic | 241 | 274 | +33 |
| comparison_predication | 16 | 16 | +0 |
| control_flow | 2 | 2 | +0 |
| conversion | 34 | 34 | +0 |
| memory_global | 8 | 8 | +0 |
| memory_param | 3 | 3 | +0 |
| memory_shared | 16 | 16 | +0 |
| other | 38 | 74 | +36 |
| synchronization | 7 | 7 | +0 |
| tensor | 32 | 64 | +32 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `add.s32` | 43 | 61 | +18 |
| `add.s64` | 33 | 50 | +17 |
| `and.b32` | 13 | 13 | +0 |
| `and.pred` | 1 | 1 | +0 |
| `bar.sync` | 7 | 7 | +0 |
| `bfe.s32` | 1 | 1 | +0 |
| `bfe.u32` | 2 | 2 | +0 |
| `bra` | 1 | 1 | +0 |
| `cp.async.cg.shared.global` | 18 | 36 | +18 |
| `cp.async.commit_group` | 6 | 12 | +6 |
| `cp.async.wait_group` | 2 | 2 | +0 |
| `cvt.rn.f16x2.f32` | 32 | 32 | +0 |
| `cvt.u32.u64` | 1 | 1 | +0 |
| `cvt.u64.u32` | 1 | 1 | +0 |
| `ld.param.b64` | 3 | 3 | +0 |
| `ld.shared.v4.b32` | 8 | 8 | +0 |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 8 | 16 | +8 |
| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | 4 | 8 | +4 |
| `mad.lo.s32` | 1 | 1 | +0 |
| `mad.wide.s32` | 10 | 10 | +0 |
| `mad.wide.u32` | 2 | 3 | +1 |
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 32 | 64 | +32 |
| `mov.b32` | 68 | 68 | +0 |
| `mov.b64` | 1 | 1 | +0 |
| `mov.u32` | 3 | 3 | +0 |
| `mul.lo.s32` | 7 | 7 | +0 |
| `mul.wide.s32` | 2 | 2 | +0 |
| `mul.wide.u32` | 3 | 2 | -1 |
| `or.b32` | 15 | 15 | +0 |
| `or.b64` | 2 | 0 | -2 |
| `ret` | 1 | 1 | +0 |
| `selp.b32` | 8 | 8 | +0 |
| `setp.eq.b32` | 1 | 1 | +0 |
| `setp.gt.s32` | 2 | 2 | +0 |
| `setp.lt.s32` | 2 | 2 | +0 |
| `setp.lt.u32` | 2 | 2 | +0 |
| `setp.ne.b32` | 1 | 1 | +0 |
| `shl.b32` | 18 | 18 | +0 |
| `shl.b64` | 1 | 1 | +0 |
| `shr.u32` | 2 | 2 | +0 |
| `st.global.v4.b32` | 8 | 8 | +0 |
| `st.shared.v4.b32` | 8 | 8 | +0 |
| `xor.b32` | 13 | 13 | +0 |

### PTX-declared resources

These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final physical register allocation, hardware resource usage, or occupancy estimates.

| Register type | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| b32 | 305 | 371 | +66 |
| b64 | 56 | 72 | +16 |
| pred | 10 | 10 | +0 |

- PTX-declared static shared bytes: 0 → 0
- Dynamic shared declaration: False → False

### Baseline shared-memory declarations

None in the selected entry.

### Candidate shared-memory declarations

None in the selected entry.

## Conclusions

- The input PTX files are different after the documented conservative normalization.
- Instruction and declaration differences are static PTX observations; they do not establish SASS changes or runtime performance effects.

## Limitations

Normalization removes only `.file`/`.loc` source annotations and whitespace differences described in the JSON report. Comments, predicates, constants, operands, labels, symbols, and instruction order are retained. Per-entry statistics exclude called device-function bodies. PTX instructions do not map one-to-one to SASS instructions.
