# NVIDIA SASS comparison

- Baseline: `evaluations/trusted-two-slab/20261027T161358957319Z/baseline.sass`
- Candidate: `evaluations/trusted-two-slab/20261027T161358957319Z/trusted.sass`
- Functions: `matmul_kernel` → `matmul_kernel`

## Full-file comparison

| Layer | Result | Changed lines |
|---|---|---:|
| Raw | different | 924 |
| Normalized presentation | different | 812 |

## Selected-function comparison

Address comments are omitted only from normalized text. Original addresses remain in structured instruction records. Normalized similarity is neither binary nor semantic equivalence.

- Normalized instruction text: different
- Allocated registers/thread: 120 → 129

### Instruction categories

| Category | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| comparison_predicate | 7 | 10 | +3 |
| control_flow | 2 | 2 | +0 |
| conversion | 32 | 32 | +0 |
| exit_return | 2 | 2 | +0 |
| integer_address_arithmetic | 209 | 216 | +7 |
| memory_constant | 1 | 1 | +0 |
| memory_global | 8 | 8 | +0 |
| memory_other | 36 | 72 | +36 |
| memory_shared | 22 | 22 | +0 |
| other | 72 | 58 | -14 |
| synchronization | 9 | 9 | +0 |
| tensor | 32 | 64 | +32 |

### Full mnemonics

| Mnemonic | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `BAR.SYNC.DEFER_BLOCKING` | 7 | 7 | +0 |
| `BRA` | 2 | 2 | +0 |
| `CS2R` | 32 | 32 | +0 |
| `DEPBAR.LE` | 2 | 2 | +0 |
| `EXIT` | 2 | 2 | +0 |
| `F2FP.F16.F32.PACK_AB` | 32 | 32 | +0 |
| `HMMA.16816.F32` | 32 | 64 | +32 |
| `IADD3` | 35 | 46 | +11 |
| `IADD3.X` | 12 | 9 | -3 |
| `IMAD` | 25 | 25 | +0 |
| `IMAD.HI` | 8 | 8 | +0 |
| `IMAD.MOV.U32` | 35 | 38 | +3 |
| `IMAD.SHL.U32` | 13 | 14 | +1 |
| `IMAD.U32` | 9 | 0 | -9 |
| `IMAD.WIDE` | 2 | 2 | +0 |
| `IMAD.WIDE.U32` | 6 | 5 | -1 |
| `IMAD.X` | 13 | 26 | +13 |
| `ISETP.GE.AND` | 1 | 1 | +0 |
| `ISETP.GE.U32.AND` | 0 | 2 | +2 |
| `ISETP.GT.AND` | 0 | 2 | +2 |
| `ISETP.LT.AND` | 1 | 1 | +0 |
| `ISETP.NE.U32.AND` | 3 | 4 | +1 |
| `LDGDEPBAR` | 6 | 12 | +6 |
| `LDGSTS.E.BYPASS.128` | 18 | 36 | +18 |
| `LDS` | 6 | 6 | +0 |
| `LDS.128` | 8 | 8 | +0 |
| `LDSM.16.M88.4` | 8 | 16 | +8 |
| `LDSM.16.MT88.4` | 4 | 8 | +4 |
| `LEA` | 1 | 2 | +1 |
| `LEA.HI.X` | 1 | 2 | +1 |
| `LEA.HI.X.SX32` | 2 | 2 | +0 |
| `LOP3.LUT` | 34 | 30 | -4 |
| `NOP` | 15 | 13 | -2 |
| `PLOP3.LUT` | 2 | 0 | -2 |
| `S2R` | 3 | 3 | +0 |
| `S2UR` | 2 | 0 | -2 |
| `SEL` | 2 | 6 | +4 |
| `SGXT` | 1 | 1 | +0 |
| `SGXT.U32` | 3 | 3 | +0 |
| `SHF.R.S32.HI` | 1 | 1 | +0 |
| `SHF.R.U32.HI` | 6 | 6 | +0 |
| `STG.E.128` | 8 | 8 | +0 |
| `STS.128` | 8 | 8 | +0 |
| `UIADD3` | 4 | 0 | -4 |
| `UIADD3.X` | 1 | 0 | -1 |
| `UISETP.GE.U32.AND` | 2 | 0 | -2 |
| `UISETP.GT.AND` | 2 | 0 | -2 |
| `UISETP.NE.U32.AND` | 1 | 0 | -1 |
| `ULDC.64` | 1 | 1 | +0 |
| `UMOV` | 5 | 0 | -5 |
| `USEL` | 4 | 0 | -4 |
| `USHF.L.U32` | 1 | 0 | -1 |

### Helper calls

| Capture | Address | Mnemonic | Target |
|---|---|---|---|
| — | — | — | No calls detected |

## Conclusions

- Differences above are static machine-instruction and compiler-metadata observations for the selected functions.
- Static counts do not explain or predict measured runtime behavior.
- Normalized textual agreement would not prove binary identity or semantic equivalence.

## Limitations

The parser consumes nvdisasm presentation text, not raw instruction encodings. It does not infer occupancy, scheduling, dependencies, memory traffic, or performance. Embedded helper bodies are excluded from primary-function counts.
