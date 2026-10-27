"""Reviewed two-pass FP16 GEMM with FP32 Split-K partials and reduction."""

import torch
import triton
import triton.language as tl


SPLIT_FACTORS = (2, 4)


@triton.jit
def partial_kernel(A, B, Workspace,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   SPLITS: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    tiles_per_split = tl.cdiv(tl.cdiv(K, BK), SPLITS)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tiles_per_split):
        global_k = (pid_s * tiles_per_split + tile) * BK + k
        a = tl.load(A + m[:, None] * K + global_k[None, :],
                    mask=(m[:, None] < M) & (global_k[None, :] < K), other=0)
        b = tl.load(B + global_k[:, None] * N + n[None, :],
                    mask=(global_k[:, None] < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    offsets = pid_s * M * N + m[:, None] * N + n[None, :]
    tl.store(Workspace + offsets, acc,
             mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def reduction_kernel(Workspace, C, M: tl.constexpr, N: tl.constexpr,
                     SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    acc = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLITS):
        acc += tl.load(Workspace + split * total + indices,
                       mask=indices < total, other=0)
    tl.store(C + indices, acc.to(tl.float16), mask=indices < total)


def allocate_workspace(a, b, split_factor):
    if type(split_factor) is not int or split_factor not in SPLIT_FACTORS:
        raise ValueError("split_factor must be 2 or 4")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible GEMM dimensions")
    return torch.empty((split_factor, a.shape[0], b.shape[1]),
                       device=a.device, dtype=torch.float32)


def launch_split_k(a, b, out, config, workspace):
    """Overwrite every valid workspace element before reducing; no atomics."""
    splits = config.get("split_factor")
    if type(splits) is not int or splits not in SPLIT_FACTORS:
        raise ValueError("split_factor must be 2 or 4")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible GEMM dimensions")
    m, k = a.shape
    n = b.shape[1]
    bm, bn, bk = (config[name] for name in ("block_m", "block_n", "block_k"))
    if type(bm) is not int or type(bn) is not int or type(bk) is not int or min(bm, bn, bk) < 1:
        raise ValueError("invalid tile dimensions")
    if splits > triton.cdiv(k, bk):
        raise ValueError("split_factor exceeds the number of K tiles")
    if (a.dtype != torch.float16 or b.dtype != torch.float16
            or out.dtype != torch.float16 or workspace.dtype != torch.float32):
        raise ValueError("Split-K requires FP16 inputs/output and FP32 workspace")
    if (not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous()
            or not workspace.is_contiguous() or a.device != b.device
            or a.device != out.device or a.device != workspace.device
            or out.shape != (m, n) or workspace.shape != (splits, m, n)):
        raise ValueError("invalid Split-K tensor layout, shape, or device")
    partial_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        a, b, workspace, m, n, k, bm, bn, bk, splits,
        num_warps=config["num_warps"], num_stages=config["num_stages"])
    reduction_kernel[(triton.cdiv(m * n, 256),)](
        workspace, out, m, n, splits, 256, num_warps=4)
