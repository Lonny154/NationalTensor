"""One FP16 GEMM body with row-major, grouped-M4, or Hilbert tile assignment."""
from __future__ import annotations

import triton
import triton.language as tl
import torch

from grouped_matmul import tile_for_pid as grouped_tile_for_pid

SCHEDULES = ("row_major", "grouped_m4", "hilbert")


def hilbert_xy(side, rank):
    """Standard Hilbert distance-to-coordinate mapping for a power-of-two square."""
    if side < 1 or side & (side - 1) or not 0 <= rank < side * side:
        raise ValueError("invalid Hilbert domain or rank")
    x = y = 0
    t = rank
    scale = 1
    while scale < side:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = scale - 1 - x, scale - 1 - y
            x, y = y, x
        x += scale * rx
        y += scale * ry
        t //= 4
        scale *= 2
    return x, y


def tile_order(schedule, tiles_m, tiles_n):
    """Return each valid output tile once, including rectangular grids."""
    if schedule not in SCHEDULES or not 1 <= tiles_m <= 65536 or not 1 <= tiles_n <= 65536:
        raise ValueError("invalid schedule or tile grid")
    if tiles_m * tiles_n > 1_000_000:
        raise ValueError("tile grid exceeds bounded mapping size")
    if schedule == "row_major":
        order = [(pid // tiles_n, pid % tiles_n) for pid in range(tiles_m * tiles_n)]
    elif schedule == "grouped_m4":
        order = [grouped_tile_for_pid(pid, tiles_m, tiles_n)
                 for pid in range(tiles_m * tiles_n)]
    else:
        side = 1 << (max(tiles_m, tiles_n) - 1).bit_length()
        if side * side > 4_000_000:
            raise ValueError("padded Hilbert domain is too large")
        order = [xy for rank in range(side * side)
                 if (xy := hilbert_xy(side, rank))[0] < tiles_m and xy[1] < tiles_n]
    if len(order) != tiles_m * tiles_n or len(set(order)) != len(order) or any(
            not (0 <= m < tiles_m and 0 <= n < tiles_n) for m, n in order):
        raise ValueError("tile mapping is incomplete, duplicate, or out of bounds")
    return order


@triton.jit
def matmul_kernel(A, B, C, TILE_MAP,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  SCHEDULE: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    if SCHEDULE == 0:
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(0)
    elif SCHEDULE == 1:
        group_id = pid // (GROUP_M * tiles_n)
        first_m = group_id * GROUP_M
        group_size_m = tl.minimum(tiles_m - first_m, GROUP_M)
        within = pid % (GROUP_M * tiles_n)
        pid_m = first_m + within % group_size_m
        pid_n = within // group_size_m
    else:
        pid_m = tl.load(TILE_MAP + 2 * pid)
        pid_n = tl.load(TILE_MAP + 2 * pid + 1)

    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a_ptrs = A + m[:, None] * K + k[None, :]
    b_ptrs = B + k[:, None] * N + n[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    for offset in range(tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(m[:, None] < M) & ((offset * BK + k[None, :]) < K), other=0)
        b = tl.load(b_ptrs, mask=((offset * BK + k[:, None]) < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = C + m[:, None] * N + n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(m[:, None] < M) & (n[None, :] < N))


def launch(a, b, out, config):
    schedule = config["schedule"]
    if schedule not in SCHEDULES:
        raise ValueError("unknown schedule")
    m, k = a.shape
    _, n = b.shape
    bm, bn, bk = (config[key] for key in ("block_m", "block_n", "block_k"))
    tiles_m, tiles_n = triton.cdiv(m, bm), triton.cdiv(n, bn)
    mapping = config.get("tile_map")
    if schedule == "hilbert":
        if mapping is None or mapping.shape != (tiles_m * tiles_n, 2) or mapping.dtype != torch.int32 or mapping.device != a.device:
            raise ValueError("Hilbert tile map shape, dtype or device mismatch")
    else:
        mapping = a  # never dereferenced in the compiled non-Hilbert branches
    grid = (tiles_n, tiles_m) if schedule == "row_major" else (tiles_m * tiles_n,)
    return matmul_kernel[grid](
        a, b, out, mapping, m, n, k, bm, bn, bk, SCHEDULES.index(schedule), 4,
        num_warps=config["num_warps"], num_stages=config["num_stages"])
