"""Human-authored test of mask hoisting and explicit K-loop staging.

This module is trusted experiment code. It is independent of saved Muse source.
"""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)

    rows = tile_m * BM + tl.arange(0, BM)
    columns = tile_n * BN + tl.arange(0, BN)
    reduction_lanes = tl.arange(0, BK)
    valid_rows = rows < M
    valid_columns = columns < N

    left = A + rows[:, None] * K + reduction_lanes[None, :]
    right = B + reduction_lanes[:, None] * N + columns[None, :]
    accumulator = tl.full((BM, BN), 0, tl.float32)

    for step in tl.range(0, tl.cdiv(K, BK), num_stages=3):
        reduction_start = step * BK
        valid_left = valid_rows[:, None] & (
            reduction_start + reduction_lanes[None, :] < K
        )
        valid_right = (
            reduction_start + reduction_lanes[:, None] < K
        ) & valid_columns[None, :]
        left_tile = tl.load(left, mask=valid_left, other=0)
        right_tile = tl.load(right, mask=valid_right, other=0)
        accumulator = tl.dot(left_tile, right_tile, accumulator)
        left += BK
        right += BK * N

    destination = C + rows[:, None] * N + columns[None, :]
    tl.store(destination, accumulator.to(tl.float16),
             mask=valid_rows[:, None] & valid_columns[None, :])


def launch(a, b, out, config):
    """Use the baseline two-dimensional grid and numeric configuration."""
    m, k = a.shape
    _, n = b.shape
    bm = config["block_m"]
    bn = config["block_n"]
    bk = config["block_k"]
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    return matmul_kernel[grid](
        a, b, out, m, n, k, bm, bn, bk,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
