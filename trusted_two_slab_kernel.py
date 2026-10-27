"""Human-authored two-slab K-loop GEMM for a bounded trusted experiment."""

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
    lanes = tl.arange(0, BK)

    left = A + rows[:, None] * K + lanes[None, :]
    right = B + lanes[:, None] * N + columns[None, :]
    accumulator = tl.full((BM, BN), 0, tl.float32)
    slab_count = tl.cdiv(K, BK)

    for pair in range(slab_count // 2):
        first_k = pair * (2 * BK)
        a0 = tl.load(left, mask=(rows[:, None] < M)
                     & (first_k + lanes[None, :] < K), other=0)
        b0 = tl.load(right, mask=(first_k + lanes[:, None] < K)
                     & (columns[None, :] < N), other=0)
        accumulator = tl.dot(a0, b0, accumulator)

        second_k = first_k + BK
        a1 = tl.load(left + BK, mask=(rows[:, None] < M)
                     & (second_k + lanes[None, :] < K), other=0)
        b1 = tl.load(right + BK * N, mask=(second_k + lanes[:, None] < K)
                     & (columns[None, :] < N), other=0)
        accumulator = tl.dot(a1, b1, accumulator)
        left += 2 * BK
        right += 2 * BK * N

    if slab_count % 2:
        final_k = (slab_count - 1) * BK
        a_tail = tl.load(left, mask=(rows[:, None] < M)
                         & (final_k + lanes[None, :] < K), other=0)
        b_tail = tl.load(right, mask=(final_k + lanes[:, None] < K)
                         & (columns[None, :] < N), other=0)
        accumulator = tl.dot(a_tail, b_tail, accumulator)

    destination = C + rows[:, None] * N + columns[None, :]
    tl.store(destination, accumulator.to(tl.float16),
             mask=(rows[:, None] < M) & (columns[None, :] < N))


def launch(a, b, out, config):
    """Use the same direct-2D grid and launch parameters as the baseline."""
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
