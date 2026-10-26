"""Reviewed grouped-M GEMM schedule and explicit schedule dispatcher.

The direct kernel stays in benchmark_proposal.py. Only the mapping from a
one-dimensional program ID to an output tile differs in the grouped kernel.
"""

import triton
import triton.language as tl

from benchmark_proposal import launch as launch_direct


DIRECT_2D = "direct_2d"
GROUPED_M4 = "grouped_m4"
GROUP_M = 4


def tile_for_pid(pid, num_pid_m, num_pid_n):
    """CPU mirror of the grouped schedule for exhaustive mapping tests."""
    if num_pid_m < 1 or num_pid_n < 1 or not 0 <= pid < num_pid_m * num_pid_n:
        raise ValueError("invalid tile grid or program ID")
    group_id = pid // (GROUP_M * num_pid_n)
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % (GROUP_M * num_pid_n)
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def grouped_matmul_kernel(
    A, B, C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    group_id = pid // (GROUP_SIZE_M * num_pid_n)
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid % (GROUP_SIZE_M * num_pid_n)
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)

    a_ptrs = A + m[:, None] * K + k[None, :]
    b_ptrs = B + k[:, None] * N + n[None, :]

    acc = tl.full((BM, BN), 0, tl.float32)

    for offset in range(tl.cdiv(K, BK)):
        a = tl.load(
            a_ptrs,
            mask=(m[:, None] < M)
            & ((offset * BK + k[None, :]) < K),
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=((offset * BK + k[:, None]) < K)
            & (n[None, :] < N),
            other=0,
        )

        acc = tl.dot(a, b, acc)

        a_ptrs += BK
        b_ptrs += BK * N

    c_ptrs = C + m[:, None] * N + n[None, :]
    tl.store(
        c_ptrs,
        acc.to(tl.float16),
        mask=(m[:, None] < M) & (n[None, :] < N),
    )


def launch_grouped(a, b, out, config):
    m, k = a.shape
    _, n = b.shape
    bm = config["block_m"]
    bn = config["block_n"]
    bk = config["block_k"]
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    grouped_matmul_kernel[grid](
        a, b, out, m, n, k, bm, bn, bk, GROUP_M,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def launch_scheduled(a, b, out, config):
    """Dispatch only the two reviewed schedules; require an explicit ID."""
    schedule = config.get("schedule")
    if schedule == DIRECT_2D:
        launch_direct(a, b, out, config)
    elif schedule == GROUPED_M4:
        launch_grouped(a, b, out, config)
    else:
        raise ValueError(f"unknown or missing schedule: {schedule!r}")
