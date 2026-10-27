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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a_ptrs = A + m[:, None] * K + k[None, :]
    b_ptrs = B + k[:, None] * N + n[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    num_iters = tl.cdiv(K, BK)
    num_pairs = num_iters // 2
    for p in range(num_pairs):
        k0 = p * 2 * BK
        a0 = tl.load(a_ptrs, mask=(m[:, None] < M) & ((k0 + k[None, :]) < K), other=0)
        b0 = tl.load(b_ptrs, mask=((k0 + k[:, None]) < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a0, b0, acc)
        a1 = tl.load(a_ptrs + BK, mask=(m[:, None] < M) & ((k0 + BK + k[None, :]) < K), other=0)
        b1 = tl.load(b_ptrs + BK * N, mask=((k0 + BK + k[:, None]) < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a1, b1, acc)
        a_ptrs += 2 * BK
        b_ptrs += 2 * BK * N
    if num_iters % 2 == 1:
        k_tail = num_pairs * 2 * BK
        a = tl.load(a_ptrs, mask=(m[:, None] < M) & ((k_tail + k[None, :]) < K), other=0)
        b = tl.load(b_ptrs, mask=((k_tail + k[:, None]) < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    c_ptrs = C + m[:, None] * N + n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(m[:, None] < M) & (n[None, :] < N))
