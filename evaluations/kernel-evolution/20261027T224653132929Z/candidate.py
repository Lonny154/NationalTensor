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
    m_mask = m[:, None] < M
    n_mask = n[None, :] < N
    a_base = A + m[:, None] * K
    b_base = B + n[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    for offset in tl.range(tl.cdiv(K, BK), num_stages=3):
        k_off = offset * BK
        k_a = k_off + k[None, :]
        k_b = k_off + k[:, None]
        b_ptrs = b_base + k_b * N
        b_ptrs = tl.max_contiguous(b_ptrs, BN)
        b = tl.load(b_ptrs, mask=(k_b < K) & n_mask, other=0)
        a_ptrs = a_base + k_a
        a_ptrs = tl.max_contiguous(a_ptrs, BK)
        a = tl.load(a_ptrs, mask=m_mask & (k_a < K), other=0)
        acc = tl.dot(a, b, acc)
    c_ptrs = C + m[:, None] * N + n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask & n_mask)