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
    m_mask = m[:, None] < M
    n_mask = n[None, :] < N
    n_full = K // BK
    k_rem = K % BK
    for i in range(n_full):
        a = tl.load(a_ptrs, mask=m_mask, other=0)
        b = tl.load(b_ptrs, mask=n_mask, other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    if k_rem != 0:
        k_tail = n_full * BK + k
        a_tail = tl.load(a_ptrs, mask=m_mask & (k_tail[None, :] < K), other=0)
        b_tail = tl.load(b_ptrs, mask=(k_tail[:, None] < K) & n_mask, other=0)
        acc = tl.dot(a_tail, b_tail, acc)
    c_ptrs = C + m[:, None] * N + n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(m[:, None] < M) & (n[None, :] < N))
