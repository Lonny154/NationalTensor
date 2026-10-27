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
    m_mask = m[:, None] < M
    n_mask = n[None, :] < N
    a_cur = tl.load(a_ptrs, mask=m_mask & (k[None, :] < K), other=0)
    b_cur = tl.load(b_ptrs, mask=(k[:, None] < K) & n_mask, other=0)
    acc = tl.full((BM, BN), 0, tl.float32)
    for offset in range(tl.cdiv(K, BK) - 1):
        k_next = (offset + 1) * BK
        a_nxt = tl.load(a_ptrs + BK, mask=m_mask & ((k_next + k[None, :]) < K), other=0)
        b_nxt = tl.load(b_ptrs + BK * N, mask=((k_next + k[:, None]) < K) & n_mask, other=0)
        acc = tl.dot(a_cur, b_cur, acc)
        a_cur = a_nxt
        b_cur = b_nxt
        a_ptrs += BK
        b_ptrs += BK * N
    acc = tl.dot(a_cur, b_cur, acc)
    c_ptrs = C + m[:, None] * N + n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask & n_mask)
