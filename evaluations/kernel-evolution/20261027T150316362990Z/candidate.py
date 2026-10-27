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
    m_mask = m < M
    n_mask = n < N
    a_ptrs = A + m[:, None] * K + k[None, :]
    b_ptrs = B + k[:, None] * N + n[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    loop_k = tl.cdiv(K, BK)
    for offset in tl.range(0, loop_k, num_stages=3):
        k_off = offset * BK
        a = tl.load(a_ptrs, mask=m_mask[:, None] & ((k_off + k[None, :]) < K), other=0)
        b = tl.load(b_ptrs, mask=((k_off + k[:, None]) < K) & n_mask[None, :], other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = C + m[:, None] * N + n[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])
