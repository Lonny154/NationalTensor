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
    pid_m0 = tl.program_id(0)
    pid_n0 = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    pid = pid_m0 * num_pid_n + pid_n0
    GROUP_M = 4
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = num_pid_m - first_pid_m
    group_size_m = tl.minimum(group_size_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
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
