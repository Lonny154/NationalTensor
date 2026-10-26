
import json
import statistics

import torch
import triton
import triton.language as tl
import triton.testing


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


def launch(a, b, out, config):
    m, k = a.shape
    _, n = b.shape

    bm = config["block_m"]
    bn = config["block_n"]
    bk = config["block_k"]

    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))

    matmul_kernel[grid](
        a, b, out, m, n, k, bm, bn, bk,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def main():
    with open("proposal.json") as f:
        candidate = json.load(f)

    baseline = {
        "block_m": 64,
        "block_n": 128,
        "block_k": 32,
        "num_warps": 4,
        "num_stages": 3,
    }

    torch.manual_seed(42)
    m, n, k = 512, 3072, 768

    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    reference = torch.mm(a, b)
    configs = {"baseline": baseline, "candidate": candidate}
    outputs = {}

    for name, config in configs.items():
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        launch(a, b, out, config)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            out, reference, rtol=0.02, atol=0.02
        )
        outputs[name] = out
        print(f"{name}: correctness PASS")

    measurements = {name: [] for name in configs}

    # Alternate benchmark ordering to reduce drift bias.
    for round_idx in range(8):
        names = list(configs)
        if round_idx % 2:
            names.reverse()

        for name in names:
            config = configs[name]
            out = outputs[name]

            ms = triton.testing.do_bench(
                lambda: launch(a, b, out, config),
                warmup=100,
                rep=200,
            )
            measurements[name].append(ms)

    baseline_ms = statistics.median(measurements["baseline"])
    candidate_ms = statistics.median(measurements["candidate"])
    speedup = baseline_ms / candidate_ms

    result = {
        "baseline": baseline,
        "candidate": candidate,
        "hypothesis": candidate.get("hypothesis", ""),
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": speedup,
        "correctness": True,
        "measurements_ms": measurements,
    }

    with open("experiment_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBaseline:  {baseline_ms:.6f} ms")
    print(f"Candidate: {candidate_ms:.6f} ms")
    print(f"Speedup:   {speedup:.4f}x")
    print("Results saved to experiment_result.json")


if __name__ == "__main__":
    main()
