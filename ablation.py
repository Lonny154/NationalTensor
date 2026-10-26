
import json
import random
import statistics

import torch
import triton.testing

from benchmark_proposal import launch

configs = {
    "baseline": (64, 4),
    "warps_only": (64, 8),
    "tile_only": (128, 4),
    "muse": (128, 8),
}

def main():
    m, n, k = 512, 3072, 768
    torch.manual_seed(42)

    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    reference = torch.mm(a, b)

    results = {}
    outputs = {}
    settings = {}

    for name, (bm, warps) in configs.items():
        config = {
            "block_m": bm,
            "block_n": 128,
            "block_k": 32,
            "num_warps": warps,
            "num_stages": 3,
        }

        out = torch.empty(
            (m, n), device="cuda", dtype=torch.float16
        )

        launch(a, b, out, config)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            out, reference, rtol=0.02, atol=0.02
        )

        print(f"{name}: correctness PASS")
        settings[name] = config
        outputs[name] = out
        results[name] = []

    rng = random.Random(42)

    for round_idx in range(12):
        names = list(configs)
        rng.shuffle(names)

        for name in names:
            config = settings[name]
            out = outputs[name]

            ms = triton.testing.do_bench(
                lambda: launch(a, b, out, config),
                warmup=100,
                rep=200,
            )
            results[name].append(ms)

    baseline_ms = statistics.median(results["baseline"])

    summary = {}
    print("\n=== ABLATION RESULTS ===")

    for name, times in results.items():
        median_ms = statistics.median(times)
        speedup = baseline_ms / median_ms

        summary[name] = {
            "config": settings[name],
            "median_ms": median_ms,
            "speedup": speedup,
            "samples_ms": times,
        }

        print(
            f"{name:12s} "
            f"{median_ms:.6f} ms "
            f"{speedup:.4f}x"
        )

    with open("ablation_result.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
