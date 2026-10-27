"""FP16 matmul correctness reference with scoped CUDA precision settings."""


REDUCTION = "allow_fp16_reduced_precision_reduction"
SPLIT_K = "allow_fp16_reduced_precision_reduction_split_k"
ACCUMULATION = "allow_fp16_accumulation"
TF32 = "allow_tf32"


def _settings(matmul_backend):
    return {
        name: getattr(matmul_backend, name)
        for name in (REDUCTION, SPLIT_K, ACCUMULATION, TF32)
        if hasattr(matmul_backend, name)
    }


def fp16_mm_reference(a, b, *, torch_module=None):
    """Return ``torch.mm(a, b)`` and an audit of its precision policy.

    The CUDA backend flags are process-wide. Keep the reference call scoped and
    restore both FP16 reduction and split-K flags as a pair where supported.
    ``torch_module`` allows CPU-only tests with a fake backend and mm function.
    """
    if torch_module is None:
        import torch
        torch_module = torch

    backend = torch_module.backends.cuda.matmul
    before = _settings(backend)
    split_k_supported = SPLIT_K in before
    accumulation_supported = ACCUMULATION in before
    active = None
    try:
        # Keep the caller's split-K choice. Forcing it off requires cuBLASLt
        # on some PyTorch/CUDA builds, even with reduced precision disabled.
        backend.allow_fp16_reduced_precision_reduction = (
            (False, before[SPLIT_K]) if split_k_supported else False
        )
        if accumulation_supported:
            backend.allow_fp16_accumulation = False
        active = _settings(backend)
        if active[REDUCTION] or (split_k_supported and active[SPLIT_K] != before[SPLIT_K]) or (
            accumulation_supported and active[ACCUMULATION]
        ):
            raise RuntimeError("FP16 reference precision policy was not applied")
        reference = torch_module.mm(a, b)
    finally:
        try:
            if accumulation_supported:
                backend.allow_fp16_accumulation = before[ACCUMULATION]
        finally:
            backend.allow_fp16_reduced_precision_reduction = (
                (before[REDUCTION], before[SPLIT_K])
                if split_k_supported else before[REDUCTION]
            )

    after = _settings(backend)
    if after != before:
        raise RuntimeError("FP16 reference backend settings were not restored")
    return reference, {
        "policy": "fp16_torch_mm_without_reduced_precision_reduction",
        "operation": "torch.mm on original FP16 inputs",
        "split_k_setting_supported": split_k_supported,
        "fp16_accumulation_setting_supported": accumulation_supported,
        "settings_before": before,
        "settings_during": active,
        "settings_after": after,
    }
