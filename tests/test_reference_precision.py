"""CPU-only checks of the scoped FP16 correctness reference policy."""

from types import SimpleNamespace
import unittest

from reference_precision import fp16_mm_reference


class FakeMatmulBackend:
    def __init__(self, reduced=True, split_k=False, accumulation=True):
        self.reduced = reduced
        self.split_k = split_k
        self.allow_fp16_accumulation = accumulation
        self.allow_tf32 = False

    @property
    def allow_fp16_reduced_precision_reduction(self):
        return self.reduced

    @allow_fp16_reduced_precision_reduction.setter
    def allow_fp16_reduced_precision_reduction(self, setting):
        # Mirror PyTorch's setter: a bare bool resets split-K to True.
        self.reduced, self.split_k = setting if isinstance(setting, tuple) else (setting, True)

    @property
    def allow_fp16_reduced_precision_reduction_split_k(self):
        return self.split_k


def fake_torch(backend, mm):
    return SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=backend)), mm=mm
    )


class ReferencePrecisionTests(unittest.TestCase):
    def test_policy_metadata_and_exact_state_restoration(self):
        backend = FakeMatmulBackend(reduced=True, split_k=False, accumulation=True)
        seen = []

        def mm(a, b):
            seen.append((a, b, backend.reduced, backend.split_k,
                         backend.allow_fp16_accumulation))
            return "reference"

        result, policy = fp16_mm_reference("a", "b", torch_module=fake_torch(backend, mm))
        self.assertEqual(result, "reference")
        self.assertEqual(seen, [("a", "b", False, False, False)])
        self.assertEqual(policy["policy"], "fp16_torch_mm_without_reduced_precision_reduction")
        self.assertTrue(policy["split_k_setting_supported"])
        self.assertEqual(policy["settings_before"], policy["settings_after"])
        self.assertEqual(policy["settings_before"]["allow_fp16_reduced_precision_reduction_split_k"], False)
        self.assertEqual(policy["settings_during"]["allow_fp16_reduced_precision_reduction"], False)
        self.assertEqual(policy["settings_during"]["allow_fp16_accumulation"], False)
        self.assertEqual((backend.reduced, backend.split_k, backend.allow_fp16_accumulation),
                         (True, False, True))

    def test_state_restored_when_mm_raises(self):
        backend = FakeMatmulBackend(reduced=True, split_k=False, accumulation=True)

        def failing_mm(_a, _b):
            self.assertEqual((backend.reduced, backend.split_k,
                              backend.allow_fp16_accumulation), (False, False, False))
            raise RuntimeError("reference failed")

        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            fp16_mm_reference(None, None, torch_module=fake_torch(backend, failing_mm))
        self.assertEqual((backend.reduced, backend.split_k, backend.allow_fp16_accumulation),
                         (True, False, True))

    def test_split_k_stays_at_prior_value_during_reference(self):
        backend = FakeMatmulBackend(reduced=True, split_k=True, accumulation=False)

        def mm(_a, _b):
            self.assertEqual((backend.reduced, backend.split_k), (False, True))
            return "reference"

        _, policy = fp16_mm_reference(None, None,
                                      torch_module=fake_torch(backend, mm))
        self.assertTrue(policy["settings_during"]["allow_fp16_reduced_precision_reduction_split_k"])
        self.assertEqual(policy["settings_before"], policy["settings_after"])

    def test_backend_without_optional_split_k_or_accumulation(self):
        backend = SimpleNamespace(allow_fp16_reduced_precision_reduction=True,
                                  allow_tf32=False)

        def mm(_a, _b):
            self.assertFalse(backend.allow_fp16_reduced_precision_reduction)
            return "legacy reference"

        result, policy = fp16_mm_reference(None, None,
                                            torch_module=fake_torch(backend, mm))
        self.assertEqual(result, "legacy reference")
        self.assertFalse(policy["split_k_setting_supported"])
        self.assertFalse(policy["fp16_accumulation_setting_supported"])
        self.assertEqual(policy["settings_before"], policy["settings_after"])
        self.assertTrue(backend.allow_fp16_reduced_precision_reduction)


if __name__ == "__main__":
    unittest.main()
