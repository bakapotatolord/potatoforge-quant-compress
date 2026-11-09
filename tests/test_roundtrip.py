from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors import safe_open

from fixtures import H4, bf16_bits, mixed_checkpoint, rewrite_manifest, save_fixture
from pf_quant_compress.archive import compress_checkpoint, decompress_archive
from pf_quant_compress.optimize import optimize_weight, rows_per_batch
from pf_quant_compress.zstd import decompress_into
from pf_quant_compress.safetensors_io import Checkpoint
from pf_quant_compress.verify import verify_archive


ROOT = Path(__file__).resolve().parents[1]
WEIGHT = "blocks.0.proj.weight"
SCALE = "blocks.0.proj.weight_scale"
PROTECT = ("blocks.18.",)


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.source = self.folder / "source.safetensors"
        self.archive = self.folder / "model.pfqc.safetensors"
        self.restored = self.folder / "restored.safetensors"
        self.bf16 = self.folder / "reference.safetensors"
        self.metadata = {"title": "Mixed precision ✓", "config": '{"value": "kept"}'}
        self.original = mixed_checkpoint(self.source)
        self.original.update({
            WEIGHT: np.random.default_rng(817).integers(-16, 16, (24, 1024), dtype=np.int8),
            SCALE: np.full((24, 1), 0.125, dtype="<f4"),
            "empty": np.empty((2, 0, 3), dtype="<f2"),
            "scalar": np.array(42, dtype="<i8"),
        })
        self.original["blocks.18.proj.weight"] = self.original[WEIGHT].copy()
        self.original["blocks.18.proj.weight_scale"] = self.original[SCALE].copy()
        self.save_source()

    def save_source(self):
        save_fixture(self.source, self.original, bf16=("ordinary.weight",), metadata=self.metadata)

    def test_roundtrip_preserves_metadata_and_every_passthrough_bit(self):
        source_bytes = self.source.read_bytes()
        source = Checkpoint(self.source)
        settings = {"target_bpw": 4.8, "max_error": 0.003, "protect": PROTECT}
        expected = optimize_weight(source, WEIGHT, **settings)
        self.assertGreater(expected.selected.multiplier, 1)
        report = compress_checkpoint(self.source, self.archive, **settings)
        self.assertEqual(report["optimized_family_count"], 1)
        self.assertEqual(report["protected_family_count"], 1)
        self.assertEqual(report["eligible_family_count"], 1)
        self.assertEqual(report["preserved_tensor_count"], len(self.original) - 2)
        self.assertFalse(report["rate_target_misses"])
        self.assertEqual(report["payload_bpw"], expected.payload_bpw)
        self.assertTrue(report["checksums_verified"])
        self.assertGreater(report["wider_scale_savings_bytes"], 0)

        compressed = Checkpoint(self.archive)
        with safe_open(self.archive, framework="np") as reader:
            codes = reader.get_tensor(WEIGHT)
            self.assertEqual((codes.dtype, codes.ndim), (np.dtype("uint8"), 1))
            decoded = bytearray(expected.codes.nbytes)
            decompress_into([codes.tobytes()], decoded)
            self.assertEqual(decoded, expected.codes.tobytes())
            np.testing.assert_array_equal(reader.get_tensor(SCALE), expected.scales)
        for name, tensor in source.tensors.items():
            if name in (WEIGHT, SCALE):
                continue
            original = compressed.tensors[name]
            self.assertEqual((original.dtype, original.shape), (tensor.dtype, tensor.shape))
            self.assertEqual(b"".join(compressed.chunks(name)), self.original[name].tobytes(), name)

        verified = verify_archive(self.archive, reference_quantized=self.source)
        self.assertEqual(verified["bit_identical_preserved_tensors"], len(self.original) - 2)
        self.assertEqual(len(verified["optimized_distortion"]), 1)
        self.assertAlmostEqual(verified["optimized_distortion"][0]["relative_sse_vs_quantized"], expected.selected.added_error)
        self.assertEqual(self.source.read_bytes(), source_bytes)

        decompress_archive(self.archive, self.restored)
        with safe_open(self.source, framework="np") as before, safe_open(self.restored, framework="np") as after:
            self.assertEqual(before.metadata(), after.metadata())
            self.assertEqual(before.offset_keys(), after.offset_keys())
            for name in before.keys():
                a, b = before.get_slice(name), after.get_slice(name)
                self.assertEqual(a.get_dtype(), b.get_dtype())
                self.assertEqual(a.get_shape(), b.get_shape())
        restored = Checkpoint(self.restored)
        wanted = {**self.original, WEIGHT: expected.codes, SCALE: expected.scales}
        for name, array in wanted.items():
            self.assertEqual(b"".join(restored.chunks(name)), array.tobytes(), name)

    def test_reference_verify_detects_differences_in_a_protected_scale(self):
        compress_checkpoint(self.source, self.archive, target_bpw=4.8, max_error=0.003, protect=PROTECT)
        self.original["blocks.18.proj.weight_scale"][0, 0] *= np.float32(1.5)
        self.save_source()
        with self.assertRaisesRegex(ValueError, "blocks.18.proj.weight_scale"):
            verify_archive(self.archive, reference_quantized=self.source)

    def test_reference_verify_checks_the_measured_distortion_cap(self):
        compress_checkpoint(self.source, self.archive, target_bpw=4.8, max_error=0.003, protect=PROTECT)
        edited = self.folder / "edited.pfqc.safetensors"

        def claim_no_error(manifest):
            manifest["settings"]["max_error"] = 0
            manifest["families"][WEIGHT]["fit"]["added_relative_sse"] = 0

        rewrite_manifest(self.archive, edited, claim_no_error)
        self.assertTrue(verify_archive(edited)["checksums_verified"])
        with self.assertRaisesRegex(ValueError, "distortion violates the cap"):
            verify_archive(edited, reference_quantized=self.source)

    def test_reference_verification_matches_fit_at_strict_cap_across_batches(self):
        self.assertEqual(rows_per_batch(4096), 16)
        rng = np.random.default_rng(201)
        self.original[WEIGHT] = np.clip(np.rint(rng.normal(0, 19, (96, 4096))), -128, 127).astype(np.int8)
        self.original[SCALE] = rng.uniform(0.002, 0.01, (96, 1)).astype("<f4")
        self.save_source()
        source = Checkpoint(self.source)
        fit = optimize_weight(source, WEIGHT, target_bpw=6.1, max_error=0.001, protect=PROTECT)
        self.assertGreater(fit.selected.multiplier, 1)
        cap = fit.selected.added_error
        report = compress_checkpoint(self.source, self.archive, target_bpw=6.1, max_error=cap, protect=PROTECT)
        selected = next(item for item in report["families"] if item["weight"] == WEIGHT)
        self.assertEqual(selected["multiplier"], fit.selected.multiplier)
        self.assertEqual(selected["added_relative_sse"], cap)
        verified = verify_archive(self.archive, reference_quantized=self.source)
        self.assertEqual(verified["optimized_distortion"][0]["relative_sse_vs_quantized"], cap)
        tighter = optimize_weight(source, WEIGHT, target_bpw=6.1, max_error=np.nextafter(cap, 0), protect=PROTECT)
        self.assertLessEqual(tighter.selected.added_error, np.nextafter(cap, 0))
        self.assertNotEqual(tighter.selected.multiplier, fit.selected.multiplier)

    def test_bf16_guided_archive_decodes_after_both_inputs_are_gone(self):
        rng = np.random.default_rng(7)
        weights = rng.integers(-32, 33, (32, 2048))
        rotated = (weights.reshape(-1, 4) @ H4.T).reshape(weights.shape)
        self.original[WEIGHT] = np.clip(rotated * 2 + rng.choice([-3, 3], size=weights.shape), -128, 127).astype(np.int8)
        self.original[SCALE] = np.full((len(weights), 1), 0.5, dtype="<f4")
        self.save_source()
        save_fixture(self.bf16, {WEIGHT: bf16_bits(weights)}, bf16=(WEIGHT,))
        report = compress_checkpoint(
            self.source, self.archive, target_bpw=7, max_error=0, protect=PROTECT, reference=self.bf16,
        )
        fit = next(family for family in report["families"] if family["weight"] == WEIGHT)
        self.assertEqual(report["settings"]["reference_mode"], "bf16")
        self.assertEqual(fit["added_relative_sse"], 0)
        self.assertLess(fit["optimized_reference_error"], fit["baseline_reference_error"])
        self.assertLess(fit["raw_reference_error_delta"], 0)
        verified = verify_archive(self.archive, reference_quantized=self.source)
        self.assertGreater(verified["optimized_distortion"][0]["relative_sse_vs_quantized"], 0)
        self.source.unlink()
        self.bf16.unlink()
        self.assertTrue(verify_archive(self.archive)["checksums_verified"])
        decompress_archive(self.archive, self.restored)
        with safe_open(self.restored, framework="np") as reader:
            self.assertEqual(reader.metadata(), self.metadata)
            self.assertEqual(reader.get_slice(WEIGHT).get_shape(), list(weights.shape))


if __name__ == "__main__":
    unittest.main()
