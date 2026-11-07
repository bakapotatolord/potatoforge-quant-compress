from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from fixtures import H4, bf16_bits, family, mixed_checkpoint, save_fixture
from pf_quant_compress.archive import compress_checkpoint
from pf_quant_compress.decoder import open_archive
from pf_quant_compress.native import libdeflate
from pf_quant_compress.optimize import (
    SCALE_MULTIPLIERS, _measure_batch, _requantize, _rotate_bf16, encoded_chunks, optimize_weight,
)
from pf_quant_compress.safetensors_io import Checkpoint
from pf_quant_compress.verify import verify_archive


ROOT = Path(__file__).resolve().parents[1]


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.source = Path(temp.name) / "source.safetensors"
        self.reference = Path(temp.name) / "reference.safetensors"

    def write_source(self, codes, scales):
        tensors = family("linear")
        tensors.update({"linear.weight": codes, "linear.weight_scale": scales})
        save_fixture(self.source, tensors)
        return Checkpoint(self.source)

    def write_reference(self, bits, key="linear.weight"):
        save_fixture(self.reference, {key: bits}, bf16=(key,))
        return Checkpoint(self.reference)

    def assert_encoded_bytes(self, array, member):
        stream = b"".join(encoded_chunks(array, member.codec))
        self.assertEqual(len(stream), member.nbytes)
        decoded = zlib.decompress(stream, wbits=-15) if member.codec == "deflate" else stream
        self.assertEqual(decoded, array.tobytes())
        return decoded

    def test_quantized_only_meets_target_and_round_trips_exactly(self):
        rng = np.random.default_rng(201)
        codes = np.clip(np.rint(rng.normal(0, 19, (192, 4096))), -128, 127).astype(np.int8)
        codes[0] = 0
        scales = rng.uniform(0.002, 0.01, (192, 1)).astype("<f4")
        source = self.write_source(codes, scales)
        original = self.source.read_bytes()

        result = optimize_weight(source, "linear.weight", target_bpw=6.1, max_error=0.001)
        self.assertGreater(result.selected.multiplier, 1.0)
        self.assertLess(result.selected.total_bytes, result.baseline.total_bytes)
        self.assertFalse(result.rate_target_missed)
        self.assertLessEqual(result.payload_bpw, 6.1)
        self.assertGreater(result.all_in_bpw, result.payload_bpw)
        self.assertIsNone(result.selected.reference_error)
        self.assertEqual(result.scales.dtype, np.dtype("<f4"))
        self.assertEqual(result.codes.dtype, np.dtype("i1"))
        np.testing.assert_array_equal(result.codes[0], codes[0])
        np.testing.assert_array_equal(result.scales[0], scales[0])

        q1 = np.frombuffer(self.assert_encoded_bytes(result.codes, result.selected.weight), dtype="i1").reshape(codes.shape)
        s1 = np.frombuffer(self.assert_encoded_bytes(result.scales, result.selected.scale), dtype="<f4").reshape(scales.shape)
        w0 = codes.astype(np.float64) * scales
        w1 = q1.astype(np.float64) * s1
        error = np.sum((w1 - w0) ** 2) / np.sum(w0 ** 2)
        self.assertGreater(error, 0.0)
        self.assertLessEqual(error, 0.001)
        self.assertAlmostEqual(error, result.selected.added_error, places=14)
        repeated = optimize_weight(source, "linear.weight", target_bpw=6.1, max_error=0.001)
        self.assertEqual(repeated.selected, result.selected)
        self.assertEqual(repeated.codes.tobytes(), q1.tobytes())
        self.assertEqual(repeated.scales.tobytes(), s1.tobytes())
        self.assertEqual(self.source.read_bytes(), original)

    def test_selection_matches_independently_measured_grid(self):
        codes = np.random.default_rng(817).integers(-16, 16, (24, 1024), dtype=np.int8)
        scales = np.full((24, 1), 0.125, dtype="<f4")
        source = self.write_source(codes, scales)
        original = codes.astype(np.float64) * scales

        def stored_size(values):
            return min(values.nbytes, len(zlib.compress(values.tobytes(), level=3, wbits=-15)))

        baseline_bytes = stored_size(codes) + scales.nbytes
        valid = []
        for multiplier in SCALE_MULTIPLIERS[1:]:
            scale = np.full(scales.shape, 0.125 * multiplier, dtype="<f4")
            trial = np.clip(np.rint(original / scale), -128, 127).astype(np.int8)
            error = np.sum((trial.astype(np.float64) * scale - original) ** 2) / np.sum(original ** 2)
            weight_bytes = stored_size(trial)
            total_bytes = weight_bytes + scale.nbytes
            if error <= 0.003 and total_bytes < baseline_bytes:
                valid.append((multiplier, error, total_bytes, 8 * weight_bytes / codes.size))

        for target in (5.2, 1.0):
            with self.subTest(target=target):
                result = optimize_weight(source, "linear.weight", target_bpw=target, max_error=0.003)
                hits = [trial for trial in valid if trial[3] <= target]
                best = min(hits, key=lambda trial: (trial[1], trial[2], trial[0])) if hits else min(
                    valid, key=lambda trial: (trial[2], trial[1], trial[0])
                )
                self.assertEqual(result.rate_target_missed, target == 1.0)
                self.assertEqual(result.selected.multiplier, best[0])
                self.assertAlmostEqual(result.selected.added_error, best[1], places=14)
                self.assertEqual(result.selected.total_bytes, best[2])

    def test_tiny_matrix_stays_exact_when_widening_cannot_save_space(self):
        codes = np.array([[1, -4, 8, -16]], dtype=np.int8)
        scales = np.array([[1]], dtype="<f4")
        result = optimize_weight(self.write_source(codes, scales), "linear.weight", target_bpw=1, max_error=1)
        self.assertEqual(result.selected, result.baseline)
        self.assertEqual(result.selected.total_bytes, codes.nbytes + scales.nbytes)
        self.assertEqual(result.codes.tobytes(), codes.tobytes())
        self.assertEqual(result.scales.tobytes(), scales.tobytes())

    def test_batch_major_matches_reference_across_batches(self):
        rng = np.random.default_rng(17)
        codes = np.clip(np.rint(rng.normal(0, 24, (30, 1024))), -128, 127).astype(np.int8)
        codes[8] = 0
        scales = rng.uniform(.002, .01, (len(codes), 1)).astype("<f4")
        source = self.write_source(codes, scales)
        rotated = codes * scales
        guide = (rotated.reshape(-1, 4) @ H4.T).reshape(codes.shape)
        reference = self.write_reference(bf16_bits(guide))
        with patch("pf_quant_compress.optimize._BATCH_WEIGHTS", 4096):
            for ref in (None, reference):
                for target in (4.0, 6.5, 6.8):
                    with self.subTest(bf16=ref is not None, target=target):
                        settings = dict(target_bpw=target, max_error=.001, reference=ref)
                        old = optimize_weight(source, "linear.weight", search="reference", **settings)
                        new = optimize_weight(source, "linear.weight", **settings)
                        self.assertEqual((new.baseline, new.selected), (old.baseline, old.selected))
                        self.assertEqual(new.codes.tobytes(), old.codes.tobytes())
                        self.assertEqual(new.scales.tobytes(), old.scales.tobytes())

    def test_libdeflate_unavailable_or_oversized_uses_zlib(self):
        codes = np.tile(np.arange(-16, 16, dtype=np.int8), (6, 8))
        source = self.write_source(codes, np.ones((len(codes), 1), dtype="<f4"))
        options = dict(target_bpw=1, max_error=.001)
        expected = optimize_weight(source, "linear.weight", **options)
        for key, value in (("libdeflate", lambda: None), ("NATIVE_TENSOR_LIMIT", codes.nbytes - 1)):
            with self.subTest(key=key), patch(f"pf_quant_compress.optimize.{key}", value):
                fit = optimize_weight(source, "linear.weight", encoder="libdeflate", **options)
                self.assertEqual(fit.encoder, "zlib")
                self.assertEqual(fit.selected, expected.selected)

    @unittest.skipUnless(libdeflate(), "Requires the bundled Windows x64 libdeflate")
    def test_libdeflate_winner_stream_is_written_and_read_by_zlib(self):
        codes = np.random.default_rng(817).integers(-16, 16, (24, 1024), dtype=np.int8)
        source = self.write_source(codes, np.full((len(codes), 1), .125, dtype="<f4"))
        options = dict(target_bpw=1, max_error=.003, encoder="libdeflate")
        fit = optimize_weight(source, "linear.weight", **options)
        self.assertGreater(fit.selected.multiplier, 1)
        self.assertEqual(len(fit.encoded_weight), fit.selected.weight.nbytes)
        output = self.source.with_name("native.pfqc.safetensors")
        report = compress_checkpoint(self.source, output, **options)
        self.assertEqual(report["encoder_counts"], {"libdeflate": 1})
        with open_archive(output) as archive:
            self.assertEqual(archive.manifest["zlib_level"], 3)
            self.assertEqual(b"".join(Checkpoint(output).chunks("linear.weight")), fit.encoded_weight)
            self.assertEqual(b"".join(archive.chunks("linear.weight")), fit.codes.tobytes())
            self.assertEqual(b"".join(archive.chunks("linear.weight_scale")), fit.scales.tobytes())

    def test_incompressible_with_zero_cap_uses_raw_and_exact_noop(self):
        codes = np.random.default_rng(523).integers(-128, 128, (17, 4096), dtype=np.int8)
        scales = np.full((17, 1), 0.125, dtype="<f4")
        result = optimize_weight(self.write_source(codes, scales), "linear.weight", target_bpw=6.0, max_error=0)
        self.assertEqual(result.selected, result.baseline)
        self.assertTrue(result.rate_target_missed)
        self.assertEqual(result.selected.weight.codec, "stored")
        self.assertEqual(result.payload_bpw, 8.0)
        self.assertEqual(result.codes.tobytes(), codes.tobytes())
        self.assertEqual(result.scales.tobytes(), scales.tobytes())
        self.assert_encoded_bytes(result.codes, result.selected.weight)

    def test_all_zero_and_extreme_scales_are_safe(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        huge = np.finfo(np.float32).max
        scales = np.array([[tiny], [huge]], dtype="<f4")
        for codes in (np.zeros((2, 4), dtype=np.int8), np.array([[-128, 127, 1, -1]] * 2, dtype=np.int8)):
            with self.subTest(codes=codes.tolist()):
                result = optimize_weight(self.write_source(codes, scales), "linear.weight", target_bpw=1, max_error=1)
                self.assertEqual(result.selected.added_error, 0)
                self.assertEqual(result.codes.tobytes(), codes.tobytes())
                self.assertEqual(result.scales.tobytes(), scales.tobytes())
                del result  # Release the Windows read mapping before rewriting the fixture.

    def test_wider_fit_cannot_turn_finite_fp32_weights_into_infinities(self):
        codes = np.clip(np.rint(np.random.default_rng(1234).normal(0, 26, (64, 1024))), -128, 127).astype(np.int8)
        scales = np.full((64, 1), 0.008, dtype="<f4")
        codes[0], scales[0, 0] = 2, 1.69e38
        codes[1], scales[1, 0] = 3, 1.12e38
        self.assertTrue(np.isfinite(codes.astype(np.float32) * scales).all())

        baseline_bpw = 8 * len(zlib.compress(codes.tobytes(), level=3, wbits=-15)) / codes.size
        source = self.write_source(codes, scales)
        settings = {"target_bpw": baseline_bpw - 0.001, "max_error": 0.001}
        result = optimize_weight(source, "linear.weight", **settings)
        with np.errstate(over="ignore"):
            reconstructed = result.codes.astype(np.float32) * result.scales
        self.assertTrue(np.isfinite(reconstructed).all())

        # Simulate an archive produced before candidate fitting checked FP32 range.
        archive = self.source.with_name("overflow.pfqc.safetensors")
        with patch("pf_quant_compress.optimize.introduces_fp32_overflow", return_value=False):
            report = compress_checkpoint(self.source, archive, **settings)
        self.assertEqual(report["optimized_family_count"], 1)
        with self.assertRaisesRegex(ValueError, "FP32 dequantization overflows"):
            verify_archive(archive, reference_quantized=self.source)

    def test_requantization_rounds_ties_to_even_and_clamps(self):
        target = np.array([[1.5, 2.5, -1.5, -2.5, 128.0, -129.0, 127.5, -127.5]])
        codes, scales = _requantize(target, np.ones((1, 8), dtype=np.int8), np.array([[0.5]], dtype="<f4"), 2)
        np.testing.assert_array_equal(codes, [[2, 2, -2, -2, 127, -128, 127, -128]])
        np.testing.assert_array_equal(scales, [[1]])

    def test_protected_unsupported_and_invalid_weights_bypass_fitting(self):
        mixed_checkpoint(self.source)
        original = self.source.read_bytes()
        source = Checkpoint(self.source)
        reference = self.write_reference(bf16_bits([[1, 2, 4, 8]]), key="irrelevant.weight")
        for key in ("blocks.18.proj.weight", "packed.weight", "w6.weight", "unrelated.weight", "wrong_shape.weight"):
            with self.subTest(key=key):
                result = optimize_weight(source, key, target_bpw=1, max_error=1, protect=("blocks.18.",), reference=reference)
                self.assertIsNone(result)
        self.assertEqual(self.source.read_bytes(), original)

    def test_bf16_rotation_matches_known_groups_and_dense_oracle(self):
        weights = [[1, 2, 4, 8, 5, 9, 6, 4]]
        np.testing.assert_array_equal(_rotate_bf16(bf16_bits(weights), 4), [[-0.5, 3.5, 5.5, 6.5, 8, 6, 3, 7]])
        for group_size in (16, 64, 256):
            h = H4
            while h.shape[0] < group_size:
                h = np.kron(h, H4)
            weights = np.arange(-group_size, group_size).reshape(1, 2 * group_size)
            expected = (weights.reshape(-1, group_size) @ h.T).reshape(weights.shape)
            np.testing.assert_array_equal(_rotate_bf16(bf16_bits(weights), group_size), expected)

    def test_bf16_fitting_can_improve_baseline_at_zero_added_error(self):
        rng = np.random.default_rng(7)
        weights = rng.integers(-32, 33, (32, 2048))
        rotated = (weights.reshape(-1, 4) @ H4.T).reshape(weights.shape)
        codes = np.clip(rotated * 2 + rng.choice([-3, 3], size=weights.shape), -128, 127).astype(np.int8)
        scales = np.full((len(codes), 1), 0.5, dtype="<f4")
        source = self.write_source(codes, scales)
        reference = self.write_reference(bf16_bits(weights))
        result = optimize_weight(source, "linear.weight", target_bpw=7, max_error=0, reference=reference)
        self.assertGreater(result.selected.multiplier, 1)
        self.assertEqual(result.selected.added_error, 0)
        self.assertLess(result.selected.reference_error, result.baseline.reference_error)
        for codes, scales, measured in (
            (codes, scales, result.baseline.reference_error),
            (result.codes, result.scales, result.selected.reference_error),
        ):
            error = np.sum((codes.astype(np.float64) * scales - rotated) ** 2) / np.sum(rotated ** 2)
            self.assertAlmostEqual(error, measured, places=14)

    def test_bf16_fit_can_recover_from_a_partial_error_cap_miss(self):
        codes = np.ones((3, 4), dtype=np.int8)
        scales = np.ones((3, 1), dtype="<f4")
        reference = bf16_bits([[1] * 4, [2] * 4, [2] * 4])
        with patch("pf_quant_compress.optimize._BATCH_WEIGHTS", 4):
            # The first row adds squared error. The next two rows each remove it.
            [(candidate, _)] = _measure_batch(codes, scales, reference, 4, (2.0,), None, None, 0.0)
        self.assertEqual(candidate.added_error, 0.0)
        self.assertAlmostEqual(candidate.reference_error, 4 / 36)

    def test_invalid_reference_fails_even_if_source_already_meets_target(self):
        codes = np.ones((1, 4), dtype=np.int8)
        source = self.write_source(codes, np.ones((1, 1), dtype="<f4"))
        valid = bf16_bits([[1, 2, 4, 8]])
        for bits, key, message in (
            (valid, "other.weight", "missing"),
            (valid[:, :2].copy(), "linear.weight", "shape"),
            (np.array([[0x7FC1, 0, 0, 0]], dtype="<u2"), "linear.weight", "finite"),
            (np.array([[0x7F80, 0, 0, 0]], dtype="<u2"), "linear.weight", "finite"),
            (np.zeros_like(valid), "linear.weight", "zero energy"),
        ):
            with self.subTest(bits=bits.tolist(), key=key), self.assertRaisesRegex(ValueError, message):
                optimize_weight(source, "linear.weight", target_bpw=8, max_error=0, reference=self.write_reference(bits, key))
        save_fixture(self.reference, {"linear.weight": np.ones((1, 4), dtype="<f4")})
        with self.assertRaisesRegex(ValueError, "BF16"):
            optimize_weight(source, "linear.weight", target_bpw=8, max_error=0, reference=Checkpoint(self.reference))

    def test_bf16_cap_rejects_worsening_reconstructions(self):
        codes = np.random.default_rng(31).integers(-32, 32, (16, 1024), dtype=np.int8)
        scales = np.ones((len(codes), 1), dtype="<f4")
        unrotated = (codes.reshape(-1, 4) @ H4.T).reshape(codes.shape)
        source = self.write_source(codes, scales)
        reference = self.write_reference(bf16_bits(unrotated))
        result = optimize_weight(source, "linear.weight", target_bpw=1, max_error=0, reference=reference)
        self.assertEqual(result.baseline.reference_error, 0)
        self.assertEqual(result.selected, result.baseline)
        allowed = optimize_weight(source, "linear.weight", target_bpw=1, max_error=0.01, reference=reference)
        self.assertLess(allowed.selected.total_bytes, allowed.baseline.total_bytes)
        self.assertGreater(allowed.selected.added_error, 0)
        self.assertLessEqual(allowed.selected.added_error, 0.01)
        self.assertAlmostEqual(allowed.selected.added_error, allowed.selected.reference_error, places=14)

    def test_invalid_limits_are_rejected(self):
        source = self.write_source(np.zeros((1, 4), dtype=np.int8), np.ones((1, 1), dtype="<f4"))
        for bpw, error in ((0, 0), (np.inf, 0), (6, -0.001), (6, np.nan), (6, np.inf)):
            with self.subTest(bpw=bpw, error=error), self.assertRaises(ValueError):
                optimize_weight(source, "linear.weight", target_bpw=bpw, max_error=error)


if __name__ == "__main__":
    unittest.main()
