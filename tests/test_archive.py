from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from safetensors import SafetensorError, TensorSpec, safe_open, serialize_file

from fixtures import family, rewrite_manifest, save_fixture
from pf_quant_compress import native
from pf_quant_compress.archive import compress_checkpoint, decompress_archive, open_archive
from pf_quant_compress.native import libdeflate
from pf_quant_compress.safetensors_io import CHUNK_BYTES, Checkpoint, atomic_output
from pf_quant_compress.verify import verify_archive


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "potatoforge_quant_compress.manifest"
WEIGHT = "linear.weight"
SCALE = "linear.weight_scale"
MARKER = "linear.comfy_quant"


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.source = self.folder / "source.safetensors"
        self.archive = self.folder / "model.pfqc.safetensors"
        self.restored = self.folder / "restored.safetensors"
        self.bad = self.folder / "bad.pfqc.safetensors"
        self.values = np.arange(2**18 + 1, dtype=np.int16)
        save_fixture(self.source, {"values": self.values})

    def compress(self):
        return compress_checkpoint(self.source, self.archive, target_bpw=6.5, max_error=0.001)

    def test_native_passthrough_and_self_contained_streaming(self):
        report = self.compress()
        self.assertEqual(report["archive_bytes"], self.archive.stat().st_size)
        self.assertEqual(report["source_bytes"], self.source.stat().st_size)
        with safe_open(self.archive, framework="np") as reader:
            self.assertEqual(reader.keys(), ["values"])
            np.testing.assert_array_equal(reader.get_tensor("values"), self.values)
            manifest = json.loads(reader.metadata()[MANIFEST])
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["tensors"][0]["stored_bytes"], self.values.nbytes)
            self.assertEqual(manifest["tensors"][0]["codec"], "stored")

        with open_archive(self.archive) as checkpoint:
            chunks = list(checkpoint.chunks("values"))
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(len(chunk) <= CHUNK_BYTES for chunk in chunks))
            self.assertEqual(b"".join(chunks), self.values.tobytes())
        self.source.unlink()
        decompress_archive(self.archive, self.restored)
        self.assertEqual(self.restored.stat().st_size, report["reconstructed_bytes"])
        with safe_open(self.restored, framework="np") as reader:
            self.assertIsNone(reader.metadata())
            np.testing.assert_array_equal(reader.get_tensor("values"), self.values)

    def test_lossless_coded_weights_are_u8_and_keep_native_scales(self):
        rng = np.random.default_rng(201)
        tensors = family("linear")
        tensors[WEIGHT] = np.clip(np.rint(rng.normal(0, 19, (96, 4096))), -128, 127).astype(np.int8)
        tensors[SCALE] = np.ones((96, 1), dtype="<f4")
        save_fixture(self.source, tensors)
        report = compress_checkpoint(self.source, self.archive, target_bpw=8, max_error=0)
        self.assertEqual(report["encoded_weight_count"], 1)
        self.assertEqual(report["optimized_family_count"], 0)
        with safe_open(self.archive, framework="np") as reader:
            payload = reader.get_tensor(WEIGHT)
            self.assertEqual((payload.dtype, payload.ndim), (np.dtype("uint8"), 1))
            self.assertGreater(payload.nbytes, CHUNK_BYTES)
            self.assertEqual(zlib.decompress(payload.tobytes(), wbits=-15), tensors[WEIGHT].tobytes())
            np.testing.assert_array_equal(reader.get_tensor(SCALE), tensors[SCALE])
            np.testing.assert_array_equal(reader.get_tensor(MARKER), tensors[MARKER])
            manifest = json.loads(reader.metadata()[MANIFEST])
            weight = next(tensor for tensor in manifest["tensors"] if tensor["name"] == WEIGHT)
            self.assertEqual(weight["storage"], "encoded_int8")
        with open_archive(self.archive) as packed:
            chunks = list(packed.chunks(WEIGHT, chunk_size=98304))
            self.assertGreater(len(chunks), 1)
            self.assertEqual(b"".join(chunks), tensors[WEIGHT].tobytes())
        report = verify_archive(self.archive, reference_quantized=self.source)
        self.assertEqual(report["bit_identical_preserved_tensors"], len(tensors))

    def test_identical_settings_produce_identical_archives(self):
        self.compress()
        repeat = self.folder / "repeat.pfqc.safetensors"
        compress_checkpoint(self.source, repeat, target_bpw=6.5, max_error=0.001)
        self.assertEqual(self.archive.read_bytes(), repeat.read_bytes())

    def test_archive_order_is_deterministic_for_tensors_with_equal_offsets(self):
        tensors = {"value": np.array([1], dtype=np.int32)}
        tensors.update({f"empty.{index}": np.empty((0,), dtype=np.uint8) for index in range(12)})
        save_fixture(self.source, tensors)
        self.compress()
        repeat = self.folder / "repeat.pfqc.safetensors"
        compress_checkpoint(self.source, repeat, target_bpw=6.5, max_error=0.001)
        self.assertEqual(self.archive.read_bytes(), repeat.read_bytes())

    def test_incompressible_eligible_weight_remains_native_i8(self):
        tensors = family("linear")
        tensors[WEIGHT] = np.random.default_rng(523).integers(-128, 128, (2, 4096), dtype=np.int8)
        save_fixture(self.source, tensors)
        report = compress_checkpoint(self.source, self.archive, target_bpw=6.5, max_error=0)
        self.assertEqual(report["encoded_weight_count"], 0)
        self.assertEqual(report["rate_target_misses"], 1)
        self.assertEqual(report["payload_bpw"], 8)
        with safe_open(self.archive, framework="np") as reader:
            self.assertEqual(reader.get_slice(WEIGHT).get_dtype(), "I8")
            np.testing.assert_array_equal(reader.get_tensor(WEIGHT), tensors[WEIGHT])

    def test_outputs_require_force_and_cannot_alias_inputs(self):
        self.archive.write_bytes(b"old archive")
        with self.assertRaises(FileExistsError):
            self.compress()
        self.assertEqual(self.archive.read_bytes(), b"old archive")
        compress_checkpoint(self.source, self.archive, target_bpw=6.5, max_error=0, force=True)
        self.restored.write_bytes(b"old checkpoint")
        with self.assertRaises(FileExistsError):
            decompress_archive(self.archive, self.restored)
        self.assertEqual(self.restored.read_bytes(), b"old checkpoint")
        decompress_archive(self.archive, self.restored, force=True)
        with safe_open(self.restored, framework="np") as reader:
            self.assertEqual(reader.keys(), ["values"])
        with self.assertRaisesRegex(ValueError, "input"):
            decompress_archive(self.archive, self.archive, force=True)
        with self.assertRaisesRegex(ValueError, "input"):
            compress_checkpoint(self.source, self.source, target_bpw=6, max_error=0, force=True)

    def test_no_clobber_checks_again_at_publication(self):
        with self.assertRaises(FileExistsError), atomic_output(self.restored, force=False) as temporary:
            temporary.write_bytes(b"staged")
            self.restored.write_bytes(b"concurrent output")
        self.assertEqual(self.restored.read_bytes(), b"concurrent output")
        self.assertFalse(list(self.folder.glob(".pfqc-*.tmp")))

    def test_rejects_malformed_and_unsupported_metadata(self):
        for text in ("{", "[]", '{"format": "a", "format": "b"}'):
            with self.subTest(text=text):
                save_fixture(self.bad, {"values": self.values[:4]}, metadata={MANIFEST: text})
                with self.assertRaisesRegex(ValueError, "Malformed PFQC compression metadata"):
                    verify_archive(self.bad)
        self.compress()
        for update, message in (
            (lambda manifest: manifest.update(schema_version=3), "schema version"),
            (lambda manifest: manifest.update(format="other"), "format"),
            (lambda manifest: manifest.update(tensors=[]), "missing storage tensors"),
            (lambda manifest: manifest["tensors"][0].update(name="missing"), "Missing storage tensor"),
            (lambda manifest: manifest["tensors"][0].update(codec="other"), "Unsupported tensor codec"),
            (lambda manifest: manifest["tensors"][0].update(shape=[-1]), "shape"),
            (lambda manifest: manifest["tensors"][0].update(shape=[2**40]), "size limit"),
            (lambda manifest: manifest["tensors"][0].update(decoded_bytes=1), "byte counts"),
            (lambda manifest: manifest["tensors"][0].update(dtype="OBJECT"), "dtype"),
            (lambda manifest: manifest["tensors"][0].update(dtype="F4", shape=[3]), "byte boundary"),
            (lambda manifest: manifest["tensors"][0].update(storage="optimized_int8"), "storage mode"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                rewrite_manifest(self.archive, self.bad, update)
                verify_archive(self.bad)

    def test_enforces_manifest_total_and_ratio_bounds(self):
        self.compress()
        for limit, value, message in (
            ("MAX_MANIFEST_BYTES", 64, "manifest"),
            ("MAX_DECODED_BYTES", 64, "total decoded"),
        ):
            with self.subTest(limit=limit), patch(f"pf_quant_compress.decoder.{limit}", value):
                with self.assertRaisesRegex(ValueError, message):
                    verify_archive(self.archive)
        self.write_v1_fixture()
        with patch("pf_quant_compress.decoder.MAX_DEFLATE_RATIO", 1):
            with self.assertRaisesRegex(ValueError, "compression ratio"):
                verify_archive(self.archive)

    def test_rejects_corruption_and_keeps_existing_output(self):
        self.compress()
        self.restored.write_bytes(b"previous checkpoint")
        data = bytearray(self.archive.read_bytes())
        offset = Checkpoint(self.archive).tensors["values"].offset
        data[offset + 5] ^= 0xFF
        self.bad.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            decompress_archive(self.bad, self.restored, force=True)
        self.assertEqual(self.restored.read_bytes(), b"previous checkpoint")
        self.assertFalse(list(self.folder.glob(".pfqc-*.tmp")))
        self.bad.write_bytes(data[:-1])
        with self.assertRaises(SafetensorError):
            verify_archive(self.bad)

    def write_v1_fixture(self, *, damage=None, signed_payload=False):
        # Build the documented schema without the production compressor.
        tensors = family("linear")
        tensors[WEIGHT] = np.tile(np.arange(-8, 8, dtype=np.int8), (2, 8))
        raw = tensors[WEIGHT].tobytes()
        encoded = {"overlong": raw + b"x", "short": raw[:-1]}.get(damage, raw)
        stream = zlib.compress(encoded, level=3, wbits=-15)
        if damage == "trailing":
            stream += b"x"
        elif damage == "truncated":
            stream = stream[:-1]
        elif damage == "invalid":
            stream = b"\x07" + stream[1:]
        records = []
        for name, tensor in tensors.items():
            is_weight = name == WEIGHT
            records.append({
                "name": name, "dtype": {WEIGHT: "I8", SCALE: "F32", MARKER: "U8"}[name],
                "shape": list(tensor.shape), "family": WEIGHT, "storage": "encoded_int8" if is_weight else "unchanged",
                "original_bytes": tensor.nbytes, "decoded_bytes": tensor.nbytes,
                "stored_bytes": len(stream) if is_weight else tensor.nbytes,
                "codec": "deflate" if is_weight else "stored", "sha256": sha256(tensor.tobytes()).hexdigest(),
            })
        manifest = {
            "format": "potatoforge_quant_compress", "schema_version": 1, "compressor_version": "0.1.0",
            "source_format": "safetensors", "method": "wider-scale", "zlib_level": 3,
            "settings": {"target_bpw": 8, "max_error": 0.001, "protect": [], "reference_mode": "quantized-only"},
            "metadata": {"label": "independent fixture"}, "source_bytes": 1024, "tensors": records,
            "families": {WEIGHT: {
                "family": {"weight": WEIGHT, "scale": SCALE, "marker": MARKER, "format": "int8_tensorwise",
                           "status": "eligible", "reason": None, "group_size": 4},
                "fit": {"multiplier": 1.0, "baseline_weight_bytes": len(stream), "baseline_scale_bytes": tensors[SCALE].nbytes,
                        "payload_bpw": 8 * len(stream) / len(raw), "all_in_bpw": 8 * (len(stream) + tensors[SCALE].nbytes) / len(raw),
                        "rate_target_missed": False, "added_relative_sse": 0.0, "baseline_reference_error": None,
                        "optimized_reference_error": None},
            }},
        }
        tensors[WEIGHT] = np.frombuffer(stream, dtype=np.int8 if signed_payload else np.uint8)
        save_fixture(self.archive, tensors, metadata={MANIFEST: json.dumps(manifest)})
        return raw

    def test_reads_independent_native_v1_fixture(self):
        raw = self.write_v1_fixture()
        decompress_archive(self.archive, self.restored)
        checkpoint = Checkpoint(self.restored)
        self.assertEqual(checkpoint.metadata, {"label": "independent fixture"})
        self.assertEqual(checkpoint.tensors[WEIGHT].shape, (2, 128))
        self.assertEqual(b"".join(checkpoint.chunks(WEIGHT)), raw)

    def test_rejects_invalid_encoded_payloads(self):
        for damage, message in (
            ("overlong", "Decoded size exceeds"), ("trailing", "Trailing DEFLATE"), ("truncated", "Incomplete DEFLATE"),
        ):
            with self.subTest(damage=damage), self.assertRaisesRegex(ValueError, message):
                self.write_v1_fixture(damage=damage)
                verify_archive(self.archive)
            with self.subTest(damage=damage, verify_checksums=False):
                with self.assertRaisesRegex(ValueError, message):
                    decompress_archive(self.archive, self.restored, verify_checksums=False)
                self.assertFalse(self.restored.exists())
        self.write_v1_fixture(signed_payload=True)
        with self.assertRaisesRegex(ValueError, "Invalid U8 DEFLATE payload"):
            verify_archive(self.archive)

    def test_check_integrity_still_verifies_an_archive_opened_without_checksums(self):
        self.write_v1_fixture()
        rewrite_manifest(self.archive, self.bad, lambda manifest: manifest["tensors"][0].update(sha256="0" * 64))
        with open_archive(self.bad, verify_checksums=False) as archive:
            self.assertEqual(len(b"".join(archive.chunks(WEIGHT))), archive.tensors[WEIGHT].nbytes)
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                archive.check_integrity()

    def test_decode_into_verifies_both_backends(self):
        native = libdeflate()
        for library in (None, native) if native else (None,):
            with self.subTest(native=library is not None), patch("pf_quant_compress.decoder.libdeflate", return_value=library):
                raw = self.write_v1_fixture()
                with open_archive(self.archive) as archive:
                    output = bytearray(len(raw))
                    archive.decode_into(WEIGHT, output)
                    self.assertEqual(output, raw)
                    with self.assertRaisesRegex(ValueError, "Invalid output buffer"):
                        archive.decode_into(WEIGHT, bytearray(len(raw) - 1))
                rewrite_manifest(self.archive, self.bad, lambda manifest: manifest["tensors"][0].update(sha256="0" * 64))
                with open_archive(self.bad) as archive, self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    archive.decode_into(WEIGHT, output)
                for damage, message in (
                    ("invalid", "DEFLATE"), ("truncated", "DEFLATE"),
                    ("overlong", "Decoded size exceeds"), ("short", "length|size"),
                    ("trailing", "Trailing DEFLATE"),
                ):
                    with self.subTest(damage=damage):
                        self.write_v1_fixture(damage=damage)
                        with open_archive(self.archive, verify_checksums=False) as archive:
                            with self.assertRaisesRegex(ValueError, message):
                                archive.decode_into(WEIGHT, output)

    def test_optional_native_backends_fall_back(self):
        with patch.object(native.sysconfig, "get_platform", return_value="win-arm64"):
            self.assertIsNone(native.libdeflate.__wrapped__())
        with (
            patch.object(native.sysconfig, "get_platform", return_value="win-amd64"),
            patch.object(native, "_DLL", self.folder / "missing.dll"),
        ):
            self.assertIsNone(native.libdeflate.__wrapped__())
        data = memoryview(b"PFQC checksum")
        with patch.object(native, "_bcrypt_sha256", return_value=None):
            self.assertEqual(native.sha256_hex(data), sha256(data).hexdigest())

    def test_rejects_encoded_protected_weight_and_duplicate_logical_names(self):
        self.write_v1_fixture()

        def protect(manifest):
            manifest["settings"]["protect"] = ["linear."]
            family = manifest["families"][WEIGHT]
            family["fit"] = None
            family["family"].update(status="protected", reason="Matched a protected weight prefix")

        rewrite_manifest(self.archive, self.bad, protect)
        with self.assertRaisesRegex(ValueError, "Ineligible encoded weight"):
            verify_archive(self.bad)
        rewrite_manifest(self.archive, self.bad, lambda manifest: manifest["tensors"][1].update(name=WEIGHT))
        with self.assertRaisesRegex(ValueError, "duplicate tensor name"):
            verify_archive(self.bad)

    def test_preserves_subbyte_safetensors_storage_and_empty_metadata(self):
        packed = np.array([[0xAB, 0x2D]], dtype=np.uint8)
        tensor = TensorSpec(
            dtype="float4_e2m1fn_x2", shape=packed.shape, data_ptr=packed.ctypes.data, data_len=packed.nbytes,
        )
        serialize_file({"packed": tensor}, self.source, {})
        self.compress()
        decompress_archive(self.archive, self.restored)
        with safe_open(self.restored, framework="np") as reader:
            self.assertEqual(reader.metadata(), {})
            self.assertEqual(reader.get_slice("packed").get_dtype(), "F4")
            self.assertEqual(reader.get_slice("packed").get_shape(), [1, 4])
        self.assertEqual(b"".join(Checkpoint(self.restored).chunks("packed")), packed.tobytes())


if __name__ == "__main__":
    unittest.main()
