from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors import safe_open

from fixtures import family, rewrite_manifest, save_fixture
from pf_quant_compress.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.source = self.folder / "source.safetensors"
        self.archive = self.folder / "model.pfqc.safetensors"
        self.restored = self.folder / "restored.safetensors"
        self.report = self.folder / "report.json"
        self.ref = self.folder / "reference.safetensors"
        tensors = family("linear")
        tensors["linear.weight"] = np.random.default_rng(817).integers(-16, 16, (24, 1024), dtype=np.int8)
        tensors["linear.weight_scale"] = np.full((24, 1), 0.125, dtype="<f4")
        save_fixture(self.source, tensors)
        self.compress = [
            "compress", str(self.source), "--output", str(self.archive), "--target-bpw", "1", "--max-error", "0.003",
        ]

    def invoke(self, args, expected=0):
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
            self.assertEqual(main(args), expected, stderr.getvalue())
        return stdout.getvalue(), stderr.getvalue()

    def test_cli_workflow_reports_rates_and_distinguishes_verification(self):
        output, _ = self.invoke(self.compress + ["--report", str(self.report)])
        self.assertIn("1 rate-target miss", output)
        self.assertIn("Target 1.000 bpw missed for 1 of 1 eligible weights", output)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report["rate_target_misses"], 1)
        self.assertTrue(report["checksums_verified"])
        self.assertIn("peak_rss_bytes", report)
        self.assertGreater(report["archive_bytes"], report["stored_tensor_bytes"])
        self.assertEqual(report["families"][0]["weight"], "linear.weight")
        output, _ = self.invoke(["verify", str(self.archive)])
        self.assertIn("decoded tensor checksums", output)
        self.assertNotIn("byte for byte", output)
        output, _ = self.invoke([
            "verify", str(self.archive), "--reference-quantized", str(self.source),
            "--report", str(self.report), "--force",
        ])
        self.assertIn("1 preserved tensors match", output)
        self.assertIn("relative SSE vs quantized", output)
        verified = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(verified["bit_identical_preserved_tensors"], 1)
        self.assertGreater(verified["optimized_distortion"][0]["relative_sse_vs_quantized"], 0)
        self.invoke(["decompress", str(self.archive), "--output", str(self.restored)])
        self.assertTrue(self.restored.is_file())

    def test_reports_retain_order_and_force_is_explicit(self):
        args = self.compress + ["--report", str(self.report)]
        self.invoke(args)
        previous = self.report.read_text(encoding="utf-8")
        original_archive = self.archive.read_bytes()
        self.invoke(args, expected=1)
        self.assertEqual(self.report.read_text(encoding="utf-8"), previous)
        self.invoke(args + ["--force"])
        self.assertEqual(self.archive.read_bytes(), original_archive)
        first = json.loads(previous)
        second = json.loads(self.report.read_text(encoding="utf-8"))
        for key in ("encode_seconds", "decode_verify_seconds", "encode_mib_per_second", "peak_rss_bytes"):
            first.pop(key)
            second.pop(key)
        self.assertEqual(first, second)
        self.assertEqual(list(second), sorted(second))

    def test_wide_grid_is_opt_in_and_respects_error_cap(self):
        args = self.compress + ["--target-bpw", "4.3", "--max-error", "0.015", "--report", str(self.report)]
        self.invoke(args)
        default = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(default["schema_version"], 1)
        self.assertTrue(default["families"][0]["rate_target_missed"])
        self.assertLessEqual(default["families"][0]["multiplier"], 2.0)

        self.invoke(args + ["--wide-grid", "--force"])
        wide = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(wide["schema_version"], 2)
        fit = wide["families"][0]
        self.assertTrue(wide["wide_grid"])
        self.assertFalse(fit["rate_target_missed"])
        self.assertGreater(fit["multiplier"], 2.0)
        self.assertLessEqual(fit["added_relative_sse"], 0.015)
        self.invoke(["verify", str(self.archive), "--reference-quantized", str(self.source)])
        self.invoke(["decompress", str(self.archive), "--output", str(self.restored)])

        # V1 archives written during the wider-grid experiment remain readable.
        legacy = self.folder / "legacy-wide.pfqc.safetensors"
        def legacy_fit(manifest):
            manifest.update(schema_version=1)
            manifest["families"]["linear.weight"]["fit"].update(multiplier=16.0)

        rewrite_manifest(self.archive, legacy, legacy_fit)
        self.invoke(["verify", str(legacy)])

        altered = self.folder / "invalid-grid.pfqc.safetensors"
        rewrite_manifest(
            self.archive, altered, lambda manifest: manifest["families"]["linear.weight"]["fit"].update(multiplier=16.5),
        )
        _, error = self.invoke(["verify", str(altered)], expected=1)
        self.assertIn("Invalid multiplier", error)

    def test_wide_grid_requires_a_higher_cap_and_only_upgrades_when_selected(self):
        args = self.compress + ["--target-bpw", "8", "--wide-grid", "--report", str(self.report)]
        _, error = self.invoke(args + ["--max-error", "0.001"], expected=1)
        self.assertIn("requires max_error > 0.001", error)
        self.assertFalse(self.archive.exists())
        self.invoke(args)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["families"][0]["multiplier"], 1.0)

    def test_decompress_skips_checksums_only_when_requested(self):
        self.invoke(self.compress)
        altered = self.folder / "altered.pfqc.safetensors"
        rewrite_manifest(self.archive, altered, lambda manifest: manifest["tensors"][0].update(sha256="0" * 64))
        args = ["decompress", str(altered), "--output", str(self.restored)]
        _, error = self.invoke(args, expected=1)
        self.assertIn("Checksum mismatch", error)
        self.assertFalse(self.restored.exists())
        _, error = self.invoke(["verify", str(altered)], expected=1)
        self.assertIn("Checksum mismatch", error)

        output, _ = self.invoke(args + ["--skip-checksums", "--report", str(self.report)])
        self.assertIn("Skipped decoded tensor SHA-256 checks", output)
        self.assertFalse(json.loads(self.report.read_text(encoding="utf-8"))["checksums_verified"])
        self.assertTrue(self.restored.is_file())

    def test_cli_rejects_bad_arguments_and_paths_without_output(self):
        invalid = [
            self.compress[:4],
            self.compress + ["--method", "other"],
            ["decompress"],
        ]
        for args in invalid:
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
        for args in (
            self.compress[:-1] + ["nan"],
            self.compress + ["--output", str(self.folder / "missing" / "bad.pfqc.safetensors")],
            self.compress + ["--report", str(self.source), "--force"],
            self.compress + ["--report", str(self.archive), "--force"],
            self.compress[:1] + [str(self.folder / "missing.safetensors")] + self.compress[2:],
            ["compress", ".", "--target-bpw", "6.5"],
        ):
            with self.subTest(args=args):
                self.invoke(args, expected=1)
                self.assertFalse(self.archive.exists())
        self.assertFalse(list(self.folder.glob(".pfqc-*.tmp")))

    def test_missing_bf16_key_leaves_existing_output_untouched(self):
        save_fixture(self.ref, {"other.weight": np.ones((24, 1024), dtype="<u2")}, bf16=("other.weight",))
        self.archive.write_bytes(b"previous archive")
        _, error = self.invoke(self.compress + ["--reference", str(self.ref), "--force"], expected=1)
        self.assertIn("reference is missing weight", error)
        self.assertEqual(self.archive.read_bytes(), b"previous archive")
        self.assertFalse(list(self.folder.glob(".pfqc-*.tmp")))

    def test_redirected_output_handles_unicode_checkpoint_paths(self):
        self.invoke(self.compress)
        restored = self.folder / "\u6a21\u578b.safetensors"
        args = [
            sys.executable, "-B", "-m", "pf_quant_compress.cli",
            "decompress", str(self.archive), "--output", str(restored),
        ]
        env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict"}
        completed = subprocess.run(args, capture_output=True, encoding="cp1252", env=env, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(restored.is_file())
        self.assertIn("Reconstructed", completed.stdout)

        conflict = subprocess.run(args, capture_output=True, encoding="cp1252", env=env, check=False)
        self.assertEqual(conflict.returncode, 1)
        self.assertIn("Output already exists", conflict.stderr)

    def test_default_outputs_are_native_and_never_clobber_the_source(self):
        original = self.source.read_bytes()
        self.invoke(["compress", str(self.source), "--target-bpw", "6.5", "--report", str(self.report)])
        compressed = self.folder / "source.pfqc.safetensors"
        with safe_open(compressed, framework="np") as reader:
            self.assertIn("potatoforge_quant_compress.manifest", reader.metadata())
        report = json.loads(self.report.read_text(encoding="utf-8"))
        self.assertEqual(report["settings"]["max_error"], 0.001)
        self.invoke(["decompress", str(compressed)], expected=1)
        self.assertEqual(self.source.read_bytes(), original)
        self.source.unlink()
        self.invoke(["decompress", str(compressed)])
        self.assertTrue(self.source.is_file())

    def test_archive_detection_uses_metadata_and_supports_renaming(self):
        renamed = self.folder / "renamed.safetensors"
        self.invoke(self.compress + ["--output", str(renamed)])
        output, _ = self.invoke(["inspect", str(renamed)])
        self.assertIn("PFQC v1 compressed safetensors", output)
        self.invoke(["verify", str(renamed)])
        self.invoke(["decompress", str(renamed)])
        self.assertTrue((self.folder / "renamed.restored.safetensors").is_file())
        _, error = self.invoke(
            ["compress", str(renamed), "--target-bpw", "6.5"], expected=1,
        )
        self.assertIn("already", error)

        misleading = self.folder / "ordinary.pfqc.safetensors"
        misleading.write_bytes(self.source.read_bytes())
        _, error = self.invoke(["decompress", str(misleading)], expected=1)
        self.assertIn("Missing PFQC compression metadata", error)

    def test_inspect_reports_invalid_compression_metadata(self):
        save_fixture(
            self.archive, {"tensor": np.zeros(1, dtype=np.uint8)},
            metadata={"potatoforge_quant_compress.manifest": "invalid"},
        )
        _, error = self.invoke(["inspect", str(self.archive)], expected=1)
        self.assertIn("Malformed PFQC compression metadata", error)


if __name__ == "__main__":
    unittest.main()
