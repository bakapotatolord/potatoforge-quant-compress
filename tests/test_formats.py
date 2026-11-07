from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors import safe_open

from fixtures import CONVROT, family, mixed_checkpoint, save_fixture
from pf_quant_compress.cli import main
from pf_quant_compress.formats import inspect_checkpoint
from pf_quant_compress.safetensors_io import Checkpoint


ROOT = Path(__file__).resolve().parents[1]


class FormatTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "fixture.safetensors"

    def test_inspect_mixed_precision_and_exact_prefix_protection(self):
        original = mixed_checkpoint(self.path)
        source_bytes = self.path.read_bytes()
        checkpoint = Checkpoint(self.path)
        report = inspect_checkpoint(checkpoint, protect=("blocks.18.",))
        statuses = {item["weight"]: item["status"] for item in report["families"]}
        self.assertEqual(statuses, {
            "bank.weight": "unsupported",
            "blocks.0.proj.weight": "eligible",
            "blocks.18.proj.weight": "protected",
            "broken.weight": "malformed",
            "embed_tokens.weight": "unsupported",
            "packed.weight": "unsupported",
            "plain.weight": "unsupported",
            "unknown.weight": "unsupported",
            "w6.weight": "unsupported",
            "wrong_shape.weight": "malformed",
        })
        self.assertEqual(report["tensor_count"], len(original))
        self.assertEqual(report["file_bytes"], len(source_bytes))
        self.assertEqual(report["by_dtype"]["BF16"], {"count": 1, "bytes": 4})
        self.assertEqual(sum(item["bytes"] for item in report["by_format"].values()),
                         sum(tensor.nbytes for tensor in original.values()))
        self.assertEqual(report["by_format"]["convrot_w4a4"]["count"], 3)
        eligible = next(item for item in report["families"] if item["status"] == "eligible")
        self.assertEqual(eligible["entropy_bits_per_weight"], 4.0)
        with safe_open(self.path, framework="np") as reader:
            self.assertEqual(checkpoint.metadata, reader.metadata())
            self.assertEqual(set(checkpoint.tensors), set(reader.keys()))
        for name, tensor in original.items():
            info = checkpoint.tensors[name]
            self.assertEqual(info.shape, tensor.shape)
            self.assertEqual(source_bytes[info.offset:info.offset + info.nbytes], tensor.tobytes())
        self.assertEqual(self.path.read_bytes(), source_bytes)

    def test_group_and_marker_validation(self):
        cases = {
            "defaults": ({"format": "int8_tensorwise", "convrot": True}, "eligible"),
            "nested": ({"format": "int8_tensorwise", "params": {"convrot": True, "convrot_groupsize": 4}}, "eligible"),
            "overrides": ({**CONVROT, "convrot": False, "params": {"convrot": True}}, "unsupported"),
            "non_boolean": ({**CONVROT, "convrot": 1}, "malformed"),
            "power_of_two": ({**CONVROT, "convrot_groupsize": 8}, "malformed"),
            "not_divisible": ({**CONVROT, "convrot_groupsize": 16}, "malformed"),
            "duplicate": (b'{"format":"int8_tensorwise","convrot":false,"convrot":true}', "malformed"),
            "invalid_utf8": (b"\xff", "malformed"),
            "missing_format": ({"convrot": True}, "malformed"),
        }
        tensors = {}
        for name, (details, _) in cases.items():
            tensors.update(family(name, details))
        tensors["defaults.weight"] = np.zeros((2, 256), dtype=np.int8)
        save_fixture(self.path, tensors)
        result = {item["weight"]: item for item in inspect_checkpoint(Checkpoint(self.path))["families"]}
        for name, (_, status) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(result[f"{name}.weight"]["status"], status)
        self.assertEqual(result["defaults.weight"]["group_size"], 256)

    def test_partial_and_invalid_families(self):
        tensors = {}
        for name in ("missing", "wrong_dtype", "nan", "negative", "zero", "inf", "bank"):
            tensors.update(family(name))
        del tensors["missing.weight_scale"]
        tensors["wrong_dtype.comfy_quant"] = tensors["wrong_dtype.comfy_quant"].view(np.int8)
        tensors["bank.weight"] = tensors["bank.weight"].reshape(2, 1, 8)
        for name, value in [("nan", np.nan), ("negative", -1), ("zero", 0), ("inf", np.inf)]:
            tensors[f"{name}.weight_scale"][0, 0] = value
        save_fixture(self.path, tensors)
        report = inspect_checkpoint(Checkpoint(self.path))
        for item in report["families"]:
            self.assertEqual(item["status"], "unsupported" if item["weight"] == "bank.weight" else "malformed")

    def test_cli_reports_diagnostics_and_size_units(self):
        mixed_checkpoint(self.path)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["inspect", str(self.path)]), 0)
        output = stdout.getvalue()
        for part in ("GB", "GiB", "2 eligible", "packed.weight", "broken.weight", "histogram estimate"):
            self.assertIn(part, output)

    def test_cli_fails_cleanly_for_missing_or_invalid_checkpoints(self):
        for payload in (None, b"not a safetensors file"):
            if payload is not None:
                self.path.write_bytes(payload)
            with redirect_stderr(StringIO()) as stderr:
                self.assertEqual(main(["inspect", str(self.path)]), 1)
            self.assertIn("error", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
