import unittest
from unittest.mock import patch

from pf_quant_compress import zstd


def encode(data):
    compressor = zstd.Compressor()
    result = [compressor.compress(data[i:i + zstd.CHUNK_SIZE]) for i in range(0, len(data), zstd.CHUNK_SIZE)]
    return b"".join(result) + compressor.flush()


class ZstdTests(unittest.TestCase):
    def test_closed_compressor_rejects_reuse(self):
        compressor = zstd.Compressor()
        compressor.flush()
        with self.assertRaisesRegex(ValueError, "compressor is closed"):
            compressor.compress(b"data")
        with self.assertRaisesRegex(ValueError, "compressor is closed"):
            compressor.flush()

    def test_round_trip_across_chunks(self):
        data = bytes(range(256)) * 16384
        encoded = encode(data)
        output = bytearray(len(data))
        zstd.decompress_into((encoded[i:i + 31] for i in range(0, len(encoded), 31)), output)
        self.assertEqual(output, data)

    def test_large_tensor_streams_into_final_buffer(self):
        block = bytes(range(256)) * 1024
        count = 260  # 65 MiB, larger than the old 64 MiB native limit.
        compressor = zstd.Compressor()
        encoded = b"".join(compressor.compress(block) for _ in range(count)) + compressor.flush()
        output = bytearray(len(block) * count)
        zstd.decompress_into((encoded[i:i + 4096] for i in range(0, len(encoded), 4096)), output)
        for offset in range(0, len(output), len(block)):
            self.assertEqual(memoryview(output)[offset:offset + len(block)], block)
        total = 0
        for chunk in zstd.decompress_chunks([encoded], len(output), 16384):
            self.assertLessEqual(len(chunk), 16384)
            start = total % 256
            self.assertEqual(chunk, block[start:start + len(chunk)])
            total += len(chunk)
        self.assertEqual(total, len(output))

    def test_truncated_trailing_corrupt_and_wrong_size(self):
        data = bytes(range(256)) * 1024
        encoded = encode(data)
        for bad, size in ((encoded[:-1], len(data)), (encoded + b"x", len(data)),
                          (b"bad" + encoded[3:], len(data)), (encoded, len(data) - 1),
                          (encoded, len(data) + 1)):
            with self.subTest(size=size, length=len(bad)), self.assertRaises(ValueError):
                zstd.decompress_into([bad], bytearray(size))

    def test_missing_library(self):
        zstd.library.cache_clear()
        with patch.object(zstd, "_DLL", zstd._DLL.with_name("missing.dll")):
            with self.assertRaisesRegex(RuntimeError, "Zstandard library"):
                zstd.library()
        zstd.library.cache_clear()

    def test_wrong_library_hash_and_unsupported_platform(self):
        zstd.library.cache_clear()
        with patch.object(zstd, "_SHA256", "0" * 64), self.assertRaisesRegex(RuntimeError, "checksum"):
            zstd.library()
        with patch.object(zstd.sysconfig, "get_platform", return_value="win-arm64"):
            with self.assertRaisesRegex(RuntimeError, "Windows x64"):
                zstd.library()
        zstd.library.cache_clear()


if __name__ == "__main__":
    unittest.main()
