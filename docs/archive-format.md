# PFQC safetensors format v1

A `<model-name>.pfqc.safetensors` file is a valid safetensors container. Its `__metadata__` string map contains the key `potatoforge_quant_compress.manifest`. The value is a JSON object serialized as a UTF-8 string. Readers use that key, the manifest's `format`, and its `schema_version` to identify the file, regardless of its name. Original checkpoint metadata lives in the manifest's `metadata` field. [Safetensors format](https://github.com/safetensors/safetensors/blob/main/README.md)

Physical and reconstructed tensors use the same keys. Unchanged tensors, including protected weights and unsupported formats, retain their original dtypes, shapes, and bytes. FP32 row scales and U8 quantization markers are native tensors. For a fitted pair, the physical scale tensor contains the new FP32 scale values and the marker contains the original bytes.

An encoded weight is physically a one-dimensional `U8` tensor (`torch.uint8`) containing one Zstandard frame. The stream decodes to row-major signed INT8 bytes with the original matrix shape. An eligible weight may be encoded losslessly when the original codes compress well. The reconstructed codes then equal the source codes exactly. Ordinary inference loaders use the reconstructed checkpoint.

## Manifest

| Field | Value |
| --- | --- |
| `format` | `potatoforge_quant_compress` |
| `schema_version` | Integer `1` |
| `compressor_version` | Version of the writer |
| `source_format` | `safetensors` |
| `method`, `zstd_level` | `wider-scale`, `1` |
| `settings` | `target_bpw`, `max_error`, ordered `protect` prefixes, and `reference_mode` (`quantized-only` or `bf16`) |
| `metadata` | Original safetensors string map, or `null` when absent |
| `source_bytes` | Original checkpoint length including its header |
| `tensors` | Tensor records in original checkpoint payload order |
| `families` | Object keyed by logical weight name |

Every tensor record contains `name`, reconstructed `dtype` and `shape`, `family`, `storage`, `original_bytes`, `decoded_bytes`, `stored_bytes`, `codec`, and `sha256`. `family` is a logical weight key or `null`. The physical tensor key is `name`. `original_bytes` equals `decoded_bytes` because reconstructed shapes and dtypes are preserved. `stored_bytes` is the physical tensor's byte count in the safetensors payload. `sha256` is lowercase hex over the reconstructed storage bytes, including the optimized result for fitted tensors.

| `codec` | Physical layout | Reconstruction |
| --- | --- | --- |
| `stored` | Original dtype and shape | Read the native storage bytes directly |
| `zstd` | `U8[stored_bytes]` | Decode one Zstandard frame, then interpret as `I8` with the manifest shape |

`storage` distinguishes `unchanged` native tensors, `encoded_int8` weights that reconstruct the source codes exactly, and `optimized_int8` members of a fitted pair. A fitted weight uses `zstd` and its FP32 scale uses `stored`.

Each family record contains `family` and `fit`. The descriptor in `family` gives `weight`, `scale`, `marker`, `format`, `status`, `reason`, and `group_size`. `fit` is `null` for protected, unsupported, and malformed families. For an eligible family it records `multiplier`, `baseline_weight_bytes`, `baseline_scale_bytes`, `payload_bpw`, `all_in_bpw`, `rate_target_missed`, `added_relative_sse`, `baseline_reference_error`, and `optimized_reference_error`. Multiplier 1.0 records an exact numerical baseline. BF16 error fields are `null` for quantized-only fitting. These statistics describe the fit and are independent of the decoder.

V1 accepts the documented multiplier grid through 14×. It does not accept the unreleased DEFLATE archives or the research transport.

## Decoding contract

1. Open the file with a safetensors reader and parse the reserved metadata string. Require the format and schema version above. Match the physical tensor keys to the logical tensor records.
2. Validate dtypes, shapes, lengths, and codecs. Read each `stored` tensor as raw little-endian row-major storage. For `zstd`, read the U8 vector as one Zstandard frame. Require end of stream, exactly `decoded_bytes` of output, and consumption of the full input.
3. Check SHA-256 of each decoded byte sequence by default. Use the manifest dtype and shape for reconstructed weights, and load native scales and markers under the original keys. BF16-guided files include the final codes and scales needed for decoding.
4. When writing a checkpoint, use the original metadata map and manifest tensor order. Emit an eight-byte little-endian safetensors header length, JSON padded with ASCII spaces to a multiple of eight, and tensor bodies with contiguous offsets. The CLI validates the closed temporary file with `safetensors.safe_open` before publishing it.

The CLI bounds the manifest to 16 MiB and accepts up to 50,000 tensors, 64 GiB per tensor, and 256 GiB of total decoded tensor data. It checks rank and byte alignment, decoded lengths and hashes, and quantization-family consistency. Tensor names are keys and are never used as filesystem paths. The decoder uses chunks of at most 256 KiB.

`open_archive(path, verify_checksums=False)` skips comparing SHA-256 digests when reading tensors. Manifest and family validation, decoded size limits, and end-of-stream checks still apply. `Archive.check_integrity()` always checks every tensor's SHA-256. The manifest requires the same checksum fields in either mode.

Consumers with a final CPU buffer can call `Archive.decode_into(name, output)` for a Zstandard weight. The buffer must be writable, contiguous, and exactly the validated size. It streams bounded compressed input directly into the destination and rejects trailing frames or data. Readers cap the Zstandard window at 128 MiB. `Archive.check_tensor(name, data)` checks the decoded size and uses Windows CNG for SHA-256 when supported, with `hashlib` as the fallback. The standalone CLI uses the streaming `chunks()` API to keep its memory footprint low.

### Native build provenance

The package bundles `libzstd.dll` and the upstream license from the official [Zstandard 1.5.7 Windows x64 release](https://github.com/facebook/zstd/releases/tag/v1.5.7). The DLL is required and is checked before loading. A missing, unsupported, or mismatched library fails explicitly. Windows CNG accelerates SHA-256 when available; `hashlib` is the fallback.

| File | SHA-256 |
| --- | --- |
| `zstd-v1.5.7-win64.zip` | `acb4e8111511749dc7a3ebedca9b04190e37a17afeb73f55d4425dbf0b90fad9` |
| `libzstd.dll` | `8f07e1112ed283e5cd2798833e9a3c32d8961381bc36da04af57a1b0ca9bd40b` |

Compression stages encoded weights and fitted scales on disk until the header is known. Other tensors stream from the input. The archived tensor records and physical tensor layout are deterministic for identical input, options, and runtime versions.

`verify --reference-quantized` compares passthrough and losslessly encoded tensors byte for byte. For a fitted pair, it checks the stored decoded hashes and measures relative squared error against the original quantized source. It checks that error against the cap in quantized-only mode. The BF16 cap uses the rotated BF16 reference at compression time.

A hand-assembled v1 manifest fixture and mixed-precision round trips are in the [archive tests](../tests/test_archive.py) and [round-trip tests](../tests/test_roundtrip.py).
