# PotatoForge Quant Compress

A standalone CPU toolkit for rate-aware storage compression of existing INT8 ConvRot checkpoints. It writes native `<model-name>.pfqc.safetensors` files with Zstandard level-1 frames stored as U8 tensors. Decode the compressed checkpoint before using it with an ordinary ComfyUI loader. Hosting platforms apply their own upload and validation rules.

## PowerShell quick start

Run these commands from the repository root. uv uses the Python version in `.python-version` and installs the CLI, NumPy, and safetensors into `.venv`.

```powershell
uv sync
uv run pf-compress inspect .\model.safetensors
uv run pf-compress compress .\model.safetensors --target-bpw 6.5
uv run pf-compress inspect .\model.pfqc.safetensors
uv run pf-compress verify .\model.pfqc.safetensors --reference-quantized .\model.safetensors
uv run pf-compress decompress .\model.pfqc.safetensors --output .\restored.safetensors
```

`uv run pf-compress decompress .\model.pfqc.safetensors` defaults to `.\model.safetensors`. Use `--output` to write beside the original or `--force` to replace an existing destination. A renamed compressed file defaults to `<name>.restored.safetensors`. The CLI detects archives from their metadata.

`compress` verifies the completed archive before publishing it, including every tensor's SHA-256. `verify` performs the same integrity checks on an existing archive. `decompress` verifies checksums by default. Pass `--skip-checksums` to omit SHA-256 comparison while keeping manifest validation, tensor-size checks, and Zstandard frame validation. This mode can accept altered tensor bytes if the archive and stream remain structurally valid.

Compression fits up to two eligible layers concurrently and writes them in a consistent order. It displays the current layer and its position in the total, its original and stored weight-plus-scale sizes, and cumulative eligible sizes, followed by writing and verification progress. Use `--report .\compression.json` to save tensor sizes, achieved rates, error measurements, elapsed times, and peak process RSS where available. `--verbose` shows per-family fit results.

## Eligibility and distortion

Inspection groups tensor bytes by dtype and marker format, reports malformed or unsupported families, and estimates histogram entropy in bits per weight. Eligible weights have an `int8_tensorwise` marker with ConvRot enabled, a nonempty 2D I8 matrix, and finite positive FP32 scales of shape `[out_features, 1]`. The rotation group must be a power of four dividing the input width. An omitted group size uses the stock default of 256. [ComfyUI quantization format](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md), [Comfy Quants INT8 format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md)

The detector preserves conventional embedding and expert names, marked expert banks, non-2D weights, and matrices wider than 262,144 inputs. Embedding tables can share the same marker and shape as biasless linears. Use repeatable `--protect` prefixes to preserve unfamiliar embedding and expert families. Prefixes match the logical weight key.

`--target-bpw` applies independently to each eligible matrix's coded I8 payload. `--max-error` defaults to `0.001`, or 0.1% added relative squared error per tensor. Without a BF16 reference, the baseline is the decoded source INT8 pair. With `--reference .\original-bf16.safetensors`, the fit uses the matching BF16 tensor after regular Hadamard rotation in FP64. The cap applies to any increase over the source pair's error against that reference. BF16-guided archives are self-contained. [Comfy Kitchen rotation](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py)

The scale grid runs from 1.0 to 2.0. The original pair is an exact baseline. Other trials widen FP32 row scales, round ties to even, and clamp codes to `[-128, 127]`. The optimizer measures level-1 Zstandard for weight codes and prices scales at their native FP32 size. An eligible baseline can be encoded losslessly when its compressed stream is smaller. Protected and unsupported tensors are copied natively with their original names, shapes, dtypes, and bytes.

The optimizer keeps the least-distorting candidate that meets the target and saves bytes. If the target is missed, it keeps the smallest candidate within the cap that saves bytes, or preserves the original pair. The CLI reports how many eligible weights missed the target. A verified best-effort archive is still written.

The default search evaluates the scale grid in row batches. `--search reference` evaluates candidates serially. Both measure the actual Zstandard level-1 stream size. Windows x64 requires the bundled native library.

### Experimental 4 bpw MOSS fit

```powershell
uv run pf-compress compress .\moss_image_sft_int8_convrot.safetensors --reference .\moss_image_sft_bf16.safetensors --target-bpw 4 --max-error 0.015 --wide-grid --output .\moss-4bpw.pfqc.safetensors --report .\moss-4bpw.json --verbose
```

`--wide-grid` adds multipliers up to 14× and requires an explicit error cap above 0.001. The 0.015 cap admitted 4 bpw fits on sampled MOSS matrices. The target applies to weight payloads. The report includes the bitrate with scales, misses, BF16-reference error, and the actual encoder for each fit. Archives that select a multiplier above 2× use schema v2 and require an updated PFQC reader, including the bundled ComfyUI decoder.

## Verification

`verify` validates archive structure and the SHA-256 of every reconstructed tensor. With `--reference-quantized` it compares passthrough tensors byte for byte and measures optimized families' distortion against the quantized source. That measurement checks the cap for quantized-only fitting. A BF16-guided fit's cap is evaluated at compression against the rotated BF16 reference.

These measurements constrain weight-space distortion. Decoded weights retain their storage dtypes and shapes, so decoded RAM and VRAM requirements stay the same. The [archive format](docs/archive-format.md) describes the decoding contract, contains the v1 decoding contract.

## Fit one weight

```python
from pf_quant_compress.optimize import optimize_weight
from pf_quant_compress.safetensors_io import Checkpoint

result = optimize_weight(
    Checkpoint("model.safetensors"),
    "blocks.0.proj.weight",
    target_bpw=6.5,
    max_error=0.001,
    protect=("blocks.18.",),
)
```

`None` signals passthrough of the whole family. Otherwise `result` holds the chosen codes and scales, baseline and chosen encoded sizes, added relative squared error, and the rate-target status.

## Run the tests

```powershell
uv run python -B -W error -m unittest discover -s tests -v
```
