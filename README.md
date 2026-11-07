# PotatoForge Quant Compress

A standalone CPU toolkit for rate-aware storage compression of existing INT8 ConvRot checkpoints. It writes native `<model-name>.pfqc.safetensors` files with DEFLATE streams stored as U8 tensors. Decode the compressed checkpoint before using it with an ordinary ComfyUI loader. Hosting platforms apply their own upload and validation rules.

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

Use `--report .\compression.json` to save tensor sizes, achieved rates, error measurements, elapsed times, and peak process RSS where available. `--verbose` shows per-family fitting progress and results.

## Eligibility and distortion

Inspection groups tensor bytes by dtype and marker format, reports malformed or unsupported families, and estimates histogram entropy in bits per weight. Eligible weights have an `int8_tensorwise` marker with ConvRot enabled, a nonempty 2D I8 matrix, and finite positive FP32 scales of shape `[out_features, 1]`. The rotation group must be a power of four dividing the input width. An omitted group size uses the stock default of 256. [ComfyUI quantization format](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md), [Comfy Quants INT8 format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md)

The detector preserves conventional embedding and expert names, marked expert banks, non-2D weights, and matrices wider than 262,144 inputs. Embedding tables can share the same marker and shape as biasless linears. Use repeatable `--protect` prefixes to preserve unfamiliar embedding and expert families. Prefixes match the logical weight key.

`--target-bpw` applies independently to each eligible matrix's coded I8 payload. `--max-error` defaults to `0.001`, or 0.1% added relative squared error per tensor. Without a BF16 reference, the baseline is the decoded source INT8 pair. With `--reference .\original-bf16.safetensors`, the fit uses the matching BF16 tensor after regular Hadamard rotation in FP64. The cap applies to any increase over the source pair's error against that reference. BF16-guided archives are self-contained. [Comfy Kitchen rotation](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py)

The scale grid runs from 1.0 to 2.0. The original pair is an exact baseline. Other trials widen FP32 row scales, round ties to even, and clamp codes to `[-128, 127]`. The optimizer measures level-3 DEFLATE for weight codes and prices scales at their native FP32 size. An eligible baseline can be encoded losslessly when its compressed stream is smaller. Protected and unsupported tensors are copied natively with their original names, shapes, dtypes, and bytes.

The optimizer keeps the least-distorting candidate that meets the target and saves bytes. If the target is missed, it keeps the smallest candidate within the cap that saves bytes, or preserves the original pair.

## Verification

`verify` validates archive structure and the SHA-256 of every reconstructed tensor. With `--reference-quantized` it compares passthrough tensors byte for byte and measures optimized families' distortion against the quantized source. That measurement checks the cap for quantized-only fitting. A BF16-guided fit's cap is evaluated at compression against the rotated BF16 reference.

These measurements constrain weight-space distortion. Decoded weights retain their storage dtypes and shapes, so decoded RAM and VRAM requirements stay the same. The [archive format](docs/archive-format.md) describes the decoding contract, and the [implementation plan](docs/implementation-plan.md) covers the v0.1 scope and validation gates.

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
