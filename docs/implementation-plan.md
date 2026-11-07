# PotatoForge Quant Compress v0.1.0 implementation plan

## Mission and scope

Build a standalone Python CLI for rate-aware lossy storage compression of existing INT8 ConvRot safetensors checkpoints. Compression can adjust selected INT8 weight codes and row scales within a per-tensor distortion cap. Decoding restores an ordinary INT8 ConvRot checkpoint with the original quantization markers, tensor layout, and metadata.

The four commands are `inspect`, `compress`, `decompress`, and `verify`. Version 0.1 uses the `wider-scale` optimization method and Python's standard-library zlib implementation of DEFLATE. The original BF16 checkpoint is optional during fitting. The compressed checkpoint contains everything required to decode.

The CLI runs on CPU and requires Python 3.11+, NumPy, and safetensors. Keep the implementation independent of PotatoForge's training and quantization code. Prior Anima and Kroma research informs the scale search and baseline comparisons. Direct model loading belongs in a separate ComfyUI loader repository. The v0.1 scope excludes runtime inference, CUDA kernels, activation capture, sensitivity optimization, lossy optimization of other quantization formats, GUI, asynchronous jobs, and runtime configuration files.

## Commands

```powershell
uv sync
uv run pf-compress inspect .\model.safetensors
uv run pf-compress compress .\model.safetensors --target-bpw 6.5
uv run pf-compress compress .\model.safetensors --target-bpw 6.5 --max-error 0.001 --method wider-scale --protect 'blocks.18.' --reference .\original-bf16.safetensors --output .\guided.pfqc.safetensors
uv run pf-compress inspect .\model.pfqc.safetensors
uv run pf-compress verify .\model.pfqc.safetensors
uv run pf-compress verify .\model.pfqc.safetensors --reference-quantized .\model.safetensors
uv run pf-compress decompress .\model.pfqc.safetensors
uv run pf-compress decompress .\model.pfqc.safetensors --output .\restored.safetensors
```

- `inspect` reports sizes by dtype and quantization family, eligible INT8 ConvRot weights, histogram entropy estimates, and unsupported or malformed markers for ordinary checkpoints. For compressed checkpoints it reports the recorded sizes, eligibility counts, and coded rates. Report decimal GB and binary GiB explicitly.
- `compress` defaults to `<model-name>.pfqc.safetensors`. `--target-bpw` is a finite positive per-eligible-matrix target for the actual I8 payload, independent of whole-file size. `--max-error` is finite and nonnegative, and defaults to `0.001`. `--protect` is repeatable and matches exact string prefixes against the logical weight key. A matching family preserves its original weight, scale, and marker bytes. `--reference` optionally supplies the original BF16 weights.
- `decompress` removes the `.pfqc` component from the default output filename. A valid archive with a different name defaults to `<name>.restored.safetensors`. Use `--output` to choose another location. Recognition uses the internal format identifier and schema version. Existing destinations require `--force`.
- `verify` validates the native container, manifest, decoded hashes, and reconstructed quantization families. `--reference-quantized` checks passthrough and losslessly encoded tensors bit for bit, reports optimized-family distortion against the quantized source, and rechecks the cap for quantized-only fits.
- `--report` writes JSON and `--verbose` provides fit progress and details. Invalid inputs, unsupported versions, overwrite conflicts, and integrity or distortion failures produce clear errors and nonzero exit status. Protect input checkpoints from output path aliases.

## Eligibility and distortion

Require a UTF-8 JSON `.comfy_quant` marker with `format: int8_tensorwise` and ConvRot enabled. Match a nonempty 2D signed I8 `.weight` and finite positive FP32 `.weight_scale` of shape `[out_features, 1]`. The rotation group must be a power of four dividing the input width. Use the stock default of 256 when omitted and honor top-level options before nested `params`. Use bounded row sizes. Preserve packed W4/W6, plain or unmarked INT8, identifiable embeddings and MoE banks, unsupported shapes, and malformed families. Biases and markers pass through byte for byte. Use explicit prefixes where model naming leaves an embedding or expert role ambiguous. [ComfyUI quantization format](https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md), [Comfy Quants INT8 format](https://github.com/Comfy-Org/comfy-quants/blob/main/docs/formats/int8_tensorwise.md)

In the stored rotated domain, let `W0 = q0 * s0` and `W1 = q1 * s1`. Without BF16, added distortion is `sum((W1-W0)^2) / max(sum(W0^2), epsilon)`. All-zero source matrices require exact zero reconstruction. The cap is a fraction of relative squared error, so `0.001` means 0.1%.

With BF16, require the exact logical weight key, shape, and BF16 dtype. Rotate the BF16 values with the marker's normalized regular Hadamard transform and check parity against independent fixtures. Compute `E0 = SSE(W0,Wbf16) / ||Wbf16||²` and `E1 = SSE(W1,Wbf16) / ||Wbf16||²` in the rotated domain. Cap `max(0,E1-E0)` and report both errors and the signed delta. Reject nonfinite or irreconcilable references. A zero-energy BF16 reference requires an all-zero baseline. These weight-space metrics are engineering constraints. [Comfy Kitchen rotation](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py)

## Wider-scale selection

1. Start with multiplier 1.0 and preserve the exact source codes and scales. Operate on one tensor and bounded row batches. Keep FP32 row scales.
2. Use the deterministic grid `1.0, 1.025, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0`. Widen the scales, round `Wref / s1` to nearest with ties to even, and clamp to `[-128,127]`. Use `W0` or the rotated BF16 reference as `Wref`. Preserve zero rows and reject invalid scales, excess distortion, or new nonfinite values in FP32 dequantization.
3. Price the actual level-3 raw DEFLATE payload for each candidate, with native I8 fallback. Include the native FP32 scale bytes in the size comparison. Use the original pair with its best weight storage representation as the baseline. Native passthrough tensors retain their source bytes.
4. Prefer minimum distortion among candidates that meet the target and save bytes over the baseline. If the target is missed, choose the smallest valid representation that saves bytes or use the baseline. Record rate misses explicitly. Apply the same policy independently to every eligible tensor.
5. Encode the selected I8 codes and retain their matching native FP32 row scales. The selected payload decodes bit for bit to the post-optimization pair.

## Native compressed checkpoint

The distributable file is `<model-name>.pfqc.safetensors`, for example `kroma-int8-convrot.pfqc.safetensors` or `anima-int8-mixed.pfqc.safetensors`. It uses the standard safetensors header and tensor buffer. The reserved metadata string `potatoforge_quant_compress.manifest` contains the format identifier, schema version 1, original metadata, logical tensor order and reconstructed dtypes and shapes, codecs and byte counts, and SHA-256 of the decoded tensors.

Use the original tensor keys. Store compressed weight streams as one-dimensional U8 tensors and preserve native tensors with their original shapes, dtypes, and values. An eligible baseline weight can use lossless storage encoding. Scales and markers use their original native layout. The manifest distinguishes unchanged tensors, losslessly coded weights, and fitted pairs. Include fit settings, actual rates, baseline sizes, and error metrics. [PFQC decoding contract](archive-format.md)

Validate input with `safetensors.safe_open` and use bounded raw reads to preserve every storage dtype. Validate the manifest size, schema, tensor keys, shapes, dtype and byte counts, decoded size limits, compression ratios, hashes, and family consistency. Reject incomplete or trailing DEFLATE data. Tensor names are never used as filesystem paths.

Stage encoded weights and fitted scales until the native header can be written. Stream native and staged data into a sibling temporary file. Restore checkpoints by computing aligned safetensors headers and contiguous offsets from the manifest and streaming decoded bodies in order. Close and validate the result before atomic publication. Clean up staged and incomplete files on failure.

The native file structure supports standard safetensors readers. Ordinary inference loaders need the decoded tensors. Hosting platforms retain their own validation rules. A future direct loader can implement the documented decoder with safetensors and DEFLATE independently of the CLI and optimizer.

## Resource and reporting requirements

Target Windows and a CPU-first workflow on an i5-12400 with a GTX 1660 Super and 32 GB RAM. Keep one selected tensor pair, bounded computation batches, and bounded codec buffers in memory beyond OS-managed mappings. Measure elapsed time, throughput, peak RSS where available, and input, compressed, and restored sizes. The restored checkpoint retains the source tensor dtypes and shapes and therefore the same decoded weight footprint.

Report actual stored bytes for the original baseline and selected representation, per-weight and aggregate payload bpw, bpw including native scales, whole-file overhead, and rate misses. Include counts for optimized, protected, skipped, preserved, unsupported, and malformed tensors or families. Report per-family scale factors and errors, aggregate minimum, mean, and maximum where relevant, and checksum results. Keep report ordering deterministic. Archive bytes and decoded hashes should repeat for the same input, settings, and runtime versions.

## Phases and acceptance gates

1. **Phase 0. Format reconnaissance.** Confirm the format and tooling, examine available research, and inspect a mixed-precision fixture. Packed and unrelated I8 weights must remain ineligible.
2. **Phase 1. Single-tensor proof.** Test measured rates, the distortion cap, no-op and raw fallback, reference parity, invalid values, clipping, and rounding. A selected pair must round-trip exactly to its optimized storage bytes.
3. **Phase 2. Native archive and CLI.** Exercise all four commands, metadata-based detection after renaming, default output paths, overwrite protection, streaming, and failure cleanup. A native compressed checkpoint must open with `safe_open`. Mixed precision, protected families, and metadata must survive decoding with their original keys, dtypes, shapes, and unchanged bytes. Fitted pairs must match their decoded hashes. BF16-guided archives must decode after both inputs are removed.
4. **Phase 3. Anima validation.** On an available Anima W4/INT8 checkpoint, verify W4 tensors and side data bit for bit and compare quantized-only and BF16-guided fitting on available layers. Validate the restored checkpoint in the existing inference workflow.
5. **Phase 4. Kroma-scale validation.** Run inspect, compress, decompress, and verify on an available Kroma INT8 ConvRot checkpoint with protections and a conservative cap. Measure storage, wall time, throughput, and peak RSS on the target hardware. Compare earlier research only where methods and rate definitions align.

Keep a synthetic compatibility fixture and malformed-archive tests for unsupported versions and codecs, invalid tensor shapes, duplicate logical names, oversized declarations, trailing or truncated streams, and hash failures. Before release, confirm native files and restored checkpoints across the supported size range, including files larger than 4 GiB. Run the relevant tests at each phase and report the evidence and remaining validation.
