import math
from pathlib import Path
from time import perf_counter

import numpy as np

from .archive import Archive, open_archive, ordinary_checkpoint
from .formats import classify_weight
from .optimize import introduces_fp32_overflow, rows_per_batch
from .safetensors_io import Checkpoint


def _relative_sse(archive: Archive, original: Checkpoint, weight: str, scale: str) -> float:
    q0, s0 = original.array(weight), original.array(scale)
    width = q0.shape[1]
    batch_rows = rows_per_batch(width)
    rows_read = 0
    error = energy = 0.0
    weights = archive.chunks(weight, chunk_size=batch_rows * width)
    scales = archive.chunks(scale, chunk_size=batch_rows * 4)
    for raw_weights, raw_scales in zip(weights, scales, strict=True):
        q1 = np.frombuffer(raw_weights, dtype=np.int8).reshape(-1, width)
        s1 = np.frombuffer(raw_scales, dtype="<f4").reshape(-1, 1)
        rows = slice(rows_read, rows_read + len(q1))
        w0 = np.multiply(q0[rows], s0[rows], dtype=np.float64)
        w1 = np.multiply(q1, s1, dtype=np.float64)
        if introduces_fp32_overflow(w0, w1, s1):
            raise ValueError(f"FP32 dequantization overflows finite source values for {weight!r}")
        residual = w1 - w0
        energy += float(np.sum(w0 * w0))
        error += float(np.sum(residual * residual))
        rows_read += len(q1)
    if energy == 0.0:
        if error != 0.0:
            raise ValueError(f"An all-zero source reconstructed as nonzero for {weight!r}")
        return 0.0
    return error / energy


def _compare_reference(archive: Archive, original: Checkpoint) -> tuple[int, list[dict]]:
    archive.validate_families()
    if archive.metadata != original.metadata or archive.tensors.keys() != original.tensors.keys():
        raise ValueError("Reference checkpoint tensor keys or metadata differ from the archive")
    unchanged = 0
    for name, tensor in archive.tensors.items():
        reference = original.tensors[name]
        if (tensor.dtype, tensor.shape, tensor.nbytes) != (reference.dtype, reference.shape, reference.nbytes):
            raise ValueError(f"Reference checkpoint tensor metadata differs for {name!r}")
        if archive.records[name]["storage"] != "optimized_int8":
            for restored, source in zip(archive.chunks(name), original.chunks(name), strict=True):
                if restored != source:
                    raise ValueError(f"Preserved tensor {name!r} differs from the quantized reference")
            unchanged += 1

    distortions = []
    quantized_only = archive.settings["reference_mode"] == "quantized-only"
    for name, record in archive.families.items():
        fit = record["fit"]
        if fit is None or fit["multiplier"] == 1:
            continue
        family = classify_weight(original, name)
        if family.status != "eligible":
            raise ValueError(f"Invalid source quantization family for {name!r}: {family.reason}")
        relative_sse = _relative_sse(archive, original, name, family.scale)
        if quantized_only and (
            relative_sse > archive.settings["max_error"]
            or not math.isclose(relative_sse, fit["added_relative_sse"], rel_tol=1e-12)
        ):
            raise ValueError(f"Verified distortion violates the cap or recorded error for {name!r}")
        distortions.append({
            "weight": name,
            "relative_sse_vs_quantized": relative_sse,
            "quantized_error_cap_verified": quantized_only,
        })
    return unchanged, distortions


def verify_archive(source: Path | str, *, reference_quantized: Path | str | None = None) -> dict:
    started = perf_counter()
    unchanged, distortions = None, []
    with open_archive(source) as packed:
        if reference_quantized is None:
            packed.check_integrity()
        else:
            unchanged, distortions = _compare_reference(packed, ordinary_checkpoint(reference_quantized))
        report = packed.report()
    report.update(
        operation="verify", decode_verify_seconds=perf_counter() - started, checksums_verified=True,
        reference_quantized_compared=reference_quantized is not None,
        bit_identical_preserved_tensors=unchanged, optimized_distortion=distortions,
    )
    return report
