from collections.abc import Iterator
from dataclasses import dataclass
import math
import zlib

import numpy as np

from .formats import classify_weight
from .safetensors_io import Checkpoint, array_chunks


DEFLATE_LEVEL = 3
SCALE_MULTIPLIERS = (1.0, 1.025, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)
_BATCH_WEIGHTS = 256 * 1024
_FP32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class MemberSize:
    codec: str
    nbytes: int


@dataclass(frozen=True)
class Candidate:
    multiplier: float
    weight: MemberSize
    scale: MemberSize
    added_error: float
    reference_error: float | None

    @property
    def total_bytes(self) -> int:
        return self.weight.nbytes + self.scale.nbytes


@dataclass(frozen=True)
class Optimization:
    codes: np.ndarray
    scales: np.ndarray
    baseline: Candidate
    selected: Candidate
    target_bpw: float

    @property
    def payload_bpw(self) -> float:
        return 8 * self.selected.weight.nbytes / self.codes.size

    @property
    def all_in_bpw(self) -> float:
        return 8 * self.selected.total_bytes / self.codes.size

    @property
    def rate_target_missed(self) -> bool:
        return self.payload_bpw > self.target_bpw


def _deflater():
    return zlib.compressobj(DEFLATE_LEVEL, wbits=-zlib.MAX_WBITS)


def encoded_chunks(array: np.ndarray, codec: str) -> Iterator[bytes | memoryview]:
    """Encode contiguous native storage bytes with the measured member codec."""
    if codec == "stored":
        yield from array_chunks(array)
        return
    if codec != "deflate":
        raise ValueError(f"Unsupported codec {codec!r}")
    compressor = _deflater()
    for chunk in array_chunks(array):
        encoded = compressor.compress(chunk)
        if encoded:
            yield encoded
    yield compressor.flush()


def _storage_size(deflated_bytes: int, raw_bytes: int) -> MemberSize:
    if deflated_bytes < raw_bytes:
        return MemberSize("deflate", deflated_bytes)
    return MemberSize("stored", raw_bytes)


def _rotate_bf16(bits: np.ndarray, group_size: int) -> np.ndarray:
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("BF16 reference values must be finite")
    rotated = values.astype(np.float64)
    stride = 1
    while stride < group_size:
        stage = rotated.reshape(-1, 4, stride)
        # Each row of the normalized regular H4 subtracts one reversed input.
        stage[:] = stage.sum(axis=1, keepdims=True) * 0.5 - stage[:, ::-1, :]
        stride *= 4
    return rotated


def _requantize(target: np.ndarray, codes: np.ndarray, scales: np.ndarray, multiplier: float):
    widened = (scales.astype(np.float64) * multiplier).astype("<f4")
    zero_rows = ~np.any(codes, axis=1)
    widened[zero_rows] = scales[zero_rows]
    trial = target / widened
    np.rint(trial, out=trial)
    np.clip(trial, -128, 127, out=trial)
    trial[zero_rows] = 0
    return trial.astype(np.int8), widened


def rows_per_batch(width: int) -> int:
    return max(1, _BATCH_WEIGHTS // width)


def _candidate_rows(codes, scales, reference, group_size, multiplier):
    batch_rows = rows_per_batch(codes.shape[1])
    for start in range(0, len(codes), batch_rows):
        rows = slice(start, start + batch_rows)
        q0, s0 = codes[rows], scales[rows]
        if multiplier == 1.0 and reference is None:
            yield rows, q0, s0, None, None
            continue

        original = np.multiply(q0, s0, dtype=np.float64)
        target = original if reference is None else _rotate_bf16(reference[rows], group_size)
        q1, s1 = (q0, s0) if multiplier == 1.0 else _requantize(target, q0, s0, multiplier)
        yield rows, q1, s1, original, target


def introduces_fp32_overflow(original: np.ndarray, reconstructed: np.ndarray, scales: np.ndarray) -> bool:
    if float(scales.max()) <= _FP32_MAX / 128:
        return False
    with np.errstate(over="ignore"):
        finite_before = np.isfinite(original.astype(np.float32))
        finite_after = np.isfinite(reconstructed.astype(np.float32))
    return bool(np.any(finite_before & ~finite_after))


def _measure(codes, scales, reference, group_size, multiplier) -> Candidate | None:
    weight_stream = _deflater()
    weight_bytes = 0
    energy = error = added_sse = 0.0

    for _, q1, s1, original, target in _candidate_rows(codes, scales, reference, group_size, multiplier):
        weight_bytes += sum(len(weight_stream.compress(chunk)) for chunk in array_chunks(q1))
        if target is None:
            continue
        reconstructed = np.multiply(q1, s1, dtype=np.float64)
        if multiplier != 1.0 and introduces_fp32_overflow(original, reconstructed, s1):
            return None
        residual = reconstructed - target
        energy += float(np.sum(target * target))
        error += float(np.sum(residual * residual))
        if reference is not None and multiplier != 1.0:
            change = reconstructed - original
            # Difference of squared residuals without subtracting two large SSE totals.
            added_sse += float(np.sum(change * (residual + (original - target))))

    weight_bytes += len(weight_stream.flush())
    if energy == 0.0 and error != 0.0:
        raise ValueError("BF16 reference has zero energy for a nonzero reconstruction")
    denominator = max(energy, float(np.finfo(np.float64).tiny))
    return Candidate(
        multiplier=multiplier,
        weight=_storage_size(weight_bytes, codes.nbytes),
        scale=MemberSize("stored", scales.nbytes),
        added_error=max(0.0, added_sse if reference is not None else error) / denominator,
        reference_error=error / denominator if reference is not None else None,
    )


def _rank(candidate: Candidate, weight_count: int, target_bpw: float) -> tuple:
    distortion = candidate.added_error if candidate.reference_error is None else candidate.reference_error
    if 8 * candidate.weight.nbytes / weight_count <= target_bpw:
        return (False, distortion, candidate.total_bytes, candidate.multiplier)
    return (True, candidate.total_bytes, distortion, candidate.multiplier)


def validate_limits(target_bpw: float, max_error: float) -> None:
    if not math.isfinite(target_bpw) or target_bpw <= 0:
        raise ValueError("target_bpw must be finite and positive")
    if not math.isfinite(max_error) or max_error < 0:
        raise ValueError("max_error must be finite and nonnegative")


def optimize_weight(
    checkpoint: Checkpoint,
    weight: str,
    *,
    target_bpw: float,
    max_error: float,
    protect: tuple[str, ...] = (),
    reference: Checkpoint | None = None,
) -> Optimization | None:
    """Fit one supported family. None signals passthrough of the entire family."""
    validate_limits(target_bpw, max_error)
    family = classify_weight(checkpoint, weight, protect)
    if family.status != "eligible":
        return None

    codes = checkpoint.array(weight)
    scales = checkpoint.array(family.scale)
    reference_bits = None
    if reference is not None:
        info = reference.tensors.get(weight)
        if info is None:
            raise ValueError(f"BF16 reference is missing weight {weight!r}")
        if info.dtype != "BF16" or info.shape != codes.shape:
            raise ValueError(f"Expected BF16 reference weight {weight!r} with shape {codes.shape}")
        reference_bits = reference.array(weight)

    group_size = family.group_size
    try:
        baseline = _measure(codes, scales, reference_bits, group_size, SCALE_MULTIPLIERS[0])
    except ValueError as error:
        raise ValueError(f"{weight!r}: {error}") from error
    assert baseline is not None
    best = baseline
    if 8 * baseline.weight.nbytes / codes.size > target_bpw:
        max_scale = float(scales.max())
        for multiplier in SCALE_MULTIPLIERS[1:]:
            if max_scale * multiplier > _FP32_MAX:
                break
            trial = _measure(codes, scales, reference_bits, group_size, multiplier)
            if trial is None or trial.added_error > max_error or trial.total_bytes >= baseline.total_bytes:
                continue
            if _rank(trial, codes.size, target_bpw) < _rank(best, codes.size, target_bpw):
                best = trial

    if best is not baseline:
        optimized_codes = np.empty_like(codes, subok=False)
        optimized_scales = np.empty_like(scales, subok=False)
        for rows, q1, s1, _, _ in _candidate_rows(codes, scales, reference_bits, group_size, best.multiplier):
            optimized_codes[rows] = q1
            optimized_scales[rows] = s1
        codes, scales = optimized_codes, optimized_scales
    return Optimization(codes, scales, baseline, best, target_bpw)
