from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .decoder import SCALE_MULTIPLIERS, WIDE_SCALE_MULTIPLIERS, validate_limits
from .formats import classify_weight
from .zstd import Compressor
from .safetensors_io import Checkpoint, array_chunks


_BATCH_WEIGHTS = 64 * 1024
_FP32_MAX = float(np.finfo(np.float32).max)
DEFAULT_MAX_ERROR = 0.001


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


def validate_optimizer_options(search: str, wide_grid: bool, max_error: float) -> None:
    if search not in ("batch-major", "reference"):
        raise ValueError(f"Unsupported search {search!r}")
    if wide_grid and max_error <= DEFAULT_MAX_ERROR:
        raise ValueError(f"The experimental wide grid requires max_error > {DEFAULT_MAX_ERROR:g}")


def encoded_chunks(array: np.ndarray, codec: str) -> Iterator[bytes | memoryview]:
    """Encode contiguous native storage bytes with the measured member codec."""
    if codec == "stored":
        yield from array_chunks(array)
        return
    if codec != "zstd":
        raise ValueError(f"Unsupported codec {codec!r}")
    compressor = Compressor()
    for chunk in array_chunks(array):
        encoded = compressor.compress(chunk)
        if encoded:
            yield encoded
    yield compressor.flush()


def _storage_size(compressed_bytes: int, raw_bytes: int) -> MemberSize:
    if compressed_bytes < raw_bytes:
        return MemberSize("zstd", compressed_bytes)
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
    return _requantize_batch(target, scales, scales.astype(np.float64), ~np.any(codes, axis=1), multiplier)


def _requantize_batch(target, scales, scales_fp64, zero_rows, multiplier):
    widened = (scales_fp64 * multiplier).astype("<f4")
    widened[zero_rows] = scales[zero_rows]
    trial = target / widened
    np.rint(trial, out=trial)
    np.clip(trial, -128, 127, out=trial)
    trial[zero_rows] = 0
    return trial.astype(np.int8), widened


def rows_per_batch(width: int) -> int:
    return max(1, _BATCH_WEIGHTS // width)


def _row_slices(codes):
    batch_rows = rows_per_batch(codes.shape[1])
    for start in range(0, len(codes), batch_rows):
        yield slice(start, start + batch_rows)


def _candidate_rows(codes, scales, reference, group_size, multiplier):
    for rows in _row_slices(codes):
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
    weight_stream = Compressor()
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


@dataclass(eq=False)
class _Trial:
    multiplier: float
    stream: Compressor
    weight_bytes: int = 0
    error: float = 0.0
    added_sse: float = 0.0


def _quantized_denominator(codes, scales):
    energy = 0.0
    for rows in _row_slices(codes):
        original = np.multiply(codes[rows], scales[rows], dtype=np.float64)
        energy += float(np.sum(original * original))
    return max(energy, float(np.finfo(np.float64).tiny))


def _measure_batch(codes, scales, reference, group_size, multipliers, denominator, max_error):
    trials = [_Trial(m, Compressor()) for m in multipliers]
    energy = 0.0
    for rows in _row_slices(codes):
        q0, s0 = codes[rows], scales[rows]
        original = np.multiply(q0, s0, dtype=np.float64)
        target = original if reference is None else _rotate_bf16(reference[rows], group_size)
        if denominator is None:
            energy += float(np.sum(target * target))
        original_residual = None if reference is None else original - target
        scales_fp64 = s0.astype(np.float64)
        zero_rows = ~np.any(q0, axis=1)
        for trial in trials[:]:
            m = trial.multiplier
            q1, s1 = (q0, s0) if m == 1.0 else _requantize_batch(target, s0, scales_fp64, zero_rows, m)
            reconstructed = np.multiply(q1, s1, dtype=np.float64)
            if m != 1.0 and introduces_fp32_overflow(original, reconstructed, s1):
                trials.remove(trial)
                continue
            residual = reconstructed - target
            trial.error += float(np.sum(residual * residual))
            if reference is None:
                # Partial SSE is nondecreasing against the fixed, full-tensor denominator.
                if trial.error / denominator > max_error:
                    trials.remove(trial)
                    continue
            elif m != 1.0:
                change = reconstructed - original
                trial.added_sse += float(np.sum(change * (residual + original_residual)))
            trial.weight_bytes += sum(len(trial.stream.compress(chunk)) for chunk in array_chunks(q1))
        if not trials:
            break

    if denominator is None:
        denominator = max(energy, float(np.finfo(np.float64).tiny))
    for trial in trials:
        if reference is not None and energy == 0.0 and trial.error != 0.0:
            raise ValueError("BF16 reference has zero energy for a nonzero reconstruction")
        added_error = max(0.0, trial.error if reference is None else trial.added_sse) / denominator
        if added_error > max_error:
            continue
        weight_bytes = trial.weight_bytes + len(trial.stream.flush())
        yield Candidate(
            trial.multiplier, _storage_size(weight_bytes, codes.nbytes), MemberSize("stored", scales.nbytes),
            added_error, trial.error / denominator if reference is not None else None,
        )


def _rank(candidate: Candidate, weight_count: int, target_bpw: float) -> tuple:
    distortion = candidate.added_error if candidate.reference_error is None else candidate.reference_error
    if 8 * candidate.weight.nbytes / weight_count <= target_bpw:
        return (False, distortion, candidate.total_bytes, candidate.multiplier)
    return (True, candidate.total_bytes, distortion, candidate.multiplier)


def optimize_weight(
    checkpoint: Checkpoint,
    weight: str,
    *,
    target_bpw: float,
    max_error: float,
    protect: tuple[str, ...] = (),
    reference: Checkpoint | None = None,
    search: str = "batch-major",
    wide_grid: bool = False,
) -> Optimization | None:
    """Fit one supported family. None signals passthrough of the entire family."""
    validate_limits(target_bpw, max_error)
    validate_optimizer_options(search, wide_grid, max_error)
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
        baseline = _measure(codes, scales, reference_bits, group_size, 1.0)
    except ValueError as error:
        raise ValueError(f"{weight!r}: {error}") from error
    assert baseline is not None
    best = baseline
    if 8 * baseline.weight.nbytes / codes.size > target_bpw:
        max_scale = float(scales.max())
        grid = WIDE_SCALE_MULTIPLIERS if wide_grid else SCALE_MULTIPLIERS
        multipliers = tuple(m for m in grid[1:] if max_scale * m <= _FP32_MAX)
        if search == "reference":
            candidates = (_measure(codes, scales, reference_bits, group_size, m) for m in multipliers)
        else:
            denominator = _quantized_denominator(codes, scales) if multipliers and reference_bits is None else None
            candidates = _measure_batch(codes, scales, reference_bits, group_size, multipliers, denominator, max_error)
        for trial in candidates:
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
