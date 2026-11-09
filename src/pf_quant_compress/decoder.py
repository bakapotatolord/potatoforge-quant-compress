"""Validate and reconstruct PFQC archives without compression or optimization."""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import math
from pathlib import Path
import sys

from .formats import Family, classify_families, tensor_family
from .native import sha256_hex
from .zstd import ZSTD_LEVEL, decompress_chunks, decompress_into
from .safetensors_io import (
    CHUNK_BYTES, MAX_HEADER_BYTES, Checkpoint, TensorInfo, read_json_object, safetensors_header, tensor_nbytes,
)


FORMAT = "potatoforge_quant_compress"
METADATA_KEY = FORMAT + ".manifest"
SCHEMA_VERSION = 1
METHOD = "wider-scale"
MAX_MANIFEST_BYTES = 16 * 2**20
MAX_TENSORS = 50_000
MAX_TENSOR_BYTES = 64 * 2**30
MAX_DECODED_BYTES = 256 * 2**30

SCALE_MULTIPLIERS = (1.0, 1.025, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)
WIDE_SCALE_MULTIPLIERS = SCALE_MULTIPLIERS + (
    2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.5, 13.0, 13.5, 14.0,
)


def validate_limits(target_bpw: float, max_error: float) -> None:
    if not math.isfinite(target_bpw) or target_bpw <= 0:
        raise ValueError("target_bpw must be finite and positive")
    if not math.isfinite(max_error) or max_error < 0:
        raise ValueError("max_error must be finite and nonnegative")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value) -> bool:
    return type(value) in (int, float) and -sys.float_info.max <= value <= sys.float_info.max


def _count(value, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _validate_settings(settings: dict) -> None:
    _require(isinstance(settings, dict), "Invalid compression settings")
    target, error = settings.get("target_bpw"), settings.get("max_error")
    _require(_number(target) and _number(error), "Invalid compression target or error cap")
    validate_limits(target, error)
    protect = settings.get("protect")
    _require(
        isinstance(protect, list) and len(protect) <= 1024
        and all(isinstance(prefix, str) and prefix for prefix in protect),
        "Expected at most 1024 nonempty protection prefixes",
    )
    _require(settings.get("reference_mode") in ("quantized-only", "bf16"), "Invalid reference mode")


@contextmanager
def open_archive(path: Path | str, *, verify_checksums: bool = True) -> Iterator["Archive"]:
    yield Archive(Checkpoint(path), verify_checksums=verify_checksums)


class Archive:
    """Validated PFQC index. Exhaust tensor iterators to finish validation."""

    def __init__(self, container: Checkpoint, *, verify_checksums: bool = True):
        self._container = container
        self._verify_checksums = verify_checksums
        self.file_bytes = container.file_bytes
        text = (container.metadata or {}).get(METADATA_KEY)
        _require(text is not None, f"Missing PFQC compression metadata ({METADATA_KEY})")
        _require(0 < len(text) <= MAX_MANIFEST_BYTES, "PFQC manifest is too large or empty")
        encoded = text.encode("utf-8")
        _require(len(encoded) <= MAX_MANIFEST_BYTES, "PFQC manifest is too large")
        try:
            self.manifest = read_json_object(encoded)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"Malformed PFQC compression metadata: {error}") from error
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        _require(manifest.get("format") == FORMAT, "Unrecognized PFQC format")
        version = manifest.get("schema_version")
        _require(type(version) is int and version == SCHEMA_VERSION,
                 "Unsupported PFQC schema version")
        self.schema_version = version
        _require(
            isinstance(manifest.get("compressor_version"), str) and bool(manifest["compressor_version"])
            and manifest.get("source_format") == "safetensors"
            and manifest.get("method") == METHOD
            and type(manifest.get("zstd_level")) is int and manifest["zstd_level"] == ZSTD_LEVEL,
            "Invalid PFQC format settings",
        )
        self.settings = manifest.get("settings")
        _validate_settings(self.settings)
        self.metadata = manifest.get("metadata")
        _require(
            "metadata" in manifest and (self.metadata is None or (
                isinstance(self.metadata, dict) and all(isinstance(value, str) for value in self.metadata.values())
            )),
            "Invalid safetensors metadata",
        )
        self.families = manifest.get("families")
        _require(isinstance(self.families, dict), "Invalid family index")
        records = manifest.get("tensors")
        _require(isinstance(records, list) and len(records) <= MAX_TENSORS, "Invalid tensor index")
        _require(len(self._container.tensors) == len(records), "Unexpected or missing storage tensors")

        self.records: dict[str, dict] = {}
        self.tensors: dict[str, TensorInfo] = {}
        offset = 0
        for record in records:
            _require(isinstance(record, dict), "Invalid tensor record")
            name = record.get("name")
            _require(
                isinstance(name, str) and name != "__metadata__" and name not in self.tensors,
                "Invalid or duplicate tensor name",
            )
            dtype, shape = record.get("dtype"), record.get("shape")
            nbytes = tensor_nbytes(dtype, shape)
            _require(nbytes <= MAX_TENSOR_BYTES, f"Tensor {name!r} exceeds the decoded size limit")
            physical = self._container.tensors.get(name)
            _require(physical is not None, f"Missing storage tensor for {name!r}")
            _require(
                all(type(record.get(key)) is int and record[key] == nbytes for key in ("original_bytes", "decoded_bytes"))
                and type(record.get("stored_bytes")) is int and record["stored_bytes"] == physical.nbytes,
                f"Invalid tensor byte counts for {name!r}",
            )
            codec = record.get("codec")
            if codec == "stored":
                _require(
                    (physical.dtype, physical.shape, physical.nbytes) == (dtype, tuple(shape), nbytes),
                    f"Native tensor metadata does not match for {name!r}",
                )
            elif codec == "zstd":
                _require(
                    dtype == "I8" and len(shape) == 2 and physical.dtype == "U8"
                    and physical.shape == (physical.nbytes,) and 0 < physical.nbytes < nbytes,
                    f"Invalid U8 Zstandard payload for {name!r}",
                )
            else:
                raise ValueError(f"Unsupported tensor codec for {name!r}")
            digest = record.get("sha256")
            _require(
                isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
                f"Invalid checksum for {name!r}",
            )
            _require(
                record.get("storage") in ("unchanged", "encoded_int8", "optimized_int8"),
                f"Invalid storage mode for {name!r}",
            )
            self.records[name] = record
            self.tensors[name] = TensorInfo(dtype, tuple(shape), offset, nbytes)
            offset += nbytes
            _require(offset <= MAX_DECODED_BYTES, "Archive exceeds the total decoded size limit")

        source_bytes = manifest.get("source_bytes")
        _require(
            _count(source_bytes, offset + 8 + MAX_HEADER_BYTES) and source_bytes >= offset + 8,
            "Invalid original checkpoint size",
        )
        self.header = safetensors_header(self.tensors, self.metadata)
        self.reconstructed_bytes = len(self.header) + offset

    def chunks(self, name: str, chunk_size: int = CHUNK_BYTES) -> Iterator[bytes]:
        return self._chunks(name, chunk_size, self._verify_checksums)

    def check_tensor(self, name: str, data) -> None:
        """Validate a contiguous buffer against its decoded manifest record."""
        raw = memoryview(data).cast("B")
        _require(len(raw) == self.tensors[name].nbytes, f"Decoded size does not match for {name!r}")
        if self._verify_checksums:
            _require(sha256_hex(raw) == self.records[name]["sha256"], f"Checksum mismatch for {name!r}")

    def decode_into(self, name: str, output) -> None:
        """Reconstruct a Zstandard tensor into an exact-sized writable buffer."""
        buffer = memoryview(output).cast("B")
        record = self.records[name]
        _require(record["codec"] == "zstd", f"Expected a Zstandard tensor for {name!r}")
        _require(not buffer.readonly and len(buffer) == record["decoded_bytes"], f"Invalid output buffer for {name!r}")
        decompress_into(self._container.chunks(name), buffer)
        self.check_tensor(name, buffer)

    def _chunks(self, name: str, chunk_size: int, verify_checksums: bool) -> Iterator[bytes]:
        _require(type(chunk_size) is int and 0 < chunk_size <= CHUNK_BYTES, "Invalid tensor chunk size")
        record = self.records[name]
        size = record["decoded_bytes"]
        if record["codec"] == "zstd":
            chunks = decompress_chunks(self._container.chunks(name), size, chunk_size)
        else:
            chunks = self._container.chunks(name, chunk_size)
        digest = sha256() if verify_checksums else None
        count = 0
        for chunk in chunks:
            count += len(chunk)
            if digest is not None:
                digest.update(chunk)
            yield chunk
        _require(count == size, f"Decoded size does not match for {name!r}")
        if digest is not None:
            _require(digest.hexdigest() == record["sha256"], f"Checksum mismatch for {name!r}")

    def validate_families(self) -> None:
        observed = classify_families(self, tuple(self.settings["protect"]))
        _require(self.families.keys() == observed.keys(), "Quantization family index does not match decoded markers")
        modified = set()
        eligible = set()
        for name, family in observed.items():
            declared = self.families[name]
            _require(
                isinstance(declared, dict) and "fit" in declared and declared.get("family") == asdict(family),
                f"Quantization family metadata does not match decoded tensors for {name!r}",
            )
            fit = declared.get("fit")
            if family.status == "eligible":
                eligible.add(name)
                if self._validate_fit(family, fit):
                    modified.update((family.weight, family.scale))
            else:
                _require(fit is None, f"Ineligible or protected family {name!r} has a fit record")
        for name, record in self.records.items():
            encoded = record["codec"] == "zstd"
            mode = "encoded_int8" if encoded else "unchanged"
            if name in modified:
                mode = "optimized_int8"
            _require(
                "family" in record and record["family"] == tensor_family(name, observed) and record["storage"] == mode,
                f"Invalid family linkage or storage mode for {name!r}",
            )
            _require(not encoded or name in eligible, f"Ineligible encoded weight {name!r}")

    def _validate_fit(self, family: Family, fit: dict) -> bool:
        name = family.weight
        _require(
            isinstance(fit, dict) and {"baseline_reference_error", "optimized_reference_error"} <= fit.keys(),
            f"Missing fit record or reference metrics for {name!r}",
        )
        weight, scale = self.records[name], self.records[family.scale]
        multiplier, error = fit.get("multiplier"), fit.get("added_relative_sse")
        allowed = WIDE_SCALE_MULTIPLIERS
        _require(
            _number(multiplier) and multiplier in allowed
            and _number(error) and 0 <= error <= self.settings["max_error"],
            f"Invalid multiplier or distortion cap for {name!r}",
        )
        baseline_weight, baseline_scale = fit.get("baseline_weight_bytes"), fit.get("baseline_scale_bytes")
        _require(
            _count(baseline_weight, weight["decoded_bytes"]) and baseline_weight > 0
            and type(baseline_scale) is int and baseline_scale == scale["decoded_bytes"],
            f"Invalid baseline sizes for {name!r}",
        )
        weight_count = weight["decoded_bytes"]
        selected_total = weight["stored_bytes"] + scale["stored_bytes"]
        payload_bpw = 8 * weight["stored_bytes"] / weight_count
        _require(
            fit.get("payload_bpw") == payload_bpw and fit.get("all_in_bpw") == 8 * selected_total / weight_count
            and fit.get("rate_target_missed") is (payload_bpw > self.settings["target_bpw"]),
            f"Inconsistent rate metrics for {name!r}",
        )
        e0, e1 = fit.get("baseline_reference_error"), fit.get("optimized_reference_error")
        if self.settings["reference_mode"] == "bf16":
            _require(_number(e0) and _number(e1) and e0 >= 0 and e1 >= 0, f"Invalid BF16 errors for {name!r}")
            # The optimizer evaluates the SSE delta directly to avoid cancellation.
            _require(abs(max(0, e1 - e0) - error) <= 1e-12 * max(e0, e1, 1.0), f"Inconsistent BF16 delta for {name!r}")
        else:
            _require(e0 is None and e1 is None, f"Unexpected BF16 errors for {name!r}")
        if multiplier == 1:
            _require(
                error == 0 and e0 == e1 and baseline_weight == weight["stored_bytes"]
                and baseline_scale == scale["stored_bytes"],
                f"Inconsistent no-op fit for {name!r}",
            )
            return False
        _require(
            weight["codec"] == "zstd" and selected_total < baseline_weight + baseline_scale
            and 8 * baseline_weight / weight_count > self.settings["target_bpw"],
            f"Optimized family {name!r} violates the baseline storage rule",
        )
        return True

    def check_integrity(self) -> None:
        self.validate_families()
        for name in self.tensors:
            for _ in self._chunks(name, CHUNK_BYTES, verify_checksums=True):
                pass

    def report(self) -> dict:
        families = []
        counts = Counter()
        weights = stored_weights = stored_scales = baseline_bytes = 0
        multipliers, errors = [], []
        misses = 0
        for record in self.families.values():
            family, fit = record["family"], record["fit"]
            details = dict(family)
            counts[family["status"]] += 1
            if fit is not None:
                weight, scale = self.records[family["weight"]], self.records[family["scale"]]
                details.update(fit)
                details.update(
                    weight_count=weight["decoded_bytes"], stored_weight_bytes=weight["stored_bytes"],
                    stored_scale_bytes=scale["stored_bytes"], scale_shape=scale["shape"], scale_dtype=scale["dtype"],
                    weight_codec=weight["codec"],
                )
                if fit["baseline_reference_error"] is not None:
                    details["raw_reference_error_delta"] = fit["optimized_reference_error"] - fit["baseline_reference_error"]
                weights += weight["decoded_bytes"]
                stored_weights += weight["stored_bytes"]
                stored_scales += scale["stored_bytes"]
                baseline_bytes += fit["baseline_weight_bytes"] + fit["baseline_scale_bytes"]
                misses += fit["rate_target_missed"]
                if fit["multiplier"] != 1:
                    multipliers.append(fit["multiplier"])
                    errors.append(fit["added_relative_sse"])
            families.append(details)

        source_bytes = self.manifest["source_bytes"]
        stored_bytes = sum(tensor["stored_bytes"] for tensor in self.records.values())
        fitted_bytes = stored_weights + stored_scales
        optimized = len(multipliers)
        return {
            "format": FORMAT, "schema_version": self.schema_version, "method": METHOD, "zstd_level": ZSTD_LEVEL,
            "settings": self.settings, "tensor_count": len(self.records), "source_bytes": source_bytes,
            "archive_bytes": self.file_bytes, "reconstructed_bytes": self.reconstructed_bytes,
            "archive_fraction_of_source": self.file_bytes / source_bytes,
            "bytes_saved_vs_source": source_bytes - self.file_bytes,
            "baseline_tensor_bytes": stored_bytes - fitted_bytes + baseline_bytes,
            "stored_tensor_bytes": stored_bytes,
            "archive_header_bytes": self.file_bytes - stored_bytes,
            "wider_scale_savings_bytes": baseline_bytes - fitted_bytes,
            "eligible_family_count": counts["eligible"], "optimized_family_count": optimized,
            "protected_family_count": counts["protected"], "unsupported_family_count": counts["unsupported"],
            "malformed_family_count": counts["malformed"], "skipped_family_count": counts["eligible"] - optimized,
            "preserved_tensor_count": len(self.records) - 2 * optimized,
            "encoded_weight_count": sum(record["codec"] == "zstd" for record in self.records.values()),
            "rate_target_misses": misses,
            "payload_bpw": 8 * stored_weights / weights if weights else None,
            "all_in_bpw": 8 * fitted_bytes / weights if weights else None,
            "min_multiplier": min(multipliers, default=None), "max_multiplier": max(multipliers, default=None),
            "mean_multiplier": sum(multipliers) / optimized if optimized else None,
            "max_added_relative_sse": max(errors, default=None),
            "mean_added_relative_sse": sum(errors) / optimized if optimized else None,
            "families": families,
        }
