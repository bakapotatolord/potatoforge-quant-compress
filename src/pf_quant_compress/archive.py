from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
from tempfile import TemporaryFile
from time import perf_counter
from typing import BinaryIO
import zlib

from safetensors import safe_open

from . import __version__
from .formats import Family, classify_families, tensor_family
from .optimize import (
    DEFLATE_LEVEL, SCALE_MULTIPLIERS, MemberSize, Optimization, encoded_chunks, optimize_weight, validate_limits,
)
from .safetensors_io import (
    CHUNK_BYTES, MAX_HEADER_BYTES, Checkpoint, TensorInfo, atomic_output, read_chunks,
    read_json_object, safetensors_header, tensor_alignment, tensor_nbytes,
)


FORMAT = "potatoforge_quant_compress"
METADATA_KEY = FORMAT + ".manifest"
SCHEMA_VERSION = 1
METHOD = "wider-scale"
MAX_MANIFEST_BYTES = 16 * 2**20
MAX_TENSORS = 50_000
MAX_TENSOR_BYTES = 64 * 2**30
MAX_DECODED_BYTES = 256 * 2**30
MAX_DEFLATE_RATIO = 1100

_LOG = logging.getLogger(__name__)


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
def open_archive(path: Path | str) -> Iterator["Archive"]:
    yield Archive(Checkpoint(path))


def has_compression_metadata(checkpoint: Checkpoint) -> bool:
    return METADATA_KEY in (checkpoint.metadata or {})


def ordinary_checkpoint(path: Path | str) -> Checkpoint:
    checkpoint = Checkpoint(path)
    if has_compression_metadata(checkpoint):
        Archive(checkpoint)
        raise ValueError(f"{checkpoint.path} is already a compressed PFQC safetensors checkpoint")
    return checkpoint


def _inflate(
    chunks: Iterable[bytes], decoded_bytes: int, chunk_size: int, name: str,
) -> Iterator[bytes]:
    decoder = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
    remaining = decoded_bytes
    pending = bytearray()
    try:
        for compressed in chunks:
            while True:
                limit = min(chunk_size - len(pending), remaining + 1)
                decoded = decoder.decompress(compressed, limit)
                _require(len(decoded) <= remaining, f"Decoded size exceeds the declared length for {name!r}")
                _require(not decoder.unused_data, f"Trailing DEFLATE data for {name!r}")
                remaining -= len(decoded)
                pending.extend(decoded)
                if len(pending) == chunk_size:
                    yield bytes(pending)
                    pending.clear()
                compressed = decoder.unconsumed_tail
                if not compressed and len(decoded) < limit:
                    break
    except zlib.error as error:
        raise ValueError(f"Invalid DEFLATE stream for {name!r}: {error}") from error
    _require(decoder.eof and remaining == 0, f"Incomplete DEFLATE data or wrong decoded length for {name!r}")
    if pending:
        yield bytes(pending)


class Archive:
    """Validated PFQC index. Exhaust tensor iterators to check their checksums."""

    def __init__(self, container: Checkpoint):
        self._container = container
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
        _require(type(version) is int and version == SCHEMA_VERSION, "Unsupported PFQC schema version")
        _require(
            isinstance(manifest.get("compressor_version"), str) and bool(manifest["compressor_version"])
            and manifest.get("source_format") == "safetensors"
            and manifest.get("method") == METHOD
            and type(manifest.get("zlib_level")) is int and manifest["zlib_level"] == DEFLATE_LEVEL,
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
            elif codec == "deflate":
                _require(
                    dtype == "I8" and len(shape) == 2 and physical.dtype == "U8"
                    and physical.shape == (physical.nbytes,) and 0 < physical.nbytes < nbytes,
                    f"Invalid U8 DEFLATE payload for {name!r}",
                )
                _require(nbytes <= physical.nbytes * MAX_DEFLATE_RATIO, f"Suspicious compression ratio for {name!r}")
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
        _require(type(chunk_size) is int and 0 < chunk_size <= CHUNK_BYTES, "Invalid tensor chunk size")
        record = self.records[name]
        size = record["decoded_bytes"]
        if record["codec"] == "deflate":
            chunks = _inflate(self._container.chunks(name), size, chunk_size, name)
        else:
            chunks = self._container.chunks(name, chunk_size)
        digest = sha256()
        count = 0
        for chunk in chunks:
            count += len(chunk)
            digest.update(chunk)
            yield chunk
        _require(count == size, f"Decoded size does not match for {name!r}")
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
            encoded = record["codec"] == "deflate"
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
        _require(
            _number(multiplier) and multiplier in SCALE_MULTIPLIERS
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
            weight["codec"] == "deflate" and selected_total < baseline_weight + baseline_scale
            and 8 * baseline_weight / weight_count > self.settings["target_bpw"],
            f"Optimized family {name!r} violates the baseline storage rule",
        )
        return True

    def check_integrity(self) -> None:
        self.validate_families()
        for name in self.tensors:
            for _ in self.chunks(name):
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
            "format": FORMAT, "schema_version": SCHEMA_VERSION, "method": METHOD, "zlib_level": DEFLATE_LEVEL,
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
            "encoded_weight_count": sum(record["codec"] == "deflate" for record in self.records.values()),
            "rate_target_misses": misses,
            "payload_bpw": 8 * stored_weights / weights if weights else None,
            "all_in_bpw": 8 * fitted_bytes / weights if weights else None,
            "min_multiplier": min(multipliers, default=None), "max_multiplier": max(multipliers, default=None),
            "mean_multiplier": sum(multipliers) / optimized if optimized else None,
            "max_added_relative_sse": max(errors, default=None),
            "mean_added_relative_sse": sum(errors) / optimized if optimized else None,
            "families": families,
        }


def _stage_tensor(staged: BinaryIO, record: dict, values, measured: MemberSize) -> int:
    raw = memoryview(values).cast("B")
    start = staged.tell()
    for chunk in encoded_chunks(values, measured.codec):
        staged.write(chunk)
    size = staged.tell() - start
    _require(
        len(raw) == record["decoded_bytes"] and size == measured.nbytes,
        f"Written tensor sizes changed for {record['name']!r}",
    )
    record.update(sha256=sha256(raw).hexdigest(), stored_bytes=size, codec=measured.codec)
    return start


def _fit_record(result: Optimization) -> dict:
    return {
        "multiplier": result.selected.multiplier,
        "baseline_weight_bytes": result.baseline.weight.nbytes,
        "baseline_scale_bytes": result.baseline.scale.nbytes,
        "payload_bpw": result.payload_bpw,
        "all_in_bpw": result.all_in_bpw,
        "rate_target_missed": result.rate_target_missed,
        "added_relative_sse": result.selected.added_error,
        "baseline_reference_error": result.baseline.reference_error,
        "optimized_reference_error": result.selected.reference_error,
    }


def compress_checkpoint(
    source: Path | str, output: Path | str, *, target_bpw: float, max_error: float,
    protect: tuple[str, ...] = (), reference: Path | str | None = None, force: bool = False,
) -> dict:
    source, output = Path(source), Path(output)
    inputs = (source,) if reference is None else (source, Path(reference))
    settings = {
        "target_bpw": target_bpw, "max_error": max_error, "protect": list(protect),
        "reference_mode": "quantized-only" if reference is None else "bf16",
    }
    _validate_settings(settings)
    started = perf_counter()
    with atomic_output(output, force=force, inputs=inputs) as temporary:
        checkpoint = ordinary_checkpoint(source)
        bf16 = ordinary_checkpoint(reference) if reference is not None else None
        _require(len(checkpoint.tensors) <= MAX_TENSORS, "Source exceeds the tensor count limit")
        _require(
            sum(tensor.nbytes for tensor in checkpoint.tensors.values()) <= MAX_DECODED_BYTES
            and all(tensor.nbytes <= MAX_TENSOR_BYTES for tensor in checkpoint.tensors.values()),
            "Source exceeds the decoded size limits",
        )
        families = classify_families(checkpoint, protect)
        tensors = {
            name: {
                "name": name, "dtype": tensor.dtype, "shape": tensor.shape,
                "family": tensor_family(name, families), "storage": "unchanged",
                "original_bytes": tensor.nbytes, "decoded_bytes": tensor.nbytes,
            }
            for name, tensor in checkpoint.tensors.items()
        }
        family_records = {}
        # Only encoded weights and changed scales need staging before the safetensors header is known.
        staged_offsets = {}
        with TemporaryFile(mode="w+b", dir=output.parent, prefix=".pfqc-", suffix=".tmp") as staged:
            for name, family in families.items():
                family_records[name] = {"family": asdict(family), "fit": None}
                if family.status != "eligible":
                    continue
                _LOG.info("Fitting %r", name)
                fit = optimize_weight(
                    checkpoint, name, target_bpw=target_bpw, max_error=max_error, protect=protect, reference=bf16,
                )
                _require(fit is not None, f"Source family changed while reading {name!r}")
                family_records[name]["fit"] = _fit_record(fit)
                modified = fit.selected.multiplier != 1
                for key, values, size in (
                    (name, fit.codes, fit.selected.weight), (family.scale, fit.scales, fit.selected.scale),
                ):
                    encoded = size.codec == "deflate"
                    if encoded or modified:
                        staged_offsets[key] = _stage_tensor(staged, tensors[key], values, size)
                    if modified:
                        tensors[key]["storage"] = "optimized_int8"
                    elif encoded:
                        tensors[key]["storage"] = "encoded_int8"
                del fit, values

            for name, record in tensors.items():
                if name not in staged_offsets:
                    digest = sha256()
                    for chunk in checkpoint.chunks(name):
                        digest.update(chunk)
                    record.update(sha256=digest.hexdigest(), stored_bytes=record["decoded_bytes"], codec="stored")
            manifest = {
                "format": FORMAT, "schema_version": SCHEMA_VERSION, "compressor_version": __version__,
                "source_format": "safetensors", "method": METHOD, "zlib_level": DEFLATE_LEVEL,
                "settings": settings, "metadata": checkpoint.metadata, "source_bytes": checkpoint.file_bytes,
                "tensors": list(tensors.values()), "families": family_records,
            }
            payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            _require(len(payload.encode("utf-8")) <= MAX_MANIFEST_BYTES, "PFQC manifest exceeds the size limit")
            physical = {}
            for name, record in tensors.items():
                size = record["stored_bytes"]
                if record["codec"] == "deflate":
                    physical[name] = TensorInfo("U8", (size,), 0, size)
                else:
                    physical[name] = checkpoint.tensors[name]
            # Put wide native dtypes first so their storage offsets stay aligned.
            physical = dict(sorted(physical.items(), key=lambda item: -tensor_alignment(item[1].dtype)))
            with temporary.open("wb") as stream:
                stream.write(safetensors_header(physical, {METADATA_KEY: payload}))
                for name, tensor in physical.items():
                    if name in staged_offsets:
                        staged.seek(staged_offsets[name])
                        chunks = read_chunks(staged, tensor.nbytes)
                    else:
                        chunks = checkpoint.chunks(name)
                    for chunk in chunks:
                        stream.write(chunk)

        encoded_at = perf_counter()
        with open_archive(temporary) as packed:
            packed.check_integrity()
            report = packed.report()
        encode_seconds = encoded_at - started
        report.update(
            operation="compress", encode_seconds=encode_seconds, decode_verify_seconds=perf_counter() - encoded_at,
            encode_mib_per_second=report["source_bytes"] / (encode_seconds * 2**20),
            checksums_verified=True,
        )
    return report


def decompress_archive(source: Path | str, output: Path | str, *, force: bool = False) -> dict:
    source, output = Path(source), Path(output)
    started = perf_counter()
    with open_archive(source) as packed, atomic_output(output, force=force, inputs=(source,)) as temporary:
        packed.validate_families()
        with temporary.open("wb") as stream:
            stream.write(packed.header)
            for name in packed.tensors:
                for chunk in packed.chunks(name):
                    stream.write(chunk)
        with safe_open(temporary, framework="np"):
            pass
        report = packed.report()
        decode_seconds = perf_counter() - started
        report.update(
            operation="decompress", decode_seconds=decode_seconds, checksums_verified=True,
            decode_mib_per_second=report["reconstructed_bytes"] / (decode_seconds * 2**20),
        )
    return report
