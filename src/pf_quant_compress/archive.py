from dataclasses import asdict
from hashlib import sha256
import json
import logging
from pathlib import Path
from tempfile import TemporaryFile
from time import perf_counter
from typing import BinaryIO

from safetensors import safe_open

from . import __version__
from .decoder import (
    Archive, DEFLATE_LEVEL, FORMAT, METADATA_KEY, METHOD, SCHEMA_VERSION, MAX_DECODED_BYTES,
    MAX_MANIFEST_BYTES, MAX_TENSOR_BYTES, MAX_TENSORS, _require, _validate_settings, open_archive,
)
from .formats import classify_families, tensor_family
from .optimize import MemberSize, Optimization, encoded_chunks, optimize_weight
from .safetensors_io import Checkpoint, TensorInfo, atomic_output, read_chunks, safetensors_header, tensor_alignment


_LOG = logging.getLogger(__name__)


def has_compression_metadata(checkpoint: Checkpoint) -> bool:
    return METADATA_KEY in (checkpoint.metadata or {})


def ordinary_checkpoint(path: Path | str) -> Checkpoint:
    checkpoint = Checkpoint(path)
    if has_compression_metadata(checkpoint):
        Archive(checkpoint)
        raise ValueError(f"{checkpoint.path} is already a compressed PFQC safetensors checkpoint")
    return checkpoint


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


def decompress_archive(
    source: Path | str, output: Path | str, *, force: bool = False, verify_checksums: bool = True,
) -> dict:
    source, output = Path(source), Path(output)
    started = perf_counter()
    with (
        open_archive(source, verify_checksums=verify_checksums) as packed,
        atomic_output(output, force=force, inputs=(source,)) as temporary,
    ):
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
            operation="decompress", decode_seconds=decode_seconds, checksums_verified=verify_checksums,
            decode_mib_per_second=report["reconstructed_bytes"] / (decode_seconds * 2**20),
        )
    return report
