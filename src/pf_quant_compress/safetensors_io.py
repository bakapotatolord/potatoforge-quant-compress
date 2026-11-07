from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Protocol

import numpy as np
from safetensors import safe_open


CHUNK_BYTES = 256 * 1024
MAX_HEADER_BYTES = 100_000_000
_ARRAY_DTYPES = {"I8": "i1", "U8": "u1", "F32": "<f4", "BF16": "<u2"}
_DTYPE_BITS = {
    dtype: bits
    for bits, dtypes in (
        (4, "F4"),
        (6, "F6_E2M3 F6_E3M2"),
        (8, "BOOL I8 U8 F8_E5M2 F8_E4M3 F8_E8M0 F8_E4M3FNUZ F8_E5M2FNUZ"),
        (16, "I16 U16 F16 BF16"),
        (32, "I32 U32 F32"),
        (64, "I64 U64 F64 C64"),
    )
    for dtype in dtypes.split()
}


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant {value!r}")


def read_json_object(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


class TensorReader(Protocol):
    @property
    def tensors(self) -> Mapping[str, TensorInfo]: ...

    def chunks(self, name: str, chunk_size: int = CHUNK_BYTES) -> Iterator[bytes]: ...


def read_chunks(stream: BinaryIO, nbytes: int, chunk_size: int = CHUNK_BYTES) -> Iterator[bytes]:
    while nbytes:
        chunk = stream.read(min(nbytes, chunk_size))
        if not chunk:
            raise ValueError("Tensor data ended before its declared length")
        nbytes -= len(chunk)
        yield chunk


def array_chunks(array: np.ndarray) -> Iterator[memoryview]:
    raw = memoryview(array).cast("B")
    for start in range(0, len(raw), CHUNK_BYTES):
        yield raw[start:start + CHUNK_BYTES]


class Checkpoint:
    """Validated safetensors index with on-demand, read-only tensor views."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with self.path.open("rb") as stream, safe_open(self.path, framework="np") as checked:
            self.file_bytes = os.fstat(stream.fileno()).st_size
            header_size = int.from_bytes(stream.read(8), "little")
            if not 0 < header_size <= min(self.file_bytes - 8, MAX_HEADER_BYTES):
                raise ValueError("Invalid safetensors header size")
            header = read_json_object(stream.read(header_size))
            self.metadata = checked.metadata()
            header.pop("__metadata__", None)
            data_start = 8 + header_size
            self.tensors: dict[str, TensorInfo] = {}
            # Stable sorting keeps header order when empty tensors share offsets.
            for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"]):
                start, end = entry["data_offsets"]
                self.tensors[name] = TensorInfo(entry["dtype"], tuple(entry["shape"]), data_start + start, end - start)

    def chunks(self, name: str, chunk_size: int = CHUNK_BYTES) -> Iterator[bytes]:
        tensor = self.tensors[name]
        with self.path.open("rb") as stream:
            stream.seek(tensor.offset)
            yield from read_chunks(stream, tensor.nbytes, chunk_size)

    def array(self, name: str) -> np.ndarray:
        """Map selected storage dtypes. BF16 views expose their raw uint16 words."""
        tensor = self.tensors[name]
        return np.memmap(
            self.path, mode="r", dtype=_ARRAY_DTYPES[tensor.dtype], offset=tensor.offset, shape=tensor.shape
        )


def tensor_nbytes(dtype: str, shape: list[int]) -> int:
    if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}")
    if not isinstance(shape, list) or len(shape) > 64 or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError("Invalid tensor shape")
    bits = _DTYPE_BITS[dtype] * math.prod(dim for dim in shape if dim)
    if bits >= 2**64 or (0 not in shape and bits % 8):
        raise ValueError("Tensor shape overflows or does not end on a byte boundary")
    return 0 if 0 in shape else bits // 8


def tensor_alignment(dtype: str) -> int:
    bits = _DTYPE_BITS.get(dtype)
    if bits is None:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}")
    return max(1, bits // 8)


def safetensors_header(tensors: Mapping[str, TensorInfo], metadata: dict[str, str] | None) -> bytes:
    header = {} if metadata is None else {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    for name, tensor in tensors.items():
        end = offset + tensor.nbytes
        header[name] = {"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError("Reconstructed safetensors header is too large")
    return len(encoded).to_bytes(8, "little") + encoded


def validate_output(path: Path, *, force: bool, inputs: tuple[Path, ...] = ()) -> None:
    resolved = path.resolve()
    exists = os.path.lexists(path)
    for source in inputs:
        if resolved == source.resolve() or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"Output {path} aliases an input or another output")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    if path.is_dir():
        raise IsADirectoryError(f"Output is a directory: {path}")
    if exists and not force:
        raise FileExistsError(f"Output already exists: {path}. Pass --force to replace it.")


@contextmanager
def atomic_output(path: Path, *, force: bool, inputs: tuple[Path, ...] = ()) -> Iterator[Path]:
    validate_output(path, force=force, inputs=inputs)
    with NamedTemporaryFile(dir=path.parent, prefix=".pfqc-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        validate_output(path, force=force, inputs=inputs)
        if force:
            os.replace(temporary, path)
        elif os.name == "nt":
            os.rename(temporary, path)
        else:
            # A same-directory hard link publishes atomically and fails if the name exists.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
