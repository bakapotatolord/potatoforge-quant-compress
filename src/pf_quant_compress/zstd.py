"""Bounded streaming access to the bundled Zstandard library."""

import ctypes as ct
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import sysconfig


_DLL = Path(__file__).resolve().parent / "_vendor" / "zstd-win-x64" / "libzstd.dll"
_SHA256 = "8f07e1112ed283e5cd2798833e9a3c32d8961381bc36da04af57a1b0ca9bd40b"
CHUNK_SIZE = 256 * 1024
ZSTD_LEVEL = 1


class _Buffer(ct.Structure):
    _fields_ = [("ptr", ct.c_void_p), ("size", ct.c_size_t), ("pos", ct.c_size_t)]


@lru_cache(maxsize=1)
def library():
    if sysconfig.get_platform() != "win-amd64":
        raise RuntimeError("PFQC requires the bundled Zstandard library on Windows x64")
    try:
        if sha256(_DLL.read_bytes()).hexdigest() != _SHA256:
            raise RuntimeError("Bundled Zstandard library has an unexpected checksum")
        dll = ct.CDLL(str(_DLL), winmode=0x900)
        for name, result, args in (
            ("createCStream", ct.c_void_p, []), ("freeCStream", ct.c_size_t, [ct.c_void_p]),
            ("initCStream", ct.c_size_t, [ct.c_void_p, ct.c_int]),
            ("compressStream2", ct.c_size_t, [ct.c_void_p, ct.POINTER(_Buffer), ct.POINTER(_Buffer), ct.c_int]),
            ("createDStream", ct.c_void_p, []), ("freeDStream", ct.c_size_t, [ct.c_void_p]),
            ("initDStream", ct.c_size_t, [ct.c_void_p]),
            ("DCtx_setParameter", ct.c_size_t, [ct.c_void_p, ct.c_int, ct.c_int]),
            ("decompressStream", ct.c_size_t, [ct.c_void_p, ct.POINTER(_Buffer), ct.POINTER(_Buffer)]),
            ("isError", ct.c_uint, [ct.c_size_t]), ("getErrorName", ct.c_char_p, [ct.c_size_t]),
        ):
            fn = getattr(dll, "ZSTD_" + name)
            fn.restype, fn.argtypes = result, args
        return dll
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"Cannot load bundled Zstandard library: {error}") from error


def _check(dll, result):
    if dll.ZSTD_isError(result):
        raise ValueError("Invalid Zstandard data: " + dll.ZSTD_getErrorName(result).decode("ascii"))
    return result


class Compressor:
    def __init__(self):
        self.context = None
        self.dll = library()
        self.context = self.dll.ZSTD_createCStream()
        if not self.context:
            raise MemoryError("Cannot allocate Zstandard compressor")
        try:
            _check(self.dll, self.dll.ZSTD_initCStream(self.context, ZSTD_LEVEL))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.context is not None:
            self.dll.ZSTD_freeCStream(self.context)
            self.context = None

    def __del__(self):
        self.close()

    def _run(self, data, mode):
        if self.context is None:
            raise ValueError("Zstandard compressor is closed")
        if len(data) > CHUNK_SIZE:
            raise ValueError("Zstandard input chunk exceeds the limit")
        data = bytes(data)
        source = ct.create_string_buffer(data)
        inp = _Buffer(ct.addressof(source), len(data), 0)
        result = []
        while True:
            dest = ct.create_string_buffer(CHUNK_SIZE)
            out = _Buffer(ct.addressof(dest), CHUNK_SIZE, 0)
            remaining = _check(self.dll, self.dll.ZSTD_compressStream2(self.context, ct.byref(out), ct.byref(inp), mode))
            if out.pos:
                result.append(dest.raw[:out.pos])
            if inp.pos == inp.size and (mode == 0 or remaining == 0):
                return b"".join(result)

    def compress(self, data):
        return self._run(data, 0)

    def flush(self):
        try:
            return self._run(b"", 2)
        finally:
            self.close()


def decompress_into(chunks, output):
    """Decode exactly one frame into a writable contiguous destination."""
    view = memoryview(output).cast("B")
    if view.readonly:
        raise ValueError("Zstandard destination is read-only")
    dll = library()
    context = dll.ZSTD_createDStream()
    if not context:
        raise MemoryError("Cannot allocate Zstandard decompressor")
    try:
        _check(dll, dll.ZSTD_initDStream(context))
        # Reject frames that would require a decoder window larger than 128 MiB.
        _check(dll, dll.ZSTD_DCtx_setParameter(context, 100, 27))
        address = ct.addressof(ct.c_ubyte.from_buffer(view))
        out = _Buffer(address, len(view), 0)
        finished = False
        for chunk in chunks:
            if finished:
                raise ValueError("Trailing Zstandard data")
            source = ct.create_string_buffer(chunk)
            inp = _Buffer(ct.addressof(source), len(chunk), 0)
            while inp.pos < inp.size:
                old_in, old_out = inp.pos, out.pos
                remaining = _check(dll, dll.ZSTD_decompressStream(context, ct.byref(out), ct.byref(inp)))
                if remaining == 0:
                    if inp.pos != inp.size:
                        raise ValueError("Trailing Zstandard data")
                    finished = True
                    break
                if old_in == inp.pos and old_out == out.pos:
                    raise ValueError("Zstandard output exceeds declared size")
        if not finished or out.pos != out.size:
            raise ValueError("Incomplete Zstandard data or wrong decoded size")
    finally:
        dll.ZSTD_freeDStream(context)


def decompress_chunks(chunks, decoded_bytes, chunk_size):
    """Decode exactly one frame while keeping output buffers bounded."""
    dll = library()
    context = dll.ZSTD_createDStream()
    if not context:
        raise MemoryError("Cannot allocate Zstandard decompressor")
    total = 0
    pending = bytearray()
    finished = False
    try:
        _check(dll, dll.ZSTD_initDStream(context))
        _check(dll, dll.ZSTD_DCtx_setParameter(context, 100, 27))
        for chunk in chunks:
            if finished:
                raise ValueError("Trailing Zstandard data")
            source = ct.create_string_buffer(chunk)
            inp = _Buffer(ct.addressof(source), len(chunk), 0)
            while inp.pos < inp.size:
                dest = ct.create_string_buffer(min(chunk_size - len(pending), decoded_bytes - total) or 1)
                out = _Buffer(ct.addressof(dest), len(dest), 0)
                old = inp.pos
                remaining = _check(dll, dll.ZSTD_decompressStream(context, ct.byref(out), ct.byref(inp)))
                total += out.pos
                if total > decoded_bytes:
                    raise ValueError("Zstandard output exceeds declared size")
                if out.pos:
                    pending.extend(dest.raw[:out.pos])
                    if len(pending) == chunk_size:
                        yield bytes(pending)
                        pending.clear()
                if remaining == 0:
                    if inp.pos != inp.size:
                        raise ValueError("Trailing Zstandard data")
                    finished = True
                    break
                if old == inp.pos and not out.pos:
                    raise ValueError("Invalid Zstandard stream")
        if not finished or total != decoded_bytes:
            raise ValueError("Incomplete Zstandard data or wrong decoded size")
        if pending:
            yield bytes(pending)
    finally:
        dll.ZSTD_freeDStream(context)
