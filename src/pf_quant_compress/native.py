"""Optional native implementations of PFQC v1's existing codec and hash."""

import ctypes as ct
from functools import cache
from hashlib import sha256
from pathlib import Path
import sys
import sysconfig

import numpy as np


NATIVE_TENSOR_LIMIT = 64 * 2**20
_DLL = Path(__file__).resolve().parent / "_vendor" / "libdeflate-win-x64" / "libdeflate.dll"
_DLL_SHA256 = "4d027ba4267ad402508de6c31e6b387ba1fae87790e3c4b57ae36943d1093935"
_SYSTEM32 = 0x800
_DLL_LOAD_DIR = 0x100
_SHA256_HANDLE = 0x41


@cache
def libdeflate():
    if sysconfig.get_platform() != "win-amd64":
        return None
    try:
        if sha256(_DLL.read_bytes()).hexdigest() != _DLL_SHA256:
            return None
        library = ct.CDLL(str(_DLL), winmode=_DLL_LOAD_DIR | _SYSTEM32)
        library.libdeflate_alloc_decompressor.argtypes = []
        library.libdeflate_alloc_decompressor.restype = ct.c_void_p
        library.libdeflate_free_decompressor.argtypes = [ct.c_void_p]
        library.libdeflate_free_decompressor.restype = None
        library.libdeflate_deflate_decompress_ex.argtypes = [
            ct.c_void_p, ct.c_char_p, ct.c_size_t, ct.c_void_p, ct.c_size_t,
            ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t),
        ]
        library.libdeflate_deflate_decompress_ex.restype = ct.c_int
    except (OSError, AttributeError):
        return None
    return library


def inflate_into(library, encoded: bytes, output: memoryview, name: str) -> None:
    address = ct.addressof(ct.c_ubyte.from_buffer(output))
    context = library.libdeflate_alloc_decompressor()
    if not context:
        raise MemoryError("Could not allocate a libdeflate decompressor")
    try:
        consumed, decoded = ct.c_size_t(), ct.c_size_t()
        status = library.libdeflate_deflate_decompress_ex(
            context, encoded, len(encoded), address, len(output), ct.byref(consumed), ct.byref(decoded),
        )
    finally:
        library.libdeflate_free_decompressor(context)
    if status == 3:
        raise ValueError(f"Decoded size exceeds the declared length for {name!r}")
    if status:
        raise ValueError(f"Invalid or incomplete DEFLATE stream for {name!r}")
    if consumed.value != len(encoded):
        raise ValueError(f"Trailing DEFLATE data for {name!r}")
    if decoded.value != len(output):
        raise ValueError(f"Decoded size does not match for {name!r}")


@cache
def _bcrypt_sha256():
    if sys.platform != "win32":
        return None
    try:
        fn = ct.WinDLL("bcrypt.dll", winmode=_SYSTEM32).BCryptHash
        fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_ulong, ct.c_void_p, ct.c_ulong, ct.c_void_p, ct.c_ulong]
        fn.restype = ct.c_long
    except (OSError, AttributeError):
        return None
    return fn


def sha256_hex(data: memoryview) -> str:
    fn = _bcrypt_sha256()
    if fn is not None and len(data) <= 0xFFFFFFFF:
        output = (ct.c_ubyte * 32)()
        address = np.frombuffer(data, dtype=np.uint8).ctypes.data
        if fn(_SHA256_HANDLE, None, 0, address, len(data), output, len(output)) == 0:
            return bytes(output).hex()
    return sha256(data).hexdigest()
