"""Windows SHA-256 acceleration with a standard library fallback."""

import ctypes as ct
from functools import cache
from hashlib import sha256
import sys

import numpy as np


_SYSTEM32 = 0x800
_SHA256_HANDLE = 0x41


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
