import json

import numpy as np
from safetensors import TensorSpec, serialize_file


CONVROT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}
H4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2


def bf16_bits(exact_values):
    return (np.asarray(exact_values, dtype="<f4").view("<u4") >> 16).astype("<u2")


def marker(value):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
    return np.frombuffer(raw, dtype=np.uint8)


def family(module, details=CONVROT):
    return {
        f"{module}.weight": np.arange(-8, 8, dtype=np.int8).reshape(2, 8),
        f"{module}.weight_scale": np.array([[0.01], [0.02]], dtype="<f4"),
        f"{module}.comfy_quant": marker(details),
    }


def save_fixture(path, tensors, *, bf16=(), metadata=None):
    specs = {
        name: TensorSpec(
            dtype="bfloat16" if name in bf16 else array.dtype.name,
            shape=array.shape,
            data_ptr=array.ctypes.data,
            data_len=array.nbytes,
        )
        for name, array in tensors.items()
    }
    serialize_file(specs, path, metadata)


def rewrite_manifest(source, destination, update):
    data = source.read_bytes()
    end = 8 + int.from_bytes(data[:8], "little")
    header = json.loads(data[8:end])
    metadata = header["__metadata__"]
    key = "potatoforge_quant_compress.manifest"
    manifest = json.loads(metadata[key])
    update(manifest)
    metadata[key] = json.dumps(manifest)
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    destination.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data[end:])


def mixed_checkpoint(path):
    tensors = {
        **family("blocks.0.proj"),
        **family("blocks.18.proj", b'{ "convrot_groupsize": 4, "format": "int8_tensorwise", "convrot": true }'),
        **family("packed", {**CONVROT, "format": "convrot_w4a4"}),
        **family("w6", {**CONVROT, "format": "w6a8_int8"}),
        **family("plain", {"format": "int8_tensorwise"}),
        **family("unknown", {**CONVROT, "format": "custom_format"}),
        **family("broken", b'{"format": "int8_tensorwise",'),
        **family("wrong_shape"),
        **family("embed_tokens"),
        **family("bank", {**CONVROT, "num_experts": 2}),
        "unrelated.weight": np.arange(16, dtype=np.int8).reshape(2, 8),
        "ordinary.weight": np.array([0x3F80, 0x7FC1], dtype="<u2"),
        "ordinary.bias": np.array([0x80000000, 0x7FC00123], dtype="<u4").view("<f4"),
        "extra_bytes": np.array([0, 128, 255], dtype=np.uint8),
    }
    tensors["w6.weight"] = tensors["w6.weight"].view(np.uint8)
    tensors["wrong_shape.weight_scale"] = np.ones((1, 2), dtype="<f4")
    save_fixture(path, tensors, bf16=("ordinary.weight",), metadata={"origin": "synthetic ✓"})
    return tensors
