from dataclasses import asdict, dataclass

import numpy as np

from .safetensors_io import Checkpoint, TensorReader, read_json_object


_MARKER_LIMIT = 16 * 1024
_CHUNK_ELEMENTS = 256 * 1024
MAX_CONVROT_WIDTH = 256 * 1024
_NON_LINEAR_MODULES = {
    "emb", "embed", "embed_in", "embed_tokens", "embed_tokens_per_layer", "embedding", "embeddings",
    "expert", "experts", "moe", "shared", "token_embd", "wpe", "wte",
}


@dataclass(frozen=True)
class Family:
    weight: str
    scale: str
    marker: str
    format: str | None
    status: str
    reason: str | None
    group_size: int | None


def valid_group_size(size: int, width: int) -> bool:
    return (
        type(size) is int and 4 <= size <= width and size.bit_count() == 1
        and size.bit_length() % 2 == 1 and width % size == 0
    )


def classify_weight(checkpoint: TensorReader, weight: str, protect: tuple[str, ...] = ()) -> Family:
    if weight != "weight" and not weight.endswith(".weight"):
        raise ValueError(f"Expected a logical weight key, got {weight!r}")
    prefix = weight.removesuffix("weight")
    scale_key = prefix + "weight_scale"
    marker_key = prefix + "comfy_quant"
    format_name = None

    def family(status, reason=None, group_size=None):
        return Family(weight, scale_key, marker_key, format_name, status, reason, group_size)

    marker = checkpoint.tensors.get(marker_key)
    if marker is None:
        return family("unsupported", "No quantization marker")
    if marker.dtype != "U8" or len(marker.shape) != 1 or not 0 < marker.nbytes <= _MARKER_LIMIT:
        return family("malformed", "Expected a nonempty, one-dimensional U8 marker of at most 16 KiB")

    raw_marker = b"".join(checkpoint.chunks(marker_key))
    try:
        details = read_json_object(raw_marker)
    except (ValueError, RecursionError):
        return family("malformed", "Marker is not a valid UTF-8 JSON object")
    if not isinstance(details.get("format"), str) or not details["format"]:
        return family("malformed", "Marker has no valid format name")
    format_name = details["format"]
    if format_name != "int8_tensorwise":
        return family("unsupported", "Preserve-only quantization format")

    params = details.get("params", {})
    if not isinstance(params, dict):
        return family("malformed", "Marker params must be an object")
    convrot = details.get("convrot", params.get("convrot", False))
    if type(convrot) is not bool:
        return family("malformed", "Marker convrot must be a boolean")
    if not convrot:
        return family("unsupported", "ConvRot is disabled or unspecified")
    segments = prefix.lower().split(".")
    non_linear = any(
        segment in _NON_LINEAR_MODULES or segment.endswith(("_embedding", "_embeddings"))
        for segment in segments
    )
    if "num_experts" in details or "num_experts" in params or non_linear:
        return family("unsupported", "Embedding or expert bank")

    codes = checkpoint.tensors.get(weight)
    scales = checkpoint.tensors.get(scale_key)
    if codes is None or scales is None:
        return family("malformed", "Missing weight or weight_scale")
    if len(codes.shape) != 2 or not all(codes.shape):
        return family("unsupported", "Only nonempty 2D weights are supported")
    if codes.dtype != "I8":
        return family("malformed", "Expected signed INT8 weights")
    rows, width = codes.shape
    if width > MAX_CONVROT_WIDTH:
        return family("unsupported", "Input width exceeds the bounded optimizer's row size")
    group_size = details.get("convrot_groupsize", params.get("convrot_groupsize", 256))
    if not valid_group_size(group_size, width):
        return family("malformed", "ConvRot group size must be a power of four dividing the input width")
    if scales.dtype != "F32" or scales.shape != (rows, 1):
        return family("malformed", "Expected FP32 scales of shape [out_features, 1]")

    for raw in checkpoint.chunks(scale_key):
        chunk = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(chunk).all() or (chunk <= 0).any():
            return family("malformed", "Scales must be finite and positive")
    if weight.startswith(protect):
        return family("protected", "Matched a protected weight prefix", group_size)
    return family("eligible", group_size=group_size)


def classify_families(checkpoint: TensorReader, protect: tuple[str, ...] = ()) -> dict[str, Family]:
    families = {}
    for key in sorted(checkpoint.tensors):
        if key == "comfy_quant" or key.endswith(".comfy_quant"):
            weight = key.removesuffix("comfy_quant") + "weight"
            families[weight] = classify_weight(checkpoint, weight, protect)
    return families


def tensor_family(name: str, families: dict) -> str | None:
    module, separator, suffix = name.rpartition(".")
    weight = module + separator + "weight"
    is_member = suffix.startswith("weight_") or suffix in ("weight", "comfy_quant", "input_scale", "pre_quant_scale")
    return weight if is_member and weight in families else None


def _histogram_entropy(codes: np.ndarray) -> float:
    symbols = codes.view(np.uint8).reshape(-1)
    histogram = np.zeros(256, dtype=np.int64)
    for start in range(0, symbols.size, _CHUNK_ELEMENTS):
        histogram += np.bincount(symbols[start:start + _CHUNK_ELEMENTS], minlength=256)
    probabilities = histogram[histogram != 0] / symbols.size
    return max(0.0, float(-np.sum(probabilities * np.log2(probabilities))))


def inspect_checkpoint(checkpoint: Checkpoint, *, protect: tuple[str, ...] = ()) -> dict:
    families = classify_families(checkpoint, protect)
    by_dtype = {}
    by_format = {}
    for name, tensor in sorted(checkpoint.tensors.items()):
        family = families.get(tensor_family(name, families))
        category = (family.format or "malformed marker") if family else "unmarked"
        for sizes, key in ((by_dtype, tensor.dtype), (by_format, category)):
            total = sizes.setdefault(key, {"count": 0, "bytes": 0})
            total["count"] += 1
            total["bytes"] += tensor.nbytes

    details = []
    for family in families.values():
        item = asdict(family)
        if family.status in ("eligible", "protected"):
            item["entropy_bits_per_weight"] = _histogram_entropy(checkpoint.array(family.weight))
        details.append(item)
    return {
        "file_bytes": checkpoint.file_bytes,
        "tensor_count": len(checkpoint.tensors),
        "by_dtype": dict(sorted(by_dtype.items())),
        "by_format": dict(sorted(by_format.items())),
        "families": details,
    }
