import argparse
from io import TextIOWrapper
import json
import logging
from pathlib import Path
import sys

from safetensors import SafetensorError

from .archive import Archive, METHOD, compress_checkpoint, decompress_archive, has_compression_metadata
from .formats import inspect_checkpoint
from .safetensors_io import Checkpoint, atomic_output, validate_output
from .verify import verify_archive


_COMPRESSED_SUFFIX = ".pfqc.safetensors"
_SAFETENSORS_SUFFIX = ".safetensors"
_DEFAULT_MAX_ERROR = 0.001


def _default_output(command: str, model: Path) -> Path | None:
    name = model.name
    if command == "decompress" and name.lower().endswith(_COMPRESSED_SUFFIX):
        return model.with_name(name[:-len(_COMPRESSED_SUFFIX)] + _SAFETENSORS_SUFFIX)
    if command in ("compress", "decompress"):
        stem = name[:-len(_SAFETENSORS_SUFFIX)] if name.lower().endswith(_SAFETENSORS_SUFFIX) else name
        suffix = _COMPRESSED_SUFFIX if command == "compress" else ".restored.safetensors"
        return model.with_name(stem + suffix)
    return None


def _size(byte_count: int) -> str:
    return f"{byte_count:,} bytes ({byte_count / 10**9:.6f} GB, {byte_count / 2**30:.6f} GiB)"


def _print_inspection(report: dict) -> None:
    print(f"{report['tensor_count']:,} tensors, {_size(report['file_bytes'])}")
    for label, key in (("Dtype", "by_dtype"), ("Quantization format", "by_format")):
        print(label)
        for name, totals in report[key].items():
            print(f"  {name!r}  {totals['count']:,} tensors, {_size(totals['bytes'])}")

    eligible = sum(item["status"] == "eligible" for item in report["families"])
    print(f"{eligible:,} eligible INT8 ConvRot weights")
    for item in report["families"]:
        if item["status"] == "eligible":
            print(f"  {item['weight']!r}  {item['entropy_bits_per_weight']:.3f} bpw (histogram estimate)")
        else:
            print(f"  {item['weight']!r}  {item['status']}, {item['reason']}")


def _print_archive_report(report: dict, output: Path | None, verbose: bool) -> None:
    operation = report["operation"]
    if operation == "inspect":
        print(f"PFQC v{report['schema_version']} compressed safetensors, {_size(report['archive_bytes'])}")
        print(
            f"{report['tensor_count']:,} tensors, {report['encoded_weight_count']:,} DEFLATE weight streams, "
            f"{report['optimized_family_count']:,} fitted families"
        )
        print(f"Reconstructed checkpoint {_size(report['reconstructed_bytes'])}")
        if report["payload_bpw"] is not None:
            print(f"{report['payload_bpw']:.3f} weight payload bpw, {report['all_in_bpw']:.3f} bpw with scales")
        return
    if operation == "compress":
        print(f"Source {_size(report['source_bytes'])}")
        print(f"Archive {_size(report['archive_bytes'])}, {report['archive_fraction_of_source']:.2%} of source")
        print(f"Reconstructed checkpoint {_size(report['reconstructed_bytes'])}")
        print(
            f"{report['eligible_family_count']:,} eligible, {report['optimized_family_count']:,} optimized, "
            f"{report['protected_family_count']:,} protected, {report['skipped_family_count']:,} eligible no-ops"
        )
        print(
            f"{report['unsupported_family_count']:,} unsupported and {report['malformed_family_count']:,} malformed families, "
            f"{report['preserved_tensor_count']:,} preserved tensors"
        )
        print(
            f"Stored tensor payloads {report['baseline_tensor_bytes']:,} -> {report['stored_tensor_bytes']:,} bytes "
            f"({report['wider_scale_savings_bytes']:,} bytes saved by wider scales)"
        )
        if report["payload_bpw"] is not None:
            print(
                f"{report['payload_bpw']:.3f} weight payload bpw, {report['all_in_bpw']:.3f} bpw with scales, "
                f"{report['rate_target_misses']:,} rate-target misses"
            )
        if report["rate_target_misses"]:
            print(
                f"Target {report['settings']['target_bpw']:.3f} bpw missed for "
                f"{report['rate_target_misses']:,} of {report['eligible_family_count']:,} eligible weights. "
                "Best admissible results were written."
            )
        if report["encoder_counts"]:
            encoders = ", ".join(f"{name}: {count:,}" for name, count in sorted(report["encoder_counts"].items()))
            print(f"Fit encoders per weight ({encoders})")
        if report["max_added_relative_sse"] is not None:
            print(
                f"Added relative SSE {report['mean_added_relative_sse']:.6g} mean, "
                f"{report['max_added_relative_sse']:.6g} maximum"
            )
        if verbose:
            for fit in report["families"]:
                if "payload_bpw" not in fit:
                    continue
                before = fit["baseline_weight_bytes"] + fit["baseline_scale_bytes"]
                after = fit["stored_weight_bytes"] + fit["stored_scale_bytes"]
                delta = ""
                if fit["baseline_reference_error"] is not None:
                    delta = f", BF16 error {fit['baseline_reference_error']:.6g} -> {fit['optimized_reference_error']:.6g}"
                miss = ", target missed" if fit["rate_target_missed"] else ""
                print(
                    f"  {fit['weight']!r}  {before:,} -> {after:,} bytes, {fit['payload_bpw']:.3f} bpw, "
                    f"scale x{fit['multiplier']:g}, added SSE {fit['added_relative_sse']:.6g}{delta}{miss}"
                )
    elif operation == "decompress":
        print(f"Reconstructed {report['tensor_count']:,} tensors to {output} ({_size(report['reconstructed_bytes'])})")

    if report["checksums_verified"]:
        print(f"Verified decoded tensor checksums for {report['tensor_count']:,} tensors")
    else:
        print("Skipped decoded tensor SHA-256 checks (--skip-checksums)")
    if operation == "verify" and report["reference_quantized_compared"]:
        print(f"{report['bit_identical_preserved_tensors']:,} preserved tensors match the quantized reference byte for byte")
        for fit in report["optimized_distortion"]:
            print(f"  {fit['weight']!r}  relative SSE vs quantized source {fit['relative_sse_vs_quantized']:.6g}")
    timings = (
        f"{label} {report[key]:.2f} s"
        for key, label in (("encode_seconds", "Encode"), ("decode_seconds", "Decode"), ("decode_verify_seconds", "Verify"))
        if key in report
    )
    print(", ".join(timings))
    if report["peak_rss_bytes"] is not None:
        print(f"Peak process RSS {report['peak_rss_bytes'] / 2**20:.1f} MiB")


def _peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
                )
            ]

        counters = ProcessMemoryCounters()
        counters.size = ctypes.sizeof(counters)
        try:
            current_process = ctypes.windll.kernel32.GetCurrentProcess
            current_process.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            if get_memory(current_process(), ctypes.byref(counters), counters.size):
                return counters.peak_working_set
        except (AttributeError, OSError):
            pass
        return None
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(prog="pf-compress", description="INT8 ConvRot checkpoint storage tools")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--report", type=Path, help="Write a JSON report")
    common.add_argument("--force", action="store_true", help="Replace existing output files")
    common.add_argument("--verbose", action="store_true", help="Show detailed progress and fit metrics")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", parents=[common], help="Inspect safetensors and ConvRot eligibility")
    inspect.add_argument("model", type=Path)
    compress = commands.add_parser("compress", parents=[common], help="Compress a safetensors checkpoint")
    compress.add_argument("model", type=Path)
    compress.add_argument("--target-bpw", required=True, type=float, help="Target coded bits per eligible weight")
    compress.add_argument(
        "--max-error", default=_DEFAULT_MAX_ERROR, type=float,
        help="Per-tensor added relative squared error cap (default %(default)s)",
    )
    compress.add_argument("--method", choices=[METHOD], default=METHOD)
    compress.add_argument("--encoder", choices=["zlib", "libdeflate"], default="zlib", help="Raw DEFLATE encoder")
    compress.add_argument("--search", choices=["batch-major", "reference"], default="batch-major",
                          help="Exhaustive evaluation order. Reference search uses zlib.")
    compress.add_argument("--wide-grid", action="store_true", help="Try experimental scale multipliers up to 16x")
    compress.add_argument("--protect", action="append", default=[], metavar="PREFIX", help="Protect a logical weight prefix")
    compress.add_argument("--reference", type=Path, help="Optional original BF16 checkpoint")
    compress.add_argument("--output", type=Path, help="Defaults to <model>.pfqc.safetensors")
    decompress = commands.add_parser("decompress", parents=[common], help="Reconstruct a standard safetensors file")
    decompress.add_argument("model", type=Path, metavar="archive")
    decompress.add_argument("--output", type=Path, help="Defaults to the checkpoint name without .pfqc")
    decompress.add_argument("--skip-checksums", action="store_true", help="Skip decoded tensor SHA-256 checks")
    verify = commands.add_parser("verify", parents=[common], help="Verify archive structure and reconstructed bytes")
    verify.add_argument("model", type=Path, metavar="archive")
    verify.add_argument("--reference-quantized", type=Path, help="Compare against the original quantized checkpoint")
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        output = getattr(args, "output", None) or _default_output(args.command, args.model)
        report_inputs = tuple(path for path in (
            args.model, output, getattr(args, "reference", None), getattr(args, "reference_quantized", None),
        ) if path is not None)
        if args.report is not None:
            validate_output(args.report, force=args.force, inputs=report_inputs)
        if args.command == "inspect":
            checkpoint = Checkpoint(args.model)
            if has_compression_metadata(checkpoint):
                packed = Archive(checkpoint)
                packed.validate_families()
                report = packed.report()
                report["operation"] = "inspect"
            else:
                report = inspect_checkpoint(checkpoint)
        elif args.command == "compress":
            report = compress_checkpoint(
                args.model, output, target_bpw=args.target_bpw, max_error=args.max_error,
                protect=tuple(args.protect), reference=args.reference, force=args.force,
                encoder=args.encoder, search=args.search, wide_grid=args.wide_grid,
            )
        elif args.command == "decompress":
            report = decompress_archive(
                args.model, output, force=args.force, verify_checksums=not args.skip_checksums,
            )
        else:
            report = verify_archive(args.model, reference_quantized=args.reference_quantized)

        report["peak_rss_bytes"] = _peak_rss_bytes()
        if args.report is not None:
            with atomic_output(args.report, force=args.force, inputs=report_inputs) as temporary:
                encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                temporary.write_text(encoded + "\n", encoding="utf-8", newline="\n")
    except (OSError, SafetensorError, ValueError, RecursionError, OverflowError) as error:
        print(f"pf-compress: error: {error}", file=sys.stderr)
        return 1

    if args.command == "inspect" and "operation" not in report:
        _print_inspection(report)
    else:
        _print_archive_report(report, output, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
