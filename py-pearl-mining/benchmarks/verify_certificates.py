#!/usr/bin/env python3
"""Offline verifier for certificates emitted by bench_prover.py."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import unittest
import json
import os
import platform
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

from bench_prover import MetadataHelperTests, cgroup_limits, module_metadata


SCHEMA = "pearl-prover-certificate-v1"
U32_MAX = (1 << 32) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def rss_metrics() -> dict[str, int | None]:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(usage if sys.platform == "darwin" else usage * 1024)
    current_bytes: int | None = None
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        current_bytes = pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        pass
    return {"ru_maxrss_bytes": peak_bytes, "current_rss_bytes": current_bytes}


def require_u32(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= U32_MAX:
        raise ValueError(f"{field} must be an integer in [0, 2^32-1]")
    return value


def parse_header(pm: Any, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError("header must be a JSON object")
    version = require_u32(data.get("version"), "header.version")
    timestamp = require_u32(data.get("timestamp"), "header.timestamp")
    nbits = require_u32(data.get("nbits"), "header.nbits")
    hashes: list[bytes] = []
    for field in ("prev_block_hex", "merkle_root_hex"):
        value = data.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(f"header.{field} must contain exactly 64 hexadecimal digits")
        hashes.append(bytes.fromhex(value))
    return pm.IncompleteBlockHeader(version, hashes[0], hashes[1], timestamp, nbits)


def parse_shape(data: Any) -> dict[str, int]:
    if not isinstance(data, dict):
        raise ValueError("shape must be a JSON object")
    required = ("m", "n", "k", "rank", "rows", "cols")
    shape: dict[str, int] = {}
    for key in required:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"shape.{key} must be a positive integer")
        shape[key] = value
    return shape


def decode_payload(data: Any, field: str) -> bytes:
    encoded = data.get(field)
    if not isinstance(encoded, str):
        raise ValueError(f"{field} must be a base64 string")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f"{field} is not valid base64") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError(f"{field} is not canonical base64")
    if not raw:
        raise ValueError(f"{field} must not be empty")
    return raw


def validate_record(data: Any) -> tuple[Any, Any, bytes, bytes, str, dict[str, int], int]:
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"record schema must be {SCHEMA!r}")
    cert_version = data.get("cert_version")
    if isinstance(cert_version, bool) or not isinstance(cert_version, int) or cert_version not in (1, 2, 3):
        raise ValueError("cert_version must be 1, 2, or 3")
    expected_digest = data.get("certificate_sha256")
    if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(expected_digest):
        raise ValueError("certificate_sha256 must be a lowercase SHA-256 hex digest")
    shape = parse_shape(data.get("shape"))
    public_data = decode_payload(data, "public_data_b64")
    proof_data = decode_payload(data, "proof_data_b64")
    actual_digest = hashlib.sha256(public_data + proof_data).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise ValueError("certificate_sha256 does not match decoded public_data and proof_data")
    return data.get("header"), cert_version, public_data, proof_data, actual_digest, shape, len(public_data) + len(proof_data)


def timed_verify(call: Callable[[], tuple[bool, str]]) -> tuple[bool | None, str, bool, dict[str, float]]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        valid, message = call()
        return bool(valid), str(message), False, {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
        }
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", True, {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-metadata", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("certificates", type=Path, nargs="?", help="JSONL from bench_prover.py --output-fixture")
    parser.add_argument("--limit", type=int, help="verify at most this many certificate records")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.self_test_metadata:
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(MetadataHelperTests)
            )
            return 0 if result.wasSuccessful() else 1
        if args.certificates is None:
            raise ValueError("a certificate JSONL path is required")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be at least 1")
        import pearl_mining as pm

        module, package_wrapper = module_metadata(pm)
        run_metadata = {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "module": module,
            "package_wrapper": package_wrapper,
            "build_tag": os.environ.get("PEARL_BUILD_TAG") or os.environ.get("BUILD_TAG"),
            "cgroup_limits": cgroup_limits(),
        }

        checked = 0
        failed = 0
        with args.certificates.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if args.limit is not None and checked >= args.limit:
                    break
                try:
                    record = json.loads(line)
                    header_data, cert_version, public_data, proof_data, digest, shape, proof_size = validate_record(record)
                    header = parse_header(pm, header_data)
                    proof = pm.ZKProof(public_data, proof_data)
                except Exception as exc:
                    raise ValueError(f"line {line_number}: {type(exc).__name__}: {exc}") from exc

                positive_valid, positive_message, positive_error, positive_timing = timed_verify(
                    lambda: pm.verify_proof_for_cert_version(cert_version, header, proof)
                )
                positive_passed = positive_valid is True and not positive_error

                altered_header = pm.IncompleteBlockHeader(
                    int(header.version),
                    bytes(header.prev_block),
                    bytes(header.merkle_root),
                    (int(header.timestamp) + 1) & U32_MAX,
                    int(header.nbits),
                )
                negative_valid, negative_message, negative_error, negative_timing = timed_verify(
                    lambda: pm.verify_proof_for_cert_version(cert_version, altered_header, proof)
                )
                altered_rejected = negative_error or negative_valid is False
                negative_passed = bool(altered_rejected)
                passed = positive_passed and negative_passed
                failed += int(not passed)
                checked += 1

                output = {
                    "type": "verification_result",
                    "source_file": str(args.certificates.resolve()),
                    "source_line": line_number,
                    "certificate_sha256": digest,
                    "cert_version": cert_version,
                    "shape": shape,
                    "proof_size_bytes": proof_size,
                    "positive_check": {
                        "passed": positive_passed,
                        "verified": positive_valid is True,
                        "message": positive_message,
                        "timing": positive_timing,
                    },
                    "altered_header_negative_check": {
                        "passed": negative_passed,
                        "rejected": bool(altered_rejected),
                        "message": negative_message,
                        "timing": negative_timing,
                    },
                    "memory": rss_metrics(),
                    **run_metadata,
                    "all_checks_passed": passed,
                }
                print(json.dumps(output, separators=(",", ":")), flush=True)

        if checked == 0:
            raise ValueError("no certificate records were found")
        summary = {
            "type": "summary",
            "certificates_checked": checked,
            "failures": failed,
            "all_checks_passed": failed == 0,
            "module": module,
        }
        print(json.dumps(summary, separators=(",", ":")), flush=True)
        return 0 if failed == 0 else 1
    except Exception as exc:
        print(json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"}, separators=(",", ":")), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
