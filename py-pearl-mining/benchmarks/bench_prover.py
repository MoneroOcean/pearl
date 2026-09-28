#!/usr/bin/env python3
"""Offline Pearl ZK v3 prover benchmark. Emits compact JSON Lines on stdout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.machinery
import json
import os
import platform
import posixpath
import re
import resource
import sys
import tempfile
import time
import unittest
from types import ModuleType
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest.mock import patch


CERT_VERSION = 3
DEFAULT_K = 4096
DEFAULT_ROWS = 16
DEFAULT_COLS = 16
DEFAULT_RANK = 256
DEFAULT_REPETITIONS = 3
DEFAULT_TIMESTAMP = 0x66666666
DEFAULT_PREV_BLOCK = bytes(32)
DEFAULT_MERKLE_ROOT = bytes([1]) * 32
UINT256_MAX = (1 << 256) - 1


def parse_int(value: str) -> int:
    """Parse decimal or 0x-prefixed integers for command-line values."""
    return int(value, 0)


def compact_target(nbits: int) -> int:
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x00FFFFFF
    if exponent == 0 or mantissa == 0 or (mantissa & 0x00800000):
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def encode_compact_target(target: int) -> int:
    """Encode a positive target using the same compact convention as nbits."""
    if target <= 0:
        raise ValueError("cannot encode an empty difficulty target")
    exponent = (target.bit_length() + 7) // 8
    if exponent <= 3:
        mantissa = target << (8 * (3 - exponent))
    else:
        mantissa = target >> (8 * (exponent - 3))
    if mantissa & 0x00800000:
        mantissa >>= 8
        exponent += 1
    if exponent > 32:
        raise ValueError("target cannot be represented as a 256-bit compact target")
    return (exponent << 24) | (mantissa & 0x007FFFFF)


def make_header(pm: Any, data: dict[str, Any]) -> Any:
    try:
        prev = bytes.fromhex(data["prev_block_hex"])
        merkle = bytes.fromhex(data["merkle_root_hex"])
        version = int(data["version"])
        timestamp = int(data["timestamp"])
        nbits = int(data["nbits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid header fixture fields: {exc}") from exc
    if len(prev) != 32 or len(merkle) != 32:
        raise ValueError("header hashes must each be exactly 32 bytes")
    for label, value in (("version", version), ("timestamp", timestamp), ("nbits", nbits)):
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError(f"header {label} must fit in u32")
    return pm.IncompleteBlockHeader(version, prev, merkle, timestamp, nbits)


def header_json(header: Any) -> dict[str, Any]:
    return {
        "version": int(header.version),
        "prev_block_hex": bytes(header.prev_block).hex(),
        "merkle_root_hex": bytes(header.merkle_root).hex(),
        "timestamp": int(header.timestamp),
        "nbits": int(header.nbits),
    }


def shape_json(shape: dict[str, int]) -> dict[str, int]:
    return {key: int(value) for key, value in shape.items()}


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


def parse_cgroup_v2_path(contents: str) -> str | None:
    """Return the unified cgroup path from /proc/self/cgroup, if present."""
    for line in contents.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            path = posixpath.normpath(fields[2])
            return path if path.startswith("/") else None
    return None


def parse_cgroup2_mountinfo(contents: str) -> list[tuple[str, str]]:
    """Return (mount root, mount point) pairs for visible cgroup2 mounts."""
    mounts = []
    for line in contents.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left, right = before.split(), after.split()
        if len(left) < 5 or not right or right[0] != "cgroup2":
            continue
        root = _unescape_mountinfo_path(left[3])
        mount_point = _unescape_mountinfo_path(left[4])
        if root.startswith("/") and mount_point.startswith("/"):
            mounts.append((posixpath.normpath(root), posixpath.normpath(mount_point)))
    return mounts


def _unescape_mountinfo_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def map_cgroup_v2_path(
    process_path: str, mount_root: str, mount_point: str
) -> tuple[str, str]:
    """Map a proc cgroup path into a cgroup2 mount, including namespace-relative paths."""
    process_path = posixpath.normpath(process_path)
    mount_root = posixpath.normpath(mount_root)
    mount_point = posixpath.normpath(mount_point)
    if not process_path.startswith("/") or not mount_root.startswith("/") or not mount_point.startswith("/"):
        raise ValueError("cgroup paths and mount points must be absolute")

    root_prefix = mount_root.rstrip("/") + "/"
    if process_path == mount_root:
        relative = "."
    elif mount_root == "/" or process_path.startswith(root_prefix):
        relative = posixpath.relpath(process_path, mount_root)
    else:
        # In a cgroup namespace, proc reports paths relative to the namespace root,
        # while mountinfo can still expose the corresponding host-side mount root.
        relative = process_path.lstrip("/") or "."
        mapped = mount_point if relative == "." else posixpath.join(mount_point, relative)
        return posixpath.normpath(mapped), "namespace_relative"
    mapped = mount_point if relative == "." else posixpath.join(mount_point, relative)
    return posixpath.normpath(mapped), "mount_root_relative"


def find_cgroup_v2_process_mount(
    cgroup_contents: str,
    mountinfo_contents: str,
    *,
    is_dir: Callable[[str], bool] = os.path.isdir,
) -> dict[str, str] | None:
    process_path = parse_cgroup_v2_path(cgroup_contents)
    if process_path is None:
        return None
    candidates: list[tuple[int, int, str, str, str, str]] = []
    for mount_root, mount_point in parse_cgroup2_mountinfo(mountinfo_contents):
        mapped, mode = map_cgroup_v2_path(process_path, mount_root, mount_point)
        # Prefer the unambiguous mount-root mapping; the fallback handles a private
        # cgroup namespace whose path is relative to a mount rooted elsewhere.
        priority = 0 if mode == "mount_root_relative" else 1
        candidates.append((priority, -len(mount_root), mapped, mount_root, mount_point, mode))
    for _, _, mapped, mount_root, mount_point, mode in sorted(candidates):
        if is_dir(mapped):
            return {
                "process_path": process_path,
                "process_directory": mapped,
                "mount_root": mount_root,
                "mount_point": mount_point,
                "mapping": mode,
            }
    return None


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return None


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _ancestor_cgroup_v2_limits(
    process_directory: str,
    mount_point: str,
    *,
    read_text: Callable[[Path], str | None] = _read_optional_text,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cpu = {"source": None, "raw": None, "quota_cores": None, "ancestor_limits": []}
    memory = {"source": None, "raw": None, "limit_bytes": None, "ancestor_limits": []}
    current, mount = Path(process_directory), Path(mount_point)
    if not _within(current, mount):
        return cpu, memory

    while True:
        cpu_raw = read_text(current / "cpu.max")
        if cpu_raw is not None:
            quota_cores = None
            try:
                quota, period = cpu_raw.split()[:2]
                if quota != "max" and int(quota) >= 0 and int(period) > 0:
                    quota_cores = int(quota) / int(period)
            except (ValueError, IndexError):
                pass
            cpu["ancestor_limits"].append(
                {"source": str(current / "cpu.max"), "raw": cpu_raw, "quota_cores": quota_cores}
            )

        memory_raw = read_text(current / "memory.max")
        if memory_raw is not None:
            limit_bytes = None
            try:
                if memory_raw != "max" and int(memory_raw) >= 0:
                    limit_bytes = int(memory_raw)
            except ValueError:
                pass
            memory["ancestor_limits"].append(
                {"source": str(current / "memory.max"), "raw": memory_raw, "limit_bytes": limit_bytes}
            )

        if current == mount:
            break
        parent = current.parent
        if parent == current or not _within(parent, mount):
            break
        current = parent

    finite_cpu = [row for row in cpu["ancestor_limits"] if row["quota_cores"] is not None]
    if finite_cpu:
        winner = min(finite_cpu, key=lambda row: row["quota_cores"])
        cpu.update(source=winner["source"], raw=winner["raw"], quota_cores=winner["quota_cores"])
    elif cpu["ancestor_limits"]:
        nearest = cpu["ancestor_limits"][0]
        cpu.update(source=nearest["source"], raw=nearest["raw"])

    finite_memory = [row for row in memory["ancestor_limits"] if row["limit_bytes"] is not None]
    if finite_memory:
        winner = min(finite_memory, key=lambda row: row["limit_bytes"])
        memory.update(source=winner["source"], raw=winner["raw"], limit_bytes=winner["limit_bytes"])
    elif memory["ancestor_limits"]:
        nearest = memory["ancestor_limits"][0]
        memory.update(source=nearest["source"], raw=nearest["raw"])
    return cpu, memory


def cgroup_limits() -> dict[str, Any]:
    """Report effective cgroup v2 ancestor limits, labeling any local-only fallback."""
    try:
        cgroup_contents = Path("/proc/self/cgroup").read_text(encoding="ascii")
        mountinfo_contents = Path("/proc/self/mountinfo").read_text(encoding="ascii")
    except OSError:
        cgroup_contents = mountinfo_contents = ""

    mount = find_cgroup_v2_process_mount(cgroup_contents, mountinfo_contents)
    if mount is not None:
        cpu, memory = _ancestor_cgroup_v2_limits(mount["process_directory"], mount["mount_point"])
        return {
            "version": 2,
            "limit_scope": "effective_process_cgroup_and_visible_ancestors",
            "process_path": mount["process_path"],
            "process_directory": mount["process_directory"],
            "mount_root": mount["mount_root"],
            "mount_point": mount["mount_point"],
            "mapping": mount["mapping"],
            "cpu": cpu,
            "memory": memory,
        }

    # Compatibility fallback for v1 and restricted proc mounts. These values may
    # describe only the visible mount root, so never present them as effective.
    cpu: dict[str, Any] = {"source": None, "raw": None, "quota_cores": None}
    cpu_v2_path = Path("/sys/fs/cgroup/cpu.max")
    raw = _read_optional_text(cpu_v2_path)
    if raw is not None:
        cpu.update(source=str(cpu_v2_path), raw=raw)
        try:
            quota, period = raw.split()[:2]
            if quota != "max" and int(period) > 0:
                cpu["quota_cores"] = int(quota) / int(period)
        except (ValueError, IndexError):
            pass
    else:
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        try:
            quota = int(_read_optional_text(quota_path) or "")
            period = int(_read_optional_text(period_path) or "")
            cpu.update(
                source=f"{quota_path},{period_path}",
                raw=f"{quota} {period}",
                quota_cores=(quota / period if quota >= 0 and period > 0 else None),
            )
        except ValueError:
            pass

    memory: dict[str, Any] = {"source": None, "raw": None, "limit_bytes": None}
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        raw = _read_optional_text(path)
        if raw is None:
            continue
        memory.update(source=str(path), raw=raw)
        try:
            value = int(raw)
            if 0 <= value < (1 << 60):
                memory["limit_bytes"] = value
        except ValueError:
            pass
        break
    return {
        "version": None,
        "limit_scope": "local_mount_only_unmapped",
        "process_path": parse_cgroup_v2_path(cgroup_contents),
        "mapping": None,
        "cpu": cpu,
        "memory": memory,
    }


def _module_origin(module: ModuleType) -> Path | None:
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "frozen"}:
        return Path(origin).resolve()
    module_file = getattr(module, "__file__", None)
    return Path(module_file).resolve() if module_file else None


def find_loaded_extension_module(
    package_module: ModuleType,
    modules: Mapping[str, ModuleType] | None = None,
) -> tuple[str, Path] | None:
    """Find the already-imported native extension backing a package, without importing code."""
    package_name = getattr(package_module, "__name__", "")
    loaded = sys.modules if modules is None else modules
    direct_path = _module_origin(package_module)
    if direct_path is not None and direct_path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
        return package_name, direct_path

    package_attributes = vars(package_module)
    candidates: list[tuple[int, int, str, Path]] = []
    for name, module in loaded.items():
        path = _module_origin(module)
        if path is None or not path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
            continue
        direct_reference = int(any(value is module for value in package_attributes.values()))
        symbol_references = 0
        for value in package_attributes.values():
            defining_module = getattr(value, "__module__", None)
            if isinstance(defining_module, str) and (
                defining_module == name or defining_module.startswith(name + ".")
            ):
                symbol_references += 1
        in_package_namespace = name == package_name or name.startswith(package_name + ".")
        if not in_package_namespace and not direct_reference and not symbol_references:
            continue
        candidates.append((direct_reference, symbol_references, name, path))

    if not candidates:
        return None
    best_score = max((candidate[0], candidate[1]) for candidate in candidates)
    winners = [candidate for candidate in candidates if candidate[:2] == best_score]
    unique_paths = {(candidate[2], candidate[3]) for candidate in winners}
    if len(unique_paths) != 1:
        return None
    name, path = next(iter(unique_paths))
    return name, path


def module_metadata(package_module: ModuleType) -> tuple[dict[str, Any], dict[str, Any] | None]:
    extension = find_loaded_extension_module(package_module)
    if extension is None:
        raise RuntimeError("could not identify the already-loaded pearl_mining native extension")
    extension_name, extension_path = extension
    package_path = _module_origin(package_module)
    native_info = {
        "name": extension_name,
        "path": str(extension_path),
        "sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        "version": str(getattr(package_module, "__version__", "unknown")),
    }
    package_info = None
    if package_path is not None and package_path != extension_path:
        package_info = {
            "path": str(package_path),
            "sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        }
    return native_info, package_info


class MetadataHelperTests(unittest.TestCase):
    def test_timed_call_records_fault_deltas(self) -> None:
        marker = object()
        result, metrics = timed_call(lambda: marker)
        self.assertIs(result, marker)
        for key in ("wall_seconds", "cpu_seconds", "minor_page_faults", "major_page_faults"):
            self.assertGreaterEqual(metrics[key], 0)

    def test_cgroup_mount_root_mapping_and_escaped_mountpoint(self) -> None:
        cgroup = "12:cpu:/legacy\n0::/tenant/job\n"
        mountinfo = "29 23 0:26 /tenant /sys/fs/cgroup/job\\040root rw - cgroup2 cgroup rw\n"
        self.assertEqual(parse_cgroup_v2_path(cgroup), "/tenant/job")
        self.assertEqual(parse_cgroup2_mountinfo(mountinfo), [("/tenant", "/sys/fs/cgroup/job root")])
        self.assertEqual(
            map_cgroup_v2_path("/tenant/job", "/tenant", "/sys/fs/cgroup/job root"),
            ("/sys/fs/cgroup/job root/job", "mount_root_relative"),
        )

    def test_cgroup_namespace_relative_mapping(self) -> None:
        self.assertEqual(
            map_cgroup_v2_path("/worker", "/host/container", "/sys/fs/cgroup"),
            ("/sys/fs/cgroup/worker", "namespace_relative"),
        )

    def test_effective_limits_include_visible_ancestors(self) -> None:
        files = {
            "/cg/a/b/cpu.max": "max 100000",
            "/cg/a/b/memory.max": "9000",
            "/cg/a/cpu.max": "200000 100000",
            "/cg/a/memory.max": "8000",
            "/cg/cpu.max": "400000 100000",
            "/cg/memory.max": "max",
        }
        cpu, memory = _ancestor_cgroup_v2_limits(
            "/cg/a/b", "/cg", read_text=lambda path: files.get(str(path))
        )
        self.assertEqual(cpu["quota_cores"], 2.0)
        self.assertEqual(cpu["source"], "/cg/a/cpu.max")
        self.assertEqual(memory["limit_bytes"], 8000)
        self.assertEqual(memory["source"], "/cg/a/memory.max")

    def test_extension_discovery_uses_loaded_package_extension(self) -> None:
        package = ModuleType("pearl_mining")
        package.__file__ = "/fake/pearl_mining/__init__.py"
        extension_name = "pearl_mining._native"
        extension = ModuleType(extension_name)
        extension.__file__ = "/fake/pearl_mining/_native.abi3.so"
        package._native = extension
        package.IncompleteBlockHeader = type(
            "IncompleteBlockHeader", (), {"__module__": extension_name}
        )
        unrelated = ModuleType("other.native")
        unrelated.__file__ = "/fake/other/native.so"
        self.assertEqual(
            find_loaded_extension_module(package, {extension_name: extension, "other.native": unrelated}),
            (extension_name, Path(extension.__file__).resolve()),
        )

    def test_module_metadata_hashes_extension_and_reports_wrapper_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_path = Path(temporary_directory) / "pearl_mining" / "__init__.py"
            extension_path = Path(temporary_directory) / "native.abi3.so"
            package_path.parent.mkdir()
            package_path.write_bytes(b"package wrapper")
            extension_path.write_bytes(b"native extension")

            package = ModuleType("pearl_mining")
            package.__file__ = str(package_path)
            extension_name = "pearl_mining._native_for_test"
            extension = ModuleType(extension_name)
            extension.__file__ = str(extension_path)
            package._native = extension
            package.IncompleteBlockHeader = type(
                "IncompleteBlockHeader", (), {"__module__": extension_name}
            )
            with patch.dict(sys.modules, {extension_name: extension}):
                native, wrapper = module_metadata(package)

            self.assertEqual(native["path"], str(extension_path.resolve()))
            self.assertEqual(native["sha256"], hashlib.sha256(b"native extension").hexdigest())
            self.assertEqual(wrapper["path"], str(package_path.resolve()))
            self.assertEqual(wrapper["sha256"], hashlib.sha256(b"package wrapper").hexdigest())


def cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def timed_call(call: Any) -> tuple[Any, dict[str, float]]:
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    value = call()
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    return value, {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "minor_page_faults": usage_end.ru_minflt - usage_start.ru_minflt,
        "major_page_faults": usage_end.ru_majflt - usage_start.ru_majflt,
    }


def read_json_fixture(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"input fixture is not valid JSON: {exc}") from exc
    if data.get("schema") != "pearl-prover-input-v1":
        raise ValueError("input fixture schema must be pearl-prover-input-v1")
    for field in ("header", "shape", "plain_proof_b64"):
        if field not in data:
            raise ValueError(f"input fixture is missing {field!r}")
    return raw, data


def positive_shape(data: dict[str, Any]) -> dict[str, int]:
    required = ("m", "n", "k", "rank", "rows", "cols")
    try:
        shape = {key: int(data[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid shape metadata: {exc}") from exc
    if any(value <= 0 for value in shape.values()):
        raise ValueError("all shape values must be positive")
    return shape


def assert_shape_options_match(args: argparse.Namespace, shape: dict[str, int]) -> None:
    for key in ("m", "n", "k", "rank", "rows", "cols"):
        supplied = getattr(args, key)
        if supplied is not None and supplied != shape[key]:
            raise ValueError(f"--{key}={supplied} conflicts with fixture value {shape[key]}")


def fixture_header_args(args: argparse.Namespace) -> dict[str, int | str] | None:
    keys = {
        "version": args.header_version,
        "prev_block_hex": args.header_prev_block_hex,
        "merkle_root_hex": args.header_merkle_root_hex,
        "timestamp": args.header_timestamp,
        "nbits": args.nbits,
    }
    if all(value is None for value in keys.values()):
        return None
    missing = [key for key, value in keys.items() if value is None]
    if missing:
        raise ValueError("header overrides must be complete; missing " + ", ".join(missing))
    return keys


def verify_negative_header(pm: Any, header: Any, certificate: Any) -> tuple[bool, str, dict[str, float]]:
    altered = pm.IncompleteBlockHeader(
        int(header.version),
        bytes(header.prev_block),
        bytes(header.merkle_root),
        (int(header.timestamp) + 1) & 0xFFFFFFFF,
        int(header.nbits),
    )
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        valid, message = pm.verify_proof_for_cert_version(CERT_VERSION, altered, certificate)
        rejected = not bool(valid)
        result_message = str(message)
    except Exception as exc:  # A rejected deserialization is also a valid negative result.
        rejected = True
        result_message = f"rejected with {type(exc).__name__}"
    timing = {
        "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
    }
    return rejected, result_message, timing


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test-metadata", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--k", type=parse_int, default=None, help=f"common dimension (default {DEFAULT_K})")
    parser.add_argument("--rows", type=parse_int, default=None, help=f"row pattern size (default {DEFAULT_ROWS})")
    parser.add_argument("--cols", type=parse_int, default=None, help=f"column pattern size (default {DEFAULT_COLS})")
    parser.add_argument("--rank", type=parse_int, default=None, help=f"noise rank (default {DEFAULT_RANK})")
    parser.add_argument("--m", type=parse_int, default=None, help="matrix row count; defaults to --rows")
    parser.add_argument("--n", type=parse_int, default=None, help="matrix column count; defaults to --cols")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--limit", type=int, help="cap repetitions for a bounded run")
    parser.add_argument("--input-fixture", type=Path, help="JSON fixture with a fixed header, shape, and plain proof")
    parser.add_argument("--raw-base64-fixture", type=Path, help="raw PlainProof base64; requires explicit header metadata")
    parser.add_argument("--output-fixture", type=Path, help="write fresh certificates as JSON Lines for offline verification")
    parser.add_argument("--nbits", type=parse_int, default=None, help="compact target override, decimal or 0x-prefixed")
    parser.add_argument("--header-version", type=parse_int)
    parser.add_argument("--header-prev-block-hex")
    parser.add_argument("--header-merkle-root-hex")
    parser.add_argument("--header-timestamp", type=parse_int)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.self_test_metadata:
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(MetadataHelperTests)
            )
            return 0 if result.wasSuccessful() else 1
        if args.repetitions < 1:
            raise ValueError("--repetitions must be at least 1")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be at least 1")
        repetitions = min(args.repetitions, args.limit) if args.limit is not None else args.repetitions
        if args.input_fixture and args.raw_base64_fixture:
            raise ValueError("choose either --input-fixture or --raw-base64-fixture")
        if args.output_fixture and args.output_fixture.resolve() in {
            path.resolve() for path in (args.input_fixture, args.raw_base64_fixture) if path is not None
        }:
            raise ValueError("--output-fixture must not overwrite an input fixture")

        import pearl_mining as pm

        module_info, package_wrapper_info = module_metadata(pm)
        source = "mined"
        input_path_sha256: str | None = None
        fixed_plain: Any | None = None
        fixed_header: Any | None = None
        fixed_shape: dict[str, int] | None = None

        if args.input_fixture:
            raw_fixture, data = read_json_fixture(args.input_fixture)
            input_path_sha256 = hashlib.sha256(raw_fixture).hexdigest()
            fixed_header = make_header(pm, data["header"])
            fixed_shape = positive_shape(data["shape"])
            assert_shape_options_match(args, fixed_shape)
            override = fixture_header_args(args)
            if override is not None and header_json(make_header(pm, override)) != header_json(fixed_header):
                raise ValueError("header command-line values conflict with input fixture")
            fixed_plain = pm.PlainProof.from_base64("".join(str(data["plain_proof_b64"]).split()))
            source = "json_fixture"
            if (
                int(fixed_plain.m) != fixed_shape["m"]
                or int(fixed_plain.n) != fixed_shape["n"]
                or int(fixed_plain.k) != fixed_shape["k"]
                or int(fixed_plain.noise_rank) != fixed_shape["rank"]
            ):
                raise ValueError("plain proof dimensions do not match fixture shape metadata")
        elif args.raw_base64_fixture:
            raw_fixture = args.raw_base64_fixture.read_bytes()
            input_path_sha256 = hashlib.sha256(raw_fixture).hexdigest()
            header_fields = fixture_header_args(args)
            if header_fields is None:
                raise ValueError("--raw-base64-fixture requires --header-version, --header-prev-block-hex, --header-merkle-root-hex, --header-timestamp, and --nbits")
            fixed_header = make_header(pm, header_fields)
            encoded = "".join(raw_fixture.decode("ascii").split())
            fixed_plain = pm.PlainProof.from_base64(encoded)
            fixed_shape = {
                "m": int(fixed_plain.m),
                "n": int(fixed_plain.n),
                "k": int(fixed_plain.k),
                "rank": int(fixed_plain.noise_rank),
                "rows": int(args.rows if args.rows is not None else DEFAULT_ROWS),
                "cols": int(args.cols if args.cols is not None else DEFAULT_COLS),
            }
            assert_shape_options_match(args, fixed_shape)
            source = "raw_base64_fixture"

        if fixed_shape is None:
            shape = {
                "m": args.m if args.m is not None else (args.rows if args.rows is not None else DEFAULT_ROWS),
                "n": args.n if args.n is not None else (args.cols if args.cols is not None else DEFAULT_COLS),
                "k": args.k if args.k is not None else DEFAULT_K,
                "rank": args.rank if args.rank is not None else DEFAULT_RANK,
                "rows": args.rows if args.rows is not None else DEFAULT_ROWS,
                "cols": args.cols if args.cols is not None else DEFAULT_COLS,
            }
            shape = positive_shape(shape)
        else:
            shape = fixed_shape

        if fixed_header is None:
            factor = shape["rows"] * shape["cols"] * shape["k"]
            if factor <= 0 or factor > UINT256_MAX:
                raise ValueError("tile difficulty adjustment factor is outside the U256 range")
            nbits = args.nbits if args.nbits is not None else encode_compact_target(UINT256_MAX // factor)
            if not 0 <= nbits <= 0xFFFFFFFF:
                raise ValueError("--nbits must fit in u32")
            base_target = compact_target(nbits)
            if base_target == 0 or base_target * factor > UINT256_MAX:
                raise ValueError("nbits gives an empty or overflowing scaled difficulty target")
            fixed_header = pm.IncompleteBlockHeader(
                1,
                DEFAULT_PREV_BLOCK,
                DEFAULT_MERKLE_ROOT,
                DEFAULT_TIMESTAMP,
                nbits,
            )
        else:
            factor = shape["rows"] * shape["cols"] * shape["k"]

        if shape["m"] < shape["rows"] or shape["n"] < shape["cols"]:
            raise ValueError("m and n must be at least the corresponding pattern dimensions")
        if shape["rows"] * shape["cols"] < 32 or shape["rows"] * shape["cols"] > 256:
            raise ValueError("row pattern size times column pattern size must be in [32, 256]")
        if (
            shape["k"] < 1024
            or shape["k"] > 65536
            or shape["k"] % 64
            or shape["k"] < 16 * shape["rank"]
            or shape["k"] > 4 * shape["rank"] * shape["rank"]
        ):
            raise ValueError("sanity checks require 1024 <= k <= 65536, k divisible by 64, and 16*rank <= k <= 4*rank^2")
        if shape["rank"] < 32 or shape["rank"] > 1024 or shape["rank"] & (shape["rank"] - 1):
            raise ValueError("rank must be a power of two in [32, 1024]")
        # zk-pow/src/circuit/pearl_program.rs defines TILE_H = 2.
        if shape["rows"] % 2 or shape["cols"] % 2:
            raise ValueError("rows and cols must be multiples of 2 for the current circuit")
        if shape["m"] > (1 << 24) or shape["n"] > (1 << 24):
            raise ValueError("m and n must be at most 2^24")
        if (shape["rows"] + shape["cols"]) * shape["k"] > (1 << 22):
            raise ValueError("pattern witness size exceeds the current 4 MiB circuit limit")

        if fixed_plain is None:
            row_pattern = pm.PeriodicPattern.from_list(list(range(shape["rows"])))
            col_pattern = pm.PeriodicPattern.from_list(list(range(shape["cols"])))
            mining_config = pm.MiningConfiguration(
                common_dim=shape["k"],
                rank=shape["rank"],
                mma_type=pm.MMAType.Int7xInt7ToInt32,
                rows_pattern=row_pattern,
                cols_pattern=col_pattern,
                moe=None,
            )
        else:
            mining_config = None

        affinity = cpu_affinity()
        rayon_env = os.environ.get("RAYON_NUM_THREADS")
        rayon_threads = int(rayon_env) if rayon_env and rayon_env.isdigit() else 6
        run_info = {
            "type": "run",
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_affinity": affinity,
            "cpu_affinity_count": len(affinity) if affinity is not None else None,
            "rayon_num_threads_env": rayon_env,
            "rayon_threads_effective": rayon_threads,
            "allocator_config_env": {
                key: os.environ[key]
                for key in ("MALLOC_CONF", "_RJEM_MALLOC_CONF")
                if key in os.environ
            },
            "benchmark_stark_rate_bits_env": os.environ.get("PEARL_BENCH_STARK_RATE_BITS"),
            "cgroup_limits": cgroup_limits(),
            "build_tag": os.environ.get("PEARL_BUILD_TAG") or os.environ.get("BUILD_TAG"),
            "module": module_info,
            "package_wrapper": package_wrapper_info,
            "fixture_source": source,
            "fixture_sha256": input_path_sha256,
            "shape": shape_json(shape),
            "header": header_json(fixed_header),
            "cert_version": CERT_VERSION,
            "requested_repetitions": args.repetitions,
            "repetitions": repetitions,
        }

        output_file = args.output_fixture.open("x", encoding="utf-8") if args.output_fixture else None
        try:
            if hasattr(pm, "clear_circuit_cache_v2"):
                pm.clear_circuit_cache_v2()
            negative_result: dict[str, Any] | None = None
            for repetition in range(1, repetitions + 1):
                if fixed_plain is None:
                    mine_wall = 0.0
                    mine_cpu = 0.0
                    candidate_search_wall = 0.0
                    candidate_search_cpu = 0.0
                    rejected_candidates = 0
                    plain = None
                    plain_valid = False
                    plain_message = ""
                    plain_verify_time = {"wall_seconds": 0.0, "cpu_seconds": 0.0}
                    search_wall_start = time.perf_counter()
                    search_cpu_start = time.process_time()
                    for candidate_attempt in range(1, 1025):
                        candidate, one_mine_time = timed_call(
                            lambda: pm.mine(
                                shape["m"],
                                shape["n"],
                                shape["k"],
                                fixed_header,
                                mining_config,
                                signal_range=(-63, 63),
                                cert_version=CERT_VERSION,
                            )
                        )
                        mine_wall += one_mine_time["wall_seconds"]
                        mine_cpu += one_mine_time["cpu_seconds"]
                        (candidate_valid, candidate_message), candidate_verify_time = timed_call(
                            lambda: pm.verify_plain_proof_for_cert_version(
                                CERT_VERSION, fixed_header, candidate
                            )
                        )
                        plain_verify_time = candidate_verify_time
                        if candidate_valid:
                            plain = candidate
                            plain_valid = True
                            plain_message = str(candidate_message)
                            break
                        rejected_candidates += 1
                    candidate_search_wall = time.perf_counter() - search_wall_start
                    candidate_search_cpu = time.process_time() - search_cpu_start
                    if plain is None:
                        raise RuntimeError(
                            "failed to mine a rank-penalty-valid plain proof in 1024 fresh attempts; "
                            f"last validation result: {candidate_message}"
                        )
                    mining_time = {"wall_seconds": mine_wall, "cpu_seconds": mine_cpu}
                    candidate_search_time = {
                        "wall_seconds": candidate_search_wall,
                        "cpu_seconds": candidate_search_cpu,
                    }
                    candidate_mine_attempts = rejected_candidates + 1
                else:
                    plain = fixed_plain
                    mining_time = {"wall_seconds": 0.0, "cpu_seconds": 0.0}
                    candidate_search_time = {"wall_seconds": 0.0, "cpu_seconds": 0.0}
                    rejected_candidates = 0
                    candidate_mine_attempts = 0
                    (plain_valid, plain_message), plain_verify_time = timed_call(
                        lambda: pm.verify_plain_proof_for_cert_version(CERT_VERSION, fixed_header, plain)
                    )

                plain_b64 = plain.to_base64()
                witness_sha256 = hashlib.sha256(plain_b64.encode("ascii")).hexdigest()
                if not plain_valid:
                    raise RuntimeError(f"plain proof verification failed: {plain_message}")

                cache_state = "cold" if repetition == 1 else "warm"
                certificate, prove_time = timed_call(
                    lambda: pm.generate_proof_for_cert_version(CERT_VERSION, fixed_header, plain)
                )
                (verified, verify_message), verify_time = timed_call(
                    lambda: pm.verify_proof_for_cert_version(CERT_VERSION, fixed_header, certificate)
                )
                if not verified:
                    raise RuntimeError(f"certificate verification failed: {verify_message}")

                if negative_result is None:
                    negative_ok, negative_message, negative_time = verify_negative_header(
                        pm, fixed_header, certificate
                    )
                    if not negative_ok:
                        raise RuntimeError("certificate unexpectedly verified with altered header")
                    negative_result = {
                        "altered_header_rejected": True,
                        "altered_header_message": negative_message,
                        "altered_header_verification": negative_time,
                    }

                public_data = bytes(certificate.public_data)
                proof_data = bytes(certificate.proof_data)
                certificate_sha256 = hashlib.sha256(public_data + proof_data).hexdigest()
                record = {
                    "type": "result",
                    "repetition": repetition,
                    "fixture_source": source,
                    "fixture_sha256": input_path_sha256,
                    "plain_witness_sha256": witness_sha256,
                    "shape": shape_json(shape),
                    "prove_cache_state": cache_state,
                    "timings": {
                        "mine": mining_time,
                        "candidate_search": candidate_search_time,
                        "plain_verify": plain_verify_time,
                        "prove": prove_time,
                        "verify": verify_time,
                    },
                    "candidate_mine_attempts": candidate_mine_attempts,
                    "candidates_rejected_by_plain_check": rejected_candidates,
                    "plain_verified": bool(plain_valid),
                    "plain_verification_message": str(plain_message),
                    "verified": bool(verified),
                    "verification_message": str(verify_message),
                    "proof_size_bytes": len(public_data) + len(proof_data),
                    "public_data_size_bytes": len(public_data),
                    "proof_data_size_bytes": len(proof_data),
                    "certificate_sha256": certificate_sha256,
                    "memory": rss_metrics(),
                }
                if repetition == 1 and negative_result is not None:
                    record["negative_header_check"] = negative_result
                print(json.dumps({**run_info, **record}, separators=(",", ":")), flush=True)
                if output_file is not None:
                    fixture_record = {
                        "schema": "pearl-prover-certificate-v1",
                        "cert_version": CERT_VERSION,
                        "header": header_json(fixed_header),
                        "shape": shape_json(shape),
                        "fixture_sha256": input_path_sha256,
                        "plain_witness_sha256": witness_sha256,
                        "certificate_sha256": certificate_sha256,
                        "public_data_b64": base64.b64encode(public_data).decode("ascii"),
                        "proof_data_b64": base64.b64encode(proof_data).decode("ascii"),
                    }
                    output_file.write(json.dumps(fixture_record, separators=(",", ":")) + "\n")
                    output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()

        summary = {
            "type": "summary",
            "completed_repetitions": repetitions,
            "all_verified": True,
            "negative_header_check_passed": bool(negative_result and negative_result["altered_header_rejected"]),
            "output_fixture": str(args.output_fixture.resolve()) if args.output_fixture else None,
        }
        print(json.dumps(summary, separators=(",", ":")), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"}, separators=(",", ":")), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
