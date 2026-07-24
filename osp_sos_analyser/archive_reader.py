# osp_sos_analyser/archive_reader.py
from __future__ import annotations

import codecs
import tarfile
from pathlib import Path
from typing import Iterator

from .classification import COMMAND_TOKENS

LOG_SUFFIXES = (".log", ".txt", ".out", ".err")
PHASE1_LOG_TOKENS = ("nova", "cinder", "ovn", "openvswitch", "ovs", "neutron")
LOW_VALUE_LOG_PATTERNS = (
    "/httpd/",
    "_access.log",
    "access.log",
    "_error.log",
)


class SosReportError(RuntimeError):
    """Raised when SOS report ingestion cannot continue."""


def normalized_member_name(member: tarfile.TarInfo) -> str:
    return member.name.replace("\\", "/").lstrip("./")


def _archive_content_signature(path: Path) -> frozenset[tuple[str, int]]:
    """Fingerprint an archive by its internal (member path, size) pairs.

    This catches duplicate sosreports that were re-tarred, renamed, or
    recompressed differently — cases where the outer file bytes/hash differ
    but the actual log content inside is the same. We strip the report-name
    root prefix (the first path segment) before comparing, since two exports
    of the same sosreport often differ only in that top-level folder name
    (e.g. 'sosreport-host-case123/...' vs 'sosreport_log/sosreport/...').
    """
    signature: set[tuple[str, int]] = set()
    try:
        with tarfile.open(path, "r|xz") as archive:
            for member in archive:
                if not member.isreg():
                    continue
                name = normalized_member_name(member)
                parts = name.split("/", 1)
                relative_name = parts[1] if len(parts) > 1 else name
                signature.add((relative_name, member.size))
    except (tarfile.TarError, OSError) as exc:
        raise SosReportError(f"Failed to read archive for dedup check: {path}: {exc}") from exc
    return frozenset(signature)


def iter_report_archives(reports_dir: Path) -> Iterator[Path]:
    if not reports_dir.exists():
        return

    candidates: list[Path] = []
    for path in sorted(reports_dir.iterdir()):
        if path.is_file() and path.suffixes[-2:] == [".tar", ".xz"]:
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(sorted(path.rglob("*.tar.xz")))

    seen_signatures: dict[frozenset[tuple[str, int]], Path] = {}
    for path in candidates:
        signature = _archive_content_signature(path)
        existing = seen_signatures.get(signature)
        if existing is not None:
            print(
                f"[skip] {path} appears to be a duplicate of {existing} "
                f"(same internal file listing) — skipping ingestion.",
                flush=True,
            )
            continue
        seen_signatures[signature] = path
        yield path


def is_interesting_log_member(member: tarfile.TarInfo, max_file_size: int) -> bool:
    if not member.isreg() or member.size <= 0 or member.size > max_file_size:
        return False

    name = normalized_member_name(member).lower()
    if not name.endswith(LOG_SUFFIXES):
        return False
    if "/var/log/containers/" not in f"/{name}" and "/var/log/" not in f"/{name}":
        return False
    if any(pattern in name for pattern in LOW_VALUE_LOG_PATTERNS):
        return False
    return any(token in name for token in PHASE1_LOG_TOKENS)


def is_interesting_command_member(member: tarfile.TarInfo, max_file_size: int) -> bool:
    if not member.isreg() or member.size <= 0 or member.size > max_file_size:
        return False

    name = normalized_member_name(member).lower()
    if "sos_commands/" not in name:
        return False
    return any(token in name for token in COMMAND_TOKENS)


def iter_text_lines(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Iterator[str]:
    extracted = archive.extractfile(member)
    if extracted is None:
        return

    with extracted:
        yield from codecs.iterdecode(extracted, encoding="utf-8", errors="replace")


def read_text_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    extracted = archive.extractfile(member)
    if extracted is None:
        return ""

    with extracted:
        return extracted.read().decode("utf-8", errors="replace")


def file_size_limit(max_file_size_mb: int | None) -> int:
    if max_file_size_mb is None:
        return 25 * 1024 * 1024
    return max(1, int(max_file_size_mb)) * 1024 * 1024