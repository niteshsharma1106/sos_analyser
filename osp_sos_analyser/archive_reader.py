# osp_sos_analyser/archive_reader.py
from __future__ import annotations

import codecs
import hashlib
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
_HASH_CHUNK_SIZE = 1024 * 1024


class SosReportError(RuntimeError):
    """Raised when SOS report ingestion cannot continue."""


def normalized_member_name(member: tarfile.TarInfo) -> str:
    return member.name.replace("\\", "/").lstrip("./")


def archive_file_fingerprint(path: Path) -> tuple[int, str]:
    """Cheap exact-file fingerprint: (size, sha256) without xz member decompress."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return size, digest.hexdigest()


def member_content_signature_key(member: tarfile.TarInfo) -> tuple[str, int] | None:
    """Relative (path, size) pair used for content-level archive dedupe."""
    if not member.isreg():
        return None
    name = normalized_member_name(member)
    parts = name.split("/", 1)
    relative_name = parts[1] if len(parts) > 1 else name
    return relative_name, member.size


def iter_report_archives(reports_dir: Path) -> Iterator[Path]:
    """Yield unique SOS archives.

    Exact byte-identical copies are skipped via size+sha256 without decompressing
    the xz stream. Renamed/re-tarred content duplicates are handled during ingest
    by building a member listing signature while streaming (one decompress).
    """
    if not reports_dir.exists():
        return

    candidates: list[Path] = []
    for path in sorted(reports_dir.iterdir()):
        if path.is_file() and path.suffixes[-2:] == [".tar", ".xz"]:
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(sorted(path.rglob("*.tar.xz")))

    seen_file_fingerprints: dict[tuple[int, str], Path] = {}
    for path in candidates:
        try:
            fingerprint = archive_file_fingerprint(path)
        except OSError as exc:
            raise SosReportError(f"Failed to fingerprint archive: {path}: {exc}") from exc
        existing = seen_file_fingerprints.get(fingerprint)
        if existing is not None:
            print(
                f"[skip] {path} is a byte-identical copy of {existing} — skipping.",
                flush=True,
            )
            continue
        seen_file_fingerprints[fingerprint] = path
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
