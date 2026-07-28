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


def iter_report_archives(reports_dir: Path) -> Iterator[Path]:
    if not reports_dir.exists():
        return

    candidates: list[Path] = []
    for path in sorted(reports_dir.iterdir()):
        if path.is_file() and path.suffixes[-2:] == [".tar", ".xz"]:
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(sorted(path.rglob("*.tar.xz")))

    seen_paths: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
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
