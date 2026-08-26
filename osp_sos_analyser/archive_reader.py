from __future__ import annotations

import codecs
import gzip
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

from .classification import (
    COMMAND_TOKENS,
    LOW_VALUE_COMMAND_MARKERS,
    SYSTEMISH_COMMAND_PLUGINS,
    SYSTEMISH_COMMAND_TOKENS,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# This is a safety cap, not the point at which log tailing starts.  Large
# files are streamed and reduced to a recent time window by ingest.py.
DEFAULT_MAX_FILE_SIZE_MB = 2048

LOG_SUFFIXES = (
    ".log",
    ".txt",
    ".out",
    ".err",
    ".log.gz",
    ".txt.gz",
)

PHASE1_LOG_TOKENS = (
    "nova",
    "cinder",
    "neutron",
    "ovn",
    "openvswitch",
    "ovs",
    "heat",
    "libvirt",
    "messages",
    "dmesg"
)

# Traditional system logs in SOS reports are normally extensionless.  journalctl
# output is usually collected under sos_commands/logs rather than /var/log.
SYSTEM_LOG_BASENAMES = frozenset({"messages", "syslog", "secure", "boot.log"})

LOW_VALUE_LOG_PATTERNS = (
    r"/httpd/",
    r"access\.log",
    r"error\.log",
)

LOW_VALUE_REGEX = re.compile("|".join(LOW_VALUE_LOG_PATTERNS), re.IGNORECASE)

SERVICE_TOKEN_SET = {x.lower() for x in PHASE1_LOG_TOKENS}

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class SosReportError(RuntimeError):
    """Raised when SOS report ingestion fails."""


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

@dataclass
class ScanStats:
    archives: int = 0
    files_seen: int = 0
    interesting_logs: int = 0
    interesting_commands: int = 0
    skipped_large: int = 0
    skipped_pattern: int = 0
    skipped_extension: int = 0
    skipped_directory: int = 0


SCAN_STATS = ScanStats()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def normalized_member_name(member: tarfile.TarInfo) -> str:
    """
    Normalize member path to POSIX format.

    Example:
        .\\var\\log\\nova\\nova.log
            ->
        var/log/nova/nova.log
    """
    return member.name.replace("\\", "/").lstrip("./")


def file_size_limit(max_file_size_mb: int | None) -> int:
    if max_file_size_mb is None:
        max_file_size_mb = DEFAULT_MAX_FILE_SIZE_MB

    return max(1, int(max_file_size_mb)) * 1024 * 1024


# -----------------------------------------------------------------------------
# Archive discovery
# -----------------------------------------------------------------------------

def iter_report_archives(reports_dir: Path) -> Iterator[Path]:
    """
    Recursively discover *.tar.xz archives.
    """

    if not reports_dir.exists():
        return

    seen: set[Path] = set()

    for archive in sorted(reports_dir.rglob("*.tar.xz")):
        resolved = archive.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)
        SCAN_STATS.archives += 1
        yield archive


# -----------------------------------------------------------------------------
# Filtering helpers
# -----------------------------------------------------------------------------

def _valid_regular_file(member: tarfile.TarInfo, max_size: int) -> bool:
    if not member.isreg():
        return False

    if member.size <= 0:
        return False

    if member.size > max_size:
        SCAN_STATS.skipped_large += 1
        return False

    return True


def _has_supported_extension(name: str) -> bool:
    lower = name.lower()

    if lower.endswith(LOG_SUFFIXES):
        return True

    SCAN_STATS.skipped_extension += 1
    return False


def _is_log_directory(name: str) -> bool:
    lower = name.lower()

    if "/var/log/" in f"/{lower}" or "/var/log/containers/" in f"/{lower}":
        return True

    SCAN_STATS.skipped_directory += 1
    return False


def _contains_service(name: str) -> bool:
    parts = {p.lower() for p in PurePosixPath(name).parts}

    if SERVICE_TOKEN_SET & parts:
        return True

    lower = name.lower()

    return any(token in lower for token in SERVICE_TOKEN_SET)


def _is_system_log(name: str) -> bool:
    """Return whether *name* is a text system log worth indexing."""
    base_name = PurePosixPath(name).name.lower()
    return base_name in SYSTEM_LOG_BASENAMES or base_name.startswith("journalctl")


def is_systemd_journal_member(member: tarfile.TarInfo, max_size: int) -> bool:
    """Identify binary persistent-journal files collected by SOS."""
    if not _valid_regular_file(member, max_size):
        return False
    name = normalized_member_name(member).lower()
    if "/var/log/journal/" not in f"/{name}":
        return False
    if not (name.endswith(".journal") or name.endswith(".journal~")):
        return False
    SCAN_STATS.interesting_logs += 1
    return True


def _is_system_log_location(name: str) -> bool:
    lower = name.lower()
    if PurePosixPath(name).name.lower().startswith("journalctl"):
        # SOS stores journal output in multiple plugin directories, for
        # example sos_commands/systemd and sos_commands/openvswitch.
        return "/sos_commands/" in f"/{lower}"
    return "/var/log/" in f"/{lower}"


# -----------------------------------------------------------------------------
# Public filters
# -----------------------------------------------------------------------------

def is_interesting_log_member(
    member: tarfile.TarInfo,
    max_file_size: int,
) -> bool:

    SCAN_STATS.files_seen += 1

    if not _valid_regular_file(member, max_file_size):
        return False

    name = normalized_member_name(member)

    # /var/log/messages, /var/log/secure, and sos_commands/logs/journalctl
    # commonly have no extension, so evaluate these before the generic suffix
    # and directory filters.
    if _is_system_log(name):
        if not _is_system_log_location(name):
            SCAN_STATS.skipped_directory += 1
            return False
        if LOW_VALUE_REGEX.search(name):
            SCAN_STATS.skipped_pattern += 1
            return False
        SCAN_STATS.interesting_logs += 1
        return True

    # Containerized RHOSP services can use service names not present in the
    # phase-one token list (for example glance, keystone, or custom sidecars).
    # Retain the active log and only the latest compressed rotation; older
    # rotations add volume without contributing the recent SOS evidence.
    if "/var/log/containers/" in f"/{name.lower()}":
        if LOW_VALUE_REGEX.search(name):
            SCAN_STATS.skipped_pattern += 1
            return False
        base_name = PurePosixPath(name).name.lower()
        if base_name.endswith(".log") or base_name.endswith(".log.1.gz"):
            SCAN_STATS.interesting_logs += 1
            return True
        SCAN_STATS.skipped_extension += 1
        return False

    if not _has_supported_extension(name):
        return False

    if not _is_log_directory(name):
        return False

    if LOW_VALUE_REGEX.search(name):
        SCAN_STATS.skipped_pattern += 1
        return False

    if not _contains_service(name):
        return False

    SCAN_STATS.interesting_logs += 1
    return True


def is_interesting_command_member(
    member: tarfile.TarInfo,
    max_file_size: int,
) -> bool:

    SCAN_STATS.files_seen += 1

    if not _valid_regular_file(member, max_file_size):
        return False

    name = normalized_member_name(member).lower()

    if "sos_commands/" not in name:
        return False

    if any(marker in name for marker in LOW_VALUE_COMMAND_MARKERS):
        SCAN_STATS.skipped_pattern += 1
        return False

    matched_tokens = [token for token in COMMAND_TOKENS if token in name]
    if not matched_tokens:
        return False

    # Require system-ish tokens to live under known sos_commands plugins so we
    # do not ingest multi-hundred-MB crm_report "messages" extracts.
    if all(token in SYSTEMISH_COMMAND_TOKENS for token in matched_tokens):
        if not any(plugin in f"/{name}" for plugin in SYSTEMISH_COMMAND_PLUGINS):
            SCAN_STATS.skipped_pattern += 1
            return False

    SCAN_STATS.interesting_commands += 1
    return True


def is_config_member(member: tarfile.TarInfo, max_size: int) -> bool:
    """Return whether a regular SOS /etc capture should be retained as config evidence."""
    if not _valid_regular_file(member, max_size):
        return False
    name = normalized_member_name(member).lower()
    return "/etc/" in f"/{name}"


# -----------------------------------------------------------------------------
# Reading members
# -----------------------------------------------------------------------------

def _extract_stream(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
):

    extracted = archive.extractfile(member)

    if extracted is None:
        raise SosReportError(
            f"Unable to extract member: {member.name}"
        )

    return extracted


def _decoded_stream(stream):

    first_two = stream.peek(2)

    if first_two.startswith(b"\x1f\x8b"):
        stream = gzip.GzipFile(fileobj=stream)

    for encoding in (
        "utf-8",
        "utf-16",
        "latin-1",
    ):
        try:
            stream.seek(0)
            return io.TextIOWrapper(
                stream,
                encoding=encoding,
                errors="replace",
            )
        except Exception:
            pass

    stream.seek(0)

    return io.TextIOWrapper(
        stream,
        encoding="utf-8",
        errors="replace",
    )


def iter_text_lines(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> Iterator[str]:

    extracted = _extract_stream(
        archive,
        member,
    )

    with extracted:

        first = extracted.peek(2)

        if first.startswith(b"\x1f\x8b"):
            reader = gzip.GzipFile(fileobj=extracted)
            yield from codecs.iterdecode(
                reader,
                encoding="utf-8",
                errors="replace",
            )
            return

        yield from codecs.iterdecode(
            extracted,
            encoding="utf-8",
            errors="replace",
        )


def read_text_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> str:

    extracted = _extract_stream(
        archive,
        member,
    )

    with extracted:

        data = extracted.read()

    for enc in (
        "utf-8",
        "utf-16",
        "latin-1",
    ):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass

    return data.decode(
        "utf-8",
        errors="replace",
    )


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def get_scan_statistics() -> ScanStats:
    """
    Return current scan statistics.
    """
    return SCAN_STATS


def reset_scan_statistics() -> None:
    """
    Reset scan statistics.
    """
    global SCAN_STATS
    SCAN_STATS = ScanStats()
