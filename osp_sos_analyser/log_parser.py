# osp_sos_analyser/log_parser.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Iterator, Optional

from .classification import build_tags, classify_service
from .models import LogEntry


# Standard OpenStack-style line: TIMESTAMP PID LEVEL MODULE [req-id] MESSAGE
STANDARD_LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d{3,6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|TRACE)\s+"
    r"(?P<module>\S+)\s+"
    r"(?:\[[^\]]*\]\s+)?"
    r"(?P<message>.*)$"
)

# Native OVN/OVS daemon line: TIMESTAMP|SEQ|MODULE|LEVEL|MESSAGE
# e.g. 2026-07-12T19:23:34.545Z|01101|binding|INFO|Claiming lport ...
OVN_LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\|"
    r"(?P<seq>\d+)\|"
    r"(?P<module>[^|]+)\|"
    r"(?P<level>[A-Za-z]+)\|"
    r"(?P<message>.*)$"
)

# Traditional rsyslog messages and the default `journalctl` rendering:
# Jul 28 14:05:01 controller-0 systemd[1]: Started ...
SYSLOG_PATTERN = re.compile(
    r"^(?P<timestamp>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<module>[^:\[]+?)(?:\[(?P<pid>\d+)\])?:\s*"
    r"(?P<message>.*)$",
    re.IGNORECASE,
)

# `journalctl --output=short-iso` rendering used for binary journal files.
# Example: 2026-07-28T14:05:01+0530 ctl01 systemd[1]: Started service.
JOURNAL_ISO_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?P<host>\S+)\s+(?P<module>[^:\[]+?)(?:\[(?P<pid>\d+)\])?:\s*"
    r"(?P<message>.*)$"
)

# CRI-O/containerd stdout/stderr wrapper around a container's own log line:
# <container-runtime-timestamp> stdout|stderr F|P <original line>
CONTAINER_WRAPPER_PATTERN = re.compile(
    r"^\S+\s+(?:stdout|stderr)\s+[FP]\s+(?P<rest>.*)$"
)


def _unwrap_container_line(line: str) -> str:
    match = CONTAINER_WRAPPER_PATTERN.match(line)
    return match.group("rest") if match else line


def parse_log_timestamp(value: str) -> Optional[datetime]:
    """Parse a timestamp from any format this project ingests. Returns
    None (rather than raising) if unparseable, so a single odd line can't
    abort ingestion of an entire file."""
    candidates = (value, value.replace("T", " ").replace(",", "."))
    formats = (
        "%Y-%m-%d %H:%M:%S.%f",   # standard openstack, post-normalize
        "%Y-%m-%dT%H:%M:%S.%fZ",  # OVN native, fractional seconds
        "%Y-%m-%dT%H:%M:%SZ",     # OVN native, whole seconds
    )
    for candidate in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def _parse_syslog_timestamp(value: str, report_name: str) -> Optional[datetime]:
    """Parse a syslog timestamp, using the SOS report year when available."""
    year_match = re.search(r"(?:^|-)20(\d{2})-\d{2}-\d{2}(?:-|\.)", report_name)
    year = int(f"20{year_match.group(1)}") if year_match else datetime.now().year
    try:
        return datetime.strptime(f"{year} {value}", "%Y %b %d %H:%M:%S")
    except ValueError:
        return None


def _entry_from_standard_match(match: re.Match[str], source_file: str, report_name: str) -> LogEntry:
    module = match.group("module")
    service, category = classify_service(module, source_file)
    return LogEntry(
        timestamp=parse_log_timestamp(match.group("timestamp")),
        pid=int(match.group("pid")),
        level="WARNING" if match.group("level") == "WARN" else match.group("level"),
        module=module,
        message=match.group("message"),
        service=service,
        category=category,
        source_file=source_file,
        report_name=report_name,
        tags=build_tags(service, category, module, source_file),
    )


_OVN_LEVEL_MAP = {"WARN": "WARNING", "DBG": "DEBUG", "ERR": "ERROR"}


def _entry_from_ovn_match(match: re.Match[str], source_file: str, report_name: str) -> LogEntry:
    module = match.group("module")
    level = match.group("level").upper()
    level = _OVN_LEVEL_MAP.get(level, level)
    service, category = classify_service(module, source_file)
    return LogEntry(
        timestamp=parse_log_timestamp(match.group("timestamp")),
        pid=None,
        level=level,
        module=module,
        message=match.group("message"),
        service=service,
        category=category,
        source_file=source_file,
        report_name=report_name,
        tags=build_tags(service, category, module, source_file),
    )


def _entry_from_syslog_match(match: re.Match[str], source_file: str, report_name: str) -> LogEntry:
    module = match.group("module").strip()
    service, category = classify_service(module, source_file)
    return LogEntry(
        timestamp=_parse_syslog_timestamp(match.group("timestamp"), report_name),
        pid=int(match.group("pid")) if match.group("pid") else None,
        level="INFO",
        module=module,
        message=match.group("message"),
        service=service,
        category=category,
        source_file=source_file,
        report_name=report_name,
        tags=build_tags(service, category, module, source_file),
    )


def _entry_from_journal_iso_match(match: re.Match[str], source_file: str, report_name: str) -> LogEntry:
    module = match.group("module").strip()
    service, category = classify_service(module, source_file)
    try:
        timestamp = datetime.fromisoformat(match.group("timestamp").replace("Z", "+00:00"))
        # The rest of this project stores naive datetimes. Keep the wall-clock
        # time emitted by journalctl, which is sufficient for SOS-local RCA.
        timestamp = timestamp.replace(tzinfo=None)
    except ValueError:
        timestamp = None
    return LogEntry(
        timestamp=timestamp,
        pid=int(match.group("pid")) if match.group("pid") else None,
        level="INFO",
        module=module,
        message=match.group("message"),
        service=service,
        category=category,
        source_file=source_file,
        report_name=report_name,
        tags=build_tags(service, category, module, source_file),
    )


def _unknown_entry(line: str, source_file: str, report_name: str) -> LogEntry:
    service, category = classify_service("unknown", source_file)
    return LogEntry(
        timestamp=None,
        pid=None,
        level="UNKNOWN",
        module="unknown",
        message=line,
        service=service,
        category=category,
        source_file=source_file,
        report_name=report_name,
        tags=build_tags(service, category, "unknown", source_file),
    )


def _with_message(entry: LogEntry, message_lines: list[str]) -> LogEntry:
    return LogEntry(
        timestamp=entry.timestamp,
        pid=entry.pid,
        level=entry.level,
        module=entry.module,
        message="\n".join(message_lines),
        service=entry.service,
        category=entry.category,
        source_file=entry.source_file,
        report_name=entry.report_name,
        tags=entry.tags,
        rhosp_version=entry.rhosp_version,
        hostname=entry.hostname,
        node_role=entry.node_role,
        cluster_id=entry.cluster_id,
    )


def parse_log_lines(
    lines: Iterable[str], source_file: str, report_name: str
) -> Iterator[LogEntry]:
    current: LogEntry | None = None
    message_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            continue

        unwrapped = _unwrap_container_line(line)
        standard_match = STANDARD_LOG_PATTERN.match(unwrapped)
        ovn_match = None if standard_match else OVN_LOG_PATTERN.match(unwrapped)
        syslog_match = None if standard_match or ovn_match else SYSLOG_PATTERN.match(unwrapped)
        journal_iso_match = (
            None
            if standard_match or ovn_match or syslog_match
            else JOURNAL_ISO_PATTERN.match(unwrapped)
        )

        if standard_match or ovn_match or syslog_match or journal_iso_match:
            if current is not None:
                yield _with_message(current, message_lines)
            current = (
                _entry_from_standard_match(standard_match, source_file, report_name)
                if standard_match
                else _entry_from_ovn_match(ovn_match, source_file, report_name)
                if ovn_match
                else _entry_from_syslog_match(syslog_match, source_file, report_name)
                if syslog_match
                else _entry_from_journal_iso_match(journal_iso_match, source_file, report_name)
            )
            message_lines = [current.message]
            continue

        # Genuinely unrecognized line (e.g. a multi-line traceback) — treat
        # as a continuation of the current entry, or start a fresh UNKNOWN
        # entry if none is open. This path should now be rare, since both
        # known log formats are matched above.
        if current is None:
            current = _unknown_entry(line, source_file, report_name)
            message_lines = [line]
        else:
            message_lines.append(line)

    if current is not None:
        yield _with_message(current, message_lines)
