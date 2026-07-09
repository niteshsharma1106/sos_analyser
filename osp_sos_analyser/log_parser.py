from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Iterator

from .classification import build_tags, classify_service
from .models import LogEntry


LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d{3,6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|TRACE)\s+"
    r"(?P<module>\S+)\s+"
    r"(?:\[[^\]]*\]\s+)?"
    r"(?P<message>.*)$"
)


def parse_log_timestamp(value: str) -> datetime:
    normalized = value.replace("T", " ").replace(",", ".")
    return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S.%f")


def _entry_from_match(match: re.Match[str], source_file: str, report_name: str) -> LogEntry:
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

        match = LOG_PATTERN.match(line)
        if match:
            if current is not None:
                yield _with_message(current, message_lines)
            current = _entry_from_match(match, source_file, report_name)
            message_lines = [current.message]
            continue

        if current is None:
            service, category = classify_service("unknown", source_file)
            current = LogEntry(
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
            message_lines = [line]
        else:
            message_lines.append(line)

    if current is not None:
        yield _with_message(current, message_lines)
