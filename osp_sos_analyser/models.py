from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LogEntry:
    timestamp: datetime | None
    pid: int | None
    level: str
    module: str
    message: str
    service: str
    category: str
    source_file: str
    report_name: str
    tags: str
    rhosp_version: str = "17.x"


@dataclass(frozen=True)
class CommandArtifact:
    source: str
    command: str
    output: str
    service: str
    category: str
    source_file: str
    report_name: str
    tags: str
    rhosp_version: str = "17.x"


@dataclass(frozen=True)
class IngestionStats:
    archives: int = 0
    log_files: int = 0
    log_rows: int = 0
    command_files: int = 0
    command_rows: int = 0

    def add(self, other: "IngestionStats") -> "IngestionStats":
        return IngestionStats(
            archives=self.archives + other.archives,
            log_files=self.log_files + other.log_files,
            log_rows=self.log_rows + other.log_rows,
            command_files=self.command_files + other.command_files,
            command_rows=self.command_rows + other.command_rows,
        )
