# db.py
from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import duckdb

from .models import CommandArtifact, LogEntry

NULL_VALUE = r"\N"


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


def _copy_rows(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> int:
    if not rows:
        return 0

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        suffix=".csv",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.writer(handle)
            for row in rows:
                writer.writerow([NULL_VALUE if value is None else value for value in row])

        conn.execute(
            f"""
            COPY {table} ({", ".join(columns)})
            FROM '{_sql_string(temp_path.as_posix())}'
            (
                FORMAT CSV,
                HEADER false,
                AUTO_DETECT false,
                DELIM ',',
                QUOTE '"',
                ESCAPE '"',
                NULL '{NULL_VALUE}',
                STRICT_MODE false,
                MAX_LINE_SIZE 100000000
            )
            """
        )
        return len(rows)
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS os_logs (
            timestamp TIMESTAMP,
            pid INTEGER,
            level TEXT,
            module TEXT,
            message TEXT,
            service TEXT,
            category TEXT,
            source_file TEXT,
            report_name TEXT,
            tags TEXT,
            rhosp_version TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS os_commands (
            source TEXT,
            command TEXT,
            output TEXT,
            service TEXT,
            category TEXT,
            source_file TEXT,
            report_name TEXT,
            tags TEXT,
            rhosp_version TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_reports (
            archive_id TEXT,
            archive_path TEXT,
            archive_name TEXT,
            archive_size BIGINT,
            archive_mtime_ns BIGINT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT,
            log_rows INTEGER,
            command_rows INTEGER
        )
        """
    )
    ensure_indexes(conn)


def ensure_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Create investigation-friendly indexes (idempotent)."""
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_os_logs_service_level_ts
        ON os_logs(service, level, timestamp)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_os_logs_timestamp
        ON os_logs(timestamp)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_os_logs_report_name
        ON os_logs(report_name)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_os_commands_service
        ON os_commands(service)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_os_commands_report_name
        ON os_commands(report_name)
        """
    )

def archive_already_ingested(conn: duckdb.DuckDBPyConnection, archive_id: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM ingested_reports
        WHERE archive_id = ? AND status = 'completed'
        LIMIT 1
        """,
        [archive_id],
    ).fetchone()
    return row is not None


def mark_archive_started(
    conn: duckdb.DuckDBPyConnection,
    archive_id: str,
    archive_path: Path,
) -> None:
    stat = archive_path.stat()
    conn.execute(
        "DELETE FROM ingested_reports WHERE archive_id = ? AND status != 'completed'",
        [archive_id],
    )
    conn.execute(
        """
        INSERT INTO ingested_reports (
            archive_id, archive_path, archive_name, archive_size, archive_mtime_ns,
            started_at, completed_at, status, log_rows, command_rows
        )
        VALUES (?, ?, ?, ?, ?, current_timestamp, NULL, 'running', 0, 0)
        """,
        [
            archive_id,
            str(archive_path.resolve()),
            archive_path.name,
            stat.st_size,
            stat.st_mtime_ns,
        ],
    )


def mark_archive_completed(
    conn: duckdb.DuckDBPyConnection,
    archive_id: str,
    log_rows: int,
    command_rows: int,
) -> None:
    conn.execute(
        """
        UPDATE ingested_reports
        SET completed_at = current_timestamp,
            status = 'completed',
            log_rows = ?,
            command_rows = ?
        WHERE archive_id = ?
        """,
        [log_rows, command_rows, archive_id],
    )


def mark_archive_failed(conn: duckdb.DuckDBPyConnection, archive_id: str) -> None:
    conn.execute(
        """
        UPDATE ingested_reports
        SET completed_at = current_timestamp,
            status = 'failed'
        WHERE archive_id = ?
        """,
        [archive_id],
    )


def dedupe_ingested_rows(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    before_logs = conn.execute("SELECT COUNT(*) FROM os_logs").fetchone()[0]
    before_commands = conn.execute("SELECT COUNT(*) FROM os_commands").fetchone()[0]

    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE deduped_os_logs AS
        SELECT * EXCLUDE(row_num)
        FROM (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY report_name, source_file, timestamp, pid, level,
                                    module, message, service
                       ORDER BY report_name
                   ) AS row_num
            FROM os_logs
        )
        WHERE row_num = 1
        """
    )
    conn.execute("DELETE FROM os_logs")
    conn.execute("INSERT INTO os_logs SELECT * FROM deduped_os_logs")
    conn.execute("DROP TABLE deduped_os_logs")

    conn.execute(
        """
        CREATE OR REPLACE TEMP TABLE deduped_os_commands AS
        SELECT * EXCLUDE(row_num)
        FROM (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY report_name, source_file, command, output
                       ORDER BY report_name
                   ) AS row_num
            FROM os_commands
        )
        WHERE row_num = 1
        """
    )
    conn.execute("DELETE FROM os_commands")
    conn.execute("INSERT INTO os_commands SELECT * FROM deduped_os_commands")
    conn.execute("DROP TABLE deduped_os_commands")

    after_logs = conn.execute("SELECT COUNT(*) FROM os_logs").fetchone()[0]
    after_commands = conn.execute("SELECT COUNT(*) FROM os_commands").fetchone()[0]
    return int(before_logs - after_logs), int(before_commands - after_commands)


def insert_logs(conn: duckdb.DuckDBPyConnection, entries: Sequence[LogEntry]) -> int:
    if not entries:
        return 0

    return _copy_rows(
        conn,
        "os_logs",
        (
            "timestamp",
            "pid",
            "level",
            "module",
            "message",
            "service",
            "category",
            "source_file",
            "report_name",
            "tags",
            "rhosp_version",
        ),
        [
            (
                entry.timestamp,
                entry.pid,
                entry.level,
                entry.module,
                entry.message,
                entry.service,
                entry.category,
                entry.source_file,
                entry.report_name,
                entry.tags,
                entry.rhosp_version,
            )
            for entry in entries
        ],
    )


def insert_commands(
    conn: duckdb.DuckDBPyConnection, artifacts: Sequence[CommandArtifact]
) -> int:
    if not artifacts:
        return 0

    return _copy_rows(
        conn,
        "os_commands",
        (
            "source",
            "command",
            "output",
            "service",
            "category",
            "source_file",
            "report_name",
            "tags",
            "rhosp_version",
        ),
        [
            (
                artifact.source,
                artifact.command,
                artifact.output,
                artifact.service,
                artifact.category,
                artifact.source_file,
                artifact.report_name,
                artifact.tags,
                artifact.rhosp_version,
            )
            for artifact in artifacts
        ],
    )
