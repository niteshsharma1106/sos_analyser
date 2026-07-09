from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from .models import CommandRecord, LogRecord


IMPORTANT_LEVELS = ("ERROR", "CRITICAL", "WARNING")


class AnalysisStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=True)

    def table_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            logs = conn.execute("SELECT COUNT(*) FROM os_logs").fetchone()[0]
            commands = conn.execute("SELECT COUNT(*) FROM os_commands").fetchone()[0]
        return {"os_logs": int(logs), "os_commands": int(commands)}

    def get_error_summary(self, limit: int = 20) -> tuple[tuple[str, str, int], ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(service, 'unknown') AS service, level, COUNT(*) AS count
                FROM os_logs
                WHERE level IN ('ERROR', 'CRITICAL')
                GROUP BY service, level
                ORDER BY count DESC, service
                LIMIT ?
                """,
                [limit],
            ).fetchall()
        return tuple((str(service), str(level), int(count)) for service, level, count in rows)

    def search_logs(
        self,
        service: str | None = None,
        text_terms: Sequence[str] = (),
        levels: Sequence[str] = (),
        limit: int = 25,
        start: datetime | None = None,
        end: datetime | None = None,
        chronological: bool = False,
    ) -> tuple[LogRecord, ...]:
        clauses: list[str] = []
        params: list[object] = []

        if service:
            clauses.append("service = ?")
            params.append(service)
        if levels:
            clauses.append(f"level IN ({', '.join('?' for _ in levels)})")
            params.extend(levels)
        if start:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("timestamp <= ?")
            params.append(end)

        for term in text_terms:
            clauses.append("(message ILIKE ? OR module ILIKE ? OR source_file ILIKE ?)")
            like = f"%{term}%"
            params.extend([like, like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        order_by = (
            "timestamp NULLS LAST"
            if chronological
            else """
                    CASE
                        WHEN message ILIKE '%failed%' THEN 0
                        WHEN message ILIKE '%failure%' THEN 0
                        WHEN message ILIKE '%not ready%' THEN 0
                        WHEN message ILIKE '%exception%' THEN 0
                        WHEN message ILIKE '%traceback%' THEN 0
                        WHEN message ILIKE '%timeout%' THEN 0
                        WHEN message ILIKE '%unreachable%' THEN 0
                        WHEN message ILIKE '%error%' THEN 0
                        WHEN message ILIKE '%status: 5%' THEN 1
                        WHEN message ILIKE '%status: 4%' THEN 2
                        WHEN message ILIKE '%status: 2%' THEN 5
                        ELSE 3
                    END,
                    CASE level
                        WHEN 'CRITICAL' THEN 0
                        WHEN 'ERROR' THEN 1
                        WHEN 'WARNING' THEN 2
                        ELSE 3
                    END,
                    timestamp NULLS LAST
            """
        )

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT timestamp, level, service, module, message, source_file, report_name
                FROM os_logs
                {where}
                ORDER BY {order_by}
                LIMIT ?
                """,
                params,
            ).fetchall()
        return tuple(_log_record(row) for row in rows)

    def get_timeline(
        self,
        services: Sequence[str] = (),
        text_terms: Sequence[str] = (),
        levels: Sequence[str] = IMPORTANT_LEVELS,
        limit: int = 30,
    ) -> tuple[LogRecord, ...]:
        clauses: list[str] = []
        params: list[object] = []

        if services:
            clauses.append(f"service IN ({', '.join('?' for _ in services)})")
            params.extend(services)
        if levels:
            clauses.append(f"level IN ({', '.join('?' for _ in levels)})")
            params.extend(levels)
        if text_terms:
            term_clauses = []
            for term in text_terms:
                term_clauses.append("(message ILIKE ? OR module ILIKE ? OR source_file ILIKE ?)")
                like = f"%{term}%"
                params.extend([like, like, like])
            clauses.append(f"({' OR '.join(term_clauses)})")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT timestamp, level, service, module, message, source_file, report_name
                FROM os_logs
                {where}
                ORDER BY timestamp NULLS LAST
                LIMIT ?
                """,
                params,
            ).fetchall()
        return tuple(_log_record(row) for row in rows)

    def get_events_around(
        self,
        timestamp: datetime,
        minutes: int = 5,
        services: Sequence[str] = (),
        limit: int = 50,
    ) -> tuple[LogRecord, ...]:
        start = timestamp - timedelta(minutes=minutes)
        end = timestamp + timedelta(minutes=minutes)
        clauses = ["timestamp BETWEEN ? AND ?"]
        params: list[object] = [start, end]
        if services:
            clauses.append(f"service IN ({', '.join('?' for _ in services)})")
            params.extend(services)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT timestamp, level, service, module, message, source_file, report_name
                FROM os_logs
                WHERE {' AND '.join(clauses)}
                ORDER BY timestamp
                LIMIT ?
                """,
                params,
            ).fetchall()
        return tuple(_log_record(row) for row in rows)

    def find_identifier_events(
        self,
        identifier: str,
        services: Sequence[str] = (),
        limit: int = 50,
    ) -> tuple[LogRecord, ...]:
        clauses = ["(message ILIKE ? OR module ILIKE ? OR source_file ILIKE ?)"]
        params: list[object] = [f"%{identifier}%", f"%{identifier}%", f"%{identifier}%"]
        if services:
            clauses.append(f"service IN ({', '.join('?' for _ in services)})")
            params.extend(services)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT timestamp, level, service, module, message, source_file, report_name
                FROM os_logs
                WHERE {' AND '.join(clauses)}
                ORDER BY timestamp NULLS LAST
                LIMIT ?
                """,
                params,
            ).fetchall()
        return tuple(_log_record(row) for row in rows)

    def get_command_output(
        self,
        command_pattern: str,
        service: str | None = None,
        limit: int = 10,
    ) -> tuple[CommandRecord, ...]:
        clauses = ["(command ILIKE ? OR source_file ILIKE ?)"]
        params: list[object] = [f"%{command_pattern}%", f"%{command_pattern}%"]
        if service:
            clauses.append("service = ?")
            params.append(service)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT command, service, source_file, output, report_name
                FROM os_commands
                WHERE {' AND '.join(clauses)}
                ORDER BY source_file
                LIMIT ?
                """,
                params,
            ).fetchall()
        return tuple(_command_record(row) for row in rows)


def _log_record(row: tuple[object, ...]) -> LogRecord:
    return LogRecord(
        timestamp=row[0],
        level=str(row[1] or ""),
        service=str(row[2] or "unknown"),
        module=str(row[3] or ""),
        message=str(row[4] or ""),
        source_file=str(row[5] or ""),
        report_name=str(row[6] or ""),
    )


def _command_record(row: tuple[object, ...]) -> CommandRecord:
    return CommandRecord(
        command=str(row[0] or ""),
        service=str(row[1] or "unknown"),
        source_file=str(row[2] or ""),
        output=str(row[3] or ""),
        report_name=str(row[4] or ""),
    )
