# ingest.py
from __future__ import annotations

import os
import tarfile
import time
from pathlib import Path

import duckdb

from .archive_reader import (
    SosReportError,
    file_size_limit,
    is_interesting_command_member,
    is_interesting_log_member,
    iter_report_archives,
    iter_text_lines,
    normalized_member_name,
    read_text_member,
)
from .classification import build_tags, classify_service
from .db import ensure_schema, insert_commands, insert_logs
from .log_parser import parse_log_lines
from .models import CommandArtifact, IngestionStats, LogEntry

BATCH_SIZE = 1000


def _default_db_path(reports_dir: Path) -> Path:
    return reports_dir.parent / "sos_analysis.duckdb"


def _command_artifact(
    source_file: str, output: str, report_name: str
) -> CommandArtifact:
    command = Path(source_file).name
    service, category = classify_service(command, source_file)
    return CommandArtifact(
        source=source_file,
        command=command,
        output=output,
        service=service,
        category=category,
        source_file=source_file,
        report_name=report_name,
        tags=build_tags(service, category, command, source_file),
    )


def _ingest_archive(
    conn: duckdb.DuckDBPyConnection,
    archive_path: Path,
    max_file_size: int,
) -> IngestionStats:
    stats = IngestionStats(archives=1)
    report_name = archive_path.name
    command_batch: list[CommandArtifact] = []
    log_index = 0
    command_index = 0

    with tarfile.open(archive_path, "r|xz") as archive:
        for member in archive:
            if is_interesting_log_member(member, max_file_size):
                log_index += 1
                rows = _ingest_log_member(
                    conn=conn,
                    archive=archive,
                    member=member,
                    report_name=report_name,
                    log_index=log_index,
                )
                stats = stats.add(IngestionStats(log_files=1, log_rows=rows))
                continue

            if is_interesting_command_member(member, max_file_size):
                command_index += 1
                source_file = normalized_member_name(member)
                if command_index == 1 or command_index % 100 == 0:
                    print(
                        f"[progress] {report_name}: reading command artifact "
                        f"{command_index}: {source_file}",
                        flush=True,
                    )
                command_batch.append(
                    _command_artifact(
                        source_file=source_file,
                        output=read_text_member(archive, member),
                        report_name=report_name,
                    )
                )

    command_rows = insert_commands(conn, command_batch)
    stats = stats.add(
        IngestionStats(
            command_files=command_index,
            command_rows=command_rows,
        )
    )
    print(
        f"[progress] {report_name}: finished archive: {stats.log_files} log file(s), "
        f"{stats.command_files} command artifact(s)",
        flush=True,
    )
    return stats


def _ingest_log_member(
    conn: duckdb.DuckDBPyConnection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    report_name: str,
    log_index: int,
) -> int:
    source_file = normalized_member_name(member)
    print(
        f"[progress] {report_name}: parsing log {log_index} "
        f"{source_file} ({member.size / 1024 / 1024:.1f} MB)",
        flush=True,
    )

    batch: list[LogEntry] = []
    rows = 0
    started = time.perf_counter()
    for entry in parse_log_lines(
        iter_text_lines(archive, member), source_file, report_name
    ):
        batch.append(entry)
        if len(batch) >= BATCH_SIZE:
            rows += insert_logs(conn, batch)
            batch.clear()
            print(
                f"[progress] {report_name}: {source_file}: {rows} row(s) loaded",
                flush=True,
            )
    rows += insert_logs(conn, batch)
    elapsed = time.perf_counter() - started
    print(
        f"[progress] {report_name}: finished {source_file}: "
        f"{rows} row(s) in {elapsed:.1f}s",
        flush=True,
    )
    return rows


def ingest_sos_reports(
    reports_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
    clear_existing: bool = False,
    max_file_size_mb: int | None = 25,
) -> Path:
    """Ingest one or more RHOSP 17.x SOS report tar.xz archives into DuckDB."""
    root = Path(reports_dir or "SOS_REPORTS").resolve()
    db_target = Path(db_path).resolve() if db_path else _default_db_path(root).resolve()

    if not root.exists():
        raise SosReportError(f"SOS reports directory does not exist: {root}")

    archive_paths = list(iter_report_archives(root))
    if not archive_paths:
        raise SosReportError(f"No SOS report .tar.xz archives found in: {root}")

    if clear_existing and db_target.exists():
        db_target.unlink()
    db_target.parent.mkdir(parents=True, exist_ok=True)

    max_file_size = file_size_limit(max_file_size_mb)
    total = IngestionStats()

    print(f"[progress] Found {len(archive_paths)} SOS archive(s)", flush=True)
    with duckdb.connect(str(db_target)) as conn:
        ensure_schema(conn)
        if clear_existing:
            conn.execute("DELETE FROM os_logs")
            conn.execute("DELETE FROM os_commands")

        for archive_index, archive_path in enumerate(archive_paths, start=1):
            print(
                f"[progress] Processing archive {archive_index}/{len(archive_paths)}: "
                f"{archive_path.name}",
                flush=True,
            )
            total = total.add(_ingest_archive(conn, archive_path, max_file_size))

    print(
        "[progress] Ingestion complete: "
        f"{total.log_rows} log row(s), {total.command_rows} command row(s)",
        flush=True,
    )
    return db_target


def main() -> None:
    db_path = ingest_sos_reports()
    print(f"Ingestion complete. Database: {db_path}")


if __name__ == "__main__":
    main()
