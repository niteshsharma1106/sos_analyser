# ingest.py
from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import deque
from datetime import timedelta
from pathlib import Path

import duckdb

from .archive_reader import (
    SosReportError,
    file_size_limit,
    is_interesting_command_member,
    is_interesting_log_member,
    is_systemd_journal_member,
    iter_report_archives,
    iter_text_lines,
    normalized_member_name,
    read_text_member,
)
from .classification import build_tags, classify_service
from .db import (
    archive_already_ingested,
    dedupe_ingested_rows,
    ensure_schema,
    insert_commands,
    insert_logs,
    mark_archive_completed,
    mark_archive_failed,
    mark_archive_started,
)
from .log_parser import parse_log_lines
from .models import CommandArtifact, IngestionStats, LogEntry

BATCH_SIZE = 1000
HASH_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_LARGE_LOG_THRESHOLD_MB = 30
DEFAULT_LARGE_LOG_TAIL_HOURS = 6


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


def _archive_id(archive_path: Path) -> str:
    digest = hashlib.sha256()
    total_size = archive_path.stat().st_size
    bytes_read = 0
    last_heartbeat = time.perf_counter()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
            bytes_read += len(chunk)
            now = time.perf_counter()
            if total_size >= 256 * 1024 * 1024 and now - last_heartbeat >= 10:
                print(
                    f"[progress] {archive_path.name}: duplicate-check hash "
                    f"{bytes_read / total_size:.0%}",
                    flush=True,
                )
                last_heartbeat = now
    return digest.hexdigest()


def _ingest_archive(
    conn: duckdb.DuckDBPyConnection,
    archive_path: Path,
    max_file_size: int,
    large_log_threshold: int,
    large_log_tail_hours: float,
) -> IngestionStats:
    stats = IngestionStats(archives=1)
    report_name = archive_path.name
    command_batch: list[CommandArtifact] = []
    log_index = 0
    command_index = 0
    members_seen = 0
    last_heartbeat = time.perf_counter()

    print(f"[progress] {report_name}: walking archive stream", flush=True)
    with tarfile.open(archive_path, "r|xz") as archive:
        for member in archive:
            members_seen += 1
            now = time.perf_counter()
            if members_seen == 1 or members_seen % 1000 == 0 or now - last_heartbeat >= 15:
                print(
                    f"[progress] {report_name}: scanned {members_seen} archive member(s); "
                    f"matched {log_index} log file(s), {command_index} command artifact(s)",
                    flush=True,
                )
                last_heartbeat = now

            if is_systemd_journal_member(member, max_file_size):
                log_index += 1
                rows = _ingest_systemd_journal_member(
                    conn=conn,
                    archive=archive,
                    member=member,
                    report_name=report_name,
                    log_index=log_index,
                    retain_recent_hours=(
                        large_log_tail_hours
                        if member.size > large_log_threshold
                        else None
                    ),
                )
                stats = stats.add(IngestionStats(log_files=1, log_rows=rows))
                continue

            if is_interesting_log_member(member, max_file_size):
                log_index += 1
                rows = _ingest_log_member(
                    conn=conn,
                    archive=archive,
                    member=member,
                    report_name=report_name,
                    log_index=log_index,
                    retain_recent_hours=(
                        large_log_tail_hours
                        if member.size > large_log_threshold
                        else None
                    ),
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
        f"{stats.command_files} command artifact(s), {members_seen} archive member(s) scanned",
        flush=True,
    )
    return stats


def _ingest_log_member(
    conn: duckdb.DuckDBPyConnection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    report_name: str,
    log_index: int,
    retain_recent_hours: float | None = None,
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
    recent_entries: deque[LogEntry] = deque()
    latest_timestamp = None
    if retain_recent_hours is not None:
        print(
            f"[progress] {report_name}: {source_file}: retaining only the "
            f"last {retain_recent_hours:g} hour(s) of timestamped records",
            flush=True,
        )
    for entry in parse_log_lines(
        iter_text_lines(archive, member), source_file, report_name
    ):
        if retain_recent_hours is not None:
            if entry.timestamp is None:
                continue
            latest_timestamp = max(latest_timestamp, entry.timestamp) if latest_timestamp else entry.timestamp
            recent_entries.append(entry)
            cutoff = latest_timestamp - timedelta(hours=retain_recent_hours)
            while recent_entries and recent_entries[0].timestamp < cutoff:
                recent_entries.popleft()
            continue
        batch.append(entry)
        if len(batch) >= BATCH_SIZE:
            rows += insert_logs(conn, batch)
            batch.clear()
            print(
                f"[progress] {report_name}: {source_file}: {rows} row(s) loaded",
                flush=True,
            )
    if retain_recent_hours is not None:
        if latest_timestamp is not None:
            cutoff = latest_timestamp - timedelta(hours=retain_recent_hours)
            batch = [entry for entry in recent_entries if entry.timestamp >= cutoff]
            for offset in range(0, len(batch), BATCH_SIZE):
                rows += insert_logs(conn, batch[offset : offset + BATCH_SIZE])
        else:
            print(
                f"[progress] {report_name}: {source_file}: no timestamped records; "
                "no rows retained from large file",
                flush=True,
            )
    else:
        rows += insert_logs(conn, batch)
    elapsed = time.perf_counter() - started
    print(
        f"[progress] {report_name}: finished {source_file}: "
        f"{rows} row(s) in {elapsed:.1f}s",
        flush=True,
    )
    return rows


def _journalctl_command(journal_file: Path) -> list[str]:
    """Build the local command that decodes one binary systemd journal."""
    configured = os.getenv("OSP_SOS_JOURNALCTL_COMMAND")
    if configured:
        return configured.split() + ["--no-pager", "--output=short-iso", "--file", str(journal_file)]
    if os.name != "nt":
        return ["journalctl", "--no-pager", "--output=short-iso", "--file", str(journal_file)]

    distro = os.getenv("OSP_SOS_WSL_DISTRO", "podman-machine-default")
    mapped = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "wslpath", "-a", str(journal_file)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    return [
        "wsl.exe", "-d", distro, "--", "journalctl", "--no-pager",
        "--output=short-iso", "--file", mapped,
    ]


def _ingest_systemd_journal_member(
    conn: duckdb.DuckDBPyConnection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    report_name: str,
    log_index: int,
    retain_recent_hours: float | None,
) -> int:
    source_file = normalized_member_name(member)
    print(
        f"[progress] {report_name}: decoding binary journal {log_index} "
        f"{source_file} ({member.size / 1024 / 1024:.1f} MB)",
        flush=True,
    )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise SosReportError(f"Unable to extract journal member: {member.name}")

    with tempfile.TemporaryDirectory(prefix="osp-sos-journal-") as temporary_directory:
        journal_file = Path(temporary_directory) / Path(source_file).name
        with extracted, journal_file.open("wb") as destination:
            shutil.copyfileobj(extracted, destination, length=1024 * 1024)

        process = subprocess.Popen(
            _journalctl_command(journal_file),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        rows = _insert_parsed_log_entries(
            conn,
            parse_log_lines(process.stdout, source_file, report_name),
            source_file,
            report_name,
            retain_recent_hours,
        )
        assert process.stderr is not None
        error_output = process.stderr.read().strip()
        if process.wait() != 0:
            raise SosReportError(
                f"journalctl could not decode {source_file}: {error_output or 'unknown error'}"
            )
    return rows


def _insert_parsed_log_entries(
    conn: duckdb.DuckDBPyConnection,
    entries,
    source_file: str,
    report_name: str,
    retain_recent_hours: float | None,
) -> int:
    """Insert parsed entries, retaining a rolling recent window when requested."""
    batch: list[LogEntry] = []
    rows = 0
    recent_entries: deque[LogEntry] = deque()
    latest_timestamp = None
    if retain_recent_hours is not None:
        print(
            f"[progress] {report_name}: {source_file}: retaining only the "
            f"last {retain_recent_hours:g} hour(s) of timestamped records",
            flush=True,
        )
    for entry in entries:
        if retain_recent_hours is not None:
            if entry.timestamp is None:
                continue
            latest_timestamp = max(latest_timestamp, entry.timestamp) if latest_timestamp else entry.timestamp
            recent_entries.append(entry)
            cutoff = latest_timestamp - timedelta(hours=retain_recent_hours)
            while recent_entries and recent_entries[0].timestamp < cutoff:
                recent_entries.popleft()
            continue
        batch.append(entry)
        if len(batch) >= BATCH_SIZE:
            rows += insert_logs(conn, batch)
            batch.clear()
    if retain_recent_hours is None:
        return rows + insert_logs(conn, batch)
    if latest_timestamp is None:
        print(
            f"[progress] {report_name}: {source_file}: no timestamped records; no rows retained from large file",
            flush=True,
        )
        return 0
    cutoff = latest_timestamp - timedelta(hours=retain_recent_hours)
    batch = [entry for entry in recent_entries if entry.timestamp >= cutoff]
    for offset in range(0, len(batch), BATCH_SIZE):
        rows += insert_logs(conn, batch[offset : offset + BATCH_SIZE])
    return rows


def ingest_sos_reports(
    reports_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
    clear_existing: bool = False,
    max_file_size_mb: int | None = 2048,
    force_reingest: bool = False,
    large_log_threshold_mb: float = DEFAULT_LARGE_LOG_THRESHOLD_MB,
    large_log_tail_hours: float = DEFAULT_LARGE_LOG_TAIL_HOURS,
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
    if large_log_threshold_mb <= 0:
        raise SosReportError("large_log_threshold_mb must be greater than zero")
    if large_log_tail_hours <= 0:
        raise SosReportError("large_log_tail_hours must be greater than zero")
    large_log_threshold = int(large_log_threshold_mb * 1024 * 1024)
    total = IngestionStats()

    print(f"[progress] Found {len(archive_paths)} SOS archive(s)", flush=True)
    with duckdb.connect(str(db_target)) as conn:
        ensure_schema(conn)
        if clear_existing:
            conn.execute("DELETE FROM os_logs")
            conn.execute("DELETE FROM os_commands")
            conn.execute("DELETE FROM ingested_reports")

        for archive_index, archive_path in enumerate(archive_paths, start=1):
            print(
                f"[progress] Checking duplicate registry for {archive_path.name}",
                flush=True,
            )
            archive_id = _archive_id(archive_path)
            if not force_reingest and archive_already_ingested(conn, archive_id):
                print(
                    f"[skip] {archive_path.name}: already ingested "
                    f"(archive_id={archive_id[:12]})",
                    flush=True,
                )
                continue

            print(
                f"[progress] Processing archive {archive_index}/{len(archive_paths)}: "
                f"{archive_path.name}",
                flush=True,
            )
            mark_archive_started(conn, archive_id, archive_path)
            try:
                archive_stats = _ingest_archive(
                    conn,
                    archive_path,
                    max_file_size,
                    large_log_threshold,
                    large_log_tail_hours,
                )
                removed_logs, removed_commands = dedupe_ingested_rows(conn)
                if removed_logs or removed_commands:
                    print(
                        f"[dedupe] Removed {removed_logs} duplicate log row(s) and "
                        f"{removed_commands} duplicate command row(s)",
                        flush=True,
                    )
                mark_archive_completed(
                    conn,
                    archive_id,
                    archive_stats.log_rows,
                    archive_stats.command_rows,
                )
                total = total.add(archive_stats)
            except Exception:
                mark_archive_failed(conn, archive_id)
                raise

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
