# ingest.py
from __future__ import annotations

import hashlib
import io
import os
import tempfile
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

from .archive_reader import (
    SosReportError,
    file_size_limit,
    is_interesting_command_member,
    is_interesting_log_member,
    iter_report_archives,
    member_content_signature_key,
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


def _signature_entry(
    member: tarfile.TarInfo, content_digest: str = ""
) -> tuple[str, int, str] | None:
    key = member_content_signature_key(member)
    if key is None:
        return None
    relative_name, size = key
    return relative_name, size, content_digest


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _ingest_archive(
    conn: duckdb.DuckDBPyConnection,
    archive_path: Path,
    max_file_size: int,
) -> tuple[IngestionStats, frozenset[tuple[str, int, str]]]:
    stats = IngestionStats(archives=1)
    report_name = archive_path.name
    command_batch: list[CommandArtifact] = []
    log_index = 0
    command_index = 0
    signature: set[tuple[str, int, str]] = set()

    with tarfile.open(archive_path, "r|xz") as archive:
        for member in archive:
            if is_interesting_log_member(member, max_file_size):
                log_index += 1
                rows, content_digest = _ingest_log_member(
                    conn=conn,
                    archive=archive,
                    member=member,
                    report_name=report_name,
                    log_index=log_index,
                )
                entry = _signature_entry(member, content_digest)
                if entry is not None:
                    signature.add(entry)
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
                output = read_text_member(archive, member)
                entry = _signature_entry(member, _digest_text(output))
                if entry is not None:
                    signature.add(entry)
                command_batch.append(
                    _command_artifact(
                        source_file=source_file,
                        output=output,
                        report_name=report_name,
                    )
                )
                if len(command_batch) >= BATCH_SIZE:
                    stats = stats.add(
                        IngestionStats(command_rows=insert_commands(conn, command_batch))
                    )
                    command_batch.clear()
                continue

            entry = _signature_entry(member)
            if entry is not None:
                signature.add(entry)

    if command_batch:
        stats = stats.add(IngestionStats(command_rows=insert_commands(conn, command_batch)))
    stats = stats.add(IngestionStats(command_files=command_index))
    print(
        f"[progress] {report_name}: finished archive: {stats.log_files} log file(s), "
        f"{stats.command_files} command artifact(s)",
        flush=True,
    )
    return stats, frozenset(signature)


def _ingest_log_member(
    conn: duckdb.DuckDBPyConnection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    report_name: str,
    log_index: int,
) -> tuple[int, str]:
    source_file = normalized_member_name(member)
    print(
        f"[progress] {report_name}: parsing log {log_index} "
        f"{source_file} ({member.size / 1024 / 1024:.1f} MB)",
        flush=True,
    )

    # Read once so we can fingerprint content without a second decompress pass.
    text = read_text_member(archive, member)
    content_digest = _digest_text(text)

    batch: list[LogEntry] = []
    rows = 0
    started = time.perf_counter()
    for entry in parse_log_lines(io.StringIO(text), source_file, report_name):
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
    return rows, content_digest


def _delete_report_rows(conn: duckdb.DuckDBPyConnection, report_name: str) -> None:
    conn.execute("DELETE FROM os_logs WHERE report_name = ?", [report_name])
    conn.execute("DELETE FROM os_commands WHERE report_name = ?", [report_name])


def _ingest_archive_to_temp(
    archive_path: Path,
    max_file_size: int,
) -> tuple[Path, IngestionStats, frozenset[tuple[str, int, str]]]:
    handle = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
    handle.close()
    temp_path = Path(handle.name)
    temp_path.unlink(missing_ok=True)
    try:
        with duckdb.connect(str(temp_path)) as conn:
            ensure_schema(conn)
            stats, signature = _ingest_archive(conn, archive_path, max_file_size)
        return temp_path, stats, signature
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _merge_temp_db(conn: duckdb.DuckDBPyConnection, temp_path: Path, alias: str) -> None:
    conn.execute(f"ATTACH '{temp_path.as_posix()}' AS {alias} (READ_ONLY)")
    try:
        conn.execute(
            f"""
            INSERT INTO os_logs
            SELECT * FROM {alias}.os_logs
            """
        )
        conn.execute(
            f"""
            INSERT INTO os_commands
            SELECT * FROM {alias}.os_commands
            """
        )
    finally:
        conn.execute(f"DETACH {alias}")


def ingest_sos_reports(
    reports_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
    clear_existing: bool = False,
    max_file_size_mb: int | None = 25,
    max_workers: int | None = None,
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
    workers = max_workers if max_workers is not None else min(4, max(1, len(archive_paths)))
    total = IngestionStats()
    seen_signatures: dict[frozenset[tuple[str, int, str]], str] = {}

    print(f"[progress] Found {len(archive_paths)} SOS archive(s)", flush=True)
    with duckdb.connect(str(db_target)) as conn:
        ensure_schema(conn)
        if clear_existing:
            conn.execute("DELETE FROM os_logs")
            conn.execute("DELETE FROM os_commands")

        if len(archive_paths) == 1 or workers <= 1:
            for archive_index, archive_path in enumerate(archive_paths, start=1):
                print(
                    f"[progress] Processing archive {archive_index}/{len(archive_paths)}: "
                    f"{archive_path.name}",
                    flush=True,
                )
                stats, signature = _ingest_archive(conn, archive_path, max_file_size)
                existing = seen_signatures.get(signature)
                if existing is not None and signature:
                    print(
                        f"[skip] {archive_path.name} matches content of {existing}; "
                        "rolling back duplicate rows.",
                        flush=True,
                    )
                    _delete_report_rows(conn, archive_path.name)
                    continue
                if signature:
                    seen_signatures[signature] = archive_path.name
                total = total.add(stats)
        else:
            print(
                f"[progress] Parallel ingest with up to {workers} worker(s)",
                flush=True,
            )
            temp_results: list[
                tuple[Path, Path, IngestionStats, frozenset[tuple[str, int, str]]]
            ] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_ingest_archive_to_temp, archive_path, max_file_size): archive_path
                    for archive_path in archive_paths
                }
                for future in as_completed(futures):
                    archive_path = futures[future]
                    temp_path, stats, signature = future.result()
                    temp_results.append((archive_path, temp_path, stats, signature))

            # Merge in deterministic archive-name order for stable tests/logs.
            temp_results.sort(key=lambda item: item[0].name)
            for merge_index, (archive_path, temp_path, stats, signature) in enumerate(
                temp_results, start=1
            ):
                try:
                    existing = seen_signatures.get(signature)
                    if existing is not None and signature:
                        print(
                            f"[skip] {archive_path.name} matches content of {existing}; "
                            "not merging duplicate.",
                            flush=True,
                        )
                        continue
                    alias = f"tmp_archive_{merge_index}"
                    print(
                        f"[progress] Merging {archive_path.name} into {db_target.name}",
                        flush=True,
                    )
                    _merge_temp_db(conn, temp_path, alias)
                    if signature:
                        seen_signatures[signature] = archive_path.name
                    total = total.add(stats)
                finally:
                    temp_path.unlink(missing_ok=True)

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
