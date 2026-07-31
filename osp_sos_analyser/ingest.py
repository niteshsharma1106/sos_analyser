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
from dataclasses import replace
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
from .cluster_loader import (
    NodeManifest,
    absorb_manifest_member,
    is_cluster_manifest_member,
    new_node_manifest,
    note_service_from_path,
    resolve_cluster_id,
)
from .db import (
    archive_already_ingested,
    dedupe_ingested_rows,
    ensure_schema,
    insert_commands,
    insert_logs,
    mark_archive_completed,
    mark_archive_failed,
    mark_archive_started,
    stamp_report_identity,
    upsert_cluster_node,
)
from .evidence_index import build_evidence_index
from .relationship_graph import build_relationship_index
from .log_parser import parse_log_lines
from .models import CommandArtifact, IngestionStats, LogEntry

BATCH_SIZE = 1000
HASH_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_LARGE_LOG_THRESHOLD_MB = 30
DEFAULT_LARGE_LOG_TAIL_HOURS = 6


def _default_db_path(reports_dir: Path) -> Path:
    return reports_dir.parent / "sos_analysis.duckdb"


def _with_identity(entry: LogEntry, manifest: NodeManifest) -> LogEntry:
    return replace(
        entry,
        hostname=manifest.hostname,
        node_role=manifest.node_role,
        cluster_id=manifest.cluster_id,
        rhosp_version=manifest.rhosp_version or entry.rhosp_version,
    )


def _command_artifact(
    source_file: str,
    output: str,
    report_name: str,
    manifest: NodeManifest,
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
        rhosp_version=manifest.rhosp_version,
        hostname=manifest.hostname,
        node_role=manifest.node_role,
        cluster_id=manifest.cluster_id,
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
    *,
    archive_id: str,
    cluster_id: str,
) -> IngestionStats:
    stats = IngestionStats(archives=1)
    report_name = archive_path.name
    command_batch: list[CommandArtifact] = []
    log_index = 0
    command_index = 0
    members_seen = 0
    last_heartbeat = time.perf_counter()
    manifest = new_node_manifest(
        archive_path, archive_id=archive_id, cluster_id=cluster_id
    )

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

            manifest_text: str | None = None
            if is_cluster_manifest_member(member, max_file_size):
                manifest_text = absorb_manifest_member(manifest, archive, member)

            if is_systemd_journal_member(member, max_file_size):
                if manifest_text is not None:
                    # Binary journals are never text manifest members; keep structure clear.
                    pass
                log_index += 1
                source_file = normalized_member_name(member)
                note_service_from_path(manifest, source_file)
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
                    manifest=manifest,
                )
                stats = stats.add(IngestionStats(log_files=1, log_rows=rows))
                continue

            if is_interesting_log_member(member, max_file_size):
                if manifest_text is not None:
                    # Already consumed as text; skip re-read on streaming xz.
                    continue
                log_index += 1
                source_file = normalized_member_name(member)
                note_service_from_path(manifest, source_file)
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
                    manifest=manifest,
                )
                stats = stats.add(IngestionStats(log_files=1, log_rows=rows))
                continue

            if is_interesting_command_member(member, max_file_size):
                command_index += 1
                source_file = normalized_member_name(member)
                note_service_from_path(manifest, source_file)
                if command_index == 1 or command_index % 100 == 0:
                    print(
                        f"[progress] {report_name}: reading command artifact "
                        f"{command_index}: {source_file}",
                        flush=True,
                    )
                output = (
                    manifest_text
                    if manifest_text is not None
                    else read_text_member(archive, member)
                )
                command_batch.append(
                    _command_artifact(
                        source_file=source_file,
                        output=output,
                        report_name=report_name,
                        manifest=manifest,
                    )
                )
                if len(command_batch) >= BATCH_SIZE:
                    flushed = insert_commands(conn, command_batch)
                    stats = stats.add(IngestionStats(command_rows=flushed))
                    command_batch.clear()

    command_rows = insert_commands(conn, command_batch)
    stats = stats.add(
        IngestionStats(
            command_files=command_index,
            command_rows=command_rows,
        )
    )
    stamp_report_identity(
        conn,
        report_name=report_name,
        hostname=manifest.hostname,
        node_role=manifest.node_role,
        cluster_id=manifest.cluster_id,
        rhosp_version=manifest.rhosp_version,
    )
    upsert_cluster_node(conn, manifest.as_row())
    print(
        f"[progress] {report_name}: cluster node "
        f"host={manifest.hostname} role={manifest.node_role} "
        f"cluster_id={manifest.cluster_id}",
        flush=True,
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
    manifest: NodeManifest | None = None,
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
        if manifest is not None:
            entry = _with_identity(entry, manifest)
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


# After an infrastructure-level journalctl/WSL failure, skip remaining binary
# journals for this process so one bad WSL setup cannot abort the whole ingest.
_JOURNAL_DECODE_SKIP_REASON: str | None = None


def windows_path_to_wsl(path: Path | str) -> str:
    """Convert a Windows path to the usual WSL `/mnt/<drive>/...` form."""
    text = str(path)
    if len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = f"/{rest}"
        return f"/mnt/{drive}{rest}"
    return str(Path(path).resolve()).replace("\\", "/")


def _wsl_prefix(distro: str | None) -> list[str]:
    if distro:
        return ["wsl.exe", "-d", distro, "--"]
    return ["wsl.exe", "--"]


def _resolve_wsl_distro() -> str | None:
    """Return configured WSL distro, or None to use the user's default distro.

    Do not default to podman-machine-default: that VM often lacks journalctl
    and/or `/mnt/<drive>` mounts for Windows temp paths.
    """
    configured = os.getenv("OSP_SOS_WSL_DISTRO")
    if configured is None:
        return None
    configured = configured.strip()
    return configured or None


def _map_windows_path_for_wsl(journal_file: Path, distro: str | None) -> str:
    cmd = [*_wsl_prefix(distro), "wslpath", "-a", str(journal_file)]
    try:
        mapped = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        ).stdout.strip()
        if mapped:
            return mapped
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return windows_path_to_wsl(journal_file)


def _journalctl_command(journal_file: Path) -> list[str]:
    """Build the local command that decodes one binary systemd journal."""
    configured = os.getenv("OSP_SOS_JOURNALCTL_COMMAND")
    if configured:
        return configured.split() + [
            "--no-pager",
            "--output=short-iso",
            "--file",
            str(journal_file),
        ]
    if os.name != "nt":
        return [
            "journalctl",
            "--no-pager",
            "--output=short-iso",
            "--file",
            str(journal_file),
        ]

    distro = _resolve_wsl_distro()
    mapped = _map_windows_path_for_wsl(journal_file, distro)
    return [
        *_wsl_prefix(distro),
        "journalctl",
        "--no-pager",
        "--output=short-iso",
        "--file",
        mapped,
    ]


def _discard_archive_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> None:
    """Consume member bytes so streaming `r|xz` readers stay aligned."""
    extracted = archive.extractfile(member)
    if extracted is None:
        return
    with extracted:
        while extracted.read(1024 * 1024):
            pass


def _ingest_systemd_journal_member(
    conn: duckdb.DuckDBPyConnection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    report_name: str,
    log_index: int,
    retain_recent_hours: float | None,
    manifest: NodeManifest | None = None,
) -> int:
    global _JOURNAL_DECODE_SKIP_REASON

    source_file = normalized_member_name(member)
    size_mb = member.size / 1024 / 1024

    if os.getenv("OSP_SOS_SKIP_JOURNALS", "").strip().lower() in {"1", "true", "yes"}:
        print(
            f"[progress] {report_name}: skipping binary journal {log_index} "
            f"{source_file} (OSP_SOS_SKIP_JOURNALS is set)",
            flush=True,
        )
        _discard_archive_member(archive, member)
        return 0

    if _JOURNAL_DECODE_SKIP_REASON is not None:
        print(
            f"[progress] {report_name}: skipping binary journal {log_index} "
            f"{source_file}: {_JOURNAL_DECODE_SKIP_REASON}",
            flush=True,
        )
        _discard_archive_member(archive, member)
        return 0

    print(
        f"[progress] {report_name}: decoding binary journal {log_index} "
        f"{source_file} ({size_mb:.1f} MB)",
        flush=True,
    )
    extracted = archive.extractfile(member)
    if extracted is None:
        raise SosReportError(f"Unable to extract journal member: {member.name}")

    try:
        with tempfile.TemporaryDirectory(prefix="osp-sos-journal-") as temporary_directory:
            journal_file = Path(temporary_directory) / Path(source_file).name
            with extracted, journal_file.open("wb") as destination:
                shutil.copyfileobj(extracted, destination, length=1024 * 1024)

            command = _journalctl_command(journal_file)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                assert process.stdout is not None
                rows = _insert_parsed_log_entries(
                    conn,
                    parse_log_lines(process.stdout, source_file, report_name),
                    source_file,
                    report_name,
                    retain_recent_hours,
                    manifest=manifest,
                )
                assert process.stderr is not None
                error_output = process.stderr.read().strip()
                returncode = process.wait()
            finally:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if returncode != 0:
                detail = error_output or f"exit status {returncode}"
                raise SosReportError(
                    f"journalctl could not decode {source_file}: {detail}"
                )
        return rows
    except (SosReportError, OSError, subprocess.SubprocessError) as exc:
        # Keep text logs + sos_commands; binary journals need a working journalctl.
        _JOURNAL_DECODE_SKIP_REASON = str(exc)
        print(
            f"[progress] {report_name}: skipping binary journals after decode failure: {exc}\n"
            f"[progress] Tip: set OSP_SOS_WSL_DISTRO to a distro with journalctl "
            f"(e.g. Ubuntu), or OSP_SOS_SKIP_JOURNALS=1 to silence this, "
            f"or OSP_SOS_JOURNALCTL_COMMAND to a custom decoder.",
            flush=True,
        )
        return 0


def _insert_parsed_log_entries(
    conn: duckdb.DuckDBPyConnection,
    entries,
    source_file: str,
    report_name: str,
    retain_recent_hours: float | None,
    manifest: NodeManifest | None = None,
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
        if manifest is not None:
            entry = _with_identity(entry, manifest)
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
    cluster_id: str | None = None,
) -> Path:
    """Ingest one or more RHOSP 17.x SOS report tar.xz archives into DuckDB."""
    global _JOURNAL_DECODE_SKIP_REASON
    _JOURNAL_DECODE_SKIP_REASON = None

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
            conn.execute("DELETE FROM cluster_nodes")
            conn.execute("DELETE FROM entities")
            conn.execute("DELETE FROM entity_mentions")
            conn.execute("DELETE FROM entity_relationships")

        resolved_cluster_id = resolve_cluster_id(
            conn, root, clear_existing=clear_existing, explicit=cluster_id
        )
        print(f"[progress] Cluster id: {resolved_cluster_id}", flush=True)

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
                    archive_id=archive_id,
                    cluster_id=resolved_cluster_id,
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

        index_stats = build_evidence_index(conn)
        print(
            "[progress] Evidence index: "
            f"{index_stats['entities']} entit(y/ies), {index_stats['mentions']} mention(s)",
            flush=True,
        )
        graph_stats = build_relationship_index(conn)
        print(
            "[progress] Relationship graph: "
            f"{graph_stats['relationships']} edge(s), "
            f"{graph_stats['chassis_entities']} chassis entit(y/ies)",
            flush=True,
        )

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
