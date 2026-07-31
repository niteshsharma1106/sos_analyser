# evidence_index.py — deterministic entity + evidence index over DuckDB rows.
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .context_pack import DIGEST_MESSAGE_CHARS, truncate_text

UUID_RE = re.compile(
    r"\b(?P<id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)
REQUEST_ID_RE = re.compile(r"\b(?P<id>req-[0-9a-fA-F-]{8,})\b")

# Prefer message cues; avoid path basenames like server.log matching "server".
TYPE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("port", (" port", "port ", "port-", "port:", "vif", "binding")),
    ("volume", (" volume", "volume ", "volume-", "volume:", "attach", "detach")),
    ("network", (" network", "network ", "network-", "subnet")),
    ("router", (" router", "router ", "router-")),
    ("image", (" image", "image ", "image-", "glance")),
    ("instance", (" instance", "instance-", "server_id", "vm ", "vm-", "spawn")),
    ("request", ("request", "req-")),
)


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    mention_count: int
    first_seen: datetime | None
    last_seen: datetime | None


@dataclass(frozen=True)
class EvidenceMention:
    entity_id: str
    entity_type: str
    timestamp: datetime | None
    hostname: str
    service: str
    level: str
    source_file: str
    report_name: str
    message_excerpt: str


def classify_entity_type(entity_id: str, message: str, service: str = "") -> str:
    if entity_id.lower().startswith("req-"):
        return "request"
    lowered = f" {message.lower()} {service.lower()} "
    for entity_type, hints in TYPE_HINTS:
        if any(hint in lowered for hint in hints):
            return entity_type
    return "unknown"


def extract_entity_ids(text: str) -> list[str]:
    found = [m.group("id") for m in UUID_RE.finditer(text or "")]
    found.extend(m.group("id") for m in REQUEST_ID_RE.finditer(text or ""))
    # Preserve order, case-fold UUIDs for canonical identity.
    canonical: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.lower() if not item.lower().startswith("req-") else item.lower()
        if key in seen:
            continue
        seen.add(key)
        canonical.append(key if not item.lower().startswith("req-") else item)
    return canonical


def build_evidence_index(conn: Any, *, report_name: str | None = None) -> dict[str, int]:
    """Rebuild entities + entity_mentions from os_logs (and hosts from cluster_nodes)."""
    import sys
    import time

    if report_name:
        conn.execute("DELETE FROM entity_mentions WHERE report_name = ?", [report_name])
    else:
        conn.execute("DELETE FROM entity_mentions")
        conn.execute("DELETE FROM entities")

    # Full-table Python scans feel "hung" on large SOS DBs. Prefer rows that are
    # likely RCA-relevant: errors/warnings or messages that look like they carry IDs.
    where_bits = [
        "message IS NOT NULL",
        "length(message) > 0",
        """(
            upper(COALESCE(level, '')) IN ('ERROR', 'CRITICAL', 'WARNING', 'FATAL')
            OR regexp_matches(message, '(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
            OR message ILIKE '%req-%'
            OR message ILIKE '%instance%'
            OR message ILIKE '%port_id%'
            OR message ILIKE '%volume%'
            OR message ILIKE '%uuid%'
            OR service IN ('nova', 'neutron', 'cinder', 'ovn', 'glance', 'keystone')
        )""",
    ]
    params: list[object] = []
    if report_name:
        where_bits.append("report_name = ?")
        params.append(report_name)
    where_sql = " AND ".join(where_bits)

    total_logs = int(
        conn.execute(f"SELECT COUNT(*) FROM os_logs WHERE {where_sql}", params).fetchone()[0]
    )
    print(
        f"[progress] Evidence index: scanning {total_logs} candidate log row(s)",
        flush=True,
    )

    entity_meta: dict[str, dict[str, Any]] = {}
    mention_rows: list[tuple[object, ...]] = []
    mentions_per_entity: dict[str, int] = {}
    max_mentions_per_entity = 80
    insert_batch_size = 2000
    page_size = 2000
    scanned = 0
    started = time.perf_counter()

    def _flush_mentions(force: bool = False) -> None:
        nonlocal mention_rows
        if not mention_rows:
            return
        if not force and len(mention_rows) < insert_batch_size:
            return
        conn.executemany(
            """
            INSERT INTO entity_mentions (
                entity_id, entity_type, timestamp, hostname, service, level,
                source_file, report_name, message_excerpt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            mention_rows,
        )
        mention_rows = []

    def _process_row(row: Sequence[Any]) -> None:
        nonlocal scanned
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            raise RuntimeError(
                f"Evidence index got unexpected row shape {row!r} (len="
                f"{len(row) if isinstance(row, (list, tuple)) else 'n/a'}). "
                "This usually means the DuckDB result cursor was invalidated by "
                "a write on the same connection."
            )
        timestamp, level, service, message, source_file, report, hostname = row[:7]
        scanned += 1
        text = str(message or "")
        for entity_id in extract_entity_ids(text):
            if mentions_per_entity.get(entity_id, 0) >= max_mentions_per_entity:
                # Still update aggregate counts/timestamps without storing more digests.
                meta = entity_meta.get(entity_id)
                if meta is not None:
                    meta["mention_count"] += 1
                    if timestamp is not None:
                        if meta["first_seen"] is None or timestamp < meta["first_seen"]:
                            meta["first_seen"] = timestamp
                        if meta["last_seen"] is None or timestamp > meta["last_seen"]:
                            meta["last_seen"] = timestamp
                continue
            entity_type = classify_entity_type(entity_id, text, str(service or ""))
            meta = entity_meta.setdefault(
                entity_id,
                {
                    "entity_type": entity_type,
                    "mention_count": 0,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                },
            )
            if meta["entity_type"] == "unknown" and entity_type != "unknown":
                meta["entity_type"] = entity_type
            meta["mention_count"] += 1
            if timestamp is not None:
                if meta["first_seen"] is None or timestamp < meta["first_seen"]:
                    meta["first_seen"] = timestamp
                if meta["last_seen"] is None or timestamp > meta["last_seen"]:
                    meta["last_seen"] = timestamp
            mention_rows.append(
                (
                    entity_id,
                    entity_type,
                    timestamp,
                    str(hostname or ""),
                    str(service or ""),
                    str(level or ""),
                    str(source_file or ""),
                    str(report or ""),
                    truncate_text(text, DIGEST_MESSAGE_CHARS),
                )
            )
            mentions_per_entity[entity_id] = mentions_per_entity.get(entity_id, 0) + 1
        if scanned == 1 or scanned % 20000 == 0:
            print(
                f"[progress] Evidence index: scanned {scanned}/{total_logs} row(s); "
                f"{len(entity_meta)} entit(y/ies)",
                flush=True,
            )

    # IMPORTANT: Do not fetchmany() + INSERT on the same DuckDB connection.
    # A write invalidates the open result; the next fetchmany() returns junk
    # like [(1,)] (rowcounts) and unpacking crashes with "expected 7, got 1".
    # Materialize a numbered temp table once, then page by rid ranges.
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _evidence_scan AS
        SELECT
            row_number() OVER () AS rid,
            timestamp,
            level,
            service,
            message,
            source_file,
            report_name,
            COALESCE(hostname, '') AS hostname
        FROM os_logs
        WHERE {where_sql}
        """,
        params,
    )
    try:
        max_rid = int(
            conn.execute("SELECT COALESCE(MAX(rid), 0) FROM _evidence_scan").fetchone()[0]
        )
        rid_start = 1
        while rid_start <= max_rid:
            chunk = conn.execute(
                """
                SELECT timestamp, level, service, message, source_file, report_name, hostname
                FROM _evidence_scan
                WHERE rid >= ? AND rid < ?
                ORDER BY rid
                """,
                [rid_start, rid_start + page_size],
            ).fetchall()
            if not chunk:
                break
            for row in chunk:
                _process_row(row)
            # Cursor is closed after fetchall — safe to insert on this connection.
            _flush_mentions(force=True)
            rid_start += page_size
    finally:
        conn.execute("DROP TABLE IF EXISTS _evidence_scan")

    _flush_mentions(force=True)

    # Register hosts as first-class entities.
    host_mentions: list[tuple[object, ...]] = []
    for hostname, role, cluster_id in conn.execute(
        "SELECT hostname, node_role, cluster_id FROM cluster_nodes"
    ).fetchall():
        if not hostname:
            continue
        entity_id = str(hostname)
        meta = entity_meta.setdefault(
            entity_id,
            {
                "entity_type": "host",
                "mention_count": 0,
                "first_seen": None,
                "last_seen": None,
            },
        )
        meta["entity_type"] = "host"
        meta["mention_count"] = max(int(meta["mention_count"]), 1)
        host_mentions.append(
            (
                entity_id,
                "host",
                None,
                entity_id,
                "system",
                "INFO",
                f"cluster_nodes/{role}",
                str(cluster_id or ""),
                f"Host {entity_id} role={role}",
            )
        )
    if host_mentions:
        conn.executemany(
            """
            INSERT INTO entity_mentions (
                entity_id, entity_type, timestamp, hostname, service, level,
                source_file, report_name, message_excerpt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            host_mentions,
        )

    if report_name:
        touched = sorted(entity_meta)
        for entity_id in touched:
            conn.execute("DELETE FROM entities WHERE entity_id = ?", [entity_id])
    else:
        conn.execute("DELETE FROM entities")

    entity_rows = [
        (
            entity_id,
            meta["entity_type"],
            int(meta["mention_count"]),
            meta["first_seen"],
            meta["last_seen"],
        )
        for entity_id, meta in entity_meta.items()
    ]
    for offset in range(0, len(entity_rows), insert_batch_size):
        conn.executemany(
            """
            INSERT INTO entities (
                entity_id, entity_type, mention_count, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?)
            """,
            entity_rows[offset : offset + insert_batch_size],
        )

    mention_count = int(conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0])
    elapsed = time.perf_counter() - started
    print(
        f"[progress] Evidence index complete: {len(entity_rows)} entit(y/ies), "
        f"{mention_count} mention(s) in {elapsed:.1f}s",
        flush=True,
    )
    sys.stdout.flush()
    return {"entities": len(entity_rows), "mentions": mention_count}

def get_cluster_manifest(conn: Any) -> list[dict[str, Any]]:
    from .cluster_loader import infer_node_role

    rows = conn.execute(
        """
        SELECT cluster_id, hostname, node_role, rhosp_version, services,
               archive_name, archive_id
        FROM cluster_nodes
        ORDER BY node_role, hostname
        """
    ).fetchall()
    result: list[dict[str, Any]] = []
    for cluster_id, hostname, node_role, rhosp_version, services, archive_name, archive_id in rows:
        role = str(node_role or "unknown")
        inferred = infer_node_role(str(hostname or ""), str(archive_name or ""))
        if role.lower() in {"", "unknown"} and inferred != "unknown":
            role = inferred
        result.append(
            {
                "cluster_id": cluster_id,
                "hostname": hostname,
                "node_role": role,
                "rhosp_version": rhosp_version,
                "services": [part for part in str(services or "").split(",") if part],
                "archive_name": archive_name,
                "archive_id": archive_id,
            }
        )
    return result


def hostnames_for_node_roles(conn: Any, node_roles: Sequence[str]) -> list[str]:
    """Resolve role filters using stored + inferred roles (works on read-only DBs)."""
    wanted = {str(role).strip().lower() for role in node_roles if str(role).strip()}
    if not wanted:
        return []
    return [
        str(node["hostname"])
        for node in get_cluster_manifest(conn)
        if node.get("hostname") and str(node.get("node_role") or "").lower() in wanted
    ]


def get_entity(conn: Any, entity_id: str) -> EntityRecord | None:
    key = entity_id.lower() if not entity_id.lower().startswith("req-") else entity_id
    row = conn.execute(
        """
        SELECT entity_id, entity_type, mention_count, first_seen, last_seen
        FROM entities
        WHERE lower(entity_id) = lower(?)
        LIMIT 1
        """,
        [key],
    ).fetchone()
    if not row:
        return None
    return EntityRecord(
        entity_id=str(row[0]),
        entity_type=str(row[1]),
        mention_count=int(row[2] or 0),
        first_seen=row[3],
        last_seen=row[4],
    )


def get_evidence(
    conn: Any,
    entity_id: str,
    *,
    limit: int = 50,
    services: Sequence[str] = (),
    hostnames: Sequence[str] = (),
    node_roles: Sequence[str] = (),
) -> list[EvidenceMention]:
    clauses = ["lower(entity_id) = lower(?)"]
    params: list[object] = [entity_id]
    if services:
        clauses.append(f"service IN ({', '.join('?' for _ in services)})")
        params.extend(services)
    if hostnames:
        clauses.append(
            f"lower(hostname) IN ({', '.join('?' for _ in hostnames)})"
        )
        params.extend(h.lower() for h in hostnames)
    if node_roles:
        role_hosts = hostnames_for_node_roles(conn, node_roles)
        if not role_hosts:
            return []
        clauses.append(
            f"lower(hostname) IN ({', '.join('?' for _ in role_hosts)})"
        )
        params.extend(h.lower() for h in role_hosts)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT entity_id, entity_type, timestamp, hostname, service, level,
               source_file, report_name, message_excerpt
        FROM entity_mentions
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp NULLS LAST
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        EvidenceMention(
            entity_id=str(row[0]),
            entity_type=str(row[1] or "unknown"),
            timestamp=row[2],
            hostname=str(row[3] or ""),
            service=str(row[4] or ""),
            level=str(row[5] or ""),
            source_file=str(row[6] or ""),
            report_name=str(row[7] or ""),
            message_excerpt=str(row[8] or ""),
        )
        for row in rows
    ]


def get_host_activity(
    conn: Any,
    hostname: str,
    *,
    services: Sequence[str] = (),
    levels: Sequence[str] = (),
    limit: int = 40,
) -> list[EvidenceMention]:
    """Return recent log digests for one host (multi-node RCA helper)."""
    clauses = ["lower(COALESCE(hostname, '')) = lower(?)"]
    params: list[object] = [hostname]
    if services:
        clauses.append(f"service IN ({', '.join('?' for _ in services)})")
        params.extend(services)
    if levels:
        clauses.append(f"upper(level) IN ({', '.join('?' for _ in levels)})")
        params.extend(level.upper() for level in levels)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT timestamp, COALESCE(hostname, ''), service, level, message,
               source_file, report_name
        FROM os_logs
        WHERE {' AND '.join(clauses)}
        ORDER BY
          CASE upper(level)
            WHEN 'CRITICAL' THEN 0 WHEN 'ERROR' THEN 1
            WHEN 'WARNING' THEN 2 ELSE 3
          END,
          timestamp NULLS LAST
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        EvidenceMention(
            entity_id=hostname,
            entity_type="host",
            timestamp=row[0],
            hostname=str(row[1] or hostname),
            service=str(row[2] or ""),
            level=str(row[3] or ""),
            source_file=str(row[5] or ""),
            report_name=str(row[6] or ""),
            message_excerpt=truncate_text(str(row[4] or ""), DIGEST_MESSAGE_CHARS),
        )
        for row in rows
    ]


def compare_node_activity(
    conn: Any,
    *,
    services: Sequence[str] = (),
    levels: Sequence[str] = ("CRITICAL", "ERROR", "WARNING"),
    cluster_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Aggregate log severity counts by hostname/role for cross-node comparison."""
    clauses: list[str] = ["COALESCE(l.hostname, '') <> ''"]
    params: list[object] = []
    if services:
        clauses.append(f"l.service IN ({', '.join('?' for _ in services)})")
        params.extend(services)
    if levels:
        clauses.append(f"upper(l.level) IN ({', '.join('?' for _ in levels)})")
        params.extend(level.upper() for level in levels)
    if cluster_id:
        clauses.append("l.cluster_id = ?")
        params.append(cluster_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            COALESCE(l.hostname, '') AS hostname,
            COALESCE(n.node_role, l.node_role, '') AS node_role,
            COALESCE(l.service, '') AS service,
            upper(COALESCE(l.level, '')) AS level,
            COUNT(*) AS event_count,
            MIN(l.timestamp) AS first_seen,
            MAX(l.timestamp) AS last_seen
        FROM os_logs l
        LEFT JOIN cluster_nodes n
          ON lower(n.hostname) = lower(l.hostname)
        WHERE {' AND '.join(clauses)}
        GROUP BY 1, 2, 3, 4
        ORDER BY event_count DESC, hostname, service, level
        LIMIT ?
        """,
        params,
    ).fetchall()
    from .cluster_loader import infer_node_role

    out: list[dict[str, Any]] = []
    for row in rows:
        hostname = str(row[0] or "")
        role = str(row[1] or "")
        if role.lower() in {"", "unknown"}:
            role = infer_node_role(hostname)
        out.append(
            {
                "hostname": hostname,
                "node_role": role,
                "service": str(row[2] or ""),
                "level": str(row[3] or ""),
                "event_count": int(row[4] or 0),
                "first_seen": row[5],
                "last_seen": row[6],
            }
        )
    return out


def parse_search_terms(search_terms: str | Sequence[str]) -> tuple[list[str], bool]:
    """
    Parse agent search text into terms.

    Returns (terms, or_mode). When the agent writes ``reboot OR kernel OR panic``,
    terms are alternatives (OR). Plain whitespace-separated terms stay AND.
    """
    if isinstance(search_terms, (list, tuple)):
        raw = " ".join(str(t) for t in search_terms)
    else:
        raw = str(search_terms or "")
    raw = raw.strip()
    if not raw:
        return [], False
    or_mode = bool(re.search(r"\bOR\b|\|", raw, flags=re.I))
    if or_mode:
        parts = re.split(r"\s+OR\s+|\s*\|\s*|,", raw, flags=re.I)
    else:
        parts = raw.split()
    skip = {"AND", "OR", "NOT", "THE", "A", "AN"}
    terms: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = part.strip().strip("\"'()[]")
        if not token or token.upper() in skip:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(token)
    return terms, or_mode


def search_logs_by_node(
    conn: Any,
    *,
    hostnames: Sequence[str] = (),
    node_roles: Sequence[str] = (),
    services: Sequence[str] = (),
    search_terms: Sequence[str] | str = (),
    resource_id: str = "",
    limit: int = 30,
    term_or: bool | None = None,
) -> list[dict[str, Any]]:
    """Raw log search with optional host/role scope for multi-node investigations."""
    clauses: list[str] = []
    params: list[object] = []
    if hostnames:
        clauses.append(
            f"lower(COALESCE(hostname, '')) IN ({', '.join('?' for _ in hostnames)})"
        )
        params.extend(h.lower() for h in hostnames)
    if node_roles:
        role_hosts = hostnames_for_node_roles(conn, node_roles)
        if not role_hosts:
            return []
        clauses.append(
            f"lower(COALESCE(hostname, '')) IN ({', '.join('?' for _ in role_hosts)})"
        )
        params.extend(h.lower() for h in role_hosts)
    if services:
        clauses.append(f"service IN ({', '.join('?' for _ in services)})")
        params.extend(services)
    if resource_id:
        clauses.append("message ILIKE ?")
        params.append(f"%{resource_id}%")

    if isinstance(search_terms, str):
        terms, detected_or = parse_search_terms(search_terms)
    else:
        # Already a sequence: treat as AND unless caller sets term_or / embeds OR text.
        joined = " ".join(str(t) for t in search_terms)
        if re.search(r"\bOR\b|\|", joined, flags=re.I):
            terms, detected_or = parse_search_terms(joined)
        else:
            terms = [str(t).strip() for t in search_terms if str(t).strip()]
            detected_or = False
    use_or = detected_or if term_or is None else bool(term_or)
    if terms:
        term_sql = " OR ".join("message ILIKE ?" for _ in terms) if use_or else " AND ".join(
            "message ILIKE ?" for _ in terms
        )
        clauses.append(f"({term_sql})")
        params.extend(f"%{term}%" for term in terms)

    where = " AND ".join(clauses) if clauses else "1=1"
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT timestamp, COALESCE(hostname, ''), COALESCE(node_role, ''),
               service, level, message, source_file, report_name
        FROM os_logs
        WHERE {where}
        ORDER BY
          CASE upper(level)
            WHEN 'CRITICAL' THEN 0 WHEN 'ERROR' THEN 1
            WHEN 'WARNING' THEN 2 ELSE 3
          END,
          timestamp NULLS LAST
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "timestamp": row[0],
            "hostname": str(row[1] or ""),
            "node_role": str(row[2] or ""),
            "service": str(row[3] or ""),
            "level": str(row[4] or ""),
            "message": str(row[5] or ""),
            "source_file": str(row[6] or ""),
            "report_name": str(row[7] or ""),
        }
        for row in rows
    ]


def search_commands_by_node(
    conn: Any,
    *,
    hostnames: Sequence[str] = (),
    command_patterns: Sequence[str] = (),
    search_terms: Sequence[str] | str = (),
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search sos_commands artifacts (dmesg, last, journalctl, uptime, ...)."""
    clauses: list[str] = []
    params: list[object] = []
    if hostnames:
        clauses.append(
            f"lower(COALESCE(hostname, '')) IN ({', '.join('?' for _ in hostnames)})"
        )
        params.extend(h.lower() for h in hostnames)
    if command_patterns:
        pattern_sql = " OR ".join(
            "(command ILIKE ? OR source_file ILIKE ?)" for _ in command_patterns
        )
        clauses.append(f"({pattern_sql})")
        for pattern in command_patterns:
            like = f"%{pattern}%"
            params.extend([like, like])
    terms, use_or = parse_search_terms(search_terms)
    if terms:
        term_sql = " OR ".join("output ILIKE ?" for _ in terms) if use_or else " AND ".join(
            "output ILIKE ?" for _ in terms
        )
        clauses.append(f"({term_sql})")
        params.extend(f"%{term}%" for term in terms)
    where = " AND ".join(clauses) if clauses else "1=1"
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT COALESCE(hostname, ''), COALESCE(command, ''), COALESCE(source_file, ''),
               COALESCE(service, ''), COALESCE(output, ''), COALESCE(report_name, '')
        FROM os_commands
        WHERE {where}
        ORDER BY source_file
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "hostname": str(row[0] or ""),
            "command": str(row[1] or ""),
            "source_file": str(row[2] or ""),
            "service": str(row[3] or ""),
            "output": str(row[4] or ""),
            "report_name": str(row[5] or ""),
        }
        for row in rows
    ]


def list_entities(
    conn: Any,
    *,
    entity_type: str | None = None,
    limit: int = 100,
) -> list[EntityRecord]:
    clauses: list[str] = []
    params: list[object] = []
    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT entity_id, entity_type, mention_count, first_seen, last_seen
        FROM entities
        {where}
        ORDER BY mention_count DESC, entity_id
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        EntityRecord(
            entity_id=str(row[0]),
            entity_type=str(row[1]),
            mention_count=int(row[2] or 0),
            first_seen=row[3],
            last_seen=row[4],
        )
        for row in rows
    ]
