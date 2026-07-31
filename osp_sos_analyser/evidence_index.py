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
    if report_name:
        conn.execute("DELETE FROM entity_mentions WHERE report_name = ?", [report_name])
    else:
        conn.execute("DELETE FROM entity_mentions")
        conn.execute("DELETE FROM entities")

    log_sql = """
        SELECT timestamp, level, service, message, source_file, report_name,
               COALESCE(hostname, '') AS hostname
        FROM os_logs
    """
    params: list[object] = []
    if report_name:
        log_sql += " WHERE report_name = ?"
        params.append(report_name)

    rows = conn.execute(log_sql, params).fetchall()
    entity_meta: dict[str, dict[str, Any]] = {}
    mention_rows: list[tuple[object, ...]] = []

    for timestamp, level, service, message, source_file, report, hostname in rows:
        text = str(message or "")
        for entity_id in extract_entity_ids(text):
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

    # Register hosts as first-class entities.
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
        mention_rows.append(
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

    if report_name:
        # Refresh aggregate entity rows touched by this report.
        touched = sorted({row[0] for row in mention_rows})
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
    if entity_rows:
        conn.executemany(
            """
            INSERT INTO entities (
                entity_id, entity_type, mention_count, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?)
            """,
            entity_rows,
        )
    if mention_rows:
        conn.executemany(
            """
            INSERT INTO entity_mentions (
                entity_id, entity_type, timestamp, hostname, service, level,
                source_file, report_name, message_excerpt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            mention_rows,
        )

    return {"entities": len(entity_rows), "mentions": len(mention_rows)}


def get_cluster_manifest(conn: Any) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT cluster_id, hostname, node_role, rhosp_version, services,
               archive_name, archive_id
        FROM cluster_nodes
        ORDER BY node_role, hostname
        """
    ).fetchall()
    return [
        {
            "cluster_id": cluster_id,
            "hostname": hostname,
            "node_role": node_role,
            "rhosp_version": rhosp_version,
            "services": [part for part in str(services or "").split(",") if part],
            "archive_name": archive_name,
            "archive_id": archive_id,
        }
        for cluster_id, hostname, node_role, rhosp_version, services, archive_name, archive_id in rows
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
) -> list[EvidenceMention]:
    clauses = ["lower(entity_id) = lower(?)"]
    params: list[object] = [entity_id]
    if services:
        clauses.append(f"service IN ({', '.join('?' for _ in services)})")
        params.extend(services)
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
