# relationship_graph.py — VM↔port↔chassis↔host operation graph over SOS evidence.
from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .context_pack import DIGEST_MESSAGE_CHARS, truncate_text
from .evidence_index import classify_entity_type, extract_entity_ids

UUID = (
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

LABELLED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instance", re.compile(
        rf"\b(?:instance(?:_uuid|_id)?|server(?:_id)?|device_id|vm)\s*[:=]?\s*"
        rf"(?P<id>{UUID})\b",
        re.IGNORECASE,
    )),
    ("port", re.compile(
        rf"\b(?:port(?:_id)?|vif(?:_id)?|vnic)\s*[:=]?\s*(?P<id>{UUID})\b",
        re.IGNORECASE,
    )),
    ("volume", re.compile(
        rf"\b(?:volume(?:_id)?|vol_id)\s*[:=]?\s*(?P<id>{UUID})\b",
        re.IGNORECASE,
    )),
    ("network", re.compile(
        rf"\b(?:network(?:_id)?|net_id)\s*[:=]?\s*(?P<id>{UUID})\b",
        re.IGNORECASE,
    )),
    ("image", re.compile(
        rf"\b(?:image(?:_id)?)\s*[:=]?\s*(?P<id>{UUID})\b",
        re.IGNORECASE,
    )),
)

HOST_BIND_RE = re.compile(
    r"\b(?:binding:host_id|host_id|on host|to host|selected host|scheduled to"
    r"|compute host|host)\s*[:=]?\s*(?P<host>[A-Za-z0-9][A-Za-z0-9._-]{1,63})\b",
    re.IGNORECASE,
)
CHASSIS_RE = re.compile(
    r"\bchassis\s*[:=]?\s*(?P<chassis>[A-Za-z0-9][A-Za-z0-9._-]{1,63})\b",
    re.IGNORECASE,
)
REQUEST_RE = re.compile(r"\b(?P<id>req-[0-9a-fA-F-]{8,})\b", re.IGNORECASE)

OPERATIONAL_RELATIONS = frozenset(
    {
        "instance_port",
        "port_host",
        "port_chassis",
        "chassis_host",
        "instance_host",
        "volume_instance",
        "request_touches",
    }
)


@dataclass(frozen=True)
class EntityRelationship:
    src_entity_id: str
    src_entity_type: str
    relation_type: str
    dst_entity_id: str
    dst_entity_type: str
    evidence_count: int
    confidence: float
    first_seen: datetime | None
    last_seen: datetime | None
    hostnames: tuple[str, ...]
    services: tuple[str, ...]
    sample_excerpt: str
    sample_hostname: str
    sample_service: str
    sample_source_file: str
    sample_report_name: str


def _canonical_id(entity_id: str) -> str:
    value = (entity_id or "").strip()
    if not value:
        return ""
    if value.lower().startswith("req-"):
        return value.lower()
    if re.fullmatch(UUID, value, flags=re.IGNORECASE):
        return value.lower()
    return value


def extract_typed_entities(message: str, service: str = "") -> dict[str, set[str]]:
    """Extract labelled and fallback entity IDs grouped by type."""
    text = message or ""
    grouped: dict[str, set[str]] = defaultdict(set)
    claimed: set[str] = set()

    for entity_type, pattern in LABELLED_PATTERNS:
        for match in pattern.finditer(text):
            entity_id = _canonical_id(match.group("id"))
            if not entity_id:
                continue
            grouped[entity_type].add(entity_id)
            claimed.add(entity_id)

    for match in REQUEST_RE.finditer(text):
        entity_id = _canonical_id(match.group("id"))
        if entity_id:
            grouped["request"].add(entity_id)
            claimed.add(entity_id)

    for entity_id in extract_entity_ids(text):
        key = _canonical_id(entity_id)
        if not key or key in claimed:
            continue
        entity_type = classify_entity_type(key, text, service)
        grouped[entity_type].add(key)

    return grouped


def _known_host_aliases(conn: Any) -> dict[str, str]:
    """Map lowercase alias -> canonical hostname from cluster_nodes."""
    aliases: dict[str, str] = {}
    rows = conn.execute("SELECT hostname FROM cluster_nodes").fetchall()
    for (hostname,) in rows:
        host = str(hostname or "").strip()
        if not host:
            continue
        aliases[host.lower()] = host
        short = host.split(".", 1)[0]
        aliases.setdefault(short.lower(), host)
    return aliases


def _match_hosts(message: str, aliases: dict[str, str]) -> set[str]:
    found: set[str] = set()
    lowered = (message or "").lower()
    for alias, canonical in aliases.items():
        if alias and alias in lowered:
            found.add(canonical)
    for match in HOST_BIND_RE.finditer(message or ""):
        host = match.group("host")
        canonical = aliases.get(host.lower())
        if canonical:
            found.add(canonical)
        elif host:
            found.add(host)
    return found


def _match_chassis(message: str) -> set[str]:
    return {
        match.group("chassis")
        for match in CHASSIS_RE.finditer(message or "")
        if match.group("chassis")
    }


def _edge_key(
    src: str, relation: str, dst: str
) -> tuple[str, str, str]:
    return (src, relation, dst)


def _upsert_edge(
    edges: dict[tuple[str, str, str], dict[str, Any]],
    *,
    src: str,
    src_type: str,
    relation: str,
    dst: str,
    dst_type: str,
    confidence: float,
    timestamp: datetime | None,
    hostname: str,
    service: str,
    level: str,
    source_file: str,
    report_name: str,
    excerpt: str,
) -> None:
    src = _canonical_id(src)
    dst = _canonical_id(dst)
    if not src or not dst or src == dst:
        return
    key = _edge_key(src, relation, dst)
    meta = edges.get(key)
    if meta is None:
        edges[key] = {
            "src_entity_id": src,
            "src_entity_type": src_type,
            "relation_type": relation,
            "dst_entity_id": dst,
            "dst_entity_type": dst_type,
            "evidence_count": 1,
            "confidence": confidence,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "hostnames": {hostname} if hostname else set(),
            "services": {service} if service else set(),
            "sample_excerpt": excerpt,
            "sample_hostname": hostname,
            "sample_service": service,
            "sample_level": level,
            "sample_source_file": source_file,
            "sample_report_name": report_name,
        }
        return
    meta["evidence_count"] += 1
    meta["confidence"] = max(float(meta["confidence"]), confidence)
    if timestamp is not None:
        if meta["first_seen"] is None or timestamp < meta["first_seen"]:
            meta["first_seen"] = timestamp
        if meta["last_seen"] is None or timestamp > meta["last_seen"]:
            meta["last_seen"] = timestamp
    if hostname:
        meta["hostnames"].add(hostname)
    if service:
        meta["services"].add(service)


def relationship_candidates_from_message(
    *,
    message: str,
    service: str,
    hostname: str,
    aliases: dict[str, str],
    timestamp: datetime | None = None,
    level: str = "",
    source_file: str = "",
    report_name: str = "",
) -> list[dict[str, Any]]:
    """Deterministic relationship recipes for one log line."""
    grouped = extract_typed_entities(message, service)
    hosts = _match_hosts(message, aliases)
    if hostname and hostname in aliases.values():
        hosts.add(hostname)
    elif hostname and hostname.lower() in aliases:
        hosts.add(aliases[hostname.lower()])
    chassis_names = _match_chassis(message)
    excerpt = truncate_text(message, DIGEST_MESSAGE_CHARS)
    lowered = message.lower()
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    instances = grouped.get("instance", set())
    ports = grouped.get("port", set())
    volumes = grouped.get("volume", set())
    requests = grouped.get("request", set())

    for instance_id in instances:
        for port_id in ports:
            _upsert_edge(
                edges,
                src=instance_id,
                src_type="instance",
                relation="instance_port",
                dst=port_id,
                dst_type="port",
                confidence=0.9,
                timestamp=timestamp,
                hostname=hostname,
                service=service,
                level=level,
                source_file=source_file,
                report_name=report_name,
                excerpt=excerpt,
            )

    if any(token in lowered for token in ("attach", "detach", "bdm", "block_device", "volume")):
        for volume_id in volumes:
            for instance_id in instances:
                _upsert_edge(
                    edges,
                    src=volume_id,
                    src_type="volume",
                    relation="volume_instance",
                    dst=instance_id,
                    dst_type="instance",
                    confidence=0.85,
                    timestamp=timestamp,
                    hostname=hostname,
                    service=service,
                    level=level,
                    source_file=source_file,
                    report_name=report_name,
                    excerpt=excerpt,
                )

    for port_id in ports:
        for host in hosts:
            conf = 0.9 if "host" in lowered or "binding" in lowered else 0.7
            _upsert_edge(
                edges,
                src=port_id,
                src_type="port",
                relation="port_host",
                dst=host,
                dst_type="host",
                confidence=conf,
                timestamp=timestamp,
                hostname=hostname or host,
                service=service,
                level=level,
                source_file=source_file,
                report_name=report_name,
                excerpt=excerpt,
            )
        for chassis in chassis_names:
            _upsert_edge(
                edges,
                src=port_id,
                src_type="port",
                relation="port_chassis",
                dst=chassis,
                dst_type="chassis",
                confidence=0.85,
                timestamp=timestamp,
                hostname=hostname,
                service=service,
                level=level,
                source_file=source_file,
                report_name=report_name,
                excerpt=excerpt,
            )

    for instance_id in instances:
        for host in hosts:
            conf = 0.9 if any(t in lowered for t in ("spawn", "build", "schedul", "migrate")) else 0.7
            _upsert_edge(
                edges,
                src=instance_id,
                src_type="instance",
                relation="instance_host",
                dst=host,
                dst_type="host",
                confidence=conf,
                timestamp=timestamp,
                hostname=hostname or host,
                service=service,
                level=level,
                source_file=source_file,
                report_name=report_name,
                excerpt=excerpt,
            )

    for chassis in chassis_names:
        for host in hosts:
            _upsert_edge(
                edges,
                src=chassis,
                src_type="chassis",
                relation="chassis_host",
                dst=host,
                dst_type="host",
                confidence=0.8,
                timestamp=timestamp,
                hostname=hostname or host,
                service=service,
                level=level,
                source_file=source_file,
                report_name=report_name,
                excerpt=excerpt,
            )
        # OVN chassis often equals or aliases the compute hostname.
        if hostname:
            host_canonical = aliases.get(hostname.lower(), hostname)
            if chassis.lower() == host_canonical.lower() or chassis.lower() == host_canonical.split(".", 1)[0].lower():
                _upsert_edge(
                    edges,
                    src=chassis,
                    src_type="chassis",
                    relation="chassis_host",
                    dst=host_canonical,
                    dst_type="host",
                    confidence=0.95,
                    timestamp=timestamp,
                    hostname=host_canonical,
                    service=service,
                    level=level,
                    source_file=source_file,
                    report_name=report_name,
                    excerpt=excerpt,
                )

    # Compute-side OVN/nova logs mentioning a port/instance without explicit host label
    # still imply observation on the SOS hostname.
    if hostname and (service in {"ovn", "nova", "neutron", "openvswitch"} or "ovn" in lowered):
        host_canonical = aliases.get(hostname.lower(), hostname)
        for port_id in ports:
            if not any(e[0] == port_id and e[1] == "port_host" for e in edges):
                _upsert_edge(
                    edges,
                    src=port_id,
                    src_type="port",
                    relation="port_host",
                    dst=host_canonical,
                    dst_type="host",
                    confidence=0.55,
                    timestamp=timestamp,
                    hostname=host_canonical,
                    service=service,
                    level=level,
                    source_file=source_file,
                    report_name=report_name,
                    excerpt=excerpt,
                )
        for instance_id in instances:
            if not any(e[0] == instance_id and e[1] == "instance_host" for e in edges):
                _upsert_edge(
                    edges,
                    src=instance_id,
                    src_type="instance",
                    relation="instance_host",
                    dst=host_canonical,
                    dst_type="host",
                    confidence=0.55,
                    timestamp=timestamp,
                    hostname=host_canonical,
                    service=service,
                    level=level,
                    source_file=source_file,
                    report_name=report_name,
                    excerpt=excerpt,
                )

    touched = set().union(*grouped.values()) if grouped else set()
    for request_id in requests:
        for entity_id in touched:
            if entity_id == request_id:
                continue
            entity_type = next(
                (etype for etype, ids in grouped.items() if entity_id in ids),
                "unknown",
            )
            _upsert_edge(
                edges,
                src=request_id,
                src_type="request",
                relation="request_touches",
                dst=entity_id,
                dst_type=entity_type,
                confidence=0.6,
                timestamp=timestamp,
                hostname=hostname,
                service=service,
                level=level,
                source_file=source_file,
                report_name=report_name,
                excerpt=excerpt,
            )

    return list(edges.values())


def build_relationship_index(conn: Any) -> dict[str, int]:
    """Rebuild entity_relationships from os_logs + cluster host aliases."""
    conn.execute("DELETE FROM entity_relationships")
    aliases = _known_host_aliases(conn)
    rows = conn.execute(
        """
        SELECT timestamp, level, service, message, source_file, report_name,
               COALESCE(hostname, '') AS hostname
        FROM os_logs
        """
    ).fetchall()

    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    chassis_entities: dict[str, dict[str, Any]] = {}

    for timestamp, level, service, message, source_file, report_name, hostname in rows:
        candidates = relationship_candidates_from_message(
            message=str(message or ""),
            service=str(service or ""),
            hostname=str(hostname or ""),
            aliases=aliases,
            timestamp=timestamp,
            level=str(level or ""),
            source_file=str(source_file or ""),
            report_name=str(report_name or ""),
        )
        for candidate in candidates:
            key = _edge_key(
                candidate["src_entity_id"],
                candidate["relation_type"],
                candidate["dst_entity_id"],
            )
            existing = edges.get(key)
            if existing is None:
                edges[key] = candidate
                continue
            existing["evidence_count"] += int(candidate["evidence_count"])
            existing["confidence"] = max(
                float(existing["confidence"]), float(candidate["confidence"])
            )
            ts = candidate.get("first_seen")
            if ts is not None:
                if existing["first_seen"] is None or ts < existing["first_seen"]:
                    existing["first_seen"] = ts
                if existing["last_seen"] is None or ts > existing["last_seen"]:
                    existing["last_seen"] = ts
            existing["hostnames"] |= set(candidate.get("hostnames") or [])
            existing["services"] |= set(candidate.get("services") or [])

            if candidate["dst_entity_type"] == "chassis":
                chassis = candidate["dst_entity_id"]
                meta = chassis_entities.setdefault(
                    chassis,
                    {"mention_count": 0, "first_seen": None, "last_seen": None},
                )
                meta["mention_count"] += 1
                if ts is not None:
                    if meta["first_seen"] is None or ts < meta["first_seen"]:
                        meta["first_seen"] = ts
                    if meta["last_seen"] is None or ts > meta["last_seen"]:
                        meta["last_seen"] = ts
            if candidate["src_entity_type"] == "chassis":
                chassis = candidate["src_entity_id"]
                meta = chassis_entities.setdefault(
                    chassis,
                    {"mention_count": 0, "first_seen": None, "last_seen": None},
                )
                meta["mention_count"] += 1

    relationship_rows = [
        (
            meta["src_entity_id"],
            meta["src_entity_type"],
            meta["relation_type"],
            meta["dst_entity_id"],
            meta["dst_entity_type"],
            int(meta["evidence_count"]),
            float(meta["confidence"]),
            meta["first_seen"],
            meta["last_seen"],
            ",".join(sorted(meta["hostnames"])),
            ",".join(sorted(meta["services"])),
            meta["sample_excerpt"],
            meta["sample_hostname"],
            meta["sample_service"],
            meta.get("sample_level", ""),
            meta["sample_source_file"],
            meta["sample_report_name"],
        )
        for meta in edges.values()
    ]
    if relationship_rows:
        conn.executemany(
            """
            INSERT INTO entity_relationships (
                src_entity_id, src_entity_type, relation_type, dst_entity_id,
                dst_entity_type, evidence_count, confidence, first_seen, last_seen,
                hostnames, services, sample_excerpt, sample_hostname, sample_service,
                sample_level, sample_source_file, sample_report_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            relationship_rows,
        )

    for chassis, meta in chassis_entities.items():
        existing = conn.execute(
            "SELECT 1 FROM entities WHERE lower(entity_id) = lower(?) LIMIT 1",
            [chassis],
        ).fetchone()
        if existing:
            continue
        conn.execute(
            """
            INSERT INTO entities (
                entity_id, entity_type, mention_count, first_seen, last_seen
            ) VALUES (?, 'chassis', ?, ?, ?)
            """,
            [
                chassis,
                int(meta["mention_count"] or 1),
                meta["first_seen"],
                meta["last_seen"],
            ],
        )

    return {"relationships": len(relationship_rows), "chassis_entities": len(chassis_entities)}


def _row_to_relationship(row: Sequence[Any]) -> EntityRelationship:
    return EntityRelationship(
        src_entity_id=str(row[0]),
        src_entity_type=str(row[1] or "unknown"),
        relation_type=str(row[2]),
        dst_entity_id=str(row[3]),
        dst_entity_type=str(row[4] or "unknown"),
        evidence_count=int(row[5] or 0),
        confidence=float(row[6] or 0.0),
        first_seen=row[7],
        last_seen=row[8],
        hostnames=tuple(part for part in str(row[9] or "").split(",") if part),
        services=tuple(part for part in str(row[10] or "").split(",") if part),
        sample_excerpt=str(row[11] or ""),
        sample_hostname=str(row[12] or ""),
        sample_service=str(row[13] or ""),
        sample_source_file=str(row[15] or ""),
        sample_report_name=str(row[16] or ""),
    )


_RELATIONSHIP_SELECT = """
    src_entity_id, src_entity_type, relation_type, dst_entity_id, dst_entity_type,
    evidence_count, confidence, first_seen, last_seen, hostnames, services,
    sample_excerpt, sample_hostname, sample_service, sample_level,
    sample_source_file, sample_report_name
"""


def get_related_entities(
    conn: Any,
    entity_id: str,
    *,
    relation_types: Sequence[str] = (),
    limit: int = 50,
) -> list[EntityRelationship]:
    """Return direct neighbors, treating edges as navigable in both directions."""
    entity_id = _canonical_id(entity_id)
    if not entity_id:
        return []
    clauses = ["(lower(src_entity_id) = lower(?) OR lower(dst_entity_id) = lower(?))"]
    params: list[object] = [entity_id, entity_id]
    if relation_types:
        clauses.append(
            f"relation_type IN ({', '.join('?' for _ in relation_types)})"
        )
        params.extend(relation_types)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT {_RELATIONSHIP_SELECT}
        FROM entity_relationships
        WHERE {' AND '.join(clauses)}
        ORDER BY confidence DESC, evidence_count DESC, last_seen NULLS LAST
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_row_to_relationship(row) for row in rows]


def get_operation_path(
    conn: Any,
    start_entity_id: str,
    *,
    target_entity_id: str = "",
    target_type: str = "",
    max_hops: int = 4,
) -> list[EntityRelationship]:
    """BFS over operational edges to reach a host/chassis or explicit target."""
    start = _canonical_id(start_entity_id)
    target = _canonical_id(target_entity_id)
    want_type = (target_type or "").strip().lower()
    if not start:
        return []
    if not target and not want_type:
        want_type = "host"

    adjacency: dict[str, list[EntityRelationship]] = defaultdict(list)
    rows = conn.execute(
        f"""
        SELECT {_RELATIONSHIP_SELECT}
        FROM entity_relationships
        WHERE relation_type IN ({', '.join('?' for _ in OPERATIONAL_RELATIONS)})
        """,
        list(OPERATIONAL_RELATIONS),
    ).fetchall()
    for row in rows:
        rel = _row_to_relationship(row)
        adjacency[rel.src_entity_id.lower()].append(rel)
        # Undirected traversal for investigation convenience.
        adjacency[rel.dst_entity_id.lower()].append(rel)

    queue: deque[tuple[str, list[EntityRelationship]]] = deque([(start.lower(), [])])
    visited: set[str] = {start.lower()}
    while queue:
        current, path = queue.popleft()
        if len(path) >= max_hops:
            continue
        for edge in sorted(
            adjacency.get(current, []),
            key=lambda item: (-item.confidence, -item.evidence_count),
        ):
            nxt = (
                edge.dst_entity_id.lower()
                if edge.src_entity_id.lower() == current
                else edge.src_entity_id.lower()
            )
            if nxt in visited:
                continue
            next_path = path + [edge]
            reached_id = edge.dst_entity_id if edge.src_entity_id.lower() == current else edge.src_entity_id
            reached_type = edge.dst_entity_type if edge.src_entity_id.lower() == current else edge.src_entity_type
            if target and reached_id.lower() == target.lower():
                return next_path
            if want_type and reached_type.lower() == want_type and not target:
                return next_path
            visited.add(nxt)
            queue.append((nxt, next_path))
    return []


def format_relationships(rows: Sequence[EntityRelationship]) -> str:
    if not rows:
        return "No relationships found for that entity."
    lines = ["src|relation|dst|conf|evidence|hosts|sample"]
    for item in rows:
        lines.append(
            "|".join(
                [
                    f"{item.src_entity_id}({item.src_entity_type})",
                    item.relation_type,
                    f"{item.dst_entity_id}({item.dst_entity_type})",
                    f"{item.confidence:.2f}",
                    str(item.evidence_count),
                    ",".join(item.hostnames[:3]) or "-",
                    truncate_text(item.sample_excerpt, 120),
                ]
            )
        )
    return "\n".join(lines)


def format_operation_path(
    path: Sequence[EntityRelationship],
    *,
    start_entity_id: str = "",
) -> str:
    if not path:
        return "No operation path found."
    start = _canonical_id(start_entity_id) or path[0].src_entity_id
    current = start.lower()
    # Resolve displayed start type from first edge.
    if path[0].src_entity_id.lower() == current:
        nodes = [f"{path[0].src_entity_id}({path[0].src_entity_type})"]
    elif path[0].dst_entity_id.lower() == current:
        nodes = [f"{path[0].dst_entity_id}({path[0].dst_entity_type})"]
    else:
        nodes = [start]
    for edge in path:
        if edge.src_entity_id.lower() == current:
            nodes.append(
                f"--{edge.relation_type}--> {edge.dst_entity_id}({edge.dst_entity_type})"
            )
            current = edge.dst_entity_id.lower()
        else:
            nodes.append(
                f"--{edge.relation_type}--> {edge.src_entity_id}({edge.src_entity_type})"
            )
            current = edge.src_entity_id.lower()
    details = format_relationships(path)
    return "Path: " + " ".join(nodes) + "\n\n" + details
