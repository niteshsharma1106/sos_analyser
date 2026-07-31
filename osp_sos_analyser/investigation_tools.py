# investigation_tools.py — agent-facing wrappers over Cluster Manifest + Evidence Index.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .context_pack import truncate_text
from .evidence_index import (
    EvidenceMention,
    extract_entity_ids,
    get_cluster_manifest,
    get_entity,
    get_evidence,
    list_entities,
)
from .models import LogRecord


def format_manifest(conn: Any) -> str:
    nodes = get_cluster_manifest(conn)
    if not nodes:
        return "No cluster manifest available. Re-ingest SOS reports to build cluster_nodes."
    lines = [
        f"cluster_id={nodes[0]['cluster_id']}",
        f"nodes={len(nodes)}",
    ]
    for node in nodes:
        services = ",".join(node.get("services") or []) or "-"
        lines.append(
            f"{node['hostname']}|role={node['node_role']}|ver={node['rhosp_version']}|"
            f"services={services}|archive={node['archive_name']}"
        )
    return "\n".join(lines)


def format_evidence_mentions(mentions: Sequence[EvidenceMention]) -> str:
    if not mentions:
        return "No indexed evidence for that entity."
    lines = []
    for item in mentions:
        lines.append(
            "|".join(
                [
                    str(item.timestamp or ""),
                    item.hostname or "-",
                    item.service or "-",
                    item.level or "-",
                    item.entity_type or "-",
                    truncate_text(item.message_excerpt, 180),
                    item.source_file or "",
                ]
            )
        )
    return "\n".join(lines)


def mentions_to_log_records(mentions: Sequence[EvidenceMention]) -> tuple[LogRecord, ...]:
    return tuple(
        LogRecord(
            timestamp=item.timestamp,
            level=item.level,
            service=item.service or "unknown",
            module=item.entity_type or "",
            message=item.message_excerpt,
            source_file=item.source_file,
            report_name=item.report_name,
        )
        for item in mentions
    )


def indexed_evidence_for_hints(
    conn: Any,
    hints: Any,
    *,
    services: Sequence[str] = (),
    limit: int = 10,
) -> list[LogRecord]:
    """Prefer Evidence Index hits for identifiers; otherwise return empty."""
    identifiers = getattr(hints, "identifiers", ()) or ()
    if not identifiers:
        return []
    mentions: list[EvidenceMention] = []
    for identifier in list(identifiers)[:3]:
        mentions.extend(
            get_evidence(conn, identifier, services=services, limit=limit)
        )
    return list(mentions_to_log_records(mentions)[:limit])


def prefetch_investigation_digest(
    conn: Any,
    *,
    raw_query: str,
    resource_id: str | None = None,
    keywords: Sequence[str] = (),
    limit_per_entity: int = 12,
) -> str:
    """Build a compact digest from manifest + evidence index for the investigator."""
    sections = ["## Cluster manifest", format_manifest(conn)]

    ids = []
    if resource_id:
        ids.append(resource_id)
    ids.extend(extract_entity_ids(raw_query))
    ids = list(dict.fromkeys(ids))

    if ids:
        blocks = []
        for entity_id in ids[:5]:
            entity = get_entity(conn, entity_id)
            header = (
                f"### {entity_id} type={entity.entity_type} mentions={entity.mention_count}"
                if entity
                else f"### {entity_id} (not in index)"
            )
            mentions = get_evidence(conn, entity_id, limit=limit_per_entity)
            blocks.append(header + "\n" + format_evidence_mentions(mentions))
        sections.append("## Indexed entity evidence\n" + "\n\n".join(blocks))
    else:
        # Surface top entities so the agent has something concrete to start from.
        top = list_entities(conn, limit=15)
        if top:
            lines = [f"{e.entity_id}|{e.entity_type}|mentions={e.mention_count}" for e in top]
            sections.append("## Top indexed entities\n" + "\n".join(lines))

    if keywords:
        sections.append("## Keywords from plan\n" + ", ".join(keywords[:8]))

    return "\n\n".join(sections)


def build_langchain_tools(conn: Any):
    """Create LangChain tools bound to an open DuckDB connection."""
    from langchain_core.tools import tool

    @tool
    def get_cluster_overview() -> str:
        """Return the Cluster Manifest: hostnames, roles, RHOSP version, and services per SOS node."""
        return format_manifest(conn)

    @tool
    def get_entity_evidence(
        entity_id: str,
        service: str = "",
        limit: int = 15,
    ) -> str:
        """
        Fetch indexed evidence for one canonical entity (instance/port/volume/network/req-/host UUID).
        Prefer this over searching raw logs. Optional service filter (nova, neutron, cinder, ...).
        """
        if not entity_id.strip():
            return "entity_id is required."
        services = [service] if service.strip() else ()
        capped = max(1, min(int(limit), 30))
        entity = get_entity(conn, entity_id.strip())
        header = (
            f"entity={entity.entity_id} type={entity.entity_type} mentions={entity.mention_count}\n"
            if entity
            else f"entity={entity_id} (not registered in entities table)\n"
        )
        mentions = get_evidence(
            conn,
            entity_id.strip(),
            limit=capped,
            services=services,
        )
        return header + format_evidence_mentions(mentions)

    @tool
    def list_indexed_entities(entity_type: str = "", limit: int = 20) -> str:
        """
        List indexed entities. Optional entity_type: instance, volume, port, network,
        router, image, request, host, unknown.
        """
        capped = max(1, min(int(limit), 50))
        rows = list_entities(
            conn,
            entity_type=entity_type.strip() or None,
            limit=capped,
        )
        if not rows:
            return "No entities in the evidence index. Re-ingest SOS reports first."
        return "\n".join(
            f"{row.entity_id}|{row.entity_type}|mentions={row.mention_count}" for row in rows
        )

    @tool
    def search_os_logs(
        service: str = "",
        resource_id: str = "",
        search_terms: str = "",
        limit: int = 15,
    ) -> str:
        """
        Fallback raw log search. Prefer get_entity_evidence when you have a UUID/req-id.
        If resource_id is set, this first returns indexed evidence for that entity.
        """
        capped = max(1, min(int(limit), 30))
        if resource_id.strip():
            services = [service] if service.strip() else ()
            entity = get_entity(conn, resource_id.strip())
            mentions = get_evidence(
                conn,
                resource_id.strip(),
                limit=capped,
                services=services,
            )
            if mentions:
                header = (
                    f"entity={entity.entity_id} type={entity.entity_type} mentions={entity.mention_count}\n"
                    if entity
                    else f"entity={resource_id}\n"
                )
                return (
                    "Indexed evidence (preferred):\n"
                    + header
                    + format_evidence_mentions(mentions)
                    + "\n\n(Use get_entity_evidence for related IDs extracted from these digests.)"
                )

        clauses: list[str] = []
        params: list[object] = []
        if service:
            clauses.append("service = ?")
            params.append(service)
        if resource_id:
            clauses.append("message ILIKE ?")
            params.append(f"%{resource_id}%")
        for term in search_terms.split():
            clauses.append("message ILIKE ?")
            params.append(f"%{term}%")
        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(capped)
        rows = conn.execute(
            f"""
            SELECT timestamp, COALESCE(hostname, ''), service, level, message, source_file
            FROM os_logs
            WHERE {where}
            ORDER BY
              CASE level WHEN 'CRITICAL' THEN 0 WHEN 'ERROR' THEN 1
                   WHEN 'WARNING' THEN 2 ELSE 3 END,
              timestamp NULLS LAST
            LIMIT ?
            """,
            params,
        ).fetchall()
        if not rows:
            return "No matching logs."
        lines = []
        for ts, host, svc, level, message, source in rows:
            lines.append(
                "|".join(
                    [
                        str(ts or ""),
                        str(host or "-"),
                        str(svc or "-"),
                        str(level or "-"),
                        truncate_text(str(message or ""), 180),
                        str(source or ""),
                    ]
                )
            )
        return "\n".join(lines)

    return [get_cluster_overview, get_entity_evidence, list_indexed_entities, search_os_logs]
