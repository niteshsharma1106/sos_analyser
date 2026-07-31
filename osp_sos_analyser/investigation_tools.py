# investigation_tools.py — agent-facing wrappers over Cluster Manifest + Evidence Index.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .context_pack import truncate_text
from .evidence_index import (
    EvidenceMention,
    compare_node_activity,
    extract_entity_ids,
    get_cluster_manifest,
    get_entity,
    get_evidence,
    get_host_activity,
    list_entities,
    search_logs_by_node,
)
from .relationship_graph import (
    format_operation_path,
    format_relationships,
    get_operation_path as fetch_operation_path,
    get_related_entities as fetch_related_entities,
)


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


def format_node_comparison(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "No per-node activity matched the filters."
    lines = ["hostname|role|service|level|count|first_seen|last_seen"]
    for row in rows:
        lines.append(
            "|".join(
                [
                    str(row.get("hostname") or "-"),
                    str(row.get("node_role") or "-"),
                    str(row.get("service") or "-"),
                    str(row.get("level") or "-"),
                    str(row.get("event_count") or 0),
                    str(row.get("first_seen") or ""),
                    str(row.get("last_seen") or ""),
                ]
            )
        )
    return "\n".join(lines)


def format_node_log_rows(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "No matching logs."
    lines = []
    for row in rows:
        lines.append(
            "|".join(
                [
                    str(row.get("timestamp") or ""),
                    str(row.get("hostname") or "-"),
                    str(row.get("node_role") or "-"),
                    str(row.get("service") or "-"),
                    str(row.get("level") or "-"),
                    truncate_text(str(row.get("message") or ""), 180),
                    str(row.get("source_file") or ""),
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
            hostname=item.hostname or "",
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
    hostnames = tuple(getattr(hints, "hostnames", ()) or ())
    if not identifiers:
        return []
    mentions: list[EvidenceMention] = []
    for identifier in list(identifiers)[:3]:
        mentions.extend(
            get_evidence(
                conn,
                identifier,
                services=services,
                hostnames=hostnames,
                limit=limit,
            )
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

    nodes = get_cluster_manifest(conn)
    if len(nodes) > 1:
        comparison = compare_node_activity(conn, limit=40)
        sections.append("## Cross-node activity\n" + format_node_comparison(comparison))

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
            related = fetch_related_entities(conn, entity_id, limit=12)
            path = fetch_operation_path(conn, entity_id, target_type="host", max_hops=4)
            block = header + "\n" + format_evidence_mentions(mentions)
            if related:
                block += "\n\nRelated:\n" + format_relationships(related)
            if path:
                block += "\n\n" + format_operation_path(path, start_entity_id=entity_id)
            blocks.append(block)
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
    def compare_nodes(
        service: str = "",
        level: str = "",
        limit: int = 40,
    ) -> str:
        """
        Compare WARNING/ERROR/CRITICAL log counts across hosts in the ingested cluster.
        Use this for multi-node incidents (controller vs compute) to see which node is noisy.
        Optional service filter (nova, neutron, ...). Optional level filter (ERROR, WARNING, ...).
        """
        services = [service.strip()] if service.strip() else ()
        levels = (
            [part.strip().upper() for part in level.split(",") if part.strip()]
            if level.strip()
            else ("CRITICAL", "ERROR", "WARNING")
        )
        capped = max(1, min(int(limit), 100))
        rows = compare_node_activity(
            conn,
            services=services,
            levels=levels,
            limit=capped,
        )
        return format_node_comparison(rows)

    @tool
    def get_entity_evidence(
        entity_id: str,
        service: str = "",
        hostname: str = "",
        node_role: str = "",
        limit: int = 15,
    ) -> str:
        """
        Fetch indexed evidence for one canonical entity (instance/port/volume/network/req-/host).
        Prefer this over searching raw logs. Optional service/hostname/node_role filters.
        For hostnames, also returns recent host activity from os_logs.
        """
        if not entity_id.strip():
            return "entity_id is required."
        services = [service] if service.strip() else ()
        hostnames = [hostname] if hostname.strip() else ()
        node_roles = [node_role] if node_role.strip() else ()
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
            hostnames=hostnames,
            node_roles=node_roles,
        )
        body = format_evidence_mentions(mentions)
        if entity and entity.entity_type == "host":
            host_rows = get_host_activity(
                conn,
                entity.entity_id,
                services=services,
                limit=capped,
            )
            body = (
                body
                + "\n\nHost activity:\n"
                + format_evidence_mentions(host_rows)
            )
        return header + body

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
        hostname: str = "",
        node_role: str = "",
        limit: int = 15,
    ) -> str:
        """
        Fallback raw log search with optional hostname/node_role scope.
        Prefer get_entity_evidence when you have a UUID/req-id.
        If resource_id is set, this first returns indexed evidence for that entity.
        """
        capped = max(1, min(int(limit), 30))
        hostnames = [hostname.strip()] if hostname.strip() else ()
        node_roles = [node_role.strip()] if node_role.strip() else ()
        services = [service] if service.strip() else ()
        if resource_id.strip():
            entity = get_entity(conn, resource_id.strip())
            mentions = get_evidence(
                conn,
                resource_id.strip(),
                limit=capped,
                services=services,
                hostnames=hostnames,
                node_roles=node_roles,
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

        rows = search_logs_by_node(
            conn,
            hostnames=hostnames,
            node_roles=node_roles,
            services=services,
            search_terms=search_terms.split(),
            resource_id=resource_id.strip(),
            limit=capped,
        )
        return format_node_log_rows(rows)

    @tool
    def get_related_entities(
        entity_id: str,
        relation_type: str = "",
        limit: int = 20,
    ) -> str:
        """
        Return graph neighbors for an instance/port/volume/request/host/chassis.
        Optional relation_type filter: instance_port, port_host, port_chassis,
        chassis_host, instance_host, volume_instance, request_touches.
        """
        if not entity_id.strip():
            return "entity_id is required."
        types = [relation_type.strip()] if relation_type.strip() else ()
        rows = fetch_related_entities(
            conn,
            entity_id.strip(),
            relation_types=types,
            limit=max(1, min(int(limit), 50)),
        )
        return format_relationships(rows)

    @tool
    def get_operation_path(
        start_entity_id: str,
        target_entity_id: str = "",
        target_type: str = "host",
        max_hops: int = 4,
    ) -> str:
        """
        Find a likely operation path such as VM -> port -> chassis -> host.
        Provide start_entity_id (UUID/hostname). Optional target_entity_id or
        target_type (host, chassis, instance, port, volume).
        """
        if not start_entity_id.strip():
            return "start_entity_id is required."
        path = fetch_operation_path(
            conn,
            start_entity_id.strip(),
            target_entity_id=target_entity_id.strip(),
            target_type=target_type.strip(),
            max_hops=max(1, min(int(max_hops), 6)),
        )
        return format_operation_path(path, start_entity_id=start_entity_id.strip())

    return [
        get_cluster_overview,
        compare_nodes,
        get_entity_evidence,
        get_related_entities,
        get_operation_path,
        list_indexed_entities,
        search_os_logs,
    ]
