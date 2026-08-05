# investigation_tools.py — agent-facing wrappers over Cluster Manifest + Evidence Index.
from __future__ import annotations

import re
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
    search_commands_by_node,
    search_logs_by_node,
)
from .models import LogRecord
from .privacy import redact_sensitive_text
from .relationship_graph import (
    format_operation_path,
    format_relationships,
    get_operation_path as fetch_operation_path,
    get_related_entities as fetch_related_entities,
)

REBOOT_LOG_TERMS = (
    "reboot",
    "System is rebooting",
    "System is powering down",
    "kernel panic",
    "Oops:",
    "BUG:",
    "watchdog",
    "MCE",
    "Hardware Error",
    "Out of memory",
    "oom-kill",
    "Resetting",
    "systemd-shutdown",
    "Starting Reboot",
    "Starting Halt",
    "Linux version",
)
REBOOT_COMMAND_PATTERNS = (
    "dmesg",
    "last",
    "who_-b",
    "uptime",
    "journalctl",
    "list-boots",
    "hostnamectl",
    "ipmitool",
    "mcelog",
    "crash",
)
# Prefer these when establishing *when* a host last booted.
BOOT_TIME_COMMAND_PATTERNS = (
    "list-boots",
    "who_-b",
    "last",
    "uptime",
    "dmesg",
    "hostnamectl",
    "journalctl",
)
_BOOT_HINT_RE = re.compile(
    r"(?im)^.*(?:"
    r"system boot|"
    r"reboot\s+system|"
    r"wtmp begins|"
    r"Linux version|"
    r"-- Boot |"
    r"Startup finished|"
    r"Reached target (?:Multi-User|Graphical|Basic)|"
    r"Command line:|"
    r"Kernel panic|"
    r"watchdog:|"
    r"systemd-shutdown|"
    r"System is rebooting|"
    r"System is powering down|"
    r"Starting Power-Off|"
    r"IDX BOOT ID|"
    r"Boot ID:"
    r").*$"
)
_BOOT_DATETIME_RE = re.compile(
    r"(?:"
    r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?"
    r"|"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?\s+20\d{2}"
    r"|"
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\s+[A-Za-z_/]+)?"
    r"|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s+20\d{2})?"
    r")"
)
# journalctl --list-boots rows (discovered from SOS, not hardcoded times).
_LIST_BOOTS_LINE_RE = re.compile(
    r"(?m)^\s*(-?\d+)\s+([0-9a-fA-F]{32})\s+"
    r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\s+\S+)?)\s+"
    r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\s+\S+)?)\s*$"
)
_REBOOTISH_QUERY_RE = re.compile(
    r"\b(reboot|rebooted|crash|panic|oom|shutdown|power[\s-]?off|watchdog|auto[\s-]?reboot)\b",
    re.I,
)


def is_rebootish_query(text: str) -> bool:
    return bool(_REBOOTISH_QUERY_RE.search(text or ""))


_UNSAFE_ANALYSIS_SQL_RE = re.compile(
    r"\b(?:alter|attach|call|copy|create|delete|detach|drop|export|import|insert|install|"
    r"load|pragma|replace|set|update|vacuum)\b|--|/\*|;",
    re.IGNORECASE,
)


def validate_analysis_sql(sql: str) -> str:
    """Allow a temporary, read-only analytical query and reject control/write SQL."""
    statement = (sql or "").strip()
    if not statement:
        raise ValueError("sql is required.")
    if not re.match(r"^(?:select|with|explain)\b", statement, re.IGNORECASE):
        raise ValueError("Only a single SELECT, WITH, or EXPLAIN query is allowed.")
    if _UNSAFE_ANALYSIS_SQL_RE.search(statement):
        raise ValueError("The analysis query contains a blocked SQL operation.")
    return statement


def format_analysis_rows(cursor: Any, *, limit: int = 100) -> str:
    """Render a bounded, redacted result set for a dynamically-created analysis."""
    columns = [str(item[0]) for item in (cursor.description or [])]
    rows = cursor.fetchmany(max(1, min(int(limit), 100)))
    if not columns:
        return "Analysis completed but returned no tabular result."
    lines = ["|".join(columns)]
    for row in rows:
        lines.append(
            "|".join(
                truncate_text(redact_sensitive_text(str(value or "")), 240).replace("\n", " ")
                for value in row
            )
        )
    return "\n".join(lines) if rows else "No rows returned."


def canonicalize_analysis_hostnames(conn: Any, sql: str) -> tuple[str, list[str]]:
    """Resolve short host aliases inside a temporary analysis SQL statement.

    The agent sees the user's wording (for example ``comp008``), whereas the
    relationship graph stores canonical hostnames.  This keeps dynamically-created
    analyses portable without requiring a bespoke tool for every host query.
    """
    notes: list[str] = []

    def replace_literal(match: re.Match[str]) -> str:
        quote, value = match.group(1), match.group(2)
        # SQL escaping produces doubled quotes; do not reinterpret such literals.
        if "'" in value:
            return match.group(0)
        matches = resolve_hostnames(conn, value)
        if len(matches) == 1 and matches[0].lower() != value.lower():
            canonical = matches[0]
            notes.append(f"resolved hostname {value!r} → {canonical!r}")
            return f"{quote}{canonical}{quote}"
        return match.group(0)

    return re.sub(r"(['\"])([^'\"]+)\1", replace_literal, sql), notes


def _extract_boot_hint_lines(output: str, *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for match in _BOOT_HINT_RE.finditer(output or ""):
        line = " ".join(match.group(0).split())
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(truncate_text(line, 220))
        if len(lines) >= limit:
            break
    return lines


def parse_journalctl_list_boots(text: str) -> list[dict[str, Any]]:
    """Parse ``journalctl --list-boots`` tables discovered in SOS artifacts."""
    boots: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for match in _LIST_BOOTS_LINE_RE.finditer(text or ""):
        idx = int(match.group(1))
        boot_id = match.group(2).lower()
        key = (idx, boot_id)
        if key in seen:
            continue
        seen.add(key)
        boots.append(
            {
                "index": idx,
                "boot_id": boot_id,
                "first_entry": match.group(3).strip(),
                "last_entry": match.group(4).strip(),
            }
        )
    boots.sort(key=lambda item: item["index"])
    return boots


def fetch_list_boots_artifacts(conn: Any, hostname: str, *, limit: int = 8) -> list[dict[str, str]]:
    """
    Locate ``journalctl --list-boots`` text for a host in either os_logs or os_commands.

    SOS often stores this as a single log message with a NULL timestamp, so command-only
    searches miss it.
    """
    host = (hostname or "").strip()
    if not host:
        return []
    capped = max(1, min(int(limit), 20))
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(source_kind: str, source_file: str, body: str) -> None:
        text = (body or "").strip()
        if not text or "IDX BOOT" not in text.upper():
            return
        key = f"{source_kind}|{source_file}|{text[:160]}"
        if key in seen:
            return
        seen.add(key)
        artifacts.append(
            {
                "source_kind": source_kind,
                "source_file": source_file or "-",
                "text": text,
            }
        )

    log_rows = conn.execute(
        """
        SELECT COALESCE(source_file, ''), COALESCE(message, '')
        FROM os_logs
        WHERE lower(COALESCE(hostname, '')) = lower(?)
          AND (
            source_file ILIKE '%list-boots%'
            OR message ILIKE '%IDX BOOT ID%'
            OR message ILIKE '%IDX BOOT%'
          )
        LIMIT ?
        """,
        [host, capped],
    ).fetchall()
    for source_file, message in log_rows:
        _add("os_logs", str(source_file or ""), str(message or ""))

    cmd_rows = conn.execute(
        """
        SELECT COALESCE(source_file, ''), COALESCE(command, ''), COALESCE(output, '')
        FROM os_commands
        WHERE lower(COALESCE(hostname, '')) = lower(?)
          AND (
            source_file ILIKE '%list-boots%'
            OR command ILIKE '%list-boots%'
            OR output ILIKE '%IDX BOOT ID%'
            OR output ILIKE '%IDX BOOT%'
          )
        LIMIT ?
        """,
        [host, capped],
    ).fetchall()
    for source_file, command, output in cmd_rows:
        label = str(source_file or command or "")
        _add("os_commands", label, str(output or ""))

    return artifacts


def discover_host_boots(conn: Any, hostname: str) -> dict[str, Any]:
    """
    Discover host boot records from SOS data (no hardcoded reboot times).

    Prefer ``journalctl --list-boots``. Fall back to previous/current
    ``journalctl --boot`` source-file time bounds when list-boots is absent.
    """
    artifacts = fetch_list_boots_artifacts(conn, hostname)
    boots: list[dict[str, Any]] = []
    sources: list[str] = []
    for artifact in artifacts:
        parsed = parse_journalctl_list_boots(artifact["text"])
        if not parsed:
            continue
        sources.append(f"{artifact['source_kind']}:{artifact['source_file']}")
        for boot in parsed:
            if not any(
                existing["index"] == boot["index"] and existing["boot_id"] == boot["boot_id"]
                for existing in boots
            ):
                boots.append(boot)

    boundary: dict[str, Any] = {}
    try:
        row = conn.execute(
            """
            SELECT
              min(CASE
                    WHEN source_file ILIKE '%journalctl%boot_-1%' THEN timestamp
                  END),
              max(CASE
                    WHEN source_file ILIKE '%journalctl%boot_-1%' THEN timestamp
                  END),
              min(CASE
                    WHEN source_file ILIKE '%journalctl%--boot%'
                     AND source_file NOT ILIKE '%boot_-1%' THEN timestamp
                  END),
              max(CASE
                    WHEN source_file ILIKE '%journalctl%--boot%'
                     AND source_file NOT ILIKE '%boot_-1%' THEN timestamp
                  END)
            FROM os_logs
            WHERE lower(COALESCE(hostname, '')) = lower(?)
              AND timestamp IS NOT NULL
            """,
            [hostname],
        ).fetchone()
    except Exception:  # noqa: BLE001 - discovery must degrade gracefully
        row = None
    if row and any(row):
        boundary = {
            "previous_boot_first": row[0],
            "previous_boot_last": row[1],
            "current_boot_first_seen": row[2],
            "current_boot_last_seen": row[3],
        }

    current = next((boot for boot in boots if boot["index"] == 0), None)
    previous = next((boot for boot in boots if boot["index"] == -1), None)
    return {
        "boots": boots,
        "sources": sources,
        "current_boot": current,
        "previous_boot": previous,
        "boundary": boundary,
    }


def format_host_reboot_timeline(
    conn: Any,
    hostname: str,
    *,
    limit: int = 12,
) -> str:
    """
    First-step reboot RCA helper: discover WHEN the host last booted from SOS data.

    Discovery order (data-driven, not hardcoded times):
    1. ``journalctl --list-boots`` in os_logs / os_commands
    2. ``journalctl --boot`` / ``--boot -1`` time bounds
    3. who -b / last / uptime / dmesg / hostnamectl excerpts
    """
    hint = (hostname or "").strip()
    if not hint:
        return "hostname is required to build a reboot timeline."
    hosts = resolve_hostnames(conn, hint)
    if not hosts:
        return f"No cluster hostname matched {hint!r}."

    host = hosts[0]
    notes: list[str] = []
    if host.lower() != hint.lower():
        notes.append(f"resolved hostname {hint!r} → {host}")

    discovery = discover_host_boots(conn, host)
    boots = discovery["boots"]
    current = discovery["current_boot"]
    previous = discovery["previous_boot"]
    boundary = discovery["boundary"]

    boot_times: list[str] = []
    evidence_blocks: list[str] = []

    if boots:
        notes.append("discovered boots via journalctl --list-boots")
        for source in discovery["sources"][:3]:
            notes.append(f"list-boots source: {source}")
        table_lines = [
            "idx|boot_id|first_entry|last_entry",
        ]
        for boot in boots:
            table_lines.append(
                f"{boot['index']}|{boot['boot_id']}|"
                f"{boot['first_entry']}|{boot['last_entry']}"
            )
            for token in (boot["first_entry"], boot["last_entry"]):
                if token and token not in boot_times:
                    boot_times.append(token)
        evidence_blocks.append(
            f"{host}|journalctl --list-boots|discovered\n"
            + "\n".join(f"  {line}" for line in table_lines)
        )
        if current:
            evidence_blocks.append(
                f"{host}|current boot (idx 0)\n"
                f"  boot_id={current['boot_id']}\n"
                f"  first_entry={current['first_entry']}\n"
                f"  last_entry={current['last_entry']}"
            )
        if previous:
            evidence_blocks.append(
                f"{host}|previous boot (idx -1)\n"
                f"  boot_id={previous['boot_id']}\n"
                f"  first_entry={previous['first_entry']}\n"
                f"  last_entry={previous['last_entry']}"
            )

    if boundary and (
        boundary.get("previous_boot_last") or boundary.get("current_boot_first_seen")
    ):
        prev_last = boundary.get("previous_boot_last")
        curr_first = boundary.get("current_boot_first_seen")
        evidence_blocks.append(
            f"{host}|journalctl --boot source bounds\n"
            f"  previous(--boot -1) last={prev_last}\n"
            f"  current(--boot) first_seen={curr_first}\n"
            "  (first_seen may be later than true boot if large journals were tailed)"
        )
        if not boots and curr_first is not None:
            boot_times.append(str(curr_first))
            notes.append("list-boots missing; used journalctl --boot source bounds")

    rows = search_commands_by_node(
        conn,
        hostnames=[host],
        command_patterns=list(BOOT_TIME_COMMAND_PATTERNS),
        search_terms="",
        limit=max(1, min(int(limit), 20)),
    )
    if not rows:
        rows = search_commands_by_node(
            conn,
            hostnames=[host],
            command_patterns=list(REBOOT_COMMAND_PATTERNS),
            search_terms="",
            limit=max(1, min(int(limit), 20)),
        )
        if rows:
            notes.append("boot-time patterns empty; used broader reboot command set")

    for row in rows:
        command = str(row.get("command") or row.get("source_file") or "command")
        source_file = str(row.get("source_file") or "")
        output = str(row.get("output") or "")
        lower_cmd = f"{command} {source_file}".lower()
        # Prefer boot-time artifacts; skip unrelated ipmitool noise when we already
        # have list-boots / who -b style evidence.
        if boots and "ipmitool" in lower_cmd and "list-boots" not in lower_cmd:
            continue
        if "list-boots" in lower_cmd and parse_journalctl_list_boots(output):
            # Already rendered from discovery.
            continue
        hints = _extract_boot_hint_lines(output, limit=6)
        if not hints and not any(
            key in lower_cmd
            for key in ("who_-b", "who -b", "last", "uptime", "hostnamectl", "list-boots")
        ):
            continue
        if not hints:
            head = truncate_text(output.strip() or "(empty output)", 400)
            hints = [line.strip() for line in head.splitlines() if line.strip()][:6]
        for line in hints:
            for dt in _BOOT_DATETIME_RE.findall(line):
                token = dt.strip()
                if token and token not in boot_times:
                    boot_times.append(token)
        evidence_blocks.append(
            f"{host}|{command}|{source_file or '-'}\n"
            + "\n".join(f"  {line}" for line in hints[:6])
        )

    lines = [f"## Last reboot / boot timeline for {host}"]
    if notes:
        lines.extend(f"({n})" for n in notes)

    if current:
        lines.append(
            f"Last reboot / current boot start (from list-boots idx 0): "
            f"{current['first_entry']} (boot_id={current['boot_id']})"
        )
        if previous:
            lines.append(
                f"Previous boot ended: {previous['last_entry']} "
                f"(boot_id={previous['boot_id']})"
            )
        lines.append(
            "Next: investigate host logs/commands in the window between previous-boot "
            "last_entry and current-boot first_entry "
            "(panic/watchdog/OOM/MCE/Hardware Error). "
            "Ignore user-session 'Reached target Shutdown' noise."
        )
    elif boot_times:
        lines.append(f"Likely boot/reboot timestamp candidates: {', '.join(boot_times[:5])}")
        lines.append(
            "Next: investigate logs/commands in a window around the newest candidate "
            "(panic/watchdog/OOM/MCE/Hardware Error just before that time)."
        )
    else:
        lines.append(
            "No clear boot timestamp extracted yet. Inspect journalctl --list-boots / "
            "who -b / last / uptime / dmesg output below, then search logs around that time."
        )

    if evidence_blocks:
        lines.append("")
        lines.append("Boot-related evidence:")
        lines.extend(evidence_blocks[:10])
    else:
        lines.append("")
        lines.append(format_command_rows(rows) if rows else "No sos_commands artifacts matched.")
    return "\n".join(lines)


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


def format_evidence_mentions(
    mentions: Sequence[EvidenceMention],
    *,
    empty: str = "No indexed evidence for that entity.",
) -> str:
    if not mentions:
        return empty
    lines = []
    for item in mentions:
        lines.append(
            "|".join(
                [
                    str(item.timestamp or "-"),
                    item.hostname or "-",
                    item.service or "-",
                    item.level or "-",
                    item.source_file or "-",
                    truncate_text(item.message_excerpt, 220),
                ]
            )
        )
    return "\n".join(lines)


def format_node_comparison(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "No cross-node activity matched."
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
                    str(row.get("first_seen") or "-"),
                    str(row.get("last_seen") or "-"),
                ]
            )
        )
    return "\n".join(lines)


def format_node_log_rows(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "No log rows matched."
    lines = []
    for row in rows:
        lines.append(
            "|".join(
                [
                    str(row.get("timestamp") or "-"),
                    str(row.get("hostname") or "-"),
                    str(row.get("node_role") or "-"),
                    str(row.get("service") or "-"),
                    str(row.get("level") or "-"),
                    str(row.get("source_file") or "-"),
                    truncate_text(str(row.get("message") or ""), 220),
                ]
            )
        )
    return "\n".join(lines)


def format_command_rows(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "No sos_commands artifacts matched."
    blocks: list[str] = []
    for row in rows:
        header = (
            f"{row.get('hostname') or '-'}|{row.get('command') or '-'}|"
            f"{row.get('source_file') or '-'}|{row.get('service') or '-'}"
        )
        output = truncate_text(str(row.get("output") or ""), 1200)
        blocks.append(header + "\n" + output)
    return "\n\n".join(blocks)


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


def resolve_hostnames(conn: Any, hint: str) -> list[str]:
    """Map short tokens like comp008 to full cluster hostnames."""
    token = (hint or "").strip().lower()
    if not token:
        return []
    nodes = get_cluster_manifest(conn)
    hosts = [str(n.get("hostname") or "") for n in nodes if n.get("hostname")]
    if not hosts:
        return [hint.strip()]
    exact = [h for h in hosts if h.lower() == token]
    if exact:
        return exact
    # Prefer suffix / token boundary matches (…-comp008).
    boundary = []
    contains = []
    for host in hosts:
        lower = host.lower()
        if lower.endswith(token) or re.search(
            rf"(?:^|[-_.]){re.escape(token)}(?:$|[-_.])", lower
        ):
            boundary.append(host)
        elif token in lower:
            contains.append(host)
    matches = boundary or contains
    return matches or [hint.strip()]


def hostnames_mentioned_in_text(conn: Any, text: str) -> list[str]:
    """Find cluster hostnames (or short tokens) referenced in a question."""
    nodes = get_cluster_manifest(conn)
    hosts = [str(n.get("hostname") or "") for n in nodes if n.get("hostname")]
    found: list[str] = []
    lower = (text or "").lower()
    for host in hosts:
        if host.lower() in lower:
            found.append(host)
            continue
        # short token: last label like comp008 / ctrl001
        short = host.split(".")[0].split("-")[-1]
        if len(short) >= 4 and short.lower() in lower:
            found.append(host)
            continue
        # also match mid-token comp008 inside longer names when user says comp008
        match = re.search(r"(?:^|[-_.])((?:comp|ctrl|ceph)\d+[a-z0-9]*)", host, re.I)
        if match and match.group(1).lower() in lower:
            found.append(host)
    # Dedupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for host in found:
        key = host.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(host)
    return out


def prefetch_investigation_digest(
    conn: Any,
    *,
    raw_query: str,
    resource_id: str | None = None,
    keywords: Sequence[str] = (),
    limit_per_entity: int = 12,
) -> str:
    """Build a compact digest from manifest + evidence index for the investigator."""
    sections: list[str] = []

    focus_hosts = hostnames_mentioned_in_text(conn, raw_query)
    rebootish = is_rebootish_query(raw_query)

    # Reboot questions: establish WHEN the host last booted BEFORE other noise.
    if rebootish and focus_hosts:
        timeline_blocks = [
            format_host_reboot_timeline(conn, host, limit=12) for host in focus_hosts[:3]
        ]
        sections.append("\n\n".join(timeline_blocks))

    sections.append("## Cluster manifest\n" + format_manifest(conn))

    nodes = get_cluster_manifest(conn)
    # Skip noisy cross-node WARNING dumps for single-host reboot questions.
    if len(nodes) > 1 and not rebootish and not focus_hosts:
        comparison = compare_node_activity(conn, limit=40)
        sections.append("## Cross-node activity\n" + format_node_comparison(comparison))

    if focus_hosts:
        host_blocks: list[str] = []
        for host in focus_hosts[:3]:
            if rebootish:
                reboot_logs = search_logs_by_node(
                    conn,
                    hostnames=[host],
                    search_terms=" OR ".join(REBOOT_LOG_TERMS[:10]),
                    limit=limit_per_entity,
                )
                cmds = search_commands_by_node(
                    conn,
                    hostnames=[host],
                    command_patterns=REBOOT_COMMAND_PATTERNS,
                    limit=8,
                )
                block = (
                    f"### Host {host} reboot/crash evidence\n"
                    + "Reboot/crash log hits:\n"
                    + format_node_log_rows(reboot_logs)
                    + "\n\nHost sos_commands (dmesg/last/journal):\n"
                    + format_command_rows(cmds)
                )
            else:
                activity = get_host_activity(
                    conn,
                    host,
                    levels=("CRITICAL", "ERROR", "WARNING"),
                    limit=limit_per_entity,
                )
                relationships = fetch_related_entities(
                    conn,
                    host,
                    relation_types=("instance_host",),
                    limit=100,
                )
                block = (
                    f"### Host {host}\n"
                    "Direct instance relationships (complete observed host inventory):\n"
                    + format_relationships(relationships)
                    + "\n\nHost activity:\n"
                    + format_evidence_mentions(activity)
                )
            host_blocks.append(block)
        sections.append("## Focused host evidence\n" + "\n\n".join(host_blocks))

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
    elif not focus_hosts:
        # Surface top entities so the agent has something concrete to start from.
        top = list_entities(conn, limit=15)
        if top:
            lines = [f"{e.entity_id}|{e.entity_type}|mentions={e.mention_count}" for e in top]
            sections.append("## Top indexed entities\n" + "\n".join(lines))

    if keywords:
        sections.append("## Keywords from plan\n" + ", ".join(keywords[:8]))

    return "\n\n".join(sections)


def build_langchain_tools(conn: Any):
    """Create LangChain tools bound to an open DuckDB connection.

    All tools share one DuckDB connection. Parallel tool calls race and return
    crossed/empty results, so every tool body is serialized with a lock.
    """
    import threading

    from langchain_core.tools import tool

    db_lock = threading.RLock()

    def _safe_tool_output(value: object) -> str:
        """Tool results are model-visible; never return common credentials."""
        return redact_sensitive_text(value)

    reboot_term_re = re.compile(
        r"\b(reboot|panic|watchdog|oom|shutdown|mce|hardware error|kernel)\b",
        re.I,
    )

    @tool
    def get_cluster_overview() -> str:
        """Return the Cluster Manifest: hostnames, roles, RHOSP version, and services per SOS node."""
        with db_lock:
            return _safe_tool_output(format_manifest(conn))

    @tool
    def create_and_run_analysis(
        name: str,
        purpose: str,
        sql: str,
    ) -> str:
        """
        Create and immediately run one temporary read-only analysis when the existing
        investigation tools do not directly answer the user's question.

        Use this for counts, lists, groupings, comparisons, and other questions that
        need a new view of the ingested snapshot. `name` and `purpose` describe the
        one-off tool you are creating; `sql` must be one SELECT/WITH query.

        Available tables: cluster_nodes(cluster_id, hostname, node_role, rhosp_version,
        services, archive_name, archive_id); entities(entity_id, entity_type,
        mention_count, first_seen, last_seen); entity_mentions(entity_id, entity_type,
        timestamp, hostname, service, level, source_file, report_name, message_excerpt);
        entity_relationships(src_entity_id, src_entity_type, relation_type,
        dst_entity_id, dst_entity_type, evidence_count, confidence, first_seen,
        last_seen, hostnames, services, sample_excerpt, sample_hostname,
        sample_service, sample_level); os_logs(..., hostname, node_role, service,
        level, timestamp, message); os_commands(..., hostname, node_role, command,
        output).

        Example: to count instances observed on a host, count distinct src_entity_id
        from entity_relationships where relation_type='instance_host' and
        lower(dst_entity_id)=lower('<hostname>'). SOS reports are snapshots: describe
        results as observed/associated, not live running state, unless live API data
        is available.
        """
        label = (name or "temporary_analysis").strip()[:80]
        objective = (purpose or "Answer the user's question").strip()[:240]
        try:
            statement = validate_analysis_sql(sql)
        except ValueError as exc:
            return f"Cannot create analysis {label!r}: {exc}"
        with db_lock:
            try:
                statement, notes = canonicalize_analysis_hostnames(conn, statement)
                result = conn.execute(statement)
                rendered = format_analysis_rows(result)
            except Exception as exc:  # noqa: BLE001 - give the agent a repairable query error
                return f"Temporary analysis {label!r} failed: {type(exc).__name__}: {exc}"
        note_text = ("\n".join(f"({note})" for note in notes) + "\n") if notes else ""
        return _safe_tool_output(
            f"Temporary analysis created: {label}\nPurpose: {objective}\n{note_text}{rendered}"
        )

    @tool
    def get_host_reboot_timeline(
        hostname: str,
        limit: int = 12,
    ) -> str:
        """
        FIRST tool for reboot/crash questions. Discover WHEN the host last booted
        from SOS data: journalctl --list-boots (preferred), then journalctl --boot
        bounds, then who -b / last / uptime / dmesg / hostnamectl.
        Pass hostname (short like comp008 or FQDN). Do this before compare_nodes
        or generic log searches. Do not treat user-session 'Reached target Shutdown'
        as a host reboot.
        """
        with db_lock:
            return _safe_tool_output(
                format_host_reboot_timeline(
                    conn,
                    hostname,
                    limit=max(1, min(int(limit), 20)),
                )
            )

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
        with db_lock:
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
            return _safe_tool_output(format_node_comparison(rows))

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
        For hostnames (full or short like comp008), also returns recent host activity
        across all services (service= is ignored for the host-activity section).
        """
        with db_lock:
            if not entity_id.strip():
                return "entity_id is required."
            services = [service] if service.strip() else ()
            hostnames = resolve_hostnames(conn, hostname) if hostname.strip() else ()
            # Hostname scope already identifies the node; role filter is redundant and
            # can exclude hosts still stored as role=unknown in older DBs.
            node_roles = ()
            if node_role.strip() and not (hostname.strip() or hostnames):
                node_roles = (node_role.strip(),)
            capped = max(1, min(int(limit), 30))
            eid = entity_id.strip()
            resolved_hosts = resolve_hostnames(conn, eid)
            entity = get_entity(conn, eid)
            if entity is None and resolved_hosts:
                for host in resolved_hosts:
                    entity = get_entity(conn, host)
                    if entity:
                        eid = host
                        break
                if entity is None:
                    eid = resolved_hosts[0]
                    hostnames = resolved_hosts

            header = (
                f"entity={entity.entity_id} type={entity.entity_type} mentions={entity.mention_count}\n"
                if entity
                else f"entity={eid} (not registered in entities table)\n"
            )
            mentions = get_evidence(
                conn,
                eid,
                limit=capped,
                services=services,
                hostnames=hostnames,
                node_roles=node_roles,
            )
            body = format_evidence_mentions(mentions)
            host_for_activity = None
            if entity and entity.entity_type == "host":
                host_for_activity = entity.entity_id
            elif resolved_hosts:
                host_for_activity = resolved_hosts[0]
            if host_for_activity:
                # Host activity must not inherit service=neutron from the agent —
                # reboot/ovn/nova lines would disappear.
                host_rows = get_host_activity(
                    conn,
                    host_for_activity,
                    services=(),
                    limit=capped,
                )
                body = (
                    body
                    + "\n\nHost activity:\n"
                    + format_evidence_mentions(
                        host_rows,
                        empty="No recent WARNING/ERROR/INFO host logs for that hostname.",
                    )
                )
            return _safe_tool_output(header + body)

    @tool
    def list_indexed_entities(entity_type: str = "", limit: int = 20) -> str:
        """
        List indexed entities. Optional entity_type: instance, volume, port, network,
        router, image, request, host, unknown.
        """
        with db_lock:
            capped = max(1, min(int(limit), 50))
            rows = list_entities(
                conn,
                entity_type=entity_type.strip() or None,
                limit=capped,
            )
            if not rows:
                return "No entities in the evidence index. Re-ingest SOS reports first."
            return _safe_tool_output(
                "\n".join(
                    f"{row.entity_id}|{row.entity_type}|mentions={row.mention_count}"
                    for row in rows
                )
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
        Hostname may be short (comp008) or FQDN; it is resolved against cluster_nodes.
        For alternatives use OR, e.g. 'reboot OR panic OR watchdog'.
        For reboot/crash searches leave service empty.
        """
        with db_lock:
            capped = max(1, min(int(limit), 30))
            hostnames = resolve_hostnames(conn, hostname) if hostname.strip() else ()
            notes: list[str] = []
            if hostname.strip() and hostnames and hostnames[0].lower() != hostname.strip().lower():
                notes.append(f"resolved hostname {hostname!r} → {', '.join(hostnames)}")

            # Reboot/crash evidence is rarely tagged as neutron/nova.
            service_value = service.strip()
            if service_value and reboot_term_re.search(search_terms or ""):
                notes.append(f"ignored service={service_value!r} for reboot/crash search")
                service_value = ""
            services = [service_value] if service_value else ()

            # Prefer hostname-only scope; role filter is optional fallback.
            node_roles = ()
            if node_role.strip() and not hostnames:
                node_roles = (node_role.strip(),)
            elif node_role.strip() and hostnames:
                notes.append(f"ignored node_role={node_role!r} because hostname is set")

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
                    return _safe_tool_output(
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
                search_terms=search_terms,
                resource_id=resource_id.strip(),
                limit=capped,
            )
            prefix = ("\n".join(f"({n})" for n in notes) + "\n") if notes else ""
            return _safe_tool_output(prefix + format_node_log_rows(rows))

    @tool
    def search_sos_commands(
        hostname: str = "",
        command_pattern: str = "",
        search_terms: str = "",
        limit: int = 10,
    ) -> str:
        """
        Search sos_commands outputs (dmesg, last, journalctl, uptime, ipmitool, ...).
        Critical for host reboot/crash RCA. Hostname may be short (comp008).
        Prefer leaving command_pattern empty to search the full reboot artifact set,
        or pass a comma list such as dmesg,last,journalctl,uptime,ipmitool.
        """
        with db_lock:
            capped = max(1, min(int(limit), 20))
            hostnames = resolve_hostnames(conn, hostname) if hostname.strip() else ()
            requested = (
                [p.strip() for p in re.split(r"[,\s]+", command_pattern) if p.strip()]
                if command_pattern.strip()
                else []
            )
            patterns = requested or list(REBOOT_COMMAND_PATTERNS)
            rows = search_commands_by_node(
                conn,
                hostnames=hostnames,
                command_patterns=patterns,
                search_terms=search_terms,
                limit=capped,
            )
            notes: list[str] = []
            # Narrow patterns like dmesg,last,journalctl often miss RHOSP archives
            # that only ship ipmitool/uptime — fall back to the full reboot set.
            if (
                not rows
                and requested
                and set(p.lower() for p in requested)
                != set(p.lower() for p in REBOOT_COMMAND_PATTERNS)
            ):
                rows = search_commands_by_node(
                    conn,
                    hostnames=hostnames,
                    command_patterns=list(REBOOT_COMMAND_PATTERNS),
                    search_terms=search_terms,
                    limit=capped,
                )
                if rows:
                    notes.append(
                        "no hits for command_pattern="
                        f"{command_pattern!r}; expanded to reboot command set"
                    )
            # If term filter was too strict, return the raw command artifacts.
            if not rows and search_terms.strip():
                rows = search_commands_by_node(
                    conn,
                    hostnames=hostnames,
                    command_patterns=patterns
                    if not notes
                    else list(REBOOT_COMMAND_PATTERNS),
                    search_terms="",
                    limit=capped,
                )
                notes.append(
                    "no output matched search_terms; showing unfiltered command artifacts"
                )
            if (
                hostname.strip()
                and hostnames
                and hostnames[0].lower() != hostname.strip().lower()
            ):
                notes.insert(
                    0, f"resolved hostname {hostname!r} → {', '.join(hostnames)}"
                )
            note = ("\n".join(f"({n})" for n in notes) + "\n") if notes else ""
            return _safe_tool_output(note + format_command_rows(rows))

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
        with db_lock:
            if not entity_id.strip():
                return "entity_id is required."
            types = [relation_type.strip()] if relation_type.strip() else ()
            eid = entity_id.strip()
            if get_entity(conn, eid) is None:
                resolved = resolve_hostnames(conn, eid)
                if resolved:
                    eid = resolved[0]
            rows = fetch_related_entities(
                conn,
                eid,
                relation_types=types,
                limit=max(1, min(int(limit), 50)),
            )
            return _safe_tool_output(format_relationships(rows))

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
        with db_lock:
            if not start_entity_id.strip():
                return "start_entity_id is required."
            start = start_entity_id.strip()
            if get_entity(conn, start) is None:
                resolved = resolve_hostnames(conn, start)
                if resolved:
                    start = resolved[0]
            path = fetch_operation_path(
                conn,
                start,
                target_entity_id=target_entity_id.strip(),
                target_type=target_type.strip(),
                max_hops=max(1, min(int(max_hops), 6)),
            )
            return _safe_tool_output(format_operation_path(path, start_entity_id=start))

    return [
        create_and_run_analysis,
        get_host_reboot_timeline,
        get_cluster_overview,
        compare_nodes,
        get_entity_evidence,
        get_related_entities,
        get_operation_path,
        list_indexed_entities,
        search_os_logs,
        search_sos_commands,
    ]
