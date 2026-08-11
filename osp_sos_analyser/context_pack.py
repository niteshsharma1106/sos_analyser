# context_pack.py — compact evidence packing for LLM tool results / prefetch.
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

DIGEST_MESSAGE_CHARS = 500
DIGEST_OUTPUT_CHARS = 500
DEFAULT_PREFETCH_LIMIT = 20
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
REQUEST_ID_RE = re.compile(r"\breq-[0-9a-fA-F-]{8,}\b")
TIME_RE = re.compile(
    r"\b(?:(\d{4}-\d{2}-\d{2})[ T])?(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\b"
)


def truncate_text(value: str | None, limit: int = DIGEST_MESSAGE_CHARS) -> str:
    text = (value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def format_log_digest_row(row: Sequence[Any], message_chars: int = DIGEST_MESSAGE_CHARS) -> str:
    """Format one log row as a single compact line for LLM context."""
    timestamp = row[0] if len(row) > 0 else ""
    service = row[1] if len(row) > 1 else ""
    level = row[2] if len(row) > 2 else ""
    message = row[3] if len(row) > 3 else ""
    source = row[4] if len(row) > 4 else ""
    parts = [str(timestamp or ""), str(service or ""), str(level or ""), truncate_text(str(message or ""), message_chars)]
    if source:
        parts.append(str(source))
    return "|".join(parts)


def format_log_digest(
    rows: Sequence[Sequence[Any]],
    *,
    message_chars: int = DIGEST_MESSAGE_CHARS,
    empty: str = "No matching logs.",
) -> str:
    if not rows:
        return empty
    return "\n".join(format_log_digest_row(row, message_chars=message_chars) for row in rows)


def parse_time_hint(text: str | None) -> datetime | None:
    """Parse a loose time/date hint into a datetime (date defaults to 1970-01-01)."""
    if not text:
        return None
    match = TIME_RE.search(text.strip())
    if not match:
        return None
    date_part, time_part = match.group(1), match.group(2)
    candidates = []
    if date_part:
        candidates.append(f"{date_part} {time_part}")
        candidates.append(f"{date_part}T{time_part}")
    else:
        candidates.append(f"1970-01-01 {time_part}")
    for candidate in candidates:
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
        ):
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    return None


def extract_identifiers(text: str) -> list[str]:
    found = UUID_RE.findall(text) + REQUEST_ID_RE.findall(text)
    return list(dict.fromkeys(found))


def prefetch_evidence_digest(
    conn: Any,
    *,
    identifiers: Sequence[str] = (),
    services: Sequence[str] = (),
    keywords: Sequence[str] = (),
    time_hint: str | None = None,
    window_minutes: int = 10,
    limit: int = DEFAULT_PREFETCH_LIMIT,
) -> str:
    """Run cheap first-pass DuckDB lookups and return a digest for the LLM.

    Priority: identifier hits → time window around hint → keyword/service ERROR rows.
    """
    sections: list[str] = []
    ids = [item for item in identifiers if item]
    if ids:
        id_rows = _search_identifiers(conn, ids, services=services, limit=limit)
        sections.append("## Identifier hits\n" + format_log_digest(id_rows))

    center = parse_time_hint(time_hint)
    if center is not None:
        # If only a clock time was provided (year 1970), match any date at that clock.
        if center.year == 1970:
            clock = center.strftime("%H:%M:%S")
            rows = conn.execute(
                """
                SELECT timestamp, service, level, message, source_file
                FROM os_logs
                WHERE timestamp IS NOT NULL
                  AND strftime(timestamp, '%H:%M:%S') BETWEEN ? AND ?
                ORDER BY
                  CASE level
                    WHEN 'CRITICAL' THEN 0 WHEN 'ERROR' THEN 1
                    WHEN 'WARNING' THEN 2 ELSE 3
                  END,
                  timestamp
                LIMIT ?
                """,
                [
                    (datetime(1970, 1, 1, center.hour, center.minute, center.second) - timedelta(minutes=window_minutes)).strftime("%H:%M:%S"),
                    (datetime(1970, 1, 1, center.hour, center.minute, center.second) + timedelta(minutes=window_minutes)).strftime("%H:%M:%S"),
                    limit,
                ],
            ).fetchall()
        else:
            start = center - timedelta(minutes=window_minutes)
            end = center + timedelta(minutes=window_minutes)
            clauses = ["timestamp BETWEEN ? AND ?"]
            params: list[object] = [start, end]
            if services:
                clauses.append(f"service IN ({', '.join('?' for _ in services)})")
                params.extend(services)
            params.append(limit)
            rows = conn.execute(
                f"""
                SELECT timestamp, service, level, message, source_file
                FROM os_logs
                WHERE {' AND '.join(clauses)}
                ORDER BY timestamp
                LIMIT ?
                """,
                params,
            ).fetchall()
        sections.append(f"## Events near {time_hint}\n" + format_log_digest(rows))

    if keywords or services:
        kw_rows = _search_keywords(conn, keywords=keywords, services=services, limit=min(limit, 15))
        sections.append("## Keyword / service errors\n" + format_log_digest(kw_rows))

    summary = conn.execute(
        """
        SELECT COALESCE(service, 'unknown'), level, COUNT(*)
        FROM os_logs
        WHERE level IN ('ERROR', 'CRITICAL')
        GROUP BY 1, 2
        ORDER BY 3 DESC
        LIMIT 10
        """
    ).fetchall()
    if summary:
        lines = [f"{service}|{level}|{count}" for service, level, count in summary]
        sections.append("## Error summary\n" + "\n".join(lines))

    if not sections:
        return "No prefetched evidence."
    return "\n\n".join(sections)


def _search_identifiers(
    conn: Any,
    identifiers: Sequence[str],
    *,
    services: Sequence[str] = (),
    limit: int,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for identifier in identifiers[:5]:
        clauses = ["(message ILIKE ? OR module ILIKE ? OR source_file ILIKE ?)"]
        params: list[object] = [f"%{identifier}%", f"%{identifier}%", f"%{identifier}%"]
        if services:
            clauses.append(f"service IN ({', '.join('?' for _ in services)})")
            params.extend(services)
        params.append(max(1, limit // max(1, min(len(identifiers), 5))))
        part = conn.execute(
            f"""
            SELECT timestamp, service, level, message, source_file
            FROM os_logs
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp NULLS LAST
            LIMIT ?
            """,
            params,
        ).fetchall()
        rows.extend(part)
    return _dedupe_rows(rows)[:limit]


def _search_keywords(
    conn: Any,
    *,
    keywords: Sequence[str],
    services: Sequence[str],
    limit: int,
) -> list[tuple[Any, ...]]:
    clauses = ["level IN ('ERROR', 'CRITICAL', 'WARNING')"]
    params: list[object] = []
    if services:
        clauses.append(f"service IN ({', '.join('?' for _ in services)})")
        params.extend(services)
    # OR keywords so prefetch stays recall-oriented; digest stays small via LIMIT.
    if keywords:
        term_clauses = []
        for term in list(keywords)[:5]:
            term_clauses.append("message ILIKE ?")
            params.append(f"%{term}%")
        clauses.append(f"({' OR '.join(term_clauses)})")
    params.append(limit)
    return conn.execute(
        f"""
        SELECT timestamp, service, level, message, source_file
        FROM os_logs
        WHERE {' AND '.join(clauses)}
        ORDER BY
          CASE level WHEN 'CRITICAL' THEN 0 WHEN 'ERROR' THEN 1 WHEN 'WARNING' THEN 2 ELSE 3 END,
          timestamp NULLS LAST
        LIMIT ?
        """,
        params,
    ).fetchall()


def _dedupe_rows(rows: Sequence[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[tuple[Any, ...]] = []
    for row in rows:
        key = (row[0], row[1], row[2], truncate_text(str(row[3]) if len(row) > 3 else "", 120))
        if key in seen:
            continue
        seen.add(key)
        out.append(tuple(row))
    return out


def compact_message_history(
    messages: Sequence[Any],
    *,
    keep_last: int = 6,
    tool_result_chars: int = 800,
) -> list[Any]:
    """Trim older chat/tool messages and shrink large tool payloads.

    Accepts LangChain-style message objects or role/content dicts. Older messages
    beyond ``keep_last`` are dropped (system messages are always kept).
    """
    items = list(messages)
    systems = [m for m in items if _message_role(m) == "system"]
    non_systems = [m for m in items if _message_role(m) != "system"]
    kept = non_systems[-keep_last:] if keep_last > 0 else non_systems
    compacted: list[Any] = []
    for message in [*systems, *kept]:
        compacted.append(_shrink_message(message, tool_result_chars=tool_result_chars))
    return compacted


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "").lower()
    role = getattr(message, "type", None) or getattr(message, "role", None)
    return str(role or "").lower()


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content or "")


def _shrink_message(message: Any, *, tool_result_chars: int) -> Any:
    role = _message_role(message)
    content = _message_content(message)
    if role in {"tool", "function"} or (role == "assistant" and len(content) > tool_result_chars):
        shrunk = truncate_text(content, tool_result_chars)
        if isinstance(message, dict):
            return {**message, "content": shrunk}
        try:
            clone = message.model_copy(update={"content": shrunk})  # pydantic/langchain
            return clone
        except Exception:
            try:
                message.content = shrunk
            except Exception:
                pass
            return message
    return message
