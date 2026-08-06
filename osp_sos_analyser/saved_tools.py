# saved_tools.py — persistent, reviewable investigation tools (create → run → keep).
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PARAM_RE = re.compile(r"\{\{(\w+)\}\}")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_MAX_SQL_CHARS = 8000
_MAX_ARGS_JSON_CHARS = 4000

# Built-in LangChain tool names that must not be overwritten by saved tools.
RESERVED_TOOL_NAMES = frozenset(
    {
        "create_and_run_analysis",
        "create_and_save_investigation_tool",
        "run_saved_investigation_tool",
        "list_saved_investigation_tools",
        "get_host_reboot_timeline",
        "search_peer_mentions",
        "get_cluster_overview",
        "compare_nodes",
        "get_entity_evidence",
        "get_related_entities",
        "get_operation_path",
        "list_indexed_entities",
        "search_os_logs",
        "search_sos_commands",
    }
)


@dataclass
class SavedInvestigationTool:
    """A durable, read-only DuckDB analysis tool created during investigation."""

    name: str
    purpose: str
    sql: str
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SavedInvestigationTool:
        params = data.get("parameters") or []
        if isinstance(params, str):
            params = [p.strip() for p in params.split(",") if p.strip()]
        name = str(data.get("name") or "").strip()
        sql = str(data.get("sql") or "").strip()
        purpose = str(data.get("purpose") or "").strip()
        description = str(data.get("description") or purpose).strip()
        inferred = extract_sql_parameters(sql)
        merged = list(dict.fromkeys([*params, *inferred]))
        return cls(
            name=name,
            purpose=purpose,
            sql=sql,
            description=description or purpose,
            parameters=merged,
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


def default_saved_tools_dir() -> Path:
    custom = os.getenv("OSP_SOS_SAVED_TOOLS_DIR", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return Path(__file__).resolve().parent / "saved_investigation_tools"


def extract_sql_parameters(sql: str) -> list[str]:
    return list(dict.fromkeys(_PARAM_RE.findall(sql or "")))


def validate_tool_name(name: str) -> str:
    cleaned = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not _NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid tool name {name!r}. Use snake_case: start with a letter, "
            "3–64 chars of [a-z0-9_]."
        )
    if cleaned in RESERVED_TOOL_NAMES:
        raise ValueError(f"Tool name {cleaned!r} conflicts with a built-in tool.")
    return cleaned


def bind_sql_template(sql: str, args: dict[str, Any]) -> tuple[str, list[Any]]:
    """Replace ``{{param}}`` markers with ``?`` placeholders and collect bound values."""
    missing: list[str] = []
    values: list[Any] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in args:
            missing.append(key)
            return "?"
        values.append(args[key])
        return "?"

    bound = _PARAM_RE.sub(_replace, sql)
    if missing:
        raise ValueError(
            "Missing required parameters: "
            + ", ".join(dict.fromkeys(missing))
            + f". Provide them in args_json. Known keys: {sorted(args)}"
        )
    return bound, values


def parse_args_json(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    text = str(raw).strip()
    if not text:
        return {}
    if len(text) > _MAX_ARGS_JSON_CHARS:
        raise ValueError("args_json is too large.")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("args_json must be a JSON object.")
    return {str(k): v for k, v in data.items()}


def _tool_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.json"


def list_saved_tool_definitions(
    directory: Path | None = None,
) -> list[SavedInvestigationTool]:
    root = directory or default_saved_tools_dir()
    if not root.is_dir():
        return []
    tools: list[SavedInvestigationTool] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tools.append(SavedInvestigationTool.from_dict(data))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return tools


def load_saved_tool(
    name: str,
    *,
    directory: Path | None = None,
) -> SavedInvestigationTool | None:
    root = directory or default_saved_tools_dir()
    path = _tool_path(root, validate_tool_name(name) if name else "")
    if not path.is_file():
        # Allow loading without re-validating reserved (already on disk).
        path = root / f"{(name or '').strip()}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SavedInvestigationTool.from_dict(data)


def save_tool_definition(
    tool: SavedInvestigationTool,
    *,
    directory: Path | None = None,
    overwrite: bool = False,
) -> Path:
    root = directory or default_saved_tools_dir()
    root.mkdir(parents=True, exist_ok=True)
    name = validate_tool_name(tool.name)
    path = _tool_path(root, name)
    if path.exists() and not overwrite:
        raise ValueError(
            f"Saved tool {name!r} already exists at {path}. "
            "Pass overwrite=true to replace it, or choose a new name."
        )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tool.name = name
    if not tool.created_at:
        tool.created_at = now
    tool.updated_at = now
    if not tool.parameters:
        tool.parameters = extract_sql_parameters(tool.sql)
    path.write_text(json.dumps(tool.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Marker so the directory is clearly intentional.
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Saved investigation tools\n\n"
            "JSON definitions created by the RCA agent via "
            "`create_and_save_investigation_tool`.\n"
            "Each file is a read-only DuckDB SQL tool with `{{param}}` placeholders.\n"
            "Review before committing; set `OSP_SOS_SAVED_TOOLS_DIR` to override this path.\n",
            encoding="utf-8",
        )
    return path


def prepare_tool_sql(
    conn: Any,
    sql: str,
    args: dict[str, Any],
) -> tuple[str, list[Any], list[str]]:
    """Validate SQL, bind ``{{params}}``, and canonicalize hostname-like args."""
    from .investigation_tools import (
        canonicalize_analysis_hostnames,
        resolve_hostnames,
        validate_analysis_sql,
    )

    if len(sql or "") > _MAX_SQL_CHARS:
        raise ValueError(f"SQL exceeds {_MAX_SQL_CHARS} characters.")
    statement = validate_analysis_sql(sql)
    notes: list[str] = []
    prepared_args: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and value.strip():
            text = value.strip()
            # Resolve short host tokens when the parameter looks host-related.
            if "host" in key.lower() or key.lower() in {"node", "peer", "target"}:
                resolved = resolve_hostnames(conn, text)
                if resolved and resolved[0].lower() != text.lower():
                    notes.append(f"resolved {key} {text!r} → {resolved[0]!r}")
                    prepared_args[key] = resolved[0]
                    continue
            prepared_args[key] = text
        else:
            prepared_args[key] = value

    bound_sql, params = bind_sql_template(statement, prepared_args)
    # Do not run literal hostname rewriting on bound SQL: patterns like
    # COALESCE(x,'') + ILIKE '%' confuse quote matching, and hostname args are
    # already resolved above.
    return bound_sql, params, notes


def run_saved_tool(
    conn: Any,
    name: str,
    args: dict[str, Any] | str | None = None,
    *,
    directory: Path | None = None,
    row_limit: int = 100,
) -> str:
    """Load a saved tool by name and execute it with the given args."""
    from .investigation_tools import format_analysis_rows

    tool = load_saved_tool(name, directory=directory)
    if tool is None:
        available = ", ".join(t.name for t in list_saved_tool_definitions(directory)) or "(none)"
        return f"No saved tool named {name!r}. Available: {available}"
    try:
        arg_map = parse_args_json(args)
    except (json.JSONDecodeError, ValueError) as exc:
        return f"Invalid args_json for {tool.name}: {exc}"
    try:
        statement, params, notes = prepare_tool_sql(conn, tool.sql, arg_map)
        result = conn.execute(statement, params) if params else conn.execute(statement)
        rendered = format_analysis_rows(result, limit=row_limit)
    except Exception as exc:  # noqa: BLE001 - agent-facing repairable error
        return f"Saved tool {tool.name!r} failed: {type(exc).__name__}: {exc}"
    note_text = ("\n".join(f"({n})" for n in notes) + "\n") if notes else ""
    return (
        f"Saved tool: {tool.name}\n"
        f"Purpose: {tool.purpose}\n"
        f"{note_text}{rendered}"
    )


def create_and_persist_tool(
    conn: Any,
    *,
    name: str,
    purpose: str,
    sql: str,
    description: str = "",
    args: dict[str, Any] | str | None = None,
    persist: bool = True,
    overwrite: bool = False,
    directory: Path | None = None,
    row_limit: int = 100,
) -> str:
    """
    Validate a new read-only SQL tool, run it immediately, and optionally persist it.

    Persistence writes a reviewable JSON file under ``saved_investigation_tools/``
    (or ``OSP_SOS_SAVED_TOOLS_DIR``). The tool can be re-run later via
    ``run_saved_investigation_tool`` or as a dynamically registered tool on the
    next investigator session.
    """
    from .investigation_tools import format_analysis_rows

    try:
        tool_name = validate_tool_name(name)
    except ValueError as exc:
        return f"Cannot create tool: {exc}"
    objective = (purpose or "").strip()
    if not objective:
        return "Cannot create tool: purpose is required."
    try:
        arg_map = parse_args_json(args)
    except (json.JSONDecodeError, ValueError) as exc:
        return f"Cannot create tool {tool_name!r}: invalid args_json: {exc}"

    tool = SavedInvestigationTool(
        name=tool_name,
        purpose=objective[:500],
        sql=(sql or "").strip(),
        description=(description or objective)[:800],
        parameters=extract_sql_parameters(sql or ""),
    )
    try:
        statement, params, notes = prepare_tool_sql(conn, tool.sql, arg_map)
        result = conn.execute(statement, params) if params else conn.execute(statement)
        rendered = format_analysis_rows(result, limit=row_limit)
    except Exception as exc:  # noqa: BLE001
        return (
            f"Cannot create tool {tool_name!r}: validation/run failed: "
            f"{type(exc).__name__}: {exc}"
        )

    saved_line = "persist=false (one-shot only; not kept for future sessions)"
    if persist:
        try:
            path = save_tool_definition(tool, directory=directory, overwrite=overwrite)
            saved_line = f"Saved for future use: {path}"
        except Exception as exc:  # noqa: BLE001
            return (
                f"Tool {tool_name!r} ran successfully but was NOT saved: {exc}\n"
                f"Result:\n{rendered}"
            )

    note_text = ("\n".join(f"({n})" for n in notes) + "\n") if notes else ""
    params_help = ", ".join(tool.parameters) or "(none)"
    return (
        f"Investigation tool created: {tool_name}\n"
        f"Purpose: {tool.purpose}\n"
        f"Parameters: {params_help}\n"
        f"{saved_line}\n"
        f"Re-run later with run_saved_investigation_tool(name={tool_name!r}, "
        f"args_json=...)\n"
        f"{note_text}Result:\n{rendered}"
    )


def format_saved_tools_catalog(directory: Path | None = None) -> str:
    tools = list_saved_tool_definitions(directory)
    root = directory or default_saved_tools_dir()
    if not tools:
        return f"No saved investigation tools in {root}."
    lines = [f"Saved investigation tools ({len(tools)}) in {root}:"]
    for tool in tools:
        params = ", ".join(tool.parameters) or "(none)"
        lines.append(f"- {tool.name}: {tool.purpose} [params: {params}]")
    return "\n".join(lines)
