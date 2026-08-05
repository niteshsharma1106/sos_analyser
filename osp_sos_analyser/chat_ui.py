# chat_ui.py — React chat UI + FastAPI backend for SOS investigations.
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .detective import investigate_prompt_offline
from .env_config import describe_llm_settings, load_app_env
from .evidence_index import get_cluster_manifest
from .langgraph_investigator import (
    investigate_with_langgraph,
    render_investigation_result,
)
from .llm_client import MissingLLMConfiguration
from .observability import configure_logging, get_logger
from .privacy import redact_sensitive_text
from .relationship_graph import (
    format_operation_path,
    format_relationships,
    get_operation_path,
    get_related_entities,
)


DEFAULT_DB_PATH = "sos_analysis.duckdb"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_EXAMPLES = [
    {
        "label": "Compute network loss",
        "prompt": "Why did compute-03 lose network connectivity?",
    },
    {
        "label": "Port bind path",
        "prompt": (
            "Port 55ab45cf-6925-4811-a008-6fe60d491c5b failed to bind — "
            "show VM→port→chassis→host"
        ),
    },
    {
        "label": "Host for VM/port",
        "prompt": "What host is related to this VM/port failure?",
    },
    {
        "label": "NoValidHost fail",
        "prompt": "VM create failed with NoValidHost",
    },
]


def _cluster_banner(db_path: str) -> str:
    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            nodes = get_cluster_manifest(conn)
    except Exception:  # noqa: BLE001 - banner is best-effort
        return ""
    if not nodes:
        return ""
    roles: dict[str, list[str]] = {}
    for node in nodes:
        roles.setdefault(node.get("node_role") or "unknown", []).append(
            node.get("hostname") or "?"
        )
    role_bits = ", ".join(f"{role}={len(hosts)}" for role, hosts in sorted(roles.items()))
    return f"Cluster `{nodes[0].get('cluster_id')}` · nodes={len(nodes)} ({role_bits})"


def _offline_graph_digest(db_path: str, focus_entity: str) -> str:
    entity = (focus_entity or "").strip()
    if not entity:
        return ""
    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            related = get_related_entities(conn, entity, limit=15)
            path = get_operation_path(conn, entity, target_type="host", max_hops=4)
    except Exception as exc:  # noqa: BLE001
        return f"Graph lookup failed: {exc}"
    parts = []
    if related:
        parts.append("## Related entities\n" + format_relationships(related))
    if path:
        parts.append("## Operation path\n" + format_operation_path(path, start_entity_id=entity))
    return "\n\n".join(parts)


def _exception_chain_text(exc: BaseException) -> str:
    """Flatten ``__cause__`` / ``__context__`` so nested SSL errors are visible."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


def _normalize_chat_text(message: Any) -> str:
    """Extract plain text from chat content (str or multimodal-style blocks)."""
    if message is None:
        return ""
    if isinstance(message, str):
        text = message.strip()
        # Legacy Gradio history sometimes stringified multimodal blocks.
        if text.startswith("[{") and "'text'" in text and "'type'" in text:
            try:
                import ast

                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return text
            return _normalize_chat_text(parsed)
        return text
    if isinstance(message, dict):
        if "text" in message:
            return _normalize_chat_text(message.get("text"))
        if "content" in message:
            return _normalize_chat_text(message.get("content"))
        return str(message.get("text") or message.get("content") or "").strip()
    if isinstance(message, (list, tuple)):
        parts: list[str] = []
        for item in message:
            part = _normalize_chat_text(item)
            if part:
                parts.append(part)
        return "\n".join(parts).strip()
    return str(message).strip()


def _answer_question(
    message: str,
    history: list,
    db_path: str,
    offline: bool,
    model: str,
    focus_entity: str = "",
    include_graph: bool = True,
    answer_style: str = "Concise RCA",
    show_observability: bool = True,
) -> str:
    del history  # reserved for future multi-turn context
    log = get_logger("chat")
    prompt = _normalize_chat_text(message)
    if not prompt:
        return "Ask an OpenStack / SOS investigation question to begin."

    db = (db_path or DEFAULT_DB_PATH).strip() or DEFAULT_DB_PATH
    if not Path(db).exists():
        return (
            f"Database not found: `{db}`.\n\n"
            "Ingest SOS reports first, for example:\n"
            "`uv run python main.py ingest --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb`"
        )

    banner = _cluster_banner(db)
    prefix = f"{banner}\n\n" if banner else ""
    log.info(
        "Chat question offline=%s graph=%s style=%s focus=%s prompt=%s",
        offline,
        include_graph,
        answer_style,
        focus_entity or "-",
        redact_sensitive_text(prompt)[:160],
    )

    try:
        if offline:
            report = investigate_prompt_offline(db_path=db, prompt=prompt)
            body = report.render_markdown()
            if include_graph:
                graph = _offline_graph_digest(db, focus_entity or prompt)
                if graph:
                    body = body + "\n\n" + graph
            return prefix + body

        result = investigate_with_langgraph(
            db_path=db,
            prompt=prompt,
            model=(model or "").strip() or None,
            focus_entity=focus_entity.strip() or None,
            answer_style=answer_style,
            include_graph=include_graph,
        )
        return prefix + render_investigation_result(
            result,
            include_observability=show_observability,
        )
    except MissingLLMConfiguration as exc:
        log.warning("Missing LLM configuration: %s", exc)
        return (
            f"{exc}\n\n"
            "Configure `OSP_SOS_MODEL`, `OSP_SOS_MODEL_PROVIDER`, and the matching "
            "API key in `.env`, or enable **Offline mode** in settings."
        )
    except Exception as exc:  # noqa: BLE001 - surface runtime errors in the chat UI
        log.exception("Chat investigation failed")
        text = _exception_chain_text(exc)
        lower = text.lower()
        if "being used by another process" in text or "Cannot open file" in text:
            return (
                "Database is locked by another process (often a running ingest).\n\n"
                "Wait for ingest to finish, or stop the other process, then ask again.\n\n"
                f"Details: `{type(exc).__name__}: {exc}`"
            )
        if (
            "CERTIFICATE_VERIFY_FAILED" in text
            or "certificate verify failed" in lower
            or "unable to get local issuer certificate" in lower
        ):
            return (
                "The LLM provider connection was blocked because Python could not "
                "verify the TLS certificate (common with corporate TLS inspection).\n\n"
                "Fix options:\n"
                "1. Restart chat after pulling the latest code (uses the Windows trust store).\n"
                "2. Or set your company root CA in `.env`:\n"
                "   `OSP_SOS_CA_BUNDLE=C:\\path\\to\\company-root-ca.pem`\n"
                "3. Or enable **Offline mode** in Settings for DuckDB-only answers.\n\n"
                "Certificate verification stays enabled."
            )
        return f"Investigation failed: `{type(exc).__name__}: {exc}`"


class AskRequest(BaseModel):
    message: str
    db_path: str = DEFAULT_DB_PATH
    offline: bool = False
    model: str = ""
    focus_entity: str = ""
    include_graph: bool = True
    answer_style: str = "Concise RCA"
    show_observability: bool = False


class AskResponse(BaseModel):
    answer: str


class ChatDefaults(BaseModel):
    db_path: str
    offline: bool
    model: str = ""
    focus_entity: str = ""
    include_graph: bool = True
    answer_style: str = "Concise RCA"
    show_observability: bool = False


class BootstrapResponse(BaseModel):
    cluster_banner: str
    llm_status: str
    defaults: ChatDefaults
    examples: list[dict[str, str]] = Field(default_factory=list)


def build_chat_app(
    *,
    default_db_path: str = DEFAULT_DB_PATH,
    default_offline: bool = False,
    default_model: str | None = None,
) -> FastAPI:
    """Build the FastAPI app that serves the React UI and investigation API."""
    load_app_env()
    cli_model = (default_model or "").strip()

    app = FastAPI(title="OSP SOS", docs_url=None, redoc_url=None)

    @app.get("/api/bootstrap", response_model=BootstrapResponse)
    def bootstrap() -> BootstrapResponse:
        banner = ""
        if Path(default_db_path).exists():
            banner = _cluster_banner(default_db_path)
        return BootstrapResponse(
            cluster_banner=banner,
            llm_status=describe_llm_settings(),
            defaults=ChatDefaults(
                db_path=default_db_path,
                offline=default_offline,
                model=cli_model,
            ),
            examples=DEFAULT_EXAMPLES,
        )

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        answer = _answer_question(
            payload.message,
            [],
            payload.db_path,
            payload.offline,
            payload.model or cli_model,
            payload.focus_entity,
            payload.include_graph,
            payload.answer_style,
            payload.show_observability,
        )
        return AskResponse(answer=answer)

    if not STATIC_DIR.exists():
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        index_path.write_text(
            "<!doctype html><html><body><p>OSP SOS UI assets missing. "
            "Build with: cd web && npm install && npm run build</p></body></html>",
            encoding="utf-8",
        )

    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str = "") -> FileResponse:
        candidate = STATIC_DIR / full_path if full_path else index_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_path)

    return app


def launch_chat(
    *,
    db_path: str = DEFAULT_DB_PATH,
    offline: bool = False,
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    log_level: str | None = None,
    log_file: str | None = None,
) -> None:
    del share  # Gradio share links are no longer used
    import uvicorn

    configure_logging(level=log_level, log_file=log_file)
    get_logger("chat").info(
        "Launching React chat UI db=%s host=%s port=%s offline=%s",
        db_path,
        host,
        port,
        offline,
    )

    app = build_chat_app(
        default_db_path=db_path,
        default_offline=offline,
        default_model=model,
    )
    get_logger("chat").info("Open http://%s:%s in your browser", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the OSP SOS chat UI")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="DuckDB database path")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Default the UI to offline deterministic mode",
    )
    parser.add_argument("--model", default=None, help="Optional model override")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Ignored (kept for CLI compatibility)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("OSP_SOS_LOG_LEVEL", "INFO"),
        help="Backend log level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("OSP_SOS_LOG_FILE"),
        help="Optional log file path (also set OSP_SOS_LOG_FILE)",
    )
    args = parser.parse_args(argv)
    launch_chat(
        db_path=args.db_path,
        offline=args.offline,
        model=args.model,
        host=args.host,
        port=args.port,
        share=args.share,
        log_level=args.log_level,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
