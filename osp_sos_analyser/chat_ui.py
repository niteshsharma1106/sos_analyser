# chat_ui.py — interactive Q&A UI for SOS investigations (Gradio chat).
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

from .detective import investigate_prompt_offline
from .evidence_index import get_cluster_manifest
from .langgraph_investigator import (
    investigate_with_langgraph,
    render_investigation_result,
)
from .llm_client import MissingLLMConfiguration
from .relationship_graph import get_operation_path, get_related_entities, format_operation_path, format_relationships


DEFAULT_DB_PATH = "sos_analysis.duckdb"
DEFAULT_EXAMPLES = [
    "Why did compute-03 lose network connectivity?",
    "Port 55ab45cf-6925-4811-a008-6fe60d491c5b failed to bind — show VM→port→chassis→host",
    "What host is related to this VM/port failure?",
    "VM create failed with NoValidHost",
    "Compare controller vs compute errors for this incident",
]


def _cluster_banner(db_path: str) -> str:
    try:
        with duckdb.connect(db_path, read_only=True) as conn:
            nodes = get_cluster_manifest(conn)
    except Exception:  # noqa: BLE001 - banner is best-effort
        return ""
    if not nodes:
        return ""
    roles = {}
    for node in nodes:
        roles.setdefault(node.get("node_role") or "unknown", []).append(node.get("hostname") or "?")
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


def _answer_question(
    message: str,
    history: list,
    db_path: str,
    offline: bool,
    model: str,
    focus_entity: str = "",
    include_graph: bool = True,
    answer_style: str = "Concise RCA",
) -> str:
    prompt = (message or "").strip()
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
            model=model.strip() or None,
            focus_entity=focus_entity.strip() or None,
            answer_style=answer_style,
            include_graph=include_graph,
        )
        return prefix + render_investigation_result(result)
    except MissingLLMConfiguration as exc:
        return (
            f"{exc}\n\n"
            "Set your API key in `.env`, or enable **Offline mode** in the sidebar."
        )
    except Exception as exc:  # noqa: BLE001 - surface runtime errors in the chat UI
        return f"Investigation failed: `{type(exc).__name__}: {exc}`"


def build_chat_app(
    *,
    default_db_path: str = DEFAULT_DB_PATH,
    default_offline: bool = False,
    default_model: str | None = None,
):
    import gradio as gr

    def respond(
        message: str,
        history: list,
        db_path: str,
        offline: bool,
        model: str,
        focus_entity: str,
        include_graph: bool,
        answer_style: str,
    ):
        return _answer_question(
            message,
            history,
            db_path,
            offline,
            model,
            focus_entity,
            include_graph,
            answer_style,
        )

    model_value = default_model or os.getenv("OSP_SOS_MODEL", "")
    banner = ""
    if Path(default_db_path).exists():
        banner = _cluster_banner(default_db_path)
    description = (
        "Ask a question about your ingested RHOSP SOS reports. "
        "Uses Cluster Manifest, Evidence Index, and VM↔port↔chassis↔host graph."
    )
    if banner:
        description = f"{banner}\n\n{description}"

    examples = [
        [prompt, default_db_path, default_offline, model_value, "", True, "Concise RCA"]
        for prompt in DEFAULT_EXAMPLES
    ]

    chat = gr.ChatInterface(
        fn=respond,
        title="OSP SOS Investigator",
        description=description,
        examples=examples,
        additional_inputs=[
            gr.Textbox(
                value=default_db_path,
                label="DuckDB path",
                info="Database created by the ingest step",
            ),
            gr.Checkbox(
                value=default_offline,
                label="Offline mode (no LLM)",
                info="Deterministic specialist search only",
            ),
            gr.Textbox(
                value=model_value,
                label="Model override (optional)",
                info="Defaults to OSP_SOS_MODEL / provider settings in .env",
            ),
            gr.Textbox(
                value="",
                label="Focused entity (optional)",
                info="Seed with VM/port/volume/req-id/hostname/chassis",
            ),
            gr.Checkbox(
                value=True,
                label="Use relationship graph",
                info="Prefer VM↔port↔chassis↔host path tools",
            ),
            gr.Radio(
                choices=["Concise RCA", "Evidence-heavy", "Operation path first"],
                value="Concise RCA",
                label="Answer style",
            ),
        ],
        additional_inputs_accordion=gr.Accordion(label="Investigation settings", open=True),
        fill_height=True,
        analytics_enabled=False,
        flagging_mode="never",
    )
    return chat


def launch_chat(
    *,
    db_path: str = DEFAULT_DB_PATH,
    offline: bool = False,
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    import gradio as gr

    app = build_chat_app(
        default_db_path=db_path,
        default_offline=offline,
        default_model=model,
    )
    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="teal",
        neutral_hue="stone",
    )
    app.launch(server_name=host, server_port=port, share=share, theme=theme)


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
        help="Create a temporary public Gradio share link",
    )
    args = parser.parse_args(argv)
    launch_chat(
        db_path=args.db_path,
        offline=args.offline,
        model=args.model,
        host=args.host,
        port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
