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
from .observability import configure_logging, get_logger
from .relationship_graph import (
    format_operation_path,
    format_relationships,
    get_operation_path,
    get_related_entities,
)


DEFAULT_DB_PATH = "sos_analysis.duckdb"
DEFAULT_EXAMPLES = [
    "Why did compute-03 lose network connectivity?",
    "Port 55ab45cf-6925-4811-a008-6fe60d491c5b failed to bind — show VM→port→chassis→host",
    "What host is related to this VM/port failure?",
    "VM create failed with NoValidHost",
    "Compare controller vs compute errors for this incident",
    "Why did n1-wrkld1-b1-b12-comp008 reboot?",
]

CHAT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Sora:wght@400;500;600&display=swap');

:root {
  --osp-ink: #15202b;
  --osp-muted: #5b6b7c;
  --osp-accent: #0f766e;
  --osp-accent-soft: #ccfbf1;
  --osp-user: #134e4a;
  --osp-panel: rgba(255, 255, 255, 0.78);
  --osp-line: rgba(21, 32, 43, 0.08);
  --osp-shadow: 0 18px 50px rgba(15, 35, 45, 0.08);
}

.gradio-container {
  font-family: "Sora", sans-serif !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 100vh;
  color: var(--osp-ink);
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(15, 118, 110, 0.16), transparent 55%),
    radial-gradient(900px 500px at 90% 0%, rgba(56, 119, 160, 0.12), transparent 50%),
    linear-gradient(180deg, #eef3f6 0%, #f7f9fb 42%, #eef2f5 100%) !important;
}

.gradio-container .main,
.gradio-container .wrap {
  max-width: 920px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

#osp-shell {
  min-height: 100vh;
  padding: 1.25rem 1rem 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

#osp-brand {
  text-align: center;
  padding: 0.35rem 0.5rem 0.15rem;
  animation: osp-rise 520ms ease-out both;
}

#osp-brand .osp-mark {
  font-family: "Fraunces", Georgia, serif;
  font-weight: 700;
  font-size: clamp(2rem, 4vw, 2.75rem);
  letter-spacing: -0.03em;
  line-height: 1.05;
  color: var(--osp-ink);
  margin: 0;
}

#osp-brand .osp-mark span {
  color: var(--osp-accent);
}

#osp-brand .osp-sub {
  margin: 0.45rem auto 0;
  max-width: 34rem;
  color: var(--osp-muted);
  font-size: 0.95rem;
  line-height: 1.45;
}

#osp-brand .osp-cluster {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  margin-top: 0.7rem;
  color: var(--osp-accent);
  font-size: 0.78rem;
  font-weight: 500;
  letter-spacing: 0.02em;
}

#osp-chat {
  flex: 1 1 auto;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  animation: osp-rise 640ms ease-out both;
}

#osp-chat .bot,
#osp-chat .assistant,
#osp-chat [class*="bot"] .message,
#osp-chat .message.bot,
#osp-chat .message-row.bot .message,
#osp-chat .bubble.bot,
#osp-chat .prose {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--osp-ink) !important;
  font-size: 0.95rem !important;
  line-height: 1.55 !important;
}

#osp-chat .user,
#osp-chat [class*="user"] .message,
#osp-chat .message.user,
#osp-chat .message-row.user .message,
#osp-chat .bubble.user {
  background: var(--osp-user) !important;
  color: #f8fffc !important;
  border: none !important;
  border-radius: 1.15rem 1.15rem 0.35rem 1.15rem !important;
  box-shadow: 0 10px 24px rgba(19, 78, 74, 0.18) !important;
  padding: 0.75rem 1rem !important;
}

#osp-composer-wrap {
  position: sticky;
  bottom: 0.4rem;
  z-index: 20;
  padding: 0.35rem;
  border-radius: 1.35rem;
  background: var(--osp-panel);
  border: 1px solid var(--osp-line);
  box-shadow: var(--osp-shadow);
  backdrop-filter: blur(14px);
  animation: osp-rise 720ms ease-out both;
}

#osp-composer textarea {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  font-family: "Sora", sans-serif !important;
  font-size: 0.98rem !important;
  color: var(--osp-ink) !important;
  padding: 0.85rem 1rem !important;
}

#osp-composer textarea:focus {
  outline: none !important;
  box-shadow: none !important;
}

#osp-send {
  min-width: 3rem !important;
  max-width: 3.4rem !important;
  height: 3rem !important;
  border-radius: 999px !important;
  background: var(--osp-accent) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  box-shadow: 0 8px 18px rgba(15, 118, 110, 0.28) !important;
  transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease !important;
}

#osp-send:hover {
  background: #0d9488 !important;
  transform: translateY(-1px);
  box-shadow: 0 12px 22px rgba(15, 118, 110, 0.34) !important;
}

#osp-suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
  padding: 0.15rem 0.25rem 0.35rem;
}

#osp-suggestions button {
  border-radius: 999px !important;
  border: 1px solid var(--osp-line) !important;
  background: rgba(255, 255, 255, 0.72) !important;
  color: var(--osp-ink) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  padding: 0.45rem 0.85rem !important;
  box-shadow: none !important;
  transition: background 160ms ease, border-color 160ms ease, transform 160ms ease !important;
}

#osp-suggestions button:hover {
  background: var(--osp-accent-soft) !important;
  border-color: rgba(15, 118, 110, 0.28) !important;
  transform: translateY(-1px);
}

#osp-settings {
  border: 1px solid var(--osp-line) !important;
  background: rgba(255, 255, 255, 0.55) !important;
  border-radius: 1rem !important;
  overflow: hidden;
}

footer, .footer {
  display: none !important;
}

@keyframes osp-rise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 720px) {
  #osp-shell { padding: 0.85rem 0.65rem 1rem; }
  #osp-brand .osp-mark { font-size: 1.8rem; }
  #osp-composer-wrap { border-radius: 1.1rem; }
}
"""


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


def _brand_html(cluster_line: str = "") -> str:
    cluster = ""
    if cluster_line:
        cluster = f'<div class="osp-cluster">{cluster_line}</div>'
    return f"""
    <div id="osp-brand">
      <p class="osp-mark">OSP <span>SOS</span></p>
      <p class="osp-sub">Ask anything about your ingested OpenStack SOS reports — hosts, ports, reboots, and cross-node failures.</p>
      {cluster}
    </div>
    """


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
    show_observability: bool = True,
) -> str:
    log = get_logger("chat")
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
    log.info(
        "Chat question offline=%s graph=%s style=%s focus=%s prompt=%s",
        offline,
        include_graph,
        answer_style,
        focus_entity or "-",
        prompt[:160],
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
            model=model.strip() or None,
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
            "Set your API key in `.env`, or enable **Offline mode** in settings."
        )
    except Exception as exc:  # noqa: BLE001 - surface runtime errors in the chat UI
        log.exception("Chat investigation failed")
        return f"Investigation failed: `{type(exc).__name__}: {exc}`"


def build_chat_app(
    *,
    default_db_path: str = DEFAULT_DB_PATH,
    default_offline: bool = False,
    default_model: str | None = None,
):
    import gradio as gr

    model_value = default_model or os.getenv("OSP_SOS_MODEL", "")
    banner = ""
    if Path(default_db_path).exists():
        banner = _cluster_banner(default_db_path)

    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Sora"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        text_size=gr.themes.sizes.text_md,
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        body_background_fill="#f7f9fb",
        body_text_color="#15202b",
        block_background_fill="rgba(255,255,255,0.55)",
        block_border_width="0px",
        block_shadow="none",
        border_color_primary="rgba(21,32,43,0.08)",
        button_primary_background_fill="#0f766e",
        button_primary_background_fill_hover="#0d9488",
        button_primary_text_color="#ffffff",
        input_background_fill="rgba(255,255,255,0.9)",
    )

    with gr.Blocks(
        title="OSP SOS",
        analytics_enabled=False,
        fill_height=True,
    ) as app:
        with gr.Column(elem_id="osp-shell"):
            gr.HTML(_brand_html(banner))

            chatbot = gr.Chatbot(
                elem_id="osp-chat",
                show_label=False,
                height="62vh",
                resizable=True,
                render_markdown=True,
                layout="bubble",
                placeholder=(
                    "<strong>Start an investigation</strong><br/>"
                    "Ask about a host reboot, port bind failure, or cross-node error pattern."
                ),
                buttons=["copy"],
            )

            suggestion_buttons = []
            suggestion_prompts = DEFAULT_EXAMPLES[:4]
            suggestion_labels = [
                "Network loss on compute",
                "Port bind path",
                "Host for VM/port",
                "NoValidHost create fail",
            ]
            with gr.Row(elem_id="osp-suggestions"):
                for label in suggestion_labels:
                    suggestion_buttons.append(gr.Button(label, size="sm"))

            with gr.Group(elem_id="osp-composer-wrap"):
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        elem_id="osp-composer",
                        show_label=False,
                        placeholder="Message OSP SOS…",
                        lines=1,
                        max_lines=6,
                        scale=8,
                        autofocus=True,
                        container=False,
                    )
                    send = gr.Button("↑", elem_id="osp-send", scale=0)

            with gr.Accordion("Settings", open=False, elem_id="osp-settings"):
                db_path = gr.Textbox(
                    value=default_db_path,
                    label="DuckDB path",
                    info="Database created by ingest",
                )
                offline = gr.Checkbox(
                    value=default_offline,
                    label="Offline mode (no LLM)",
                )
                model = gr.Textbox(
                    value=model_value,
                    label="Model override",
                    placeholder="Uses OSP_SOS_MODEL from .env when empty",
                )
                focus_entity = gr.Textbox(
                    value="",
                    label="Focused entity",
                    placeholder="VM / port / volume / req-id / hostname",
                )
                include_graph = gr.Checkbox(
                    value=True,
                    label="Use relationship graph",
                )
                answer_style = gr.Radio(
                    choices=["Concise RCA", "Evidence-heavy", "Operation path first"],
                    value="Concise RCA",
                    label="Answer style",
                )
                show_observability = gr.Checkbox(
                    value=False,
                    label="Show agent observability timeline",
                )

        def _user_step(message: str, history: list):
            text = (message or "").strip()
            history = list(history or [])
            if not text:
                return "", history
            history.append({"role": "user", "content": text})
            return "", history

        def _bot_step(
            history: list,
            db_path_value: str,
            offline_value: bool,
            model_value_in: str,
            focus_value: str,
            graph_value: bool,
            style_value: str,
            obs_value: bool,
        ):
            history = list(history or [])
            if not history:
                return history
            last = history[-1]
            if isinstance(last, dict):
                prompt = str(last.get("content") or "")
            else:
                prompt = str(last[0] if last else "")
            answer = _answer_question(
                prompt,
                history,
                db_path_value,
                offline_value,
                model_value_in,
                focus_value,
                graph_value,
                style_value,
                obs_value,
            )
            history.append({"role": "assistant", "content": answer})
            return history

        inputs = [
            db_path,
            offline,
            model,
            focus_entity,
            include_graph,
            answer_style,
            show_observability,
        ]

        msg.submit(
            _user_step,
            [msg, chatbot],
            [msg, chatbot],
            queue=False,
        ).then(
            _bot_step,
            [chatbot, *inputs],
            [chatbot],
        )
        send.click(
            _user_step,
            [msg, chatbot],
            [msg, chatbot],
            queue=False,
        ).then(
            _bot_step,
            [chatbot, *inputs],
            [chatbot],
        )

        for button, prompt in zip(suggestion_buttons, suggestion_prompts):
            button.click(
                lambda p=prompt: p,
                outputs=[msg],
                queue=False,
            ).then(
                _user_step,
                [msg, chatbot],
                [msg, chatbot],
                queue=False,
            ).then(
                _bot_step,
                [chatbot, *inputs],
                [chatbot],
            )

    # Stash theme/css for launch() — Gradio 6 moved these off Blocks().
    app._osp_theme = theme  # type: ignore[attr-defined]
    app._osp_css = CHAT_CSS  # type: ignore[attr-defined]
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
    configure_logging(level=log_level, log_file=log_file)
    get_logger("chat").info(
        "Launching chat UI db=%s host=%s port=%s offline=%s",
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
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=getattr(app, "_osp_theme", None),
        css=getattr(app, "_osp_css", CHAT_CSS),
    )


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
