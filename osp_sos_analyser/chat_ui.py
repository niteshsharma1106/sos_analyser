# chat_ui.py — interactive Q&A UI for SOS investigations (Gradio chat).
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import duckdb

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
DEFAULT_EXAMPLES = [
    "Why did compute-03 lose network connectivity?",
    "Port 55ab45cf-6925-4811-a008-6fe60d491c5b failed to bind — show VM→port→chassis→host",
    "What host is related to this VM/port failure?",
    "VM create failed with NoValidHost",
    "Compare controller vs compute errors for this incident",
    "Why did n1-wrkld1-b1-b12-comp008 reboot?",
]

CHAT_CSS = """
:root {
  --osp-ink: #101820;
  --osp-ink-soft: #243040;
  --osp-muted: #667788;
  --osp-accent: #1f6f8b;
  --osp-accent-deep: #155a72;
  --osp-accent-wash: rgba(31, 111, 139, 0.12);
  --osp-accent-line: rgba(31, 111, 139, 0.28);
  --osp-user-bg: #e7f2f7;
  --osp-user-text: #101820;
  --osp-surface: #ffffff;
  --osp-line: rgba(16, 24, 32, 0.10);
  --osp-shadow: 0 10px 28px rgba(16, 32, 48, 0.07);
  --osp-radius: 0.85rem;
  --osp-col: min(1120px, calc(100vw - 2rem));
}

html, body {
  height: 100% !important;
  margin: 0 !important;
  overflow: hidden !important;
}

.gradio-container {
  font-family: "Outfit", sans-serif !important;
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 100vh !important;
  height: 100vh !important;
  color: var(--osp-ink);
  background-color: #edf1f4 !important;
  background-image:
    radial-gradient(ellipse 90% 55% at 8% -8%, rgba(31, 111, 139, 0.18), transparent 58%),
    radial-gradient(ellipse 70% 45% at 96% 4%, rgba(23, 50, 74, 0.10), transparent 52%),
    linear-gradient(165deg, #f5f7f9 0%, #e8eef2 48%, #e3e9ee 100%) !important;
  overflow: hidden !important;
}

/* Kill Gradio's default content max-width so the shell can center one column. */
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .fillable,
.gradio-container [data-testid="block-container"],
.gradio-container .columns,
.gradio-container .row {
  max-width: none !important;
}

.gradio-container > .main,
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container [data-testid="block-container"] {
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  height: 100% !important;
}

.gradio-container .gap {
  gap: 0.65rem !important;
}

#osp-shell {
  box-sizing: border-box !important;
  width: var(--osp-col) !important;
  max-width: var(--osp-col) !important;
  margin: 0 auto !important;
  height: 100vh !important;
  min-height: 100vh !important;
  padding: 1rem 0 0.85rem !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
  justify-content: flex-start !important;
  gap: 0.65rem !important;
}

#osp-shell.column,
div#osp-shell {
  display: flex !important;
  flex-direction: column !important;
}

/* One shared column width for every section. */
#osp-shell > *,
#osp-shell > div,
#osp-shell .block,
#osp-shell .form,
#osp-shell .group,
#osp-shell .row,
#osp-shell .column,
#osp-shell .svelte-1ed2p3z {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
  box-sizing: border-box !important;
  align-self: stretch !important;
}

#osp-shell > * {
  flex-shrink: 0;
}

#osp-brand {
  flex: 0 0 auto;
  text-align: left;
  padding: 0;
  background: transparent !important;
  animation: osp-rise 480ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

#osp-brand,
#osp-brand .block,
#osp-brand .html-container,
#osp-brand .prose,
#osp-brand p,
#osp-brand span,
#osp-brand div {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}

#osp-brand .osp-brand-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}

#osp-brand .osp-mark {
  font-family: "Bricolage Grotesque", sans-serif;
  font-weight: 700;
  font-size: clamp(1.9rem, 3.4vw, 2.55rem);
  letter-spacing: -0.045em;
  line-height: 0.95;
  color: var(--osp-ink) !important;
  margin: 0;
  animation: osp-mark-in 700ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

#osp-brand .osp-mark span {
  color: var(--osp-accent) !important;
  display: inline-block;
}

#osp-brand .osp-sub {
  margin: 0.45rem 0 0;
  max-width: 40rem;
  color: var(--osp-muted) !important;
  font-size: 0.94rem;
  line-height: 1.45;
  font-weight: 400;
}

#osp-brand .osp-cluster {
  margin: 0;
  color: var(--osp-accent-deep) !important;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.01em;
  white-space: nowrap;
  animation: osp-fade 900ms ease both;
}

#osp-chat {
  flex: 1 1 0% !important;
  flex-shrink: 1 !important;
  min-height: 0 !important;
  height: auto !important;
  width: 100% !important;
  max-width: 100% !important;
  border: 1px solid var(--osp-line) !important;
  background: #ffffff !important;
  border-radius: var(--osp-radius) !important;
  box-shadow: none !important;
  overflow: hidden !important;
  overflow-x: hidden !important;
  padding: 0 !important;
  animation: osp-rise 620ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

#osp-chat,
#osp-chat > .wrapper,
#osp-chat > div,
#osp-chat .bubble-wrap,
#osp-chat [class*="scroll"],
#osp-chat .chatbot,
#osp-chat .block {
  width: 100% !important;
  max-width: 100% !important;
  height: 100% !important;
  min-height: 0 !important;
  max-height: none !important;
  box-sizing: border-box !important;
  overflow-x: hidden !important;
}

/* Kill Gradio/Soft gray “selected text” style on chat content. */
#osp-chat p,
#osp-chat span,
#osp-chat code,
#osp-chat pre,
#osp-chat li,
#osp-chat strong,
#osp-chat em,
#osp-chat a,
#osp-chat .md,
#osp-chat .prose,
#osp-chat .prose * {
  background: transparent !important;
  background-color: transparent !important;
}

#osp-chat .bot,
#osp-chat .assistant,
#osp-chat [class*="bot"] .message,
#osp-chat .message.bot,
#osp-chat .message-row.bot .message,
#osp-chat .bubble.bot,
#osp-chat .prose,
#osp-chat [data-testid="bot"],
#osp-chat .message-row.role-assistant .message,
#osp-chat .message-row.role-assistant .bubble {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--osp-ink-soft) !important;
  font-size: 0.96rem !important;
  line-height: 1.6 !important;
  max-width: 100% !important;
}

#osp-chat .message-row,
#osp-chat .bubble-row {
  width: 100% !important;
  max-width: 100% !important;
  padding-left: 0.85rem !important;
  padding-right: 0.85rem !important;
}

/* Light user bubble — never near-black. */
#osp-chat .user,
#osp-chat [class*="user"] .message,
#osp-chat .message.user,
#osp-chat .message-row.user .message,
#osp-chat .bubble.user,
#osp-chat [data-testid="user"],
#osp-chat .message-row.role-user .message,
#osp-chat .message-row.role-user .bubble,
#osp-chat .message-row.role-user [class*="message"] {
  background: var(--osp-user-bg) !important;
  background-color: var(--osp-user-bg) !important;
  color: var(--osp-user-text) !important;
  border: 1px solid var(--osp-accent-line) !important;
  border-radius: 1rem 1rem 0.3rem 1rem !important;
  box-shadow: none !important;
  padding: 0.75rem 1rem !important;
  max-width: min(36rem, 90%) !important;
}

#osp-chat .user *,
#osp-chat .bubble.user *,
#osp-chat .message.user *,
#osp-chat .message-row.role-user * {
  color: var(--osp-user-text) !important;
  background: transparent !important;
  background-color: transparent !important;
}

#osp-composer-wrap {
  flex: 0 0 auto;
  position: relative;
  z-index: 20;
  padding: 0.3rem 0.35rem 0.3rem 0.45rem;
  border-radius: var(--osp-radius);
  background: var(--osp-surface) !important;
  border: 1px solid var(--osp-line);
  box-shadow: var(--osp-shadow);
  animation: osp-rise 760ms cubic-bezier(0.22, 1, 0.36, 1) both;
  transition: border-color 180ms ease, box-shadow 180ms ease;
}

#osp-composer-wrap:focus-within {
  border-color: var(--osp-accent-line);
  box-shadow: 0 14px 36px rgba(31, 111, 139, 0.12);
}

#osp-composer-wrap .row,
#osp-composer-wrap > div {
  width: 100% !important;
  display: flex !important;
  flex-direction: row !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
  gap: 0.4rem !important;
}

#osp-composer {
  flex: 1 1 auto !important;
  order: 1 !important;
  width: auto !important;
  min-width: 0 !important;
  background: transparent !important;
}

#osp-composer textarea {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  font-family: "Outfit", sans-serif !important;
  font-size: 1rem !important;
  color: var(--osp-ink) !important;
  padding: 0.75rem 0.85rem !important;
  width: 100% !important;
}

#osp-composer textarea:focus {
  outline: none !important;
  box-shadow: none !important;
}

#osp-composer textarea::placeholder {
  color: var(--osp-muted) !important;
  opacity: 0.85;
}

#osp-send {
  order: 2 !important;
  min-width: 2.85rem !important;
  max-width: 3.1rem !important;
  width: 2.85rem !important;
  height: 2.85rem !important;
  border-radius: 0.65rem !important;
  background: var(--osp-accent) !important;
  border: none !important;
  color: #f7fbff !important;
  font-weight: 600 !important;
  font-size: 1.05rem !important;
  box-shadow: none !important;
  transition: transform 150ms ease, background 150ms ease !important;
  flex: 0 0 auto !important;
}

#osp-send:hover {
  background: var(--osp-accent-deep) !important;
  transform: translateY(-1px);
}

#osp-suggestions {
  flex: 0 0 auto;
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.45rem !important;
  justify-content: flex-start !important;
  align-items: center !important;
  padding: 0 !important;
  margin: 0 !important;
  width: 100% !important;
  background: transparent !important;
}

#osp-suggestions button,
#osp-suggestions button * {
  border-radius: 0.55rem !important;
  border: 1px solid var(--osp-line) !important;
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: var(--osp-ink-soft) !important;
  font-family: "Outfit", sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  padding: 0.42rem 0.75rem !important;
  box-shadow: none !important;
  margin: 0 !important;
  transition: background 150ms ease, border-color 150ms ease, color 150ms ease !important;
}

#osp-suggestions button * {
  border: none !important;
  padding: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
}

#osp-suggestions button:hover,
#osp-suggestions button:hover * {
  background: var(--osp-accent-wash) !important;
  background-color: var(--osp-accent-wash) !important;
  border-color: var(--osp-accent-line) !important;
  color: var(--osp-accent-deep) !important;
}

#osp-suggestions button:hover * {
  border: none !important;
  background: transparent !important;
  background-color: transparent !important;
}

#osp-settings {
  flex: 0 0 auto;
  border: 1px solid var(--osp-line) !important;
  background: rgba(255, 255, 255, 0.42) !important;
  border-radius: var(--osp-radius) !important;
  overflow: hidden;
  margin-top: 0 !important;
}

#osp-settings .label-wrap span,
#osp-settings label span {
  font-family: "Outfit", sans-serif !important;
  font-weight: 500 !important;
  color: var(--osp-ink-soft) !important;
}

footer, .footer {
  display: none !important;
}

@keyframes osp-rise {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes osp-mark-in {
  from { opacity: 0; letter-spacing: 0.04em; transform: translateY(8px); }
  to { opacity: 1; letter-spacing: -0.045em; transform: translateY(0); }
}

@keyframes osp-fade {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (max-width: 720px) {
  :root { --osp-col: calc(100vw - 1.2rem); }
  #osp-shell { padding: 0.75rem 0 0.7rem !important; gap: 0.55rem !important; }
  #osp-brand .osp-mark { font-size: 1.75rem; }
  #osp-brand .osp-cluster { white-space: normal; }
  html, body, .gradio-container { overflow: auto !important; height: auto !important; min-height: 100vh !important; }
  #osp-shell { height: auto !important; min-height: 100vh !important; }
  #osp-chat { min-height: 52vh !important; }
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
      <div class="osp-brand-row">
        <p class="osp-mark">OSP <span>SOS</span></p>
        {cluster}
      </div>
      <p class="osp-sub">Investigate OpenStack SOS reports — reboots, ports, and cross-node failures.</p>
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


def _normalize_chat_text(message: Any) -> str:
    """Extract plain text from Gradio chatbot content (str or multimodal blocks)."""
    if message is None:
        return ""
    if isinstance(message, str):
        text = message.strip()
        # Gradio sometimes stringifies multimodal blocks into the history.
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
        text = str(exc)
        if "being used by another process" in text or "Cannot open file" in text:
            return (
                "Database is locked by another process (often a running ingest).\n\n"
                "Wait for ingest to finish, or stop the other process, then ask again.\n\n"
                f"Details: `{type(exc).__name__}: {exc}`"
            )
        return f"Investigation failed: `{type(exc).__name__}: {exc}`"


def build_chat_app(
    *,
    default_db_path: str = DEFAULT_DB_PATH,
    default_offline: bool = False,
    default_model: str | None = None,
):
    import gradio as gr

    load_app_env()
    # Optional CLI override only; empty means "use .env as-is".
    cli_model = (default_model or "").strip()
    banner = ""
    if Path(default_db_path).exists():
        banner = _cluster_banner(default_db_path)
    llm_status = describe_llm_settings()

    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Outfit"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        text_size=gr.themes.sizes.text_md,
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="#edf1f4",
        body_text_color="#101820",
        block_background_fill="#ffffff",
        block_border_width="0px",
        block_shadow="none",
        border_color_primary="rgba(16,24,32,0.10)",
        button_primary_background_fill="#1f6f8b",
        button_primary_background_fill_hover="#155a72",
        button_primary_text_color="#f7fbff",
        button_secondary_background_fill="#ffffff",
        button_secondary_text_color="#243040",
        input_background_fill="#ffffff",
        background_fill_primary="#ffffff",
        background_fill_secondary="#edf1f4",
    )

    with gr.Blocks(
        title="OSP SOS",
        analytics_enabled=False,
        fill_height=True,
        fill_width=True,
    ) as app:
        with gr.Column(elem_id="osp-shell", elem_classes=["osp-shell"], scale=1, min_width=320):
            gr.HTML(_brand_html(banner))

            chatbot = gr.Chatbot(
                elem_id="osp-chat",
                show_label=False,
                container=False,
                value=[
                    {
                        "role": "assistant",
                        "content": (
                            "## Start an investigation\n"
                            "Ask about a host, VM, port, reboot, or an error in your SOS reports. "
                            "Use a quick prompt below to get started."
                        ),
                    }
                ],
                height="100%",
                resizable=False,
                render_markdown=True,
                layout="bubble",
                placeholder=None,
                buttons=["copy"],
                scale=1,
            )

            suggestion_buttons = []
            suggestion_prompts = DEFAULT_EXAMPLES[:4]
            suggestion_labels = [
                "Compute network loss",
                "Port bind path",
                "Host for VM/port",
                "NoValidHost fail",
            ]
            with gr.Row(elem_id="osp-suggestions"):
                for label in suggestion_labels:
                    suggestion_buttons.append(gr.Button(label, size="sm"))

            with gr.Group(elem_id="osp-composer-wrap"):
                with gr.Row(equal_height=True):
                    msg = gr.Textbox(
                        elem_id="osp-composer",
                        show_label=False,
                        placeholder="Ask about a reboot, port, or host…",
                        lines=1,
                        max_lines=6,
                        scale=8,
                        autofocus=True,
                        container=False,
                    )
                    send = gr.Button("→", elem_id="osp-send", scale=1, min_width=48)

            with gr.Accordion("Settings", open=False, elem_id="osp-settings"):
                gr.Markdown(
                    f"**LLM config** (from `.env` only — not editable here)\n\n`{llm_status}`"
                )
                db_path = gr.Textbox(
                    value=default_db_path,
                    label="DuckDB path",
                    info="Database created by ingest",
                )
                offline = gr.Checkbox(
                    value=default_offline,
                    label="Offline mode (no LLM)",
                )
                # Hidden: optional CLI --model override only; UI never hardcodes a model.
                model_state = gr.State(cli_model)
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
            text = _normalize_chat_text(message)
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
                prompt = _normalize_chat_text(last.get("content"))
            else:
                prompt = _normalize_chat_text(last[0] if last else "")
            answer = _answer_question(
                prompt,
                history,
                db_path_value,
                offline_value,
                model_value_in or "",
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
            model_state,
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
        inbrowser=False,
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
