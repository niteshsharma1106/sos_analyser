# chat_ui.py — interactive Q&A UI for SOS investigations (Gradio chat).
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .detective import investigate_prompt_offline
from .langgraph_investigator import (
    investigate_with_langgraph,
    render_investigation_result,
)
from .llm_client import MissingLLMConfiguration


DEFAULT_DB_PATH = "sos_analysis.duckdb"
DEFAULT_EXAMPLES = [
    "Why did compute-03 lose network connectivity?",
    "Port 55ab45cf-6925-4811-a008-6fe60d491c5b failed to bind",
    "VM create failed with NoValidHost",
    "Why did the controller reboot?",
]


def _answer_question(
    message: str,
    history: list,
    db_path: str,
    offline: bool,
    model: str,
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

    try:
        if offline:
            report = investigate_prompt_offline(db_path=db, prompt=prompt)
            return report.render_markdown()

        result = investigate_with_langgraph(
            db_path=db,
            prompt=prompt,
            model=model.strip() or None,
        )
        return render_investigation_result(result)
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

    def respond(message: str, history: list, db_path: str, offline: bool, model: str):
        return _answer_question(message, history, db_path, offline, model)

    model_value = default_model or os.getenv("OSP_SOS_MODEL", "")
    # Gradio requires list-of-lists examples when additional_inputs are present.
    examples = [
        [prompt, default_db_path, default_offline, model_value]
        for prompt in DEFAULT_EXAMPLES
    ]

    chat = gr.ChatInterface(
        fn=respond,
        title="OSP SOS Investigator",
        description=(
            "Ask a question about your ingested RHOSP SOS reports. "
            "Answers use the Cluster Manifest + Evidence Index when available."
        ),
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
        help="Create a temporary public Gradio link",
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
