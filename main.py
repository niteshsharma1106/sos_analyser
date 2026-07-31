#main.py
from __future__ import annotations

import argparse
import sys
from enum import Enum
from typing import List, Optional

from osp_sos_analyser.detective import investigate_prompt_offline
from osp_sos_analyser.ingest import ingest_sos_reports
from osp_sos_analyser.llm_client import MissingLLMConfiguration
from osp_sos_analyser.langgraph_investigator import (
    investigate_with_langgraph,
    render_investigation_result,
)
from osp_sos_analyser.chat_ui import launch_chat
from osp_sos_analyser.observability import configure_logging, get_logger
from pydantic import BaseModel, Field


class IncidentType(str, Enum):
    UNKNOWN = "unknown"
    VM_CREATE_FAILURE = "vm_create_failure"
    INSTANCE_FAILURE = "instance_failure"
    VOLUME_FAILURE = "volume_failure"
    NETWORK_FAILURE = "network_failure"
    SERVICE_FAILURE = "service_failure"


class InvestigationContext(BaseModel):
    user_prompt: str
    incident_type: IncidentType = IncidentType.UNKNOWN
    symptom: Optional[str] = None
    affected_host: Optional[str] = None
    instance_ids: List[str] = Field(default_factory=list)
    volume_ids: List[str] = Field(default_factory=list)
    port_ids: List[str] = Field(default_factory=list)
    request_ids: List[str] = Field(default_factory=list)
    approximate_time: Optional[str] = None
    services: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    assumptions: List[str] = Field(default_factory=list)



def build_ingest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest RHOSP SOS reports into DuckDB")
    parser.add_argument(
        "--reports-dir",
        default="SOS_REPORTS",
        help="Directory containing one or more SOS report tar.xz archives",
    )
    parser.add_argument("--db-path", default=None, help="Destination DuckDB file path")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear prior ingested rows before loading",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=2048,
        help="Hard maximum file size in MB to process (default: 2048)",
    )
    parser.add_argument(
        "--large-log-threshold-mb",
        type=float,
        default=30,
        help="Stream files above this size and retain only their newest time window (default: 30)",
    )
    parser.add_argument(
        "--large-log-tail-hours",
        type=float,
        default=6,
        help="Hours of timestamped records retained from files above the threshold (default: 6)",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Re-ingest archives even if they are already marked completed",
    )
    return parser


def build_analyze_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Investigate an RHOSP incident prompt")
    parser.add_argument("prompt", help="Incident prompt or pasted symptom to investigate")
    parser.add_argument(
        "--db-path",
        default="sos_analysis.duckdb",
        help="DuckDB database created by the ingestion phase",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LangChain model for LLM-backed analysis. Defaults to OSP_SOS_MODEL.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the old deterministic search workflow without an LLM.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Backend log level (default OSP_SOS_LOG_LEVEL or INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (default OSP_SOS_LOG_FILE)",
    )
    return parser


def build_chat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch interactive SOS investigation chat UI")
    parser.add_argument(
        "--db-path",
        default="sos_analysis.duckdb",
        help="DuckDB database created by the ingestion phase",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Default the UI to offline deterministic mode",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override (defaults to OSP_SOS_MODEL)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="UI bind address")
    parser.add_argument("--port", type=int, default=7860, help="UI bind port")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio share link",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Backend log level (default OSP_SOS_LOG_LEVEL or INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (default OSP_SOS_LOG_FILE)",
    )
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        args = build_analyze_parser().parse_args(sys.argv[2:])
        configure_logging(level=args.log_level, log_file=args.log_file)
        log = get_logger("cli")
        if args.offline:
            report = investigate_prompt_offline(db_path=args.db_path, prompt=args.prompt)
            print(report.render_markdown())
        else:
            try:
                log.info("Analyze prompt=%s", args.prompt)
                print(f"User Prompt: {args.prompt}")
                result = investigate_with_langgraph(
                    db_path=args.db_path,
                    prompt=args.prompt,
                    model=args.model,
                )
            except MissingLLMConfiguration as exc:
                raise SystemExit(str(exc)) from exc
            print(render_investigation_result(result, include_observability=True))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        args = build_chat_parser().parse_args(sys.argv[2:])
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
        return

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        argv = sys.argv[2:]
    else:
        argv = sys.argv[1:]
    args = build_ingest_parser().parse_args(argv)
    configure_logging()
    db_path = ingest_sos_reports(
        reports_dir=args.reports_dir,
        db_path=args.db_path,
        clear_existing=args.clear_existing,
        max_file_size_mb=args.max_file_size_mb,
        force_reingest=args.force_reingest,
        large_log_threshold_mb=args.large_log_threshold_mb,
        large_log_tail_hours=args.large_log_tail_hours,
    )
    print(f"Ingestion complete. Database: {db_path}")


if __name__ == "__main__":
    main()
