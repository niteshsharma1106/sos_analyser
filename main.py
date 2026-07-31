#main.py
from __future__ import annotations

import argparse
import sys
from enum import Enum
from typing import List, Optional

from osp_sos_analyser.detective import investigate_prompt_offline
from osp_sos_analyser.ingest import ingest_sos_reports
from osp_sos_analyser.llm_client import MissingLLMConfiguration
from osp_sos_analyser.langchain_detective import investigate_prompt_with_langchain
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
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        args = build_analyze_parser().parse_args(sys.argv[2:])
        if args.offline:
            report = investigate_prompt_offline(db_path=args.db_path, prompt=args.prompt)
        else:
            try:
                print(f"User Prompt: {args.prompt}")

                report = investigate_prompt_with_langchain(
                    db_path=args.db_path,
                    prompt=args.prompt,
                    model=args.model,
                )
            except MissingLLMConfiguration as exc:
                raise SystemExit(str(exc)) from exc
        print(report.render_markdown())
        return

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        argv = sys.argv[2:]
    else:
        argv = sys.argv[1:]
    args = build_ingest_parser().parse_args(argv)
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
