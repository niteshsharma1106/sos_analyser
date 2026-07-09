from __future__ import annotations

import argparse
from pathlib import Path

from osp_sos_analyser.ingest import ingest_sos_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest RHOSP SOS reports into a DuckDB database")
    parser.add_argument("--reports-dir", default="SOS_REPORTS", help="Directory containing one or more SOS report tar.xz archives")
    parser.add_argument("--db-path", default=None, help="Destination DuckDB file path")
    parser.add_argument("--clear-existing", action="store_true", help="Clear prior ingested rows before loading")
    parser.add_argument("--max-file-size-mb", type=int, default=10, help="Maximum file size (MB) to process; larger files are skipped")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    db_path = ingest_sos_reports(
        reports_dir=args.reports_dir,
        db_path=args.db_path,
        clear_existing=args.clear_existing,
        max_file_size_mb=args.max_file_size_mb,
    )
    print(f"Ingestion complete. Database: {db_path}")


if __name__ == "__main__":
    main()
