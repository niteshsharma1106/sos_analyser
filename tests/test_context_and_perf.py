from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import duckdb

from osp_sos_analyser.analysis import AnalysisStore
from osp_sos_analyser.context_pack import (
    compact_message_history,
    format_log_digest,
    parse_time_hint,
    prefetch_evidence_digest,
    truncate_text,
)
from osp_sos_analyser.db import ensure_indexes, ensure_schema, insert_logs
from osp_sos_analyser.models import LogEntry


class ContextPackTests(unittest.TestCase):
    def test_format_log_digest_truncates_messages(self) -> None:
        long_message = "x" * 500
        digest = format_log_digest(
            [(datetime(2026, 7, 9, 14, 0, 0), "nova", "ERROR", long_message, "nova.log")]
        )
        self.assertIn("nova|ERROR|", digest)
        self.assertIn("…", digest)
        self.assertLess(len(digest), 260)

    def test_parse_time_hint_and_prefetch(self) -> None:
        self.assertEqual(parse_time_hint("at 14:00").hour, 14)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "analysis.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                insert_logs(
                    conn,
                    [
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 14, 0, 1),
                            pid=1,
                            level="ERROR",
                            module="neutron.plugins.ml2",
                            message="Port binding failed for 55ab45cf-6925-4811-a008-6fe60d491c5b",
                            service="neutron",
                            category="networking",
                            source_file="var/log/containers/neutron/server.log",
                            report_name="sample.tar.xz",
                            tags="neutron",
                        )
                    ],
                )
                digest = prefetch_evidence_digest(
                    conn,
                    identifiers=["55ab45cf-6925-4811-a008-6fe60d491c5b"],
                    services=["neutron"],
                    keywords=["binding"],
                    time_hint="14:00",
                    limit=10,
                )
            self.assertIn("Identifier hits", digest)
            self.assertIn("Port binding failed", digest)
            self.assertIn("Error summary", digest)

    def test_compact_message_history_keeps_system_and_recent(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "q1"},
            {"role": "tool", "content": "y" * 2000},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        compacted = compact_message_history(messages, keep_last=4, tool_result_chars=50)
        roles = [m["role"] for m in compacted]
        self.assertEqual(roles[0], "system")
        self.assertEqual(len(compacted), 5)  # system + last 4
        tool = next(m for m in compacted if m["role"] == "tool")
        self.assertLessEqual(len(tool["content"]), 50)
        self.assertTrue(tool["content"].endswith("…"))


class AnalysisStorePerfTests(unittest.TestCase):
    def test_persistent_connection_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "analysis.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                ensure_indexes(conn)
                insert_logs(
                    conn,
                    [
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 14, 0, 1),
                            pid=1,
                            level="INFO",
                            module="nova.compute",
                            message="heartbeat ok",
                            service="nova",
                            category="compute",
                            source_file="nova.log",
                            report_name="a.tar.xz",
                            tags="nova",
                        ),
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 14, 0, 2),
                            pid=2,
                            level="ERROR",
                            module="nova.compute",
                            message="VM spawn failed with traceback",
                            service="nova",
                            category="compute",
                            source_file="nova.log",
                            report_name="a.tar.xz",
                            tags="nova",
                        ),
                    ],
                )
                index_names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'os_logs'"
                    ).fetchall()
                }
                self.assertIn("idx_os_logs_service_level_ts", index_names)
                self.assertIn("idx_os_logs_timestamp", index_names)

            store = AnalysisStore(db_path)
            first = store._connect()
            second = store._connect()
            self.assertIs(first, second)
            rows = store.search_logs(service="nova", limit=5)
            self.assertEqual(rows[0].level, "ERROR")
            self.assertIn("failed", rows[0].message.lower())
            store.close()


class TruncateHelperTests(unittest.TestCase):
    def test_truncate_text(self) -> None:
        self.assertEqual(truncate_text("abc", 10), "abc")
        self.assertTrue(truncate_text("abcdefghij", 5).endswith("…"))
