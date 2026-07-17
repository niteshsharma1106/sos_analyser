import tempfile
import unittest
from pathlib import Path

from osp_sos_analyser.dbconnector import DatabaseConnector
from osp_sos_analyser.models import LogEntry


class DatabaseConnectorTests(unittest.TestCase):
    def test_connects_and_runs_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"

            with DatabaseConnector(db_path) as db:
                db.ensure_schema()
                entry = LogEntry(
                    timestamp=None,
                    pid=42,
                    level="INFO",
                    module="nova.compute.manager",
                    message="VM spawning failed",
                    service="nova",
                    category="compute",
                    source_file="var/log/containers/nova-compute.log",
                    report_name="sample-report.tar.xz",
                    tags="containerized",
                )
                inserted = db.insert_logs([entry])
                self.assertEqual(inserted, 1)

                rows = db.fetch_all("SELECT message FROM os_logs")
                self.assertEqual(rows, [("VM spawning failed",)])

                db.execute("CREATE TABLE IF NOT EXISTS demo (value INTEGER)")
                db.execute("INSERT INTO demo VALUES (1), (2)")
                values = db.fetch_all("SELECT value FROM demo ORDER BY value")
                self.assertEqual(values, [(1,), (2,)])
