import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from osp_sos_analyser.ingest import ingest_sos_reports


class IngestSosReportsTests(unittest.TestCase):
    def test_ingests_rhosp17_container_logs_into_duckdb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()
            archive_path = reports_dir / "sample-report.tar.xz"

            with tarfile.open(archive_path, "w:xz") as archive:
                payload = """2026-07-09 11:27:48.123 45672 INFO nova.compute.manager [-] VM spawning failed
2026-07-09 11:27:49.001 45673 ERROR cinder.volume.manager [-] Volume failed
2026-07-09 11:27:50.002 45674 WARNING ovn-controller [-] OVN northd updated
Traceback (most recent call last):
  File \"/tmp/test.py\", line 1, in <module>
    raise RuntimeError('boom')
RuntimeError: boom
"""
                data = payload.encode("utf-8")
                info = tarfile.TarInfo("var/log/containers/nova-compute.log")
                info.size = len(data)
                archive.addfile(info, fileobj=__import__("io").BytesIO(data))

                data2 = b"2026-07-09 11:27:49.001 45673 ERROR cinder.volume.manager [-] Volume failed\n"
                info2 = tarfile.TarInfo("var/log/containers/cinder-volume.log")
                info2.size = len(data2)
                archive.addfile(info2, fileobj=__import__("io").BytesIO(data2))

                command_data = b"CONTAINER ID   IMAGE   COMMAND\nabc123   quay.io/centos/centos:stream9   /bin/bash\n"
                command_info = tarfile.TarInfo("sos_commands/containers/podman_ps")
                command_info.size = len(command_data)
                archive.addfile(command_info, fileobj=__import__("io").BytesIO(command_data))

            db_path = tmp_path / "test.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            self.assertTrue(db_path.exists())
            import duckdb

            with duckdb.connect(str(db_path)) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM os_logs").fetchone()[0]
                self.assertEqual(rows, 4)

                latest = conn.execute(
                    "SELECT message FROM os_logs WHERE module = 'nova.compute.manager' ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                self.assertIsNotNone(latest)
                self.assertIn("VM spawning failed", latest[0])

                ovn_message = conn.execute(
                    "SELECT message FROM os_logs WHERE module = 'ovn-controller' LIMIT 1"
                ).fetchone()
                self.assertIsNotNone(ovn_message)
                self.assertIn("OVN northd updated", ovn_message[0])
                self.assertIn("Traceback (most recent call last):", ovn_message[0])
                self.assertIn("RuntimeError: boom", ovn_message[0])

                schema = conn.execute("PRAGMA table_info('os_logs')").fetchall()
                names = [col[1] for col in schema]
                self.assertIn("timestamp", names)
                self.assertIn("pid", names)
                self.assertIn("level", names)
                self.assertIn("module", names)
                self.assertIn("message", names)
                self.assertIn("service", names)
                self.assertIn("category", names)
                self.assertIn("source_file", names)
                self.assertIn("report_name", names)
                self.assertIn("tags", names)
                self.assertIn("rhosp_version", names)

                enriched = conn.execute(
                    "SELECT service, category, tags, rhosp_version FROM os_logs WHERE module = 'nova.compute.manager' LIMIT 1"
                ).fetchone()
                self.assertEqual(enriched[0], "nova")
                self.assertEqual(enriched[1], "compute")
                self.assertIn("containerized", enriched[2])
                self.assertEqual(enriched[3], "17.x")

                command_rows = conn.execute("SELECT COUNT(*) FROM os_commands").fetchone()[0]
                self.assertGreaterEqual(command_rows, 1)
                command_value = conn.execute(
                    "SELECT output FROM os_commands WHERE source LIKE '%podman_ps%' LIMIT 1"
                ).fetchone()
                self.assertIsNotNone(command_value)
                self.assertIn("CONTAINER ID", command_value[0])

            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path)

            with duckdb.connect(str(db_path)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM os_logs").fetchone()[0], 4)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM os_commands").fetchone()[0], 1)
                registry = conn.execute(
                    "SELECT COUNT(*) FROM ingested_reports WHERE status = 'completed'"
                ).fetchone()[0]
                self.assertEqual(registry, 1)
