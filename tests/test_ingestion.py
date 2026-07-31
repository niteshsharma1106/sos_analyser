import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from osp_sos_analyser.ingest import ingest_sos_reports
from osp_sos_analyser.log_parser import parse_log_lines


class IngestSosReportsTests(unittest.TestCase):
    def test_parses_decoded_binary_journal_output(self) -> None:
        entries = list(
            parse_log_lines(
                ["2026-07-28T14:05:01.123456+0530 ctl01 systemd[1]: Started service.\n"],
                "var/log/journal/machine/system.journal",
                "sosreport-controller-2026-07-28-test.tar.xz",
            )
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].timestamp.year, 2026)
        self.assertEqual(entries[0].module, "systemd")
        self.assertEqual(entries[0].pid, 1)
        self.assertEqual(entries[0].message, "Started service.")
        self.assertEqual(entries[0].service, "system")

    def test_large_log_retains_only_latest_six_hours(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()
            archive_path = reports_dir / "sosreport-controller-2026-07-28-tail.tar.xz"
            payload = (
                b"Jul 28 07:00:00 ctl01 kernel: old event\n"
                b"Jul 28 11:00:00 ctl01 kernel: retained boundary event\n"
                b"Jul 28 17:00:00 ctl01 kernel: newest event\n"
            )
            with tarfile.open(archive_path, "w:xz") as archive:
                info = tarfile.TarInfo("var/log/messages")
                info.size = len(payload)
                archive.addfile(info, fileobj=__import__("io").BytesIO(payload))

            db_path = tmp_path / "test.duckdb"
            ingest_sos_reports(
                reports_dir=reports_dir,
                db_path=db_path,
                clear_existing=True,
                max_file_size_mb=1,
                large_log_threshold_mb=0.0001,
                large_log_tail_hours=6,
            )

            import duckdb

            with duckdb.connect(str(db_path)) as conn:
                messages = [row[0] for row in conn.execute("SELECT message FROM os_logs ORDER BY timestamp").fetchall()]
            self.assertEqual(messages, ["retained boundary event", "newest event"])

    def test_ingests_extensionless_system_and_journalctl_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()
            archive_path = reports_dir / "sosreport-controller-2026-07-28-test.tar.xz"

            with tarfile.open(archive_path, "w:xz") as archive:
                members = {
                    "var/log/messages": (
                        b"Jul 28 14:05:01 ctl01 kernel: watchdog: BUG: soft lockup\n"
                        b"Jul 28 14:05:02 ctl01 systemd[1]: Reached target Reboot.\n"
                    ),
                    "var/log/secure": b"Jul 28 14:06:01 ctl01 sshd[42]: Accepted publickey\n",
                    "sos_commands/systemd/journalctl_--no-pager_--boot": b"Jul 28 14:07:01 ctl01 systemd[1]: Started podman.service.\n",
                }
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, fileobj=__import__("io").BytesIO(data))

            db_path = tmp_path / "test.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            import duckdb

            with duckdb.connect(str(db_path)) as conn:
                rows = conn.execute(
                    "SELECT source_file, timestamp, module, message, service "
                    "FROM os_logs ORDER BY source_file, timestamp"
                ).fetchall()

            self.assertEqual(len(rows), 4)
            self.assertEqual({row[0] for row in rows}, set(members))
            self.assertTrue(all(row[1] is not None for row in rows))
            self.assertTrue(
                any(
                    row[0] == "var/log/messages"
                    and row[2] == "kernel"
                    and row[3] == "watchdog: BUG: soft lockup"
                    and row[4] == "system"
                    for row in rows
                )
            )
            self.assertIn("Started podman.service.", "\n".join(row[3] for row in rows))

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


class IngestDedupeAndParallelTests(unittest.TestCase):
    def _write_sample_archive(self, archive_path: Path, marker: str) -> None:
        with tarfile.open(archive_path, "w:xz") as archive:
            payload = (
                f"2026-07-09 11:27:48.123 45672 ERROR nova.compute.manager [-] {marker}\n"
            ).encode("utf-8")
            info = tarfile.TarInfo("var/log/containers/nova-compute.log")
            info.size = len(payload)
            archive.addfile(info, fileobj=__import__("io").BytesIO(payload))

    def test_skips_byte_identical_archive_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()
            first = reports_dir / "report-a.tar.xz"
            second = reports_dir / "report-b.tar.xz"
            self._write_sample_archive(first, "spawn failed")
            second.write_bytes(first.read_bytes())

            db_path = tmp_path / "test.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            import duckdb

            with duckdb.connect(str(db_path)) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM os_logs").fetchone()[0]
                reports = conn.execute(
                    "SELECT DISTINCT report_name FROM os_logs"
                ).fetchall()
                registry = conn.execute(
                    "SELECT COUNT(*) FROM ingested_reports WHERE status = 'completed'"
                ).fetchone()[0]
            self.assertEqual(rows, 1)
            self.assertEqual(len(reports), 1)
            self.assertEqual(registry, 1)

    def test_ingests_distinct_archives_sequentially(self) -> None:
        """Distinct archives both land in DuckDB (phase2 sequential registry path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()
            self._write_sample_archive(reports_dir / "host-a.tar.xz", "failure-a")
            self._write_sample_archive(reports_dir / "host-b.tar.xz", "failure-b")

            db_path = tmp_path / "test.duckdb"
            ingest_sos_reports(
                reports_dir=reports_dir,
                db_path=db_path,
                clear_existing=True,
            )

            import duckdb

            with duckdb.connect(str(db_path)) as conn:
                rows = conn.execute("SELECT COUNT(*) FROM os_logs").fetchone()[0]
                messages = {
                    row[0]
                    for row in conn.execute("SELECT message FROM os_logs").fetchall()
                }
                index_names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'os_logs'"
                    ).fetchall()
                }
            self.assertEqual(rows, 2)
            self.assertTrue(any("failure-a" in msg for msg in messages))
            self.assertTrue(any("failure-b" in msg for msg in messages))
            self.assertIn("idx_os_logs_service_level_ts", index_names)
