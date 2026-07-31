from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import duckdb

from osp_sos_analyser.cluster_loader import (
    guess_hostname_from_archive_name,
    infer_node_role,
)
from osp_sos_analyser.evidence_index import get_cluster_manifest, get_evidence, get_entity
from osp_sos_analyser.ingest import ingest_sos_reports


class ClusterManifestAndEvidenceIndexTests(unittest.TestCase):
    def test_hostname_and_role_guesses(self) -> None:
        self.assertEqual(
            guess_hostname_from_archive_name("sosreport-compute-03-2026-07-09.tar.xz"),
            "compute-03",
        )
        self.assertEqual(infer_node_role("compute-03"), "compute")
        self.assertEqual(infer_node_role("controller-0"), "controller")

    def test_ingest_builds_manifest_and_evidence_index(self) -> None:
        port_id = "55ab45cf-6925-4811-a008-6fe60d491c5b"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()
            archive_path = reports_dir / "sosreport-compute-03-2026-07-09.tar.xz"

            with tarfile.open(archive_path, "w:xz") as archive:
                hostname = b"compute-03.localdomain\n"
                info = tarfile.TarInfo("hostname")
                info.size = len(hostname)
                archive.addfile(info, fileobj=io.BytesIO(hostname))

                payload = (
                    f"2026-07-09 14:00:01.123 100 ERROR neutron.plugins.ml2 [-] "
                    f"Port binding failed for {port_id}\n"
                ).encode("utf-8")
                log_info = tarfile.TarInfo("var/log/containers/neutron/server.log")
                log_info.size = len(payload)
                archive.addfile(log_info, fileobj=io.BytesIO(payload))

            db_path = tmp_path / "analysis.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            with duckdb.connect(str(db_path)) as conn:
                nodes = get_cluster_manifest(conn)
                self.assertEqual(len(nodes), 1)
                self.assertEqual(nodes[0]["hostname"], "compute-03.localdomain")
                self.assertEqual(nodes[0]["node_role"], "compute")
                self.assertTrue(nodes[0]["cluster_id"])

                host_row = conn.execute(
                    "SELECT DISTINCT hostname, node_role, cluster_id FROM os_logs"
                ).fetchone()
                self.assertEqual(host_row[0], "compute-03.localdomain")
                self.assertEqual(host_row[1], "compute")
                self.assertEqual(host_row[2], nodes[0]["cluster_id"])

                entity = get_entity(conn, port_id)
                self.assertIsNotNone(entity)
                assert entity is not None
                self.assertEqual(entity.entity_type, "port")
                self.assertGreaterEqual(entity.mention_count, 1)

                mentions = get_evidence(conn, port_id)
                self.assertGreaterEqual(len(mentions), 1)
                self.assertIn("Port binding failed", mentions[0].message_excerpt)
                self.assertEqual(mentions[0].hostname, "compute-03.localdomain")
                self.assertEqual(mentions[0].service, "neutron")

    def test_multi_node_ingest_shares_cluster_and_cross_host_evidence(self) -> None:
        from osp_sos_analyser.evidence_index import (
            compare_node_activity,
            get_evidence,
            search_logs_by_node,
        )

        port_id = "55ab45cf-6925-4811-a008-6fe60d491c5b"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()

            controller = reports_dir / "sosreport-controller-0-2026-07-09.tar.xz"
            with tarfile.open(controller, "w:xz") as archive:
                hostname = b"controller-0\n"
                info = tarfile.TarInfo("hostname")
                info.size = len(hostname)
                archive.addfile(info, fileobj=io.BytesIO(hostname))
                payload = (
                    f"2026-07-09 14:00:01.123 100 ERROR neutron.plugins.ml2 [-] "
                    f"Port binding failed for {port_id} on compute-03\n"
                ).encode()
                log_info = tarfile.TarInfo("var/log/containers/neutron/server.log")
                log_info.size = len(payload)
                archive.addfile(log_info, fileobj=io.BytesIO(payload))

            compute = reports_dir / "sosreport-compute-03-2026-07-09.tar.xz"
            with tarfile.open(compute, "w:xz") as archive:
                hostname = b"compute-03\n"
                info = tarfile.TarInfo("hostname")
                info.size = len(hostname)
                archive.addfile(info, fileobj=io.BytesIO(hostname))
                payload = (
                    f"2026-07-09 14:00:02.123 200 ERROR ovn-controller [-] "
                    f"failed to bind port {port_id}\n"
                ).encode()
                log_info = tarfile.TarInfo("var/log/containers/openvswitch/ovn-controller.log")
                log_info.size = len(payload)
                archive.addfile(log_info, fileobj=io.BytesIO(payload))

            db_path = tmp_path / "analysis.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            with duckdb.connect(str(db_path)) as conn:
                nodes = get_cluster_manifest(conn)
                self.assertEqual(len(nodes), 2)
                hostnames = {node["hostname"] for node in nodes}
                self.assertEqual(hostnames, {"controller-0", "compute-03"})
                cluster_ids = {node["cluster_id"] for node in nodes}
                self.assertEqual(len(cluster_ids), 1)

                mentions = get_evidence(conn, port_id)
                mention_hosts = {item.hostname for item in mentions}
                self.assertEqual(mention_hosts, {"controller-0", "compute-03"})

                compute_only = get_evidence(conn, port_id, hostnames=["compute-03"])
                self.assertTrue(compute_only)
                self.assertTrue(all(item.hostname == "compute-03" for item in compute_only))

                role_only = search_logs_by_node(conn, node_roles=["controller"], limit=10)
                self.assertTrue(role_only)
                self.assertTrue(all(row["hostname"] == "controller-0" for row in role_only))

                comparison = compare_node_activity(conn, levels=("ERROR",), limit=20)
                comparison_hosts = {row["hostname"] for row in comparison}
                self.assertIn("controller-0", comparison_hosts)
                self.assertIn("compute-03", comparison_hosts)
