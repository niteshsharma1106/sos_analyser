from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import duckdb

from osp_sos_analyser.ingest import ingest_sos_reports
from osp_sos_analyser.investigation_tools import (
    build_langchain_tools,
    format_manifest,
    prefetch_investigation_digest,
)


class InvestigationToolsTests(unittest.TestCase):
    def test_tools_prefer_evidence_index(self) -> None:
        port_id = "55ab45cf-6925-4811-a008-6fe60d491c5b"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()
            archive_path = reports_dir / "sosreport-compute-03-2026-07-09.tar.xz"
            with tarfile.open(archive_path, "w:xz") as archive:
                hostname = b"compute-03\n"
                info = tarfile.TarInfo("hostname")
                info.size = len(hostname)
                archive.addfile(info, fileobj=io.BytesIO(hostname))
                payload = (
                    f"2026-07-09 14:00:01.123 100 ERROR neutron.plugins.ml2 [-] "
                    f"Port binding failed for {port_id}\n"
                ).encode()
                log_info = tarfile.TarInfo("var/log/containers/neutron/server.log")
                log_info.size = len(payload)
                archive.addfile(log_info, fileobj=io.BytesIO(payload))

            db_path = tmp_path / "analysis.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            with duckdb.connect(str(db_path), read_only=True) as conn:
                manifest = format_manifest(conn)
                self.assertIn("compute-03", manifest)
                self.assertIn("role=compute", manifest)

                digest = prefetch_investigation_digest(
                    conn,
                    raw_query=f"port {port_id} binding failed",
                    resource_id=port_id,
                )
                self.assertIn("Cluster manifest", digest)
                self.assertIn(port_id, digest)
                self.assertIn("Port binding failed", digest)

                tools = {tool.name: tool for tool in build_langchain_tools(conn)}
                overview = tools["get_cluster_overview"].invoke({})
                self.assertIn("compute-03", overview)

                evidence = tools["get_entity_evidence"].invoke(
                    {"entity_id": port_id, "service": "neutron", "limit": 10}
                )
                self.assertIn("Port binding failed", evidence)
                self.assertIn("type=port", evidence)

                fallback = tools["search_os_logs"].invoke(
                    {"resource_id": port_id, "service": "", "search_terms": "", "limit": 10}
                )
                self.assertIn("Indexed evidence (preferred)", fallback)

    def test_multi_node_tools_scope_by_host_and_compare(self) -> None:
        port_id = "55ab45cf-6925-4811-a008-6fe60d491c5b"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reports_dir = tmp_path / "SOS_REPORTS"
            reports_dir.mkdir()

            for host, role, line, path in (
                (
                    "controller-0",
                    "controller",
                    f"2026-07-09 14:00:01.123 100 ERROR neutron.plugins.ml2 [-] Port binding failed for {port_id}\n",
                    "var/log/containers/neutron/server.log",
                ),
                (
                    "compute-03",
                    "compute",
                    f"2026-07-09 14:00:02.123 200 ERROR ovn-controller [-] failed to bind port {port_id}\n",
                    "var/log/containers/openvswitch/ovn-controller.log",
                ),
            ):
                archive_path = reports_dir / f"sosreport-{host}-2026-07-09.tar.xz"
                with tarfile.open(archive_path, "w:xz") as archive:
                    hostname = f"{host}\n".encode()
                    info = tarfile.TarInfo("hostname")
                    info.size = len(hostname)
                    archive.addfile(info, fileobj=io.BytesIO(hostname))
                    payload = line.encode()
                    log_info = tarfile.TarInfo(path)
                    log_info.size = len(payload)
                    archive.addfile(log_info, fileobj=io.BytesIO(payload))
                self.assertEqual(role in {"controller", "compute"}, True)

            db_path = tmp_path / "analysis.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            with duckdb.connect(str(db_path), read_only=True) as conn:
                tools = {tool.name: tool for tool in build_langchain_tools(conn)}
                overview = tools["get_cluster_overview"].invoke({})
                self.assertIn("controller-0", overview)
                self.assertIn("compute-03", overview)
                self.assertIn("nodes=2", overview)

                compare = tools["compare_nodes"].invoke(
                    {"service": "", "level": "ERROR", "limit": 20}
                )
                self.assertIn("controller-0", compare)
                self.assertIn("compute-03", compare)

                compute_evidence = tools["get_entity_evidence"].invoke(
                    {
                        "entity_id": port_id,
                        "hostname": "compute-03",
                        "service": "",
                        "node_role": "",
                        "limit": 10,
                    }
                )
                self.assertIn("compute-03", compute_evidence)
                self.assertNotIn("controller-0", compute_evidence)

                controller_logs = tools["search_os_logs"].invoke(
                    {
                        "service": "",
                        "resource_id": "",
                        "search_terms": "binding",
                        "hostname": "",
                        "node_role": "controller",
                        "limit": 10,
                    }
                )
                self.assertIn("controller-0", controller_logs)
                self.assertNotIn("compute-03", controller_logs)

                digest = prefetch_investigation_digest(
                    conn,
                    raw_query=f"port {port_id} failed across cluster",
                    resource_id=port_id,
                )
                self.assertIn("Cross-node activity", digest)
