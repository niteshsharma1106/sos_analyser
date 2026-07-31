from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import duckdb

from osp_sos_analyser.ingest import ingest_sos_reports
from osp_sos_analyser.relationship_graph import (
    build_relationship_index,
    extract_typed_entities,
    get_operation_path,
    get_related_entities,
    relationship_candidates_from_message,
)


class RelationshipGraphTests(unittest.TestCase):
    def test_extract_typed_entities_prefers_labels(self) -> None:
        message = (
            "Failed to bind port_id=55ab45cf-6925-4811-a008-6fe60d491c5b "
            "for instance_uuid=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee on host compute-03"
        )
        grouped = extract_typed_entities(message, "neutron")
        self.assertIn("55ab45cf-6925-4811-a008-6fe60d491c5b", grouped["port"])
        self.assertIn("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", grouped["instance"])

    def test_relationship_candidates_include_vm_port_host(self) -> None:
        message = (
            "Port binding failed port_id=55ab45cf-6925-4811-a008-6fe60d491c5b "
            "device_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee binding:host_id=compute-03 "
            "chassis=compute-03"
        )
        edges = relationship_candidates_from_message(
            message=message,
            service="neutron",
            hostname="controller-0",
            aliases={"compute-03": "compute-03", "controller-0": "controller-0"},
        )
        relations = {(e["src_entity_type"], e["relation_type"], e["dst_entity_type"]) for e in edges}
        self.assertIn(("instance", "instance_port", "port"), relations)
        self.assertIn(("port", "port_host", "host"), relations)
        self.assertIn(("port", "port_chassis", "chassis"), relations)

    def test_ingest_builds_operation_path_vm_port_host(self) -> None:
        instance_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
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
                    f"Port binding failed port_id={port_id} device_id={instance_id} "
                    f"binding:host_id=compute-03 chassis=compute-03\n"
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
                    f"failed to bind port {port_id} on chassis compute-03\n"
                ).encode()
                log_info = tarfile.TarInfo(
                    "var/log/containers/openvswitch/ovn-controller.log"
                )
                log_info.size = len(payload)
                archive.addfile(log_info, fileobj=io.BytesIO(payload))

            db_path = tmp_path / "analysis.duckdb"
            ingest_sos_reports(reports_dir=reports_dir, db_path=db_path, clear_existing=True)

            with duckdb.connect(str(db_path)) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM entity_relationships"
                ).fetchone()[0]
                self.assertGreaterEqual(count, 2)

                related = get_related_entities(conn, port_id)
                relation_types = {item.relation_type for item in related}
                self.assertTrue({"instance_port", "port_host"} & relation_types)

                path = get_operation_path(
                    conn, instance_id, target_type="host", max_hops=4
                )
                self.assertTrue(path)
                reached_hosts = {
                    edge.dst_entity_id
                    for edge in path
                    if edge.dst_entity_type == "host"
                } | {
                    edge.src_entity_id
                    for edge in path
                    if edge.src_entity_type == "host"
                }
                self.assertIn("compute-03", reached_hosts)

                # Rebuild is idempotent.
                stats = build_relationship_index(conn)
                self.assertGreaterEqual(stats["relationships"], 2)


if __name__ == "__main__":
    unittest.main()
