from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import duckdb

from osp_sos_analyser.db import ensure_schema, insert_commands, insert_logs
from osp_sos_analyser.detective import investigate_prompt
from osp_sos_analyser.langchain_detective import investigate_prompt_with_langchain
from osp_sos_analyser.llm_client import MissingLLMConfiguration
from osp_sos_analyser.models import CommandArtifact, LogEntry


class DetectivePhase2Tests(unittest.TestCase):
    def test_detective_routes_network_prompt_to_specialists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "analysis.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                insert_logs(
                    conn,
                    [
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 14, 0, 1),
                            pid=100,
                            level="ERROR",
                            module="neutron.plugins.ml2",
                            message="Port binding failed for compute-03",
                            service="neutron",
                            category="networking",
                            source_file="var/log/containers/neutron/server.log",
                            report_name="sample.tar.xz",
                            tags="networking,neutron,openstack,rhosp17",
                        ),
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 14, 0, 3),
                            pid=101,
                            level="WARNING",
                            module="ovn-controller",
                            message="OVN chassis for compute-03 disconnected",
                            service="ovn",
                            category="networking",
                            source_file="var/log/containers/openvswitch/ovn-controller.log",
                            report_name="sample.tar.xz",
                            tags="networking,ovn,openstack,rhosp17",
                        ),
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 14, 0, 8),
                            pid=102,
                            level="ERROR",
                            module="nova.compute.manager",
                            message="Instance network setup failed on compute-03",
                            service="nova",
                            category="compute",
                            source_file="var/log/containers/nova/nova-compute.log",
                            report_name="sample.tar.xz",
                            tags="compute,nova,openstack,rhosp17",
                        ),
                    ],
                )
                insert_commands(
                    conn,
                    [
                        CommandArtifact(
                            source="sos_commands/podman/podman_ps",
                            command="podman_ps",
                            output="ovn_controller exited 2 minutes ago",
                            service="podman",
                            category="container-runtime",
                            source_file="sos_commands/podman/podman_ps",
                            report_name="sample.tar.xz",
                            tags="containerized,podman,openstack,rhosp17",
                        )
                    ],
                )

            report = investigate_prompt(
                db_path,
                "VMs on compute-03 suddenly lost network connectivity at 14:00",
            )
            rendered = report.render_markdown()

            self.assertIn("neutron", report.hints.services)
            self.assertIn("ovn", report.hints.services)
            self.assertIn("compute-03", report.hints.hostnames)
            self.assertIn("Neutron/OVN Agent", rendered)
            self.assertIn("Nova Agent", rendered)
            self.assertIn("System/Podman Agent", rendered)
            self.assertIn("Port binding failed", rendered)
            self.assertIn("Instance network setup failed", rendered)

    def test_instance_uuid_prompt_stays_identifier_focused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "analysis.duckdb"
            instance_id = "fd27c003-5b78-4abb-a85a-aa90973f7ff0"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                insert_logs(
                    conn,
                    [
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 14, 0, 1),
                            pid=100,
                            level="ERROR",
                            module="oslo.messaging._drivers.impl_rabbit",
                            message="AMQP server is unreachable",
                            service="nova",
                            category="compute",
                            source_file="var/log/containers/nova/nova-api.log",
                            report_name="sample.tar.xz",
                            tags="compute,nova,openstack,rhosp17",
                        ),
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 15, 16, 34),
                            pid=101,
                            level="INFO",
                            module="nova.api.openstack.wsgi",
                            message=f"HTTP exception thrown: Instance {instance_id} is not ready",
                            service="nova",
                            category="compute",
                            source_file="var/log/containers/nova/nova-api.log",
                            report_name="sample.tar.xz",
                            tags="compute,nova,openstack,rhosp17",
                        ),
                        LogEntry(
                            timestamp=datetime(2026, 7, 9, 15, 16, 35),
                            pid=102,
                            level="INFO",
                            module="nova.api.openstack.requestlog",
                            message=f'POST /v2.1/servers/{instance_id}/action status: 409',
                            service="nova",
                            category="compute",
                            source_file="var/log/containers/nova/nova-api.log",
                            report_name="sample.tar.xz",
                            tags="compute,nova,openstack,rhosp17",
                        ),
                    ],
                )

            report = investigate_prompt(db_path, f"VM {instance_id} failed to create")
            rendered = report.render_markdown()

            self.assertEqual(report.hints.services, ("nova",))
            self.assertNotIn("System/Podman Agent", rendered)
            self.assertIn("Instance fd27c003-5b78-4abb-a85a-aa90973f7ff0 is not ready", rendered)
            self.assertNotIn("AMQP server is unreachable", "\n".join(event.message for event in report.timeline))

    def test_langchain_agent_requires_api_key_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "analysis.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)

            import os

            old_key = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with self.assertRaises(MissingLLMConfiguration):
                    investigate_prompt_with_langchain(db_path, "VM failed to create")
            finally:
                if old_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
