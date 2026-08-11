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
    format_host_reboot_timeline,
    format_manifest,
    parse_journalctl_list_boots,
    prefetch_investigation_digest,
)


class InvestigationToolsTests(unittest.TestCase):
    def test_agent_can_create_a_temporary_read_only_analysis(self) -> None:
        from osp_sos_analyser.db import ensure_schema

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "analysis.duckdb"
            host = "n1-wrkld1-b1-b12-comp008"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                conn.execute(
                    """
                    INSERT INTO cluster_nodes (
                        cluster_id, hostname, node_role, rhosp_version, services,
                        archive_name, archive_id
                    ) VALUES ('cluster-1', ?, 'compute', '17.x', 'nova', 'sos', 'report-1')
                    """,
                    [host],
                )
                conn.execute(
                    """
                    INSERT INTO entity_relationships (
                        src_entity_id, src_entity_type, relation_type,
                        dst_entity_id, dst_entity_type, evidence_count, confidence
                    ) VALUES
                        ('instance-a', 'instance', 'instance_host', ?, 'host', 1, 0.9),
                        ('instance-b', 'instance', 'instance_host', ?, 'host', 1, 0.9)
                    """,
                    [host, host],
                )
                tools = {tool.name: tool for tool in build_langchain_tools(conn)}
                result = tools["create_and_run_analysis"].invoke(
                    {
                        "name": "count_instances_on_compute",
                        "purpose": "Count instances observed on the selected compute host.",
                        "sql": (
                            "SELECT count(DISTINCT src_entity_id) AS observed_instance_count "
                            "FROM entity_relationships "
                            "WHERE relation_type = 'instance_host' "
                            "AND lower(dst_entity_id) = lower('comp008')"
                        ),
                    }
                )
                self.assertIn("Temporary analysis created", result)
                self.assertIn("resolved hostname 'comp008'", result)
                self.assertIn("observed_instance_count", result)
                self.assertIn("2", result)

                blocked = tools["create_and_run_analysis"].invoke(
                    {
                        "name": "unsafe",
                        "purpose": "must not run writes",
                        "sql": "DELETE FROM entity_relationships",
                    }
                )
                self.assertIn("Only a single SELECT", blocked)

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

                self.assertIn("get_related_entities", tools)
                self.assertIn("get_operation_path", tools)
                self.assertIn("search_sos_commands", tools)

    def test_reboot_investigation_helpers_resolve_comp_host(self) -> None:
        from osp_sos_analyser.db import ensure_schema
        from osp_sos_analyser.evidence_index import parse_search_terms
        from osp_sos_analyser.investigation_tools import (
            hostnames_mentioned_in_text,
            resolve_hostnames,
        )

        terms, or_mode = parse_search_terms("reboot OR kernel OR crash OR power")
        self.assertTrue(or_mode)
        self.assertEqual(terms, ["reboot", "kernel", "crash", "power"])
        self.assertNotIn("OR", terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cluster.duckdb"
            host = "n1-wrkld1-b1-b12-comp008"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                conn.execute(
                    """
                    INSERT INTO cluster_nodes (
                        cluster_id, hostname, node_role, rhosp_version, services,
                        archive_name, archive_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        "abc",
                        host,
                        "unknown",
                        "17.x",
                        "nova,ovn",
                        "sosreport-n1-wrkld1-b1-b12-comp008-2026-07-27-qrkrkek.tar.xz",
                        "id1",
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO os_logs (
                        timestamp, level, service, message, source_file,
                        report_name, hostname, node_role
                    ) VALUES (now(), 'ERROR', 'system', 'kernel panic - not syncing',
                              'var/log/messages', 'sos', ?, 'unknown')
                    """,
                    [host],
                )
                conn.execute(
                    """
                    INSERT INTO os_commands (
                        source, command, output, service, category, source_file,
                        report_name, hostname, node_role
                    ) VALUES ('sos', 'dmesg', 'Kernel panic - not syncing: Fatal',
                              'system', 'kernel', 'sos_commands/kernel/dmesg',
                              'sos', ?, 'unknown')
                    """,
                    [host],
                )
                conn.execute(
                    """
                    INSERT INTO os_commands (
                        source, command, output, service, category, source_file,
                        report_name, hostname, node_role
                    ) VALUES ('sos', 'who -b', 'system boot  2026-07-26 21:05',
                              'system', 'system', 'sos_commands/systemd/who_-b',
                              'sos', ?, 'unknown')
                    """,
                    [host],
                )

                # Overlay should report compute even though DB says unknown.
                self.assertIn("role=compute", format_manifest(conn))
                self.assertEqual(resolve_hostnames(conn, "comp008"), [host])
                self.assertEqual(
                    hostnames_mentioned_in_text(
                        conn, "why compute n1-wrkld1-b1-b12-comp008 rebooted?"
                    ),
                    [host],
                )

                timeline = format_host_reboot_timeline(conn, "comp008")
                self.assertIn("Last reboot / boot timeline", timeline)
                self.assertIn("2026-07-26 21:05", timeline)
                self.assertIn("Likely boot/reboot timestamp candidates", timeline)

                digest = prefetch_investigation_digest(
                    conn,
                    raw_query="why compute n1-wrkld1-b1-b12-comp008 rebooted?",
                    keywords=["reboot", "comp008"],
                )
                self.assertIn("Last reboot / boot timeline", digest)
                self.assertIn("Focused host evidence", digest)
                self.assertIn("kernel panic", digest.lower())
                self.assertIn("dmesg", digest.lower())
                # Reboot prefetch should not dump cross-node warning noise.
                self.assertNotIn("Cross-node activity", digest)

                tools = {tool.name: tool for tool in build_langchain_tools(conn)}
                self.assertIn("get_host_reboot_timeline", tools)
                tool_timeline = tools["get_host_reboot_timeline"].invoke(
                    {"hostname": "comp008"}
                )
                self.assertIn("2026-07-26 21:05", tool_timeline)
                logs = tools["search_os_logs"].invoke(
                    {
                        "hostname": "comp008",
                        "search_terms": "reboot OR panic OR watchdog",
                        "limit": 10,
                    }
                )
                self.assertIn("panic", logs.lower())
                cmds = tools["search_sos_commands"].invoke(
                    {
                        "hostname": "comp008",
                        "command_pattern": "dmesg",
                        "search_terms": "panic OR Fatal",
                        "limit": 5,
                    }
                )
                self.assertIn("Kernel panic", cmds)

                # Reboot searches must not keep a neutron service filter.
                logs_reboot = tools["search_os_logs"].invoke(
                    {
                        "hostname": "comp008",
                        "service": "neutron",
                        "search_terms": "reboot OR panic OR watchdog",
                        "limit": 10,
                    }
                )
                self.assertIn("ignored service='neutron'", logs_reboot)
                self.assertIn("panic", logs_reboot.lower())

                # Term-filtered sos_commands should fall back to raw artifacts.
                cmds_loose = tools["search_sos_commands"].invoke(
                    {
                        "hostname": "comp008",
                        "command_pattern": "dmesg",
                        "search_terms": "this-term-will-not-match-zzzz",
                        "limit": 5,
                    }
                )
                self.assertIn("unfiltered command artifacts", cmds_loose)
                self.assertIn("Kernel panic", cmds_loose)

                # Narrow patterns with no hits should expand to the reboot set.
                cmds_fallback = tools["search_sos_commands"].invoke(
                    {
                        "hostname": "comp008",
                        "command_pattern": "this-pattern-does-not-exist",
                        "search_terms": "",
                        "limit": 5,
                    }
                )
                self.assertIn("expanded to reboot command set", cmds_fallback)
                self.assertIn("dmesg", cmds_fallback.lower())

    def test_parse_journalctl_list_boots(self) -> None:
        blob = (
            "IDX BOOT ID                          FIRST ENTRY                 LAST ENTRY\n"
            " -1 09b773b6a3794ce485c95e0ba34695ba Sun 2026-07-26 23:50:13 IST "
            "Mon 2026-07-27 02:34:06 IST\n"
            "  0 11e3514c3a2e49148084a1e471fcded6 Mon 2026-07-27 02:38:27 IST "
            "Mon 2026-07-27 09:51:02 IST\n"
        )
        boots = parse_journalctl_list_boots(blob)
        self.assertEqual(len(boots), 2)
        self.assertEqual(boots[0]["index"], -1)
        self.assertEqual(boots[1]["index"], 0)
        self.assertEqual(boots[1]["boot_id"], "11e3514c3a2e49148084a1e471fcded6")
        self.assertIn("2026-07-27 02:38:27", boots[1]["first_entry"])

    def test_reboot_timeline_discovers_list_boots_from_os_logs(self) -> None:
        from osp_sos_analyser.db import ensure_schema

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "boots.duckdb"
            host = "n1-wrkld1-b1-b12-comp008"
            ctrl = "n1-wrkld1-b1-b13-ctrl001"
            list_boots = (
                "IDX BOOT ID                          FIRST ENTRY                 LAST ENTRY\n"
                " -1 09b773b6a3794ce485c95e0ba34695ba Sun 2026-07-26 23:50:13 IST "
                "Mon 2026-07-27 02:34:06 IST\n"
                "  0 11e3514c3a2e49148084a1e471fcded6 Mon 2026-07-27 02:38:27 IST "
                "Mon 2026-07-27 09:51:02 IST\n"
            )
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                for name, role in ((host, "compute"), (ctrl, "controller")):
                    conn.execute(
                        """
                        INSERT INTO cluster_nodes (
                            cluster_id, hostname, node_role, rhosp_version, services,
                            archive_name, archive_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        ["abc", name, role, "17.x", "nova", "sos.tar.xz", f"id-{role}"],
                    )
                # Mimic real ingest: list-boots lands in os_logs with NULL timestamp.
                conn.execute(
                    """
                    INSERT INTO os_logs (
                        timestamp, level, service, message, source_file,
                        report_name, hostname, node_role
                    ) VALUES (
                        NULL, 'UNKNOWN', 'system', ?,
                        'sos_commands/systemd/journalctl_--list-boots',
                        'sos', ?, 'compute'
                    )
                    """,
                    [list_boots, host],
                )
                conn.execute(
                    """
                    INSERT INTO os_logs (
                        timestamp, level, service, message, source_file,
                        report_name, hostname, node_role
                    ) VALUES
                    (?, 'ERROR', 'system', ?, 'pacemaker.log', 'sos', ?, 'controller'),
                    (?, 'NOTICE', 'system', ?, 'pacemaker.log', 'sos', ?, 'controller')
                    """,
                    [
                        "2026-07-27 02:33:45",
                        f"Remote connection to {host} unexpectedly dropped during monitor",
                        ctrl,
                        "2026-07-27 02:34:47",
                        f"Peer {host} was terminated (reboot) by peer-ctrl: OK",
                        ctrl,
                    ],
                )

                timeline = format_host_reboot_timeline(conn, "comp008")
                self.assertIn("list-boots", timeline.lower())
                self.assertIn("2026-07-27 02:38:27", timeline)
                self.assertIn("11e3514c3a2e49148084a1e471fcded6", timeline)
                self.assertIn("Last reboot / current boot start", timeline)
                self.assertIn("Peer/cluster mentions", timeline)
                self.assertIn("unexpectedly dropped during monitor", timeline)
                self.assertIn("terminated (reboot)", timeline)
                self.assertIn(ctrl, timeline)
                peer_lines = [line for line in timeline.splitlines() if ctrl in line]
                self.assertIn(host, peer_lines[0])
                self.assertNotIn("ipmitool", timeline.lower())

                tools = {tool.name: tool for tool in build_langchain_tools(conn)}
                tool_out = tools["get_host_reboot_timeline"].invoke({"hostname": "comp008"})
                self.assertIn("2026-07-27 02:38:27", tool_out)
                self.assertIn("terminated (reboot)", tool_out)
                peer_out = tools["search_peer_mentions"].invoke(
                    {
                        "hostname": "comp008",
                        "start_time": "2026-07-27 02:30:00",
                        "end_time": "2026-07-27 02:45:00",
                    }
                )
                self.assertIn("terminated (reboot)", peer_out)
