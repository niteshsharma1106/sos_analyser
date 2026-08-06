from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from osp_sos_analyser.db import ensure_schema
from osp_sos_analyser.investigation_tools import build_langchain_tools
from osp_sos_analyser.saved_tools import (
    bind_sql_template,
    create_and_persist_tool,
    extract_sql_parameters,
    format_saved_tools_catalog,
    list_saved_tool_definitions,
    load_saved_tool,
    run_saved_tool,
    validate_tool_name,
)


class SavedToolsTests(unittest.TestCase):
    def test_validate_and_extract_params(self) -> None:
        self.assertEqual(validate_tool_name("Peer-Mentions"), "peer_mentions")
        with self.assertRaises(ValueError):
            validate_tool_name("get_host_reboot_timeline")
        params = extract_sql_parameters(
            "SELECT 1 WHERE hostname = {{hostname}} AND ts >= {{start_time}}"
        )
        self.assertEqual(params, ["hostname", "start_time"])
        sql, values = bind_sql_template(
            "SELECT * FROM os_logs WHERE hostname = {{hostname}} LIMIT {{limit}}",
            {"hostname": "comp008", "limit": 5},
        )
        self.assertEqual(sql.count("?"), 2)
        self.assertEqual(values, ["comp008", 5])

    def test_create_persist_reload_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tools"
            db_path = Path(tmpdir) / "t.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                conn.execute(
                    """
                    INSERT INTO cluster_nodes (
                        cluster_id, hostname, node_role, rhosp_version, services,
                        archive_name, archive_id
                    ) VALUES ('c1', 'n1-wrkld1-b1-b12-comp008', 'compute', '17',
                              'nova', 'a.tar', 'id1'),
                             ('c1', 'n1-wrkld1-b1-b13-ctrl001', 'controller', '17',
                              'pacemaker', 'b.tar', 'id2')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO os_logs (
                        timestamp, level, service, message, source_file,
                        report_name, hostname, node_role
                    ) VALUES (
                        '2026-07-27 02:34:47', 'INFO', 'system',
                        'Peer n1-wrkld1-b1-b12-comp008 was terminated (reboot) by ctrl',
                        'pacemaker.log', 'sos', 'n1-wrkld1-b1-b13-ctrl001', 'controller'
                    )
                    """
                )
                out = create_and_persist_tool(
                    conn,
                    name="peer_mentions_window",
                    purpose="Peer logs naming a host in a time window",
                    sql=(
                        "SELECT timestamp, hostname, left(message, 200) AS message "
                        "FROM os_logs "
                        "WHERE lower(COALESCE(hostname,'')) != lower({{hostname}}) "
                        "AND message ILIKE '%' || {{hostname}} || '%' "
                        "AND timestamp >= {{start_time}} "
                        "AND timestamp <= {{end_time}} "
                        "ORDER BY timestamp LIMIT {{limit}}"
                    ),
                    args={
                        "hostname": "comp008",
                        "start_time": "2026-07-27 02:30:00",
                        "end_time": "2026-07-27 02:45:00",
                        "limit": 20,
                    },
                    persist=True,
                    directory=root,
                )
                self.assertIn("Saved for future use", out)
                self.assertIn("terminated (reboot)", out)

                loaded = load_saved_tool("peer_mentions_window", directory=root)
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.name, "peer_mentions_window")
                self.assertIn("hostname", loaded.parameters)

                again = run_saved_tool(
                    conn,
                    "peer_mentions_window",
                    {
                        "hostname": "n1-wrkld1-b1-b12-comp008",
                        "start_time": "2026-07-27 02:30:00",
                        "end_time": "2026-07-27 02:45:00",
                        "limit": 10,
                    },
                    directory=root,
                )
                self.assertIn("terminated (reboot)", again)
                catalog = format_saved_tools_catalog(root)
                self.assertIn("peer_mentions_window", catalog)

                tools = {t.name: t for t in build_langchain_tools(conn)}
                # Point env at our temp registry by saving into default dir? Dynamic
                # registration reads default_saved_tools_dir — use tool APIs instead.
                create_tool = tools["create_and_save_investigation_tool"]
                listed = tools["list_saved_investigation_tools"]
                self.assertTrue(callable(create_tool.invoke))
                self.assertIn("No saved investigation tools", listed.invoke({}))

    def test_canonicalize_preserves_empty_sql_strings(self) -> None:
        from osp_sos_analyser.investigation_tools import canonicalize_analysis_hostnames

        with duckdb.connect(":memory:") as conn:
            ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO cluster_nodes (
                    cluster_id, hostname, node_role, rhosp_version, services,
                    archive_name, archive_id
                ) VALUES ('c1', 'n1-wrkld1-b1-b12-comp008', 'compute', '17',
                          'nova', 'a.tar', 'id1')
                """
            )
            sql = (
                "SELECT * FROM os_logs WHERE COALESCE(hostname,'') != '' "
                "AND message ILIKE '%' || 'x' || '%' AND hostname = 'comp008'"
            )
            rewritten, notes = canonicalize_analysis_hostnames(conn, sql)
            self.assertIn("COALESCE(hostname,'')", rewritten)
            self.assertIn("ILIKE '%' || 'x' || '%'", rewritten)
            self.assertTrue(any("comp008" in n for n in notes))
            self.assertIn("n1-wrkld1-b1-b12-comp008", rewritten)

    def test_rejects_write_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "tools"
            with duckdb.connect(":memory:") as conn:
                ensure_schema(conn)
                out = create_and_persist_tool(
                    conn,
                    name="bad_tool",
                    purpose="should fail",
                    sql="DELETE FROM os_logs",
                    args={},
                    directory=root,
                )
                self.assertIn("Cannot create tool", out)
                self.assertEqual(list_saved_tool_definitions(root), [])


if __name__ == "__main__":
    unittest.main()
