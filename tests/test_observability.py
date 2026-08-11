from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from osp_sos_analyser.langgraph_investigator import render_investigation_result
from osp_sos_analyser.observability import (
    AgentRunTrace,
    close_logging,
    configure_logging,
    get_logger,
)


class ObservabilityTests(unittest.TestCase):
    def test_agent_run_trace_records_and_renders(self) -> None:
        trace = AgentRunTrace(run_id="abcd1234-test", prompt="port binding failed")
        trace.node_start("QUERY_EXPAND")
        trace.tool_start("get_cluster_overview", {})
        trace.tool_end("get_cluster_overview", "nodes=2")
        trace.node_end("QUERY_EXPAND")
        payload = trace.to_dict()
        self.assertEqual(payload["run_id"], "abcd1234-test")
        self.assertGreaterEqual(payload["event_count"], 4)

        restored = AgentRunTrace.from_dict(payload)
        assert restored is not None
        markdown = restored.render_markdown()
        self.assertIn("Agent observability", markdown)
        self.assertIn("get_cluster_overview", markdown)
        self.assertIn("node_start", markdown)

    def test_handoff_records_format_size_and_redacted_preview(self) -> None:
        trace = AgentRunTrace(run_id="handoff-test")
        trace.handoff(
            "QUERY_EXPAND",
            "INVESTIGATOR",
            {"plan": {"hostname": "comp008"}, "digest": "token=secret-value"},
        )
        event = trace.events[0]
        self.assertEqual(event.kind, "handoff")
        self.assertEqual(event.details["format"], "JSON object")
        self.assertGreater(event.details["total_chars"], 0)
        self.assertGreater(event.details["total_bytes_utf8"], 0)
        self.assertIn("comp008", event.details["fields"]["plan"]["preview"])
        self.assertNotIn("secret-value", event.details["fields"]["digest"]["preview"])
        self.assertIn("Inter-agent handoffs", trace.render_markdown())

    def test_configure_logging_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "osp_sos.log"
            configure_logging(level="INFO", log_file=str(log_path), force=True)
            try:
                logger = get_logger("test")
                logger.info("hello observability")
                for handler in logging.getLogger("osp_sos").handlers:
                    handler.flush()
                text = log_path.read_text(encoding="utf-8")
                self.assertIn("hello observability", text)
            finally:
                close_logging()

    def test_render_investigation_includes_trace(self) -> None:
        trace = AgentRunTrace(run_id="run-obs-1", prompt="test")
        trace.node_start("INVESTIGATOR")
        trace.tool_start("compare_nodes", {"level": "ERROR"})
        trace.tool_end("compare_nodes", "controller-0|ERROR|2")
        text = render_investigation_result(
            {
                "run_id": "run-obs-1",
                "final_rca": "Likely OVN binding failure",
                "agent_trace": trace.to_dict(),
            },
            include_observability=True,
        )
        self.assertIn("Likely OVN binding failure", text)
        self.assertIn("Agent observability", text)
        self.assertIn("compare_nodes", text)


if __name__ == "__main__":
    unittest.main()
