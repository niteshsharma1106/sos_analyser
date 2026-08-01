from __future__ import annotations

import unittest

from osp_sos_analyser.observability import AgentRunTrace
from osp_sos_analyser.privacy import redact_sensitive_text


class PrivacyTests(unittest.TestCase):
    def test_redacts_common_credentials(self) -> None:
        value = (
            "password=hunter2 token: abc123 Authorization: Bearer bearer-secret "
            "https://user:pass@example.test/v1"
        )
        redacted = redact_sensitive_text(value)
        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("abc123", redacted)
        self.assertNotIn("bearer-secret", redacted)
        self.assertNotIn("user:pass", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_tool_trace_redacts_output_preview(self) -> None:
        trace = AgentRunTrace(prompt="test")
        trace.tool_end("search_os_logs", "api_key=super-secret")
        preview = trace.events[-1].details["output_preview"]
        self.assertNotIn("super-secret", preview)
        self.assertIn("[REDACTED]", preview)


if __name__ == "__main__":
    unittest.main()
