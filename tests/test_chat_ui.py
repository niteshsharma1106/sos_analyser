from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from osp_sos_analyser.chat_ui import _answer_question, build_chat_app
from osp_sos_analyser.db import ensure_schema


class ChatUiTests(unittest.TestCase):
    def test_missing_db_returns_guidance(self) -> None:
        text = _answer_question(
            "Why did compute-03 lose network?",
            [],
            "/tmp/does-not-exist-sos.duckdb",
            True,
            "",
        )
        self.assertIn("Database not found", text)
        self.assertIn("ingest", text.lower())

    def test_empty_prompt_returns_hint(self) -> None:
        text = _answer_question("   ", [], "sos_analysis.duckdb", True, "")
        self.assertIn("Ask an OpenStack", text)

    def test_offline_path_uses_detective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "sos.duckdb")
            conn = duckdb.connect(db_path)
            try:
                ensure_schema(conn)
            finally:
                conn.close()

            with patch(
                "osp_sos_analyser.chat_ui.investigate_prompt_offline"
            ) as offline_fn:
                class _Report:
                    def render_markdown(self) -> str:
                        return "# Offline answer"

                offline_fn.return_value = _Report()
                text = _answer_question(
                    "Port binding failed",
                    [],
                    db_path,
                    True,
                    "",
                )
                self.assertEqual(text, "# Offline answer")
                offline_fn.assert_called_once()

    def test_build_chat_app_constructs(self) -> None:
        app = build_chat_app(default_db_path="sos_analysis.duckdb", default_offline=True)
        self.assertIsNotNone(app)
        self.assertTrue(hasattr(app, "launch"))


if __name__ == "__main__":
    unittest.main()
