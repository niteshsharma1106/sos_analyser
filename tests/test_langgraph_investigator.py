from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import duckdb

from osp_sos_analyser.db import ensure_schema
from osp_sos_analyser.langgraph_investigator import (
    ExpandedQuery,
    InvestigationEntities,
    _init_llm,
    render_investigation_result,
)
from osp_sos_analyser.llm_client import MissingLLMConfiguration


class LangGraphInvestigatorModuleTests(unittest.TestCase):
    def test_expanded_query_model_accepts_notebook_shape(self) -> None:
        plan = ExpandedQuery(
            summary="Port binding failed",
            intent="network_failure",
            entities=InvestigationEntities(
                resource_id="55ab45cf-6925-4811-a008-6fe60d491c5b",
                resource_type="port",
                service="neutron",
            ),
            keywords=["port", "binding", "55ab45cf-6925-4811-a008-6fe60d491c5b"],
            investigation_targets=["neutron", "ovn"],
            hypotheses=["OVN chassis issue"],
        )
        self.assertEqual(plan.entities.service, "neutron")
        self.assertIn("binding", plan.keywords)

    def test_render_investigation_result(self) -> None:
        text = render_investigation_result(
            {
                "expanded_plan": ExpandedQuery(
                    summary="host reboot",
                    intent="system",
                    entities=InvestigationEntities(service="system"),
                    keywords=["reboot"],
                    investigation_targets=["system"],
                ),
                "prefetch_digest": "cluster_id=abc",
                "findings": {"investigator_raw": "checking kernel logs"},
                "final_rca": "Likely kernel panic",
            }
        )
        self.assertIn("Root Cause Analysis", text)
        self.assertIn("Likely kernel panic", text)
        self.assertIn("cluster_id=abc", text)

    def test_init_llm_requires_api_key(self) -> None:
        previous = {
            key: os.environ.pop(key, None)
            for key in ("GROQ_API_KEY", "GROK_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
        }
        os.environ["OSP_SOS_SKIP_DOTENV"] = "1"
        try:
            with self.assertRaises(MissingLLMConfiguration):
                _init_llm(model="llama-3.1-8b-instant", model_provider="groq")
        finally:
            os.environ.pop("OSP_SOS_SKIP_DOTENV", None)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_init_llm_rejects_groq_model_on_google_provider(self) -> None:
        previous = {
            key: os.environ.pop(key, None)
            for key in (
                "GROQ_API_KEY",
                "GROK_API_KEY",
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "OSP_SOS_MODEL",
                "OSP_SOS_MODEL_PROVIDER",
            )
        }
        os.environ["OSP_SOS_SKIP_DOTENV"] = "1"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        try:
            with self.assertRaises(MissingLLMConfiguration) as ctx:
                _init_llm(model="openai/gpt-oss-120b", model_provider="google_genai")
            self.assertIn("not a Google Gemini model", str(ctx.exception))
        finally:
            os.environ.pop("OSP_SOS_SKIP_DOTENV", None)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_init_llm_skips_parallel_tool_calls_for_google(self) -> None:
        from unittest.mock import MagicMock, patch

        previous = {
            key: os.environ.pop(key, None)
            for key in (
                "GROQ_API_KEY",
                "GROK_API_KEY",
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "OSP_SOS_MODEL",
                "OSP_SOS_MODEL_PROVIDER",
            )
        }
        os.environ["OSP_SOS_SKIP_DOTENV"] = "1"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        fake_llm = MagicMock()
        try:
            with patch(
                "langchain.chat_models.init_chat_model", return_value=fake_llm
            ) as init_mock:
                result = _init_llm(
                    model="gemini-2.5-flash", model_provider="google_genai"
                )
            init_mock.assert_called_once()
            fake_llm.bind.assert_not_called()
            self.assertIs(result, fake_llm)
        finally:
            os.environ.pop("OSP_SOS_SKIP_DOTENV", None)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_init_llm_binds_parallel_tool_calls_for_groq(self) -> None:
        from unittest.mock import MagicMock, patch

        previous = {
            key: os.environ.pop(key, None)
            for key in (
                "GROQ_API_KEY",
                "GROK_API_KEY",
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "OSP_SOS_MODEL",
                "OSP_SOS_MODEL_PROVIDER",
            )
        }
        os.environ["OSP_SOS_SKIP_DOTENV"] = "1"
        os.environ["GROQ_API_KEY"] = "test-key"
        fake_llm = MagicMock()
        bound = MagicMock(name="bound_llm")
        fake_llm.bind.return_value = bound
        try:
            with patch(
                "langchain.chat_models.init_chat_model", return_value=fake_llm
            ):
                result = _init_llm(
                    model="gemma2-9b-it", model_provider="groq"
                )
            fake_llm.bind.assert_called_once_with(parallel_tool_calls=False)
            self.assertIs(result, bound)
        finally:
            os.environ.pop("OSP_SOS_SKIP_DOTENV", None)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_schema_ready_for_investigator_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "empty.duckdb"
            with duckdb.connect(str(db_path)) as conn:
                ensure_schema(conn)
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                    ).fetchall()
                }
            self.assertIn("cluster_nodes", tables)
            self.assertIn("entities", tables)
            self.assertIn("entity_mentions", tables)
