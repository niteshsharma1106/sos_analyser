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
    _normalize_node_role,
    _parse_partial_expanded_json,
    fallback_expanded_query,
    render_investigation_result,
    sanitize_expanded_query,
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
            self.assertIn("entity_relationships", tables)


class ExpandedQueryHardeningTests(unittest.TestCase):
    def test_normalize_node_role_salvages_keyword_dump(self) -> None:
        dump = (
            "compute-node-auto-rebooted-unexpectedly-analyze-it-root-cause-"
            "investigation-system-level-issue-host-failure-kernel-panic"
        )
        self.assertEqual(_normalize_node_role(dump), "compute")
        self.assertEqual(_normalize_node_role("controller"), "controller")
        self.assertIsNone(_normalize_node_role("totally-made-up-role-name-that-is-long"))

    def test_entities_coerce_runaway_node_role(self) -> None:
        ent = InvestigationEntities(
            hostname="comp008",
            node_role="compute-node-auto-rebooted-" + ("x" * 500),
        )
        self.assertEqual(ent.node_role, "compute")
        self.assertEqual(ent.hostname, "comp008")

    def test_fallback_expanded_query_for_compute_reboot(self) -> None:
        plan = fallback_expanded_query(
            "why compute n1-wrkld1-b1-b12-comp008 rebooted unexpectedly"
        )
        self.assertEqual(plan.entities.hostname, "n1-wrkld1-b1-b12-comp008")
        self.assertEqual(plan.entities.node_role, "compute")
        self.assertEqual(plan.entities.service, "system")
        self.assertIn("reboot", plan.keywords)
        self.assertTrue(plan.investigation_targets)

    def test_sanitize_fills_missing_keywords_from_query(self) -> None:
        plan = ExpandedQuery(
            summary="Investigate reboot",
            intent="system",
            entities=InvestigationEntities(service="system"),
            keywords=[],
            investigation_targets=[],
        )
        cleaned = sanitize_expanded_query(
            plan, "compute node comp008 auto-rebooted unexpectedly"
        )
        self.assertEqual(cleaned.entities.hostname, "comp008")
        self.assertIn("reboot", cleaned.keywords)

    def test_parse_partial_json_with_truncated_node_role(self) -> None:
        # Mimic the user failure: truncated completion with runaway node_role.
        blob = (
            '{"summary": "Investigate unexpected auto-reboot on compute node comp008.", '
            '"intent": "Analyze the cause of a compute node auto-reboot.", '
            '"entities": {"hostname": "comp008", "node_role": "compute-node-auto-rebooted-'
            + ("unexpectedly-analyze-it-" * 80)
            + "truncated"
        )
        plan = _parse_partial_expanded_json(
            blob,
            "compute node comp008 auto-rebooted unexpectedly",
        )
        self.assertIsInstance(plan, ExpandedQuery)
        self.assertEqual(plan.entities.hostname or "comp008", "comp008")
        # Either salvaged from partial JSON or heuristic fallback.
        self.assertIn(plan.entities.node_role, {None, "compute"})
        self.assertTrue(plan.keywords or plan.summary)
