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
    INVESTIGATOR_SYSTEM_PROMPT,
    _init_llm,
    _normalize_node_role,
    _parse_partial_expanded_json,
    extract_groq_failed_tool_call,
    fallback_expanded_query,
    invoke_tool_by_name,
    is_groq_tool_use_failed,
    parse_groq_failed_generation,
    render_investigation_result,
    sanitize_expanded_query,
)
from osp_sos_analyser.llm_client import MissingLLMConfiguration


class LangGraphInvestigatorModuleTests(unittest.TestCase):
    def test_investigator_prompt_prefers_reasoning_over_hardcoded_reboot_cause(self) -> None:
        prompt = INVESTIGATOR_SYSTEM_PROMPT
        self.assertIn("widen the search", prompt.lower())
        self.assertIn("not found in available SOS", prompt)
        self.assertIn("external reset", prompt.lower())
        self.assertNotIn(
            "Do NOT call compare_nodes, get_related_entities, or get_operation_path",
            prompt,
        )
        self.assertNotIn("Ignore unrelated controller/OVN", prompt)

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
        self.assertIn("Likely kernel panic", text)
        self.assertNotIn("cluster_id=abc", text)

    def test_normal_chat_render_hides_internal_agent_material(self) -> None:
        text = render_investigation_result(
            {
                "prefetch_digest": "controller warning noise",
                "findings": {"investigator_raw": "internal scratchpad"},
                "final_rca": "The requested VM list is instance-a.",
            },
            include_observability=False,
        )
        self.assertEqual(text, "The requested VM list is instance-a.")

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


class GroqToolCallRecoveryTests(unittest.TestCase):
    def test_parse_function_tag_with_inline_args(self) -> None:
        failed = (
            '<function=search_os_logs({"hostname": "n1-wrkld1-b1-b12-comp008", '
            '"limit": 15, "search_terms": "reboot OR panic OR watchdog OR '
            'oom-kill OR Hardware Error"})></function>'
        )
        parsed = parse_groq_failed_generation(failed)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        name, args = parsed
        self.assertEqual(name, "search_os_logs")
        self.assertEqual(args["hostname"], "n1-wrkld1-b1-b12-comp008")
        self.assertEqual(args["limit"], 15)
        self.assertIn("reboot", args["search_terms"])

    def test_parse_function_tag_with_body_args(self) -> None:
        failed = '<function=search_sos_commands>{"hostname": "comp008"}</function>'
        parsed = parse_groq_failed_generation(failed)
        self.assertEqual(parsed, ("search_sos_commands", {"hostname": "comp008"}))

    def test_extract_from_bad_request_shaped_error(self) -> None:
        class _FakeGroqError(Exception):
            def __init__(self) -> None:
                super().__init__(
                    "Error code: 400 - {'error': {'message': "
                    "\"tool call validation failed: attempted to call tool "
                    "'search_os_logs({\\\"hostname\\\": \\\"comp008\\\"})' "
                    "which was not in request.tools\", "
                    "'type': 'invalid_request_error', 'code': 'tool_use_failed', "
                    "'failed_generation': "
                    "'<function=search_os_logs({\\\"hostname\\\": \\\"comp008\\\"})>"
                    "</function>'}}"
                )
                self.body = {
                    "error": {
                        "message": (
                            "tool call validation failed: attempted to call tool "
                            "'search_os_logs({\"hostname\": \"comp008\"})' "
                            "which was not in request.tools"
                        ),
                        "type": "invalid_request_error",
                        "code": "tool_use_failed",
                        "failed_generation": (
                            '<function=search_os_logs({"hostname": "comp008"})>'
                            "</function>"
                        ),
                    }
                }

        exc = _FakeGroqError()
        self.assertTrue(is_groq_tool_use_failed(exc))
        parsed = extract_groq_failed_tool_call(exc)
        self.assertEqual(parsed, ("search_os_logs", {"hostname": "comp008"}))

    def test_invoke_tool_by_name(self) -> None:
        class _Tool:
            name = "search_os_logs"

            def invoke(self, args: dict) -> str:
                return f"ok:{args.get('hostname')}"

        out = invoke_tool_by_name([_Tool()], "search_os_logs", {"hostname": "comp008"})
        self.assertEqual(out, "ok:comp008")
        missing = invoke_tool_by_name([_Tool()], "nope", {})
        self.assertIn("Unknown tool", missing)

    def test_package_root_exports_dynamic_investigation(self) -> None:
        from osp_sos_analyser import (
            investigate_prompt_with_langchain,
            investigate_with_langgraph,
            render_investigation_result,
        )

        self.assertTrue(callable(investigate_prompt_with_langchain))
        self.assertTrue(callable(investigate_with_langgraph))
        self.assertTrue(callable(render_investigation_result))
