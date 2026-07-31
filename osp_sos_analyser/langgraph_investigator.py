# langgraph_investigator.py — LangGraph RCA workflow (moved out of osp.ipynb).
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .context_pack import truncate_text
from .investigation_tools import build_langchain_tools, prefetch_investigation_digest
from .llm_client import MissingLLMConfiguration
from .observability import (
    AgentObservabilityCallback,
    AgentRunTrace,
    configure_logging,
    get_logger,
)


EXPAND_SYSTEM_PROMPT = """You are a query enrichment agent for a Red Hat OpenStack investigation workflow.
Return JSON only with the fields: summary, intent, entities, keywords, search_queries, investigation_targets, hypotheses, time_window.

CRITICAL RULES:
- `keywords` MUST include any UUIDs, hostnames, or instance names verbatim from the incident report. Do NOT replace them with concept words like 'VM' or 'creation'.
- `keywords` should be lowercase tokens that would actually appear in OpenStack logs (e.g. 'failed', 'error', 'spawn', 'build', the UUID itself).
- `entities.service` may be: nova, cinder, neutron, glance, keystone, heat, octavia, ironic, system, or unknown.
- `entities.resource_id` MUST be the UUID if present in the incident.
- Use `system` for controller/compute host reboots, kernel, hardware, podman, or OS-level symptoms.
- Use `unknown` when the incident does not explicitly identify an OpenStack service. Do not select nova by default.
- Do not fetch more than 10 to 20 events or logs per query.
- Every selected service must be justified by words in the incident.
- Keep output lightweight and valid JSON.
"""

INVESTIGATOR_SYSTEM_PROMPT = """
You are an OpenStack RCA investigator.

Investigation order (mandatory):
1) Call get_cluster_overview once to understand nodes/roles.
2) If multiple nodes are present, call compare_nodes to see which hosts are noisy.
3) If you have a UUID/req-id/hostname, call get_entity_evidence first.
   Use hostname= or node_role= filters when the incident is node-specific.
4) Call get_related_entities and get_operation_path to map
   VM ↔ port ↔ chassis ↔ host (and volume ↔ instance when relevant).
5) Extract related IDs from digests and query those with get_entity_evidence
   (ports/networks -> neutron/ovn, volumes/images -> cinder/glance, instances -> nova).
6) Use list_indexed_entities if you need candidates by type.
7) Use search_os_logs only as a fallback when the evidence/graph index has no hits.
   Prefer scoping with hostname= or node_role= (controller vs compute).

Tool results are already digests. Do not paste them back in full.
Stop once you can explain or rule out a root cause. End with a concise RCA.
"""


class InvestigationEntities(BaseModel):
    investigation_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    resource_id: Optional[str] = Field(
        default=None,
        description="Any UUID at the center of the investigation — instance, volume, port, network, image, etc.",
    )
    resource_type: Optional[str] = Field(
        default=None,
        description="e.g. 'instance', 'volume', 'port', 'network', 'image'",
    )
    service: Optional[str] = Field(default=None)
    problem: Optional[str] = Field(default=None)
    cluster: Optional[str] = Field(default=None)
    hostname: Optional[str] = Field(
        default=None,
        description="Hostname if the incident is tied to one SOS node",
    )
    node_role: Optional[str] = Field(
        default=None,
        description="controller, compute, or other role when known",
    )


class TimeWindow(BaseModel):
    start: Optional[str] = Field(default=None)
    end: Optional[str] = Field(default=None)


class SearchTask(BaseModel):
    service: str
    objective: str
    query: str
    priority: int


class ExpandedQuery(BaseModel):
    summary: str
    intent: str
    entities: InvestigationEntities
    keywords: List[str]
    search_queries: List[SearchTask] = Field(
        default_factory=list,
        description="Search tasks with service, objective, query, and priority.",
    )
    investigation_targets: List[str]
    hypotheses: List[str] = Field(default_factory=list)
    time_window: Optional[TimeWindow] = Field(default=None)


class InvestigationState(TypedDict, total=False):
    investigation_id: UUID
    raw_query: str
    expanded_plan: Optional[ExpandedQuery]
    prefetch_digest: str
    gathered_evidence: List[Dict[str, Any]]
    findings: Dict[str, Any]
    final_rca: Optional[str]
    next_node: str
    agent_trace: Optional[Dict[str, Any]]
    run_id: str


def _normalize_provider(model_provider: str) -> str:
    provider = (model_provider or "").strip().lower().replace("-", "_")
    if provider in {"google", "gemini"}:
        return "google_genai"
    if provider in {"grok"}:
        return "groq"
    return provider


def _required_api_key(model_provider: str) -> str | None:
    provider = _normalize_provider(model_provider)
    if provider in {"openai"}:
        return "OPENAI_API_KEY"
    if provider in {"google_genai", "google_vertexai"}:
        return "GOOGLE_API_KEY"
    if provider in {"groq"}:
        return "GROQ_API_KEY"
    return None


def _default_model(model_provider: str) -> str:
    provider = _normalize_provider(model_provider)
    if provider == "google_genai":
        return "gemini-2.5-pro"
    if provider == "openai":
        return "gpt-4o-mini"
    # Groq-hosted open models use the org/model form.
    return "openai/gpt-oss-120b"


def _validate_model_provider(model: str, model_provider: str) -> None:
    """Fail fast on obvious model/provider mismatches before the HTTP call."""
    provider = _normalize_provider(model_provider)
    lower = (model or "").strip().lower()
    if not lower:
        return

    looks_like_groq = lower.startswith("openai/") or lower.startswith("meta-llama/")
    looks_like_gemini = lower.startswith("gemini")
    looks_like_openai_api = lower.startswith("gpt-") or lower.startswith("o1") or lower.startswith("o3")

    if provider == "google_genai" and (looks_like_groq or looks_like_openai_api):
        raise MissingLLMConfiguration(
            f"Model '{model}' is not a Google Gemini model, but "
            f"OSP_SOS_MODEL_PROVIDER={provider}.\n\n"
            "Fix your `.env` to one of:\n"
            "  # Google\n"
            "  OSP_SOS_MODEL_PROVIDER=google_genai\n"
            "  OSP_SOS_MODEL=gemini-2.5-pro\n"
            "  GOOGLE_API_KEY=...\n\n"
            "  # Groq (for openai/gpt-oss-120b)\n"
            "  OSP_SOS_MODEL_PROVIDER=groq\n"
            "  OSP_SOS_MODEL=openai/gpt-oss-120b\n"
            "  GROQ_API_KEY=...\n"
        )
    if provider == "groq" and looks_like_gemini:
        raise MissingLLMConfiguration(
            f"Model '{model}' looks like Gemini, but provider is '{provider}'. "
            "Set OSP_SOS_MODEL_PROVIDER=google_genai or use a Groq model id."
        )
    if provider == "openai" and (looks_like_groq or looks_like_gemini):
        raise MissingLLMConfiguration(
            f"Model '{model}' does not match provider '{provider}'. "
            "Use an OpenAI model (e.g. gpt-4o-mini) or change OSP_SOS_MODEL_PROVIDER."
        )


def _init_llm(model: str | None = None, model_provider: str | None = None):
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model

    if os.getenv("OSP_SOS_SKIP_DOTENV") != "1":
        load_dotenv()

    _grok_key = os.getenv("GROK_API_KEY")
    if _grok_key and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = _grok_key

    provider = _normalize_provider(
        model_provider or os.getenv("OSP_SOS_MODEL_PROVIDER", "groq")
    )
    resolved_model = (
        (model or "").strip()
        or os.getenv("OSP_SOS_MODEL", "").strip()
        or _default_model(provider)
    )
    required_key = _required_api_key(provider)
    if required_key and not os.getenv(required_key):
        raise MissingLLMConfiguration(
            f"LangGraph investigator with provider '{provider}' requires {required_key}. "
            "Set it in .env, or run analyze/chat with --offline."
        )
    _validate_model_provider(resolved_model, provider)

    llm = init_chat_model(model=resolved_model, model_provider=provider)
    # OpenAI-compatible APIs accept this; Google GenAI rejects it as an
    # unknown GenerateContentConfig field.
    if provider in {"openai", "groq"}:
        return llm.bind(parallel_tool_calls=False)
    return llm


def build_investigation_app(
    db_con,
    *,
    model: str | None = None,
    model_provider: str | None = None,
    llm=None,
    trace: AgentRunTrace | None = None,
):
    """Compile the expand → investigate → synthesize LangGraph app."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableConfig, RunnablePassthrough
    from langgraph.graph import END, START, StateGraph

    log = get_logger("investigator")
    run_trace = trace or AgentRunTrace()
    llm = llm or _init_llm(model=model, model_provider=model_provider)
    tools = build_langchain_tools(db_con)

    prompt_expand = ChatPromptTemplate.from_messages(
        [
            ("system", EXPAND_SYSTEM_PROMPT),
            ("human", "Incident Report: {query}"),
        ]
    )

    summarization = SummarizationMiddleware(
        model=llm,
        trigger=("tokens", 3000),
        keep=("messages", 10),
    )
    investigator_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=INVESTIGATOR_SYSTEM_PROMPT,
        middleware=[summarization],
    )

    def expand_query_node(state: InvestigationState) -> dict:
        run_trace.node_start("QUERY_EXPAND")
        log.info("Expanding query for run %s", run_trace.run_id[:8])
        structured_llm = llm.with_structured_output(ExpandedQuery, method="json_schema")
        enrichment_chain = {"query": RunnablePassthrough()} | prompt_expand | structured_llm
        expand_cb = AgentObservabilityCallback(run_trace, stage="query_expand")
        run_trace.llm_start("query_expand")
        result = enrichment_chain.invoke(
            state["raw_query"],
            config=RunnableConfig(callbacks=[expand_cb.handler]),
        )
        run_trace.llm_end(
            "query_expand",
            intent=getattr(result, "intent", None),
            resource_id=getattr(getattr(result, "entities", None), "resource_id", None),
        )
        run_trace.add(
            "plan",
            "Expanded investigation plan",
            details={
                "summary": getattr(result, "summary", ""),
                "intent": getattr(result, "intent", ""),
                "keywords": list(getattr(result, "keywords", []) or [])[:8],
            },
        )

        resource_id = None
        if result.entities and result.entities.resource_id:
            resource_id = result.entities.resource_id
        digest = prefetch_investigation_digest(
            db_con,
            raw_query=state["raw_query"],
            resource_id=resource_id,
            keywords=list(result.keywords or [])[:8],
            limit_per_entity=12,
        )
        run_trace.add(
            "prefetch",
            "Built cluster/evidence prefetch digest",
            details={"chars": len(digest), "preview": truncate_text(digest, 240)},
        )
        run_trace.node_end("QUERY_EXPAND")
        return {
            "expanded_plan": result,
            "prefetch_digest": digest,
            "run_id": run_trace.run_id,
        }

    def investigator_node(state: InvestigationState) -> dict:
        run_trace.node_start("INVESTIGATOR")
        plan = state["expanded_plan"]
        prefetch = state.get("prefetch_digest") or "No prefetched evidence."
        user_message = (
            "Investigate this RHOSP incident using Cluster Manifest + Evidence Index tools.\n\n"
            f"Incident: {plan.summary}\n"
            f"Known entities: {plan.entities.model_dump()}\n"
            f"Keywords: {plan.keywords}\n"
            f"Time window: {plan.time_window}\n\n"
            f"Prefetched digest (already retrieved — build on it):\n{prefetch}\n"
        )
        agent_cb = AgentObservabilityCallback(run_trace, stage="investigator")
        result = investigator_agent.invoke(
            {"messages": [("user", user_message)]},
            config=RunnableConfig(callbacks=[agent_cb.handler]),
        )
        messages = result["messages"]

        call_args: dict[str, dict[str, Any]] = {}
        for message in messages:
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    call_args[tool_call["id"]] = tool_call.get("args", {})

        gathered_evidence = []
        for message in messages:
            if isinstance(message, ToolMessage):
                args = call_args.get(message.tool_call_id, {})
                tool_name = message.name or "tool"
                output = str(message.content)
                # Callbacks usually record tool events; keep a digest fallback.
                if not any(
                    event.kind == "tool_end" and tool_name in event.message
                    for event in run_trace.events
                ):
                    run_trace.tool_start(tool_name, args)
                    run_trace.tool_end(tool_name, output)
                gathered_evidence.append(
                    {
                        "service": args.get("service") or args.get("entity_type") or "index",
                        "source": tool_name,
                        "summary": str(args),
                        "raw_logs": truncate_text(output, 1200),
                    }
                )

        run_trace.add(
            "investigator_summary",
            "Investigator finished tool loop",
            details={"tool_results": len(gathered_evidence)},
        )
        run_trace.node_end("INVESTIGATOR", tools=len(gathered_evidence))
        return {
            "gathered_evidence": gathered_evidence,
            "findings": {"investigator_raw": messages[-1].content},
        }

    def synthesize_rca(state: InvestigationState) -> dict:
        run_trace.node_start("SYNTHESIZE_RCA")
        plan = state["expanded_plan"]
        evidence = state.get("gathered_evidence", [])
        investigator_notes = state.get("findings", {}).get("investigator_raw", "")
        prefetch = truncate_text(state.get("prefetch_digest", ""), 1500)
        evidence_text = (
            "\n\n".join(f"[{item['service']}/{item['source']}] {item['raw_logs']}" for item in evidence)
            if evidence
            else "No tool evidence was gathered."
        )
        rca_prompt = f"""
    Incident: {plan.summary}
    Hypotheses considered: {plan.hypotheses}
    Prefetched cluster/evidence digest:
    {prefetch or 'None'}
    Investigator's working notes: {investigator_notes}
    Evidence gathered:
    {evidence_text}

    Write a concise root cause analysis. Cite hostnames and entity IDs when possible.
    If evidence is weak, say so and recommend next checks.
    """
        synth_cb = AgentObservabilityCallback(run_trace, stage="synthesize")
        result = llm.invoke(
            rca_prompt,
            config=RunnableConfig(callbacks=[synth_cb.handler]),
        )
        content = getattr(result, "content", result)
        run_trace.add(
            "rca",
            "Synthesized root cause analysis",
            details={"chars": len(str(content))},
        )
        run_trace.node_end("SYNTHESIZE_RCA")
        return {
            "final_rca": content,
            "agent_trace": run_trace.to_dict(),
        }

    workflow = StateGraph(InvestigationState)
    workflow.add_node("QUERY_EXPAND", expand_query_node)
    workflow.add_node("INVESTIGATOR", investigator_node)
    workflow.add_node("SYNTHESIZE_RCA", synthesize_rca)
    workflow.add_edge(START, "QUERY_EXPAND")
    workflow.add_edge("QUERY_EXPAND", "INVESTIGATOR")
    workflow.add_edge("INVESTIGATOR", "SYNTHESIZE_RCA")
    workflow.add_edge("SYNTHESIZE_RCA", END)
    return workflow.compile()


def investigate_with_langgraph(
    db_path: str | os.PathLike[str],
    prompt: str,
    *,
    model: str | None = None,
    model_provider: str | None = None,
    recursion_limit: int = 15,
    focus_entity: str | None = None,
    answer_style: str = "Concise RCA",
    include_graph: bool = True,
    trace: AgentRunTrace | None = None,
) -> dict[str, Any]:
    """Run the notebook RCA workflow from a Python entrypoint."""
    from langchain_core.utils.uuid import uuid7

    from .dbconnector import DatabaseConnector

    configure_logging()
    log = get_logger("investigator")

    enriched = (prompt or "").strip()
    extras: list[str] = []
    if focus_entity and focus_entity.strip():
        extras.append(f"Focused entity seed: {focus_entity.strip()}")
    if include_graph:
        extras.append(
            "Prefer relationship graph tools (get_related_entities, get_operation_path) "
            "to map VM↔port↔chassis↔host."
        )
    style = (answer_style or "Concise RCA").strip()
    if style == "Evidence-heavy":
        extras.append("Answer style: evidence-heavy — cite more digests and hostnames.")
    elif style == "Operation path first":
        extras.append(
            "Answer style: start with the operation path (VM→port→chassis→host), then RCA."
        )
    else:
        extras.append("Answer style: concise RCA.")
    if extras:
        enriched = enriched + "\n\n" + "\n".join(extras)

    run_trace = trace or AgentRunTrace(prompt=enriched)
    run_trace.add("run_start", "Starting LangGraph investigation", details={"db_path": str(db_path)})
    log.info("Investigation start run=%s prompt=%s", run_trace.run_id[:8], truncate_text(prompt, 120))

    try:
        with DatabaseConnector(db_path, read_only=True) as db:
            app = build_investigation_app(
                db.connect(),
                model=model,
                model_provider=model_provider,
                trace=run_trace,
            )
            initial_state: InvestigationState = {
                "raw_query": enriched,
                "expanded_plan": None,
                "prefetch_digest": "",
                "gathered_evidence": [],
                "findings": {},
                "final_rca": "",
                "next_node": "",
                "run_id": run_trace.run_id,
            }
            config = {
                "run_id": uuid7(),
                "recursion_limit": recursion_limit,
            }
            final_state = app.invoke(initial_state, config=config)
    except Exception as exc:
        run_trace.error(f"Investigation failed: {exc}")
        raise

    final_state = dict(final_state)
    final_state["agent_trace"] = run_trace.to_dict()
    final_state["run_id"] = run_trace.run_id
    run_trace.add("run_end", "Investigation complete")
    log.info(
        "Investigation complete run=%s duration_ms=%s events=%s",
        run_trace.run_id[:8],
        run_trace.to_dict()["duration_ms"],
        len(run_trace.events),
    )
    return final_state


def render_investigation_result(
    final_state: dict[str, Any],
    *,
    include_observability: bool = True,
) -> str:
    """Pretty-print a LangGraph investigation result as markdown-ish text."""
    lines = ["# LangGraph Investigation Report", ""]
    run_id = final_state.get("run_id")
    if run_id:
        lines.append(f"- run_id: `{run_id}`")
    plan = final_state.get("expanded_plan")
    if plan is not None:
        intent = getattr(plan, "intent", None) or (plan.get("intent") if isinstance(plan, dict) else None)
        window = getattr(plan, "time_window", None) or (
            plan.get("time_window") if isinstance(plan, dict) else None
        )
        lines.append(f"- Intent: {intent}")
        lines.append(f"- Time window: {window}")
        lines.append("")
    prefetch = final_state.get("prefetch_digest") or ""
    if prefetch:
        lines.extend(["## Prefetch digest", prefetch[:1500], ""])
    findings = final_state.get("findings") or {}
    if findings:
        lines.append("## Findings")
        for key, value in findings.items():
            lines.append(f"### {key}")
            lines.append(str(value))
            lines.append("")
    lines.append("## Root Cause Analysis")
    lines.append(str(final_state.get("final_rca") or "No RCA generated."))

    if include_observability:
        restored = AgentRunTrace.from_dict(final_state.get("agent_trace"))
        if restored and restored.events:
            lines.extend(["", restored.render_markdown()])
    return "\n".join(lines)
