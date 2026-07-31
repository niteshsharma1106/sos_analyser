# langgraph_investigator.py — LangGraph RCA workflow (moved out of osp.ipynb).
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

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
Return ONE compact JSON object only — no markdown, no commentary.

Required fields:
  summary, intent, entities, keywords, search_queries, investigation_targets, hypotheses, time_window

CRITICAL RULES (follow exactly):
- Keep EVERY string SHORT. summary ≤ 160 chars. intent ≤ 80 chars.
- `entities.node_role` MUST be exactly one of: controller | compute | storage | unknown
  NEVER put keywords, log phrases, or hyphenated dumps into node_role.
- `entities.hostname` is the node hostname from the incident (full name preferred), ≤ 64 chars, or null.
- `entities.service` may be: nova, cinder, neutron, glance, keystone, heat, octavia, ironic, system, or unknown.
- Use `system` for host reboots, kernel, hardware, or OS-level symptoms.
- `keywords`: 3–12 short lowercase tokens (hostnames, UUIDs, error words). Each ≤ 48 chars.
- `investigation_targets`: 2–8 short labels (e.g. "system", "kernel", "nova-compute"). Each ≤ 48 chars.
- `search_queries`: at most 5 items; each query string ≤ 120 chars.
- `hypotheses`: at most 5 short strings.
- Do NOT invent long keyword chains. Do NOT repeat the same phrase.
- Prefer null over inventing entities.
- Output must be valid, complete JSON that fits in a small response.
"""

_KNOWN_NODE_ROLES = frozenset({"controller", "compute", "storage", "network", "ceph", "unknown"})
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HOSTNAME_RE = re.compile(
    r"\b("
    r"[A-Za-z0-9][A-Za-z0-9._-]{3,60}?(?:comp|ctrl|ceph|compute|controller)\d+[A-Za-z0-9._-]*"
    r"|"
    r"(?:ctrl|comp|compute|controller|ceph|storage|wrkld|node)[\w.-]*\d[\w.-]*"
    r")\b",
    re.IGNORECASE,
)
_SHORT_HOSTNAME_RE = re.compile(r"\b([a-z][a-z0-9-]{1,30}\d{2,})\b", re.IGNORECASE)

INVESTIGATOR_SYSTEM_PROMPT = """
You are an OpenStack RCA investigator.

Investigation order (mandatory):
1) Call get_cluster_overview once to understand nodes/roles.
2) If multiple nodes are present, call compare_nodes to see which hosts are noisy.
3) If you have a UUID/req-id/hostname, call get_entity_evidence first.
   Use hostname= or node_role= filters when the incident is node-specific.
   Short names like comp008 resolve to full hostnames automatically.
4) For host reboot / crash / panic / power questions (CRITICAL):
   a) Call search_sos_commands with hostname= and command_pattern='dmesg,last,journalctl,uptime'
      (look for panic, MCE, watchdog, oom, shutdown).
   b) Call search_os_logs with hostname= and search_terms using OR, e.g.
      'reboot OR panic OR watchdog OR oom-kill OR Hardware Error'.
   Do not conclude "no evidence" until sos_commands were checked.
5) Call get_related_entities and get_operation_path to map
   VM ↔ port ↔ chassis ↔ host (and volume ↔ instance when relevant).
6) Extract related IDs from digests and query those with get_entity_evidence
   (ports/networks -> neutron/ovn, volumes/images -> cinder/glance, instances -> nova).
7) Use list_indexed_entities if you need candidates by type.
8) Use search_os_logs only as a fallback when the evidence/graph index has no hits.
   Prefer scoping with hostname= or node_role= (controller vs compute).

Tool results are already digests. Do not paste them back in full.
Stop once you can explain or rule out a root cause. End with a concise RCA.
"""


def _clip_str(value: Any, max_len: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len:
        return text[:max_len].rstrip("-_ .")
    return text


def _normalize_node_role(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _KNOWN_NODE_ROLES:
        return text
    # Models sometimes dump keywords into node_role; salvage a known token.
    for role in ("controller", "compute", "storage", "network", "ceph"):
        if re.search(rf"\b{role}\b", text) or text.startswith(role):
            return role
    if len(text) > 32:
        return None
    return None


def _clean_token_list(values: Any, *, max_items: int, max_len: int) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = _clip_str(raw, max_len)
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
        if len(out) >= max_items:
            break
    return out


class InvestigationEntities(BaseModel):
    investigation_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    resource_id: Optional[str] = Field(
        default=None,
        description="UUID at the center of the investigation, if present.",
        max_length=64,
    )
    resource_type: Optional[str] = Field(
        default=None,
        description="instance | volume | port | network | image | host",
        max_length=32,
    )
    service: Optional[str] = Field(
        default=None,
        description="nova|cinder|neutron|glance|keystone|heat|octavia|ironic|system|unknown",
        max_length=32,
    )
    problem: Optional[str] = Field(default=None, max_length=120)
    cluster: Optional[str] = Field(default=None, max_length=64)
    hostname: Optional[str] = Field(
        default=None,
        description="Hostname from the incident (prefer full name, e.g. n1-wrkld1-b1-b12-comp008)",
        max_length=64,
    )
    node_role: Optional[str] = Field(
        default=None,
        description="Exactly one of: controller, compute, storage, unknown",
        max_length=32,
    )

    @field_validator("resource_id", mode="before")
    @classmethod
    def _clip_resource_id(cls, value: Any) -> Any:
        return _clip_str(value, 64)

    @field_validator("resource_type", "service", mode="before")
    @classmethod
    def _clip_short_enums(cls, value: Any) -> Any:
        return _clip_str(value, 32)

    @field_validator("problem", mode="before")
    @classmethod
    def _clip_problem(cls, value: Any) -> Any:
        return _clip_str(value, 120)

    @field_validator("cluster", mode="before")
    @classmethod
    def _clip_cluster(cls, value: Any) -> Any:
        return _clip_str(value, 64)

    @field_validator("hostname", mode="before")
    @classmethod
    def _clip_hostname(cls, value: Any) -> Any:
        return _clip_str(value, 64)

    @field_validator("node_role", mode="before")
    @classmethod
    def _coerce_node_role(cls, value: Any) -> Any:
        return _normalize_node_role(value)


class TimeWindow(BaseModel):
    start: Optional[str] = Field(default=None, max_length=64)
    end: Optional[str] = Field(default=None, max_length=64)


class SearchTask(BaseModel):
    service: str = Field(max_length=32)
    objective: str = Field(max_length=160)
    query: str = Field(max_length=160)
    priority: int = Field(default=1, ge=1, le=10)

    @field_validator("service", "objective", "query", mode="before")
    @classmethod
    def _clip_search_fields(cls, value: Any) -> Any:
        return _clip_str(value, 160) or ""


class ExpandedQuery(BaseModel):
    summary: str = Field(max_length=240)
    intent: str = Field(max_length=120)
    entities: InvestigationEntities
    keywords: List[str] = Field(default_factory=list, max_length=16)
    search_queries: List[SearchTask] = Field(
        default_factory=list,
        description="At most 5 search tasks.",
        max_length=5,
    )
    investigation_targets: List[str] = Field(default_factory=list, max_length=12)
    hypotheses: List[str] = Field(default_factory=list, max_length=5)
    time_window: Optional[TimeWindow] = Field(default=None)

    @field_validator("summary", "intent", mode="before")
    @classmethod
    def _clip_top_strings(cls, value: Any) -> Any:
        return _clip_str(value, 240) or ""

    @field_validator("keywords", "investigation_targets", "hypotheses", mode="before")
    @classmethod
    def _clean_lists(cls, value: Any) -> Any:
        return _clean_token_list(value, max_items=16, max_len=48)

    @model_validator(mode="after")
    def _trim_collections(self) -> "ExpandedQuery":
        self.keywords = _clean_token_list(self.keywords, max_items=12, max_len=48)
        self.investigation_targets = _clean_token_list(
            self.investigation_targets, max_items=8, max_len=48
        )
        self.hypotheses = _clean_token_list(self.hypotheses, max_items=5, max_len=120)
        if len(self.search_queries) > 5:
            self.search_queries = self.search_queries[:5]
        return self


def _extract_hostname_from_text(text: str) -> Optional[str]:
    match = _HOSTNAME_RE.search(text or "")
    if match:
        return match.group(1)
    match = _SHORT_HOSTNAME_RE.search(text or "")
    if match:
        return match.group(1)
    return None


def _guess_node_role(text: str) -> Optional[str]:
    lower = (text or "").lower()
    if re.search(r"\bcomput", lower):
        return "compute"
    if re.search(r"\bcontrol", lower):
        return "controller"
    if re.search(r"\bstorage|\bceph\b", lower):
        return "storage"
    return None


def fallback_expanded_query(raw_query: str) -> ExpandedQuery:
    """Heuristic ExpandedQuery when the LLM returns truncated/invalid JSON."""
    text = (raw_query or "").strip()
    lower = text.lower()
    hostname = _extract_hostname_from_text(text)
    node_role = _guess_node_role(text)
    uuids = _UUID_RE.findall(text)

    keywords = _clean_token_list(
        [
            hostname,
            *uuids[:2],
            *(
                token
                for token in (
                    "reboot",
                    "kernel",
                    "panic",
                    "oom",
                    "crash",
                    "failed",
                    "error",
                    "timeout",
                    "nova-compute",
                    "shutdown",
                    "power",
                )
                if token in lower
            ),
        ],
        max_items=12,
        max_len=48,
    )
    if not keywords:
        keywords = _clean_token_list(re.findall(r"[a-z0-9-]{3,}", lower)[:8], max_items=8, max_len=48)

    service = "system"
    if any(w in lower for w in ("neutron", "ovn", "port binding", "chassis")):
        service = "neutron"
    elif any(w in lower for w in ("cinder", "volume")):
        service = "cinder"
    elif any(w in lower for w in ("glance", "image")):
        service = "glance"
    elif any(w in lower for w in ("nova", "instance", "spawn", "vm ")) and "reboot" not in lower:
        service = "nova"
    elif any(w in lower for w in ("reboot", "kernel", "panic", "hardware", "power")):
        service = "system"

    targets = ["system"]
    if service != "system":
        targets.append(service)
    if "reboot" in lower or "panic" in lower:
        targets.extend(["kernel", "journal", "nova-compute"])
    if hostname:
        targets.append(hostname)

    summary = truncate_text(text.replace("\n", " "), 160) or "Investigate OpenStack incident"
    return ExpandedQuery(
        summary=summary,
        intent="investigate_incident",
        entities=InvestigationEntities(
            resource_id=uuids[0] if uuids else None,
            resource_type="host" if hostname and service == "system" else None,
            service=service,
            problem=truncate_text(text.replace("\n", " "), 120),
            hostname=hostname,
            node_role=node_role or ("compute" if hostname and hostname.lower().startswith("comp") else None),
        ),
        keywords=keywords,
        search_queries=[
            SearchTask(
                service=service,
                objective="Find host/system errors around the incident",
                query=" ".join(keywords[:6]) or "error failed",
                priority=1,
            )
        ],
        investigation_targets=_clean_token_list(targets, max_items=8, max_len=48),
        hypotheses=[],
        time_window=None,
    )


def sanitize_expanded_query(plan: ExpandedQuery | dict[str, Any] | None, raw_query: str = "") -> ExpandedQuery:
    """Normalize a parsed plan; fall back if empty/invalid."""
    if plan is None:
        return fallback_expanded_query(raw_query)
    if isinstance(plan, dict):
        try:
            plan = ExpandedQuery.model_validate(plan)
        except Exception:
            return fallback_expanded_query(raw_query)
    # Re-run validators via model_validate to clip any post-parse mutations.
    try:
        cleaned = ExpandedQuery.model_validate(plan.model_dump())
    except Exception:
        return fallback_expanded_query(raw_query)
    if not cleaned.keywords and raw_query:
        fb = fallback_expanded_query(raw_query)
        cleaned.keywords = fb.keywords
        if not cleaned.entities.hostname:
            cleaned.entities.hostname = fb.entities.hostname
        if not cleaned.entities.node_role:
            cleaned.entities.node_role = fb.entities.node_role
        if not cleaned.investigation_targets:
            cleaned.investigation_targets = fb.investigation_targets
    return cleaned


def _parse_partial_expanded_json(text: str, raw_query: str) -> ExpandedQuery:
    """Best-effort parse when the model returns truncated JSON."""
    blob = (text or "").strip()
    if not blob:
        return fallback_expanded_query(raw_query)
    # Strip markdown fences if present.
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*", "", blob)
        blob = re.sub(r"\s*```$", "", blob)
    try:
        data = json.loads(blob)
        return sanitize_expanded_query(data, raw_query)
    except json.JSONDecodeError:
        pass

    # Salvage truncated JSON object by closing open braces/quotes roughly.
    start = blob.find("{")
    if start < 0:
        return fallback_expanded_query(raw_query)
    candidate = blob[start:]
    # Cut runaway node_role / string values at a reasonable length if unterminated.
    def _fix_node_role(match: re.Match[str]) -> str:
        role = _normalize_node_role(match.group(2) or "") or "unknown"
        return f'{match.group(1)}{role}"'

    candidate = re.sub(
        r'("node_role"\s*:\s*")([^"]{0,64})[^"]*',
        _fix_node_role,
        candidate,
        count=1,
    )
    # Try progressively truncating at last complete-looking key boundary.
    for end in range(len(candidate), max(len(candidate) - 8000, 40), -1):
        snippet = candidate[:end].rstrip(", \n\t")
        # Close open strings/braces naively.
        if snippet.count('"') % 2 == 1:
            snippet += '"'
        open_braces = snippet.count("{") - snippet.count("}")
        open_brackets = snippet.count("[") - snippet.count("]")
        snippet += "]" * max(open_brackets, 0)
        snippet += "}" * max(open_braces, 0)
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                data.setdefault("keywords", [])
                data.setdefault("investigation_targets", [])
                data.setdefault("summary", truncate_text(raw_query, 160) or "Investigate")
                data.setdefault("intent", "investigate_incident")
                data.setdefault("entities", {})
                return sanitize_expanded_query(data, raw_query)
        except Exception:
            continue
    return fallback_expanded_query(raw_query)


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
    from .env_config import normalize_provider

    return normalize_provider(model_provider)


def _required_api_key(model_provider: str) -> str | None:
    from .env_config import api_key_env_for_provider

    return api_key_env_for_provider(model_provider)


def _init_llm(model: str | None = None, model_provider: str | None = None):
    from langchain.chat_models import init_chat_model

    from .env_config import get_llm_settings

    settings = get_llm_settings(model=model, model_provider=model_provider)
    llm = init_chat_model(model=settings.model, model_provider=settings.provider)
    # OpenAI-compatible APIs accept this; Google GenAI rejects it as an
    # unknown GenerateContentConfig field.
    if settings.provider in {"openai", "groq"}:
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
        raw_query = state["raw_query"]
        expand_cb = AgentObservabilityCallback(run_trace, stage="query_expand")
        run_trace.llm_start("query_expand")
        result: ExpandedQuery | None = None
        parse_error: str | None = None
        try:
            structured_llm = llm.with_structured_output(
                ExpandedQuery, method="json_schema"
            )
            enrichment_chain = (
                {"query": RunnablePassthrough()} | prompt_expand | structured_llm
            )
            result = enrichment_chain.invoke(
                raw_query,
                config=RunnableConfig(callbacks=[expand_cb.handler]),
            )
            result = sanitize_expanded_query(result, raw_query)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "Structured expand failed (%s); attempting salvage/fallback",
                truncate_text(parse_error, 200),
            )
            # Prefer salvaging model text from the exception when available.
            completion = getattr(exc, "llm_output", None) or getattr(exc, "completion", None)
            if isinstance(completion, dict):
                completion = completion.get("text") or completion.get("content")
            if not completion:
                # LangChain OutputParserException often stores the raw text here.
                completion = getattr(exc, "text", None) or str(exc)
            # Strip the leading "Failed to parse ..." wrapper if present.
            if isinstance(completion, str) and "from completion" in completion:
                idx = completion.find("{")
                if idx >= 0:
                    completion = completion[idx:]
            try:
                result = _parse_partial_expanded_json(str(completion or ""), raw_query)
            except Exception:
                result = fallback_expanded_query(raw_query)
            run_trace.add(
                "plan_fallback",
                "Used salvage/fallback ExpandedQuery after parse failure",
                details={"error": truncate_text(parse_error, 240)},
            )

        assert result is not None
        run_trace.llm_end(
            "query_expand",
            intent=getattr(result, "intent", None),
            resource_id=getattr(getattr(result, "entities", None), "resource_id", None),
            fallback=bool(parse_error),
        )
        run_trace.add(
            "plan",
            "Expanded investigation plan",
            details={
                "summary": getattr(result, "summary", ""),
                "intent": getattr(result, "intent", ""),
                "hostname": getattr(getattr(result, "entities", None), "hostname", None),
                "node_role": getattr(getattr(result, "entities", None), "node_role", None),
                "keywords": list(getattr(result, "keywords", []) or [])[:8],
                "fallback": bool(parse_error),
            },
        )

        resource_id = None
        if result.entities and result.entities.resource_id:
            resource_id = result.entities.resource_id
        digest = prefetch_investigation_digest(
            db_con,
            raw_query=raw_query,
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
