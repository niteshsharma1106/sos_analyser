from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .analysis import AnalysisStore
from .evidence import extract_evidence_hints
from .llm_client import MissingLLMConfiguration
from .models import AgentFinding, CommandRecord, InvestigationReport, LogRecord


class SearchArgs(BaseModel):
    query_terms: list[str] = Field(
        default_factory=list,
        description="Exact terms from the incident context: UUIDs, request IDs, hostnames, errors, API action names.",
    )
    levels: list[str] = Field(
        default_factory=list,
        description="Optional log levels such as ERROR, CRITICAL, WARNING, INFO.",
    )
    limit: int = Field(default=15, ge=1, le=50)


class CommandSearchArgs(BaseModel):
    command_patterns: list[str] = Field(
        default_factory=list,
        description="Command/source patterns to inspect, such as podman_ps, systemctl, ovn, ovs, hostname.",
    )
    limit: int = Field(default=10, ge=1, le=30)


class TimelineArgs(BaseModel):
    query_terms: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    limit: int = Field(default=30, ge=1, le=80)


class RCAResponse(BaseModel):
    summary: str
    most_likely_root_cause: str
    confidence: Literal["none", "low", "medium", "high"]
    impacted_services: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    timeline_analysis: str
    recommended_next_checks: list[str] = Field(default_factory=list)


def investigate_prompt_with_langchain(
    db_path: str | Path,
    prompt: str,
    model: str | None = None,
) -> InvestigationReport:
    from dotenv import load_dotenv

    load_dotenv()


    model_provider = os.getenv("OSP_SOS_MODEL_PROVIDER", "openai")
    os.environ["GROQ_API_KEY"]= os.getenv("GROK_API_KEY")
    required_key = _required_api_key(model_provider)
    if required_key and not os.getenv(required_key):
        raise MissingLLMConfiguration(
            f"LangChain agent analysis with provider '{model_provider}' requires {required_key}. "
            "Set it in .env, or run analyze with --offline."
        )

    store = AnalysisStore(db_path)
    evidence_cache: list[LogRecord | CommandRecord] = []
    agent_notes: list[AgentFinding] = []

    tools = _build_tools(store, evidence_cache, agent_notes)

    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model

    # llm = init_chat_model(
    #     model=model or os.getenv("OSP_SOS_MODEL", _default_model(model_provider)),
    #     model_provider=model_provider,
    # )
    llm = init_chat_model(
        model="llama-3.1-8b-instant",
        model_provider="groq")
    
    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=RCAResponse,
        system_prompt=_system_prompt(),
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Investigate this RHOSP 17 SOS incident. Use the specialist tools; "
                        "do not answer from assumptions alone.\n\n"
                        f"Incident: {prompt}"
                    ),
                }
            ]
        }
    )
    structured = result["structured_response"]
    hints = extract_evidence_hints(prompt)
    timeline = _timeline_from_evidence(evidence_cache)

    return InvestigationReport(
        prompt=prompt,
        hints=hints,
        findings=tuple(agent_notes),
        timeline=timeline,
        error_summary=store.get_error_summary(),
        final_summary=structured.summary,
        final_root_cause=structured.most_likely_root_cause,
        final_confidence=structured.confidence,
        final_timeline_analysis=structured.timeline_analysis,
        final_recommendations=tuple(structured.recommended_next_checks),
    )


def _build_tools(
    store: AnalysisStore,
    evidence_cache: list[LogRecord | CommandRecord],
    agent_notes: list[AgentFinding],
):
    from langchain_core.tools import tool

    def search_services(
        agent_name: str,
        services: list[str],
        args: SearchArgs,
    ) -> str:
        rows: list[LogRecord] = []
        for service in services:
            rows.extend(
                store.search_logs(
                    service=service,
                    text_terms=args.query_terms,
                    levels=args.levels,
                    limit=args.limit,
                )
            )
        rows = _dedupe_logs(rows)[: args.limit]
        evidence_cache.extend(rows)
        summary = _summarize_logs(rows)
        agent_notes.append(
            AgentFinding(
                agent=agent_name,
                summary=summary,
                evidence=tuple(rows[:8]),
                recommendations=(),
                confidence=_confidence_from_logs(rows),
            )
        )
        return json.dumps([_log_to_dict(row) for row in rows], default=str)

    @tool(args_schema=SearchArgs)
    def ask_nova_agent(
        query_terms: list[str] | None = None,
        levels: list[str] | None = None,
        limit: int = 15,
    ) -> str:
        """Search Nova logs using terms chosen from the user context. Use for VM create, spawn, scheduling, instance action, migration, metadata, compute, or API failures."""
        return search_services(
            "Nova Agent",
            ["nova"],
            SearchArgs(query_terms=query_terms or [], levels=levels or [], limit=limit),
        )

    @tool(args_schema=SearchArgs)
    def ask_network_agent(
        query_terms: list[str] | None = None,
        levels: list[str] | None = None,
        limit: int = 15,
    ) -> str:
        """Search Neutron, OVN, and OVS logs. Use for ports, binding, network loss, routers, chassis, tunnels, datapath, and VM networking context."""
        return search_services(
            "Neutron/OVN Agent",
            ["neutron", "ovn"],
            SearchArgs(query_terms=query_terms or [], levels=levels or [], limit=limit),
        )

    @tool(args_schema=SearchArgs)
    def ask_cinder_agent(
        query_terms: list[str] | None = None,
        levels: list[str] | None = None,
        limit: int = 15,
    ) -> str:
        """Search Cinder logs. Use for volume create/delete, attach/detach, backend, scheduler, snapshot, or storage symptoms."""
        return search_services(
            "Cinder Agent",
            ["cinder"],
            SearchArgs(query_terms=query_terms or [], levels=levels or [], limit=limit),
        )

    @tool(args_schema=CommandSearchArgs)
    def ask_system_agent(
        command_patterns: list[str] | None = None,
        limit: int = 10,
    ) -> str:
        """Search SOS command artifacts for Podman, systemctl, OVN/OVS command output, hostname, and host/service state."""
        patterns = command_patterns or []
        rows: list[CommandRecord] = []
        for pattern in patterns:
            rows.extend(store.get_command_output(pattern, limit=limit))
        rows = _dedupe_commands(rows)[:limit]
        evidence_cache.extend(rows)
        agent_notes.append(
            AgentFinding(
                agent="System/Podman Agent",
                summary=f"Found {len(rows)} command artifact(s) for {', '.join(patterns) or 'requested patterns'}.",
                evidence=tuple(rows[:8]),
                recommendations=(),
                confidence="low" if rows else "none",
            )
        )
        return json.dumps([_command_to_dict(row) for row in rows], default=str)

    @tool(args_schema=TimelineArgs)
    def build_context_timeline(
        query_terms: list[str] | None = None,
        services: list[str] | None = None,
        limit: int = 30,
    ) -> str:
        """Build a cross-service timeline from model-selected query terms. Use exact UUIDs/request IDs when present."""
        rows: list[LogRecord] = []
        terms = query_terms or []
        selected_services = services or []
        if terms:
            for term in terms:
                rows.extend(store.find_identifier_events(term, services=selected_services, limit=limit))
        if not rows:
            rows.extend(store.get_timeline(services=selected_services, text_terms=terms, limit=limit))
        rows = _dedupe_logs(rows)[:limit]
        evidence_cache.extend(rows)
        return json.dumps([_log_to_dict(row) for row in rows], default=str)

    @tool
    def get_error_summary() -> str:
        """Return top ERROR/CRITICAL counts by service for broad context. Use after context-specific searches, not as the main evidence."""
        return json.dumps(
            [
                {"service": service, "level": level, "count": count}
                for service, level, count in store.get_error_summary()
            ]
        )

    return [
        ask_nova_agent,
        ask_network_agent,
        ask_cinder_agent,
        ask_system_agent,
        build_context_timeline,
        get_error_summary,
    ]


def _system_prompt() -> str:
    return (
        "You are a Coordinator Agent for RHOSP 17 SOS report analysis. "
        "You must reason from the user's actual incident context and call tools dynamically. "
        "Never use fixed search strings unless they appear in the prompt or are a direct OpenStack synonym of the symptom. "
        "Use exact UUIDs, request IDs, hostnames, timestamps, resource IDs, and pasted log text as primary search terms. "
        "Dispatch specialist tools based on the symptom: Nova for VM lifecycle/API/scheduler/compute; "
        "Network for Neutron/OVN/OVS ports, binding and connectivity; Cinder for volume/storage; "
        "System for podman/systemctl/container/host state. "
        "If evidence is only status 200 polling, say it is weak. "
        "Prioritize ERROR/CRITICAL, HTTP 4xx/5xx, tracebacks, 'not ready', failed, timeout, NoValidHost, binding, backend, and unreachable evidence. "
        "Return a concise RCA with confidence and recommended next checks."
    )


def _required_api_key(model_provider: str) -> str | None:
    provider = model_provider.lower().replace("-", "_")
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider in {"google_genai", "google_vertexai", "google"}:
        return "GOOGLE_API_KEY"
    return None


def _default_model(model_provider: str) -> str:
    provider = model_provider.lower().replace("-", "_")
    if provider in {"google_genai", "google"}:
        return "gemini-2.5-pro"
    return "gpt-5.5"


def _timeline_from_evidence(
    evidence: list[LogRecord | CommandRecord],
) -> tuple[LogRecord, ...]:
    logs = [item for item in evidence if isinstance(item, LogRecord)]
    return tuple(
        sorted(
            _dedupe_logs(logs),
            key=lambda item: (item.timestamp is None, item.timestamp),
        )[:30]
    )


def _summarize_logs(rows: list[LogRecord]) -> str:
    if not rows:
        return "No matching log evidence found."
    first = rows[0]
    return f"Found {len(rows)} matching log row(s); strongest signal: {first.level} {first.module}: {first.message[:180]}"


def _confidence_from_logs(rows: list[LogRecord]) -> str:
    if any(row.level in {"ERROR", "CRITICAL"} for row in rows):
        return "medium"
    if rows:
        return "low"
    return "none"


def _dedupe_logs(items: list[LogRecord]) -> list[LogRecord]:
    seen = set()
    unique = []
    for item in items:
        key = (item.timestamp, item.service, item.module, item.message, item.source_file)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _dedupe_commands(items: list[CommandRecord]) -> list[CommandRecord]:
    seen = set()
    unique = []
    for item in items:
        key = (item.command, item.source_file, item.output[:200])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _log_to_dict(row: LogRecord) -> dict[str, object]:
    return {
        "timestamp": row.timestamp,
        "level": row.level,
        "service": row.service,
        "module": row.module,
        "message": row.message[:1000],
        "source_file": row.source_file,
        "report_name": row.report_name,
    }


def _command_to_dict(row: CommandRecord) -> dict[str, object]:
    return {
        "command": row.command,
        "service": row.service,
        "source_file": row.source_file,
        "output": row.output[:1000],
        "report_name": row.report_name,
    }
