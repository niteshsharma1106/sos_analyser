# langgraph_investigator.py — LangGraph RCA workflow (moved out of osp.ipynb).
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .context_pack import truncate_text
from .investigation_tools import build_langchain_tools, is_rebootish_query, prefetch_investigation_digest
from .llm_client import MissingLLMConfiguration
from .observability import (
    AgentObservabilityCallback,
    AgentRunTrace,
    configure_logging,
    get_logger,
)
from .privacy import redact_sensitive_text


EXPAND_SYSTEM_PROMPT = """You are a query enrichment agent for a Red Hat OpenStack investigation workflow.
Return ONE compact JSON object only — no markdown, no commentary.

Required fields:
  summary, intent, entities, keywords, search_queries, investigation_targets, hypotheses, time_window

CRITICAL RULES (follow exactly):
- Keep EVERY string SHORT. summary ≤ 160 chars. intent ≤ 80 chars.
- `entities.node_role` MUST be exactly one of: controller | compute | storage | unknown
  NEVER put keywords, log phrases, or hyphenated dumps into node_role.
- `entities.hostname` is the node hostname from the incident (full name preferred), ≤ 64 chars, or null.
- `entities.service` may be: nova, cinder, neutron, glance, keystone, heat, octavia, ironic, system, rabbitmq, pacemaker, pcs, chronosync or unknown.
- Use `system` for host reboots, kernel, hardware, or OS-level symptoms.
- `keywords`: 3–12 short lowercase tokens (hostnames, UUIDs, error words). Each ≤ 48 chars.
- `investigation_targets`: 2–8 short labels based on the investigation (for example: "system", "kernel", "nova-compute"). Each ≤ 48 chars.
- `search_queries`: at most 5 objects. Every object MUST contain:
  `service` (for example "system"), `objective` (what to find),
  `query` (the search text), and `priority` (1–10).
    Example:
    "search_queries": [
    {{
        "service": "system",
        "objective": "Find evidence preceding the compute reboot",
        "query": "comp008 reboot panic watchdog",
        "priority": 1
    }}
    ]
- `hypotheses`: at most 5 short strings.
- `time_window`: either null, or an object with optional `start`, `end`, or
  `relative` fields. Use `{{"relative": "last 24 hours"}}` when exact timestamps
  are unavailable; never return a bare string.
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
    r"(?:ctrl|comp|compute|controller|ceph|storage|wrkld|node|ctl|cmp)[\w.-]*\d[\w.-]*"
    r")\b",
    re.IGNORECASE,
)
_SHORT_HOSTNAME_RE = re.compile(r"\b([a-z][a-z0-9-]{1,30}\d{2,})\b", re.IGNORECASE)

INVESTIGATOR_SYSTEM_PROMPT = """
You are an OpenStack RCA investigator.

CRITICAL EXECUTION RULES:
- Call tools ONE AT A TIME via the native tool-calling API only.
- Never write XML tags, <function=...>, or Python-style tool_name({...}) text.
- Wait for each tool result before choosing the next tool.
- Stay on the user question. Do not chase unrelated noise.
- Before every tool call, check that its result reduces uncertainty in the user's
  exact question. Prefer the next most informative check over broad dumps.
- If none of the named tools can directly answer a read-only question, use
  create_and_run_analysis to create a temporary SQL analysis for this request. Give
  it a clear name and purpose, query only the supplied snapshot, and then use its
  result. This is preferred over guessing or exploring unrelated evidence.
- Stop when the question is answered with evidence. Do not invent causes.

Reasoning style (apply to every investigation):
- Separate facts (timestamps, tool quotes) from hypotheses.
- After each tool result, ask: what is still unexplained? Choose the next tool
  that would falsify or confirm the leading uncertainty.
- Absence of one class of evidence is not proof of a specific alternative cause.
  Say "not found in available SOS" when that is all you know.
- When the affected host's own logs are silent or inconclusive around an event,
  widen the search to other ingested cluster nodes for the same time window and
  for mentions of that hostname (short or FQDN). Management-plane or peer logs
  often explain what the host itself never wrote.
- Prefer positive evidence that names the host and action over generic guesses
  (do not default to BMC / manual power-cycle / "external reset" without quotes).

Investigation order:
A) Host reboot / crash / panic / power / auto-reboot questions (HIGHEST PRIORITY):
   1. Call get_host_reboot_timeline with the hostname first → establish WHEN.
      The timeline also includes Peer/cluster mentions from OTHER hosts in the
      boot gap — read that section before concluding "unknown".
   2. On that host, search around the boot window for local OS crash evidence
      (panic/watchdog/oom/MCE/Hardware Error, abrupt halt). Leave service empty.
   3. If local cause is missing: use the peer mentions already in the timeline
      (or call search_peer_mentions with the boot-gap times). Do NOT set
      hostname=<affected host> on search_os_logs when looking for peer reactions —
      that only searches the host's own SOS and will miss controller/peer evidence.
   4. Answer: last boot time → evidenced cause from quotes (or unknown) → open gaps.
      Cite hostnames and times. Do not fill gaps with unverified BMC/external-reset stories.

B) Other incidents:
   1) get_cluster_overview once if hostname/role is unclear.
   2) get_entity_evidence for UUID/req-id/hostname.
   3) compare_nodes when multi-node contrast helps.
   4) get_related_entities / get_operation_path for relationship/path questions.
   5) For configuration questions, use search_sos_configs. If logs suggest a
      configuration cause, you MUST call search_sos_configs before claiming a
      setting is missing, incorrect, or overridden. Cite the node, config path,
      and observed setting; if no matching configuration is available, label the
      configuration cause as unconfirmed rather than presenting it as the cause.
   6) Otherwise use search_os_logs / list_indexed_entities /
      create_and_run_analysis as needed.

C) Inventory / count / list / summary questions:
   1) Answer from the entity and relationship snapshot, not incident-RCA tools.
   2) Use create_and_run_analysis when a count, grouping, or filter is needed.
   3) State snapshot limitations clearly (observed/associated is not live state).

Tool results are already digests. Do not paste them back in full.
End with a concise, evidence-backed answer.
"""

# Groq Llama / gpt-oss models sometimes emit tool calls as
# <function=name({...})></function> instead of structured tool_calls; the API
# then 400s with tool_use_failed. We salvage name+args and continue.
_GROQ_FUNCTION_TAG_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][\w]*)\s*"
    r"(?:"
    r"\(\s*(\{.*\})\s*\)\s*(?:></function>|>)?"  # name({...})
    r"|"
    r">\s*(\{.*?\})\s*</function>"  # name>{"k":...}</function>
    r"|"
    r">\s*</function>"  # empty body
    r"|"
    r"\s+(\{.*?\})\s*(?:></function>|</function>)?"  # name {...}
    r"|"
    r"(\{.*?\})\s*</function>"  # malformed name{"k": ...}</function>
    r")",
    re.DOTALL,
)
_GROQ_INLINE_TOOL_RE = re.compile(
    r"attempted to call tool\s+'([A-Za-z_][\w]*)\((\{.*\})\)'",
    re.DOTALL,
)
_MAX_GROQ_TOOL_RECOVERIES = 3


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
    relative: Optional[str] = Field(default=None, max_length=64)

    @field_validator("start", "end", "relative", mode="before")
    @classmethod
    def _clip_time_fields(cls, value: Any) -> Any:
        return _clip_str(value, 64)


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

    @field_validator("search_queries", mode="before")
    @classmethod
    def _normalize_search_queries(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []

        normalized = []
        for item in value[:5]:
            if isinstance(item, str):
                query = _clip_str(item, 160)
                if query:
                    normalized.append(
                        {
                            "service": "system",
                            "objective": "Search incident evidence",
                            "query": query,
                            "priority": 1,
                        }
                    )
            elif isinstance(item, dict):
                normalized.append(item)
        return normalized

    @field_validator("time_window", mode="before")
    @classmethod
    def _normalize_time_window(cls, value: Any) -> Any:
        """Preserve legacy LLM values such as ``"last 24 hours"``."""
        if isinstance(value, str):
            relative = _clip_str(value, 64)
            return {"relative": relative} if relative else None
        return value

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
                    "fenced",

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
    if any(w in lower for w in ("neutron", "ovn", "port binding", "chassis", "ovs")):
        service = "neutron"
    elif any(w in lower for w in ("cinder", "volume")):
        service = "cinder"
    elif any(w in lower for w in ("glance", "image")):
        service = "glance"
    elif any(w in lower for w in ("nova", "instance", "spawn", "spin")) and "reboot" not in lower:
        service = "nova"
    elif any(w in lower for w in ("reboot", "kernel", "panic", "hardware", "power")):
        service = "system"

    targets = ["system"]
    if service != "system":
        targets.append(service)
    if "reboot" in lower or "panic" in lower:
        targets.extend(["kernel", "journal", "nova-compute", "pacemaker", "audit", "message"])
    if hostname:
        targets.append(hostname)

    summary = truncate_text(text.replace("\n", " "), 500) or "Investigate OpenStack incident"
    return ExpandedQuery(
        summary=summary,
        intent="investigate_incident",
        entities=InvestigationEntities(
            resource_id=uuids[0] if uuids else None,
            resource_type="host" if hostname and service == "system" else None,
            service=service,
            problem=truncate_text(text.replace("\n", " "), 500),
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
                data.setdefault("summary", truncate_text(raw_query, 500) or "Investigate")
                data.setdefault("intent", "investigate_incident")
                data.setdefault("entities", {})
                return sanitize_expanded_query(data, raw_query)
        except Exception:
            continue
    return fallback_expanded_query(raw_query)


def _expanded_query_parse_diagnostic(completion: Any) -> str:
    """Return a concise reason a structured expansion completion was rejected."""
    text = str(completion or "").strip()
    start = text.find("{")
    if start < 0:
        return "No JSON object was present in the provider completion."
    text = text[start:]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
    try:
        ExpandedQuery.model_validate(payload)
    except Exception as exc:
        errors = getattr(exc, "errors", lambda: [])()
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            return f"Schema validation failed at {location}: {first.get('msg', str(exc))}"
        return f"Schema validation failed: {type(exc).__name__}: {exc}"
    return "JSON and ExpandedQuery validation succeeded; inspect provider parser settings/version."


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


def _structured_expand_methods(model_provider: str | None = None) -> list[str]:
    """
    Prefer native json_schema when available. Groq supports it only for a
    subset of models, so use json_mode there and rely on the JSON fallback for
    malformed completions.
    """
    from .env_config import get_llm_settings

    try:
        provider = (model_provider or "").strip() or get_llm_settings().provider
    except Exception:
        provider = (model_provider or "").strip()
    if provider == "groq":
        return ["json_mode"]
    return ["json_schema", "json_mode"]


def _required_api_key(model_provider: str) -> str | None:
    from .env_config import api_key_env_for_provider

    return api_key_env_for_provider(model_provider)


def _init_llm(model: str | None = None, model_provider: str | None = None):
    from .env_config import get_llm_settings, init_chat_model_from_env

    settings = get_llm_settings(model=model, model_provider=model_provider)
    llm = init_chat_model_from_env(model=model, model_provider=model_provider)
    # OpenAI-compatible APIs accept this; Google GenAI rejects it as an
    # unknown GenerateContentConfig field.
    if settings.provider in {"openai", "groq"}:
        return llm.bind(parallel_tool_calls=False)
    return llm


def _extract_json_object(text: str) -> dict[str, Any] | None:
    blob = (text or "").strip()
    if not blob:
        return None
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(blob[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_groq_failed_generation(text: str) -> tuple[str, dict[str, Any]] | None:
    """
    Parse Groq tool_use_failed ``failed_generation`` payloads.

    Handles mangled forms such as::

        <function=search_os_logs({"hostname": "comp008"})></function>
        <function=search_os_logs>{"hostname": "comp008"}</function>
    """
    blob = (text or "").strip()
    if not blob:
        return None

    match = _GROQ_FUNCTION_TAG_RE.search(blob)
    if match:
        name = match.group(1)
        raw_args = (
            match.group(2)
            or match.group(3)
            or match.group(4)
            or match.group(5)
            or ""
        )
        args = _extract_json_object(raw_args) or {}
        return name, args

    # Some failures only include the mangled name in the error message.
    match = _GROQ_INLINE_TOOL_RE.search(blob)
    if match:
        args = _extract_json_object(match.group(2)) or {}
        return match.group(1), args

    # JSON object shape: {"name": "...", "arguments": {...}}
    data = _extract_json_object(blob)
    if data and isinstance(data.get("name"), str):
        raw = data.get("arguments", data.get("parameters", {}))
        if isinstance(raw, str):
            raw = _extract_json_object(raw) or {}
        if isinstance(raw, dict):
            return data["name"], raw
    return None


def extract_groq_failed_tool_call(exc: BaseException) -> tuple[str, dict[str, Any]] | None:
    """Extract (tool_name, args) from a Groq BadRequestError / wrapper."""
    candidates: list[str] = []

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict):
            failed = err.get("failed_generation")
            if isinstance(failed, str) and failed.strip():
                candidates.append(failed)
            message = err.get("message")
            if isinstance(message, str) and message.strip():
                candidates.append(message)

    # LangChain often wraps the provider error; walk the cause chain.
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current)
        if text.strip():
            candidates.append(text)
        current = current.__cause__ or current.__context__

    for candidate in candidates:
        parsed = parse_groq_failed_generation(candidate)
        if parsed:
            return parsed
        # Error message may embed the whole mangled call as the tool name.
        inline = _GROQ_INLINE_TOOL_RE.search(candidate)
        if inline:
            args = _extract_json_object(inline.group(2)) or {}
            return inline.group(1), args
    return None


def is_groq_tool_use_failed(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "tool_use_failed" in text or "tool call validation failed" in text:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict) and err.get("code") == "tool_use_failed":
            return True
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException) and cause is not exc:
        return is_groq_tool_use_failed(cause)
    return False


def invoke_tool_by_name(tools: list[Any], name: str, args: dict[str, Any] | None = None) -> str:
    """Invoke a LangChain tool by name; returns a string digest or error text."""
    log = get_logger("Invoking Tool")
    args = args or {}
    tool_map = {getattr(tool, "name", ""): tool for tool in tools}
    tool = tool_map.get(name)
    log.info("[Invoking Tool]: Invoking tool %r with args %s", name, args)
    if tool is None:
        available = ", ".join(sorted(k for k in tool_map if k)) or "(none)"
        return f"Unknown tool {name!r}. Available: {available}"
    try:
        result = tool.invoke(args)
    except Exception as tool_exc:  # noqa: BLE001 - surface to the agent loop
        return f"Tool {name} failed: {type(tool_exc).__name__}: {tool_exc}"
    return str(result)


def _evidence_from_agent_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Collect tool digests from an agent message list."""
    from langchain_core.messages import AIMessage, ToolMessage

    call_args: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                call_args[tool_call["id"]] = tool_call.get("args", {}) or {}

    gathered: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            args = call_args.get(message.tool_call_id, {})
            tool_name = message.name or "tool"
            gathered.append(
                {
                    "service": args.get("service") or args.get("entity_type") or "index",
                    "source": tool_name,
                    "summary": str(args),
                    "raw_logs": truncate_text(str(message.content), 1200),
                }
            )
    return gathered


def _reboot_timeline_evidence_excerpt(output: str, *, max_lines: int = 8) -> str:
    """Keep the causal reboot chain when a timeline is passed to synthesis.

    Tool output begins with boot metadata, so a simple character truncation can
    drop the peer monitor failure and the later fence completion.
    """
    text = str(output or "")
    host_match = re.search(r"^## Last reboot / boot timeline for ([^\n(]+)", text, re.MULTILINE)
    target = host_match.group(1).strip().lower() if host_match else ""
    selected: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if (
            "last reboot / current boot start" in lowered
            or "previous boot ended" in lowered
            or (
                (not target or target in lowered)
                and any(
                    phrase in lowered
                    for phrase in (
                        "unexpectedly dropped during monitor",
                        "lost connection to remote executor",
                        "state is now lost",
                        "operation 'reboot' targeting",
                        "was terminated (reboot)",
                        "was unfenced",
                    )
                )
            )
        ):
            selected.append(line)
            if len(selected) >= max_lines:
                break
    return "\n".join(selected) or truncate_text(text, 2400)


def synthesize_rca_text(
    *,
    plan: "ExpandedQuery",
    evidence_text: str,
    investigator_notes: str,
    raw_query: str,
    prefetch: str = "",
    hypotheses: list[str] | None = None,
    llm,
    run_trace: "AgentRunTrace",
) -> str:
    """Shared RCA synthesis step. Used by both the linear investigator graph
    and the planner/analysis-designer graph so the prompt only lives once."""
    from langchain_core.runnables import RunnableConfig

    rca_prompt = f"""
Original user question: {raw_query or plan.summary}
Interpreted request: {plan.summary}
Hypotheses considered: {hypotheses if hypotheses is not None else getattr(plan, "hypotheses", [])}
Prefetched cluster/evidence digest:
{prefetch or 'None'}
Investigator's working notes: {investigator_notes}
Evidence gathered:
{evidence_text}

Answer the user's actual request directly. Do not force an RCA format for an
inventory, count, list, relationship, or summary question: state the requested
result first and use the evidence only to qualify it. Cite hostnames and entity
IDs when possible.
If this is a reboot/crash question: state the last boot/reboot time first (or say
it could not be determined). Prefer causes supported by direct quotes from tools.
If the host's own logs lack a crash signature, weigh peer/controller evidence from
the same window that names this host. Do not invent BMC/manual/"external reset"
explanations without positive evidence — say unknown and what to check next.
Separate facts from hypotheses. If evidence is weak, say so.
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
    return str(content)

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
        last_exc: Exception | None = None
        salvage_completions: list[Any] = []
        for method in _structured_expand_methods(model_provider):
            try:
                structured_llm = llm.with_structured_output(
                    ExpandedQuery, method=method
                )
                enrichment_chain = (
                    {"query": RunnablePassthrough()} | prompt_expand | structured_llm
                )
                result = enrichment_chain.invoke(
                    raw_query,
                    config=RunnableConfig(callbacks=[expand_cb.handler]),
                )
                result = sanitize_expanded_query(result, raw_query)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                completion = getattr(exc, "llm_output", None) or getattr(
                    exc, "completion", None
                ) or getattr(exc, "text", None)
                if completion:
                    salvage_completions.append(completion)
                diagnostic_completion = completion or str(exc)
                log.warning(
                    "Structured expand via %s parse diagnosis: %s",
                    method,
                    _expanded_query_parse_diagnostic(diagnostic_completion),
                )
                # Keep the full (redacted) provider payload available only when
                # debug logging is enabled; the normal warning stays compact.
                log.debug(
                    "Structured expand via %s raw completion: %s",
                    method,
                    truncate_text(redact_sensitive_text(str(diagnostic_completion)), 8000),
                )
                log.warning(
                    "Structured expand via %s failed (%s)",
                    method,
                    truncate_text(f"{type(exc).__name__}: {exc}", 500),
                )
                continue

        if result is None and last_exc is not None:
            parse_error = f"{type(last_exc).__name__}: {last_exc}"
            log.warning(
                "Structured expand failed (%s); attempting salvage/fallback",
                truncate_text(parse_error, 500),
            )
            # The first JSON-mode call can contain a useful partial plan even if a
            # later json_schema retry is rejected by the provider. Salvage in order.
            for completion in [*salvage_completions, str(last_exc)]:
                if isinstance(completion, dict):
                    completion = completion.get("text") or completion.get("content")
                if isinstance(completion, str) and "from completion" in completion:
                    idx = completion.find("{")
                    if idx >= 0:
                        completion = completion[idx:]
                try:
                    candidate = _parse_partial_expanded_json(str(completion or ""), raw_query)
                except Exception:
                    continue
                if candidate:
                    result = candidate
                    break
            if result is None:
                result = fallback_expanded_query(raw_query)
            run_trace.add(
                "plan_fallback",
                "Used salvage/fallback ExpandedQuery after parse failure",
                details={"error": truncate_text(parse_error, 500)},
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
            details={"chars": len(digest), "preview": truncate_text(digest, 500)},
        )
        run_trace.handoff(
            "QUERY_EXPAND",
            "INVESTIGATOR",
            {
                "expanded_plan": result,
                "prefetch_digest": digest,
            },
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
        hostname = getattr(getattr(plan, "entities", None), "hostname", None)
        rebootish = is_rebootish_query(
            " ".join(
                [
                    str(getattr(plan, "summary", "") or ""),
                    str(getattr(getattr(plan, "entities", None), "problem", "") or ""),
                    " ".join(getattr(plan, "keywords", []) or []),
                    str(state.get("raw_query") or ""),
                ]
            )
        )
        if rebootish:
            host_hint = hostname or "the compute hostname from the question"
            mission = (
                "REBOOT MISSION (reason step-by-step; do not invent causes):\n"
                f"1) Call get_host_reboot_timeline for {host_hint} — read WHEN and the "
                "Peer/cluster mentions section (other hosts naming this hostname in the "
                "boot gap).\n"
                "2) Only if needed, check local crash signatures on that host.\n"
                "3) If peer mentions are thin, call search_peer_mentions with the boot-gap "
                "times. Never search peer reactions with hostname set to the affected host "
                "(that only returns its own SOS).\n"
                "4) Final answer: boot time first, then causes supported by quotes from "
                "peer or local evidence. If still unknown, say what is missing — do not "
                "default to BMC/manual external reset without positive evidence.\n"
            )
        else:
            mission = (
                "Investigate this RHOSP incident. After each tool result, pick the next "
                "check that most reduces uncertainty in the user question.\n"
            )
        base_user_message = (
            f"{mission}\n"
            f"Incident: {plan.summary}\n"
            f"Known entities: {plan.entities.model_dump()}\n"
            f"Keywords: {plan.keywords}\n"
            f"Time window: {plan.time_window}\n\n"
            f"Prefetched digest (already retrieved — build on it; for reboot questions "
            f"the boot timeline is at the top):\n{prefetch}\n"
        )
        agent_cb = AgentObservabilityCallback(run_trace, stage="investigator")
        config = RunnableConfig(
            callbacks=[agent_cb.handler],
            recursion_limit=8,)

        gathered_evidence: list[dict[str, Any]] = []
        recovered_digests: list[str] = []
        investigator_raw = ""
        user_message = base_user_message

        # A reboot RCA has one non-negotiable first evidence-collection step.
        # Prompting alone is not enforcement: a model can answer from prefetch
        # without making a native tool call, producing an incomplete RCA.
        if rebootish and hostname:
            required_name = "get_host_reboot_timeline"
            required_args = {"hostname": hostname}
            run_trace.tool_start(required_name, required_args)
            required_output = invoke_tool_by_name(tools, required_name, required_args)
            run_trace.tool_end(required_name, required_output)
            gathered_evidence.append(
                {
                    "service": "system",
                    "source": required_name,
                    "summary": str(required_args),
                    "raw_logs": _reboot_timeline_evidence_excerpt(required_output),
                }
            )
            user_message += (
                "\nRequired evidence already collected:\n"
                f"[{required_name}] {truncate_text(required_output, 2400)}\n"
                "Do not repeat that exact call. Determine whether a local crash-signature "
                "or wider peer search is needed to explain the trigger.\n"
            )

        run_trace.handoff(
            "INVESTIGATOR_ORCHESTRATOR",
            "INVESTIGATOR_AGENT",
            {
                "mission": mission,
                "agent_input": user_message,
            },
        )

        for attempt in range(_MAX_GROQ_TOOL_RECOVERIES + 1):
            last_messages: list[Any] = []
            try:
                # Stream so prior successful tool turns survive a late Groq 400.
                for chunk in investigator_agent.stream(
                    {"messages": [("user", user_message)]},
                    config=config,
                    stream_mode="values",
                ):
                    if isinstance(chunk, dict) and chunk.get("messages") is not None:
                        last_messages = list(chunk.get("messages") or [])
                gathered_from_agent = _evidence_from_agent_messages(last_messages)
                # Keep manually recovered tool digests from earlier Groq failures.
                if recovered_digests and gathered_evidence:
                    existing = {
                        (item.get("source"), item.get("summary"))
                        for item in gathered_from_agent
                    }
                    merged = [
                        item
                        for item in gathered_evidence
                        if (item.get("source"), item.get("summary")) not in existing
                    ]
                    gathered_evidence = merged + gathered_from_agent
                else:
                    # Preserve the deterministic reboot baseline even when the
                    # agent itself makes no calls.
                    if gathered_evidence:
                        existing = {
                            (item.get("source"), item.get("summary"))
                            for item in gathered_from_agent
                        }
                        gathered_evidence = [
                            item
                            for item in gathered_evidence
                            if (item.get("source"), item.get("summary")) not in existing
                        ] + gathered_from_agent
                    else:
                        gathered_evidence = gathered_from_agent
                for item in gathered_evidence:
                    tool_name = str(item.get("source") or "tool")
                    if not any(
                        event.kind == "tool_end" and tool_name in event.message
                        for event in run_trace.events
                    ):
                        run_trace.tool_start(tool_name, {})
                        run_trace.tool_end(tool_name, str(item.get("raw_logs") or ""))
                if last_messages:
                    investigator_raw = getattr(
                        last_messages[-1], "content", last_messages[-1]
                    )
                break
            except Exception as exc:
                # Preserve any evidence collected before the failing LLM turn.
                if last_messages:
                    prior = _evidence_from_agent_messages(last_messages)
                    if prior:
                        # Keep recovered digests from earlier attempts, then add prior.
                        existing_sources = {
                            (item.get("source"), item.get("summary"))
                            for item in gathered_evidence
                        }
                        for item in prior:
                            key = (item.get("source"), item.get("summary"))
                            if key not in existing_sources:
                                gathered_evidence.append(item)

                parsed = (
                    extract_groq_failed_tool_call(exc)
                    if is_groq_tool_use_failed(exc)
                    else None
                )
                if parsed is None or attempt >= _MAX_GROQ_TOOL_RECOVERIES:
                    if gathered_evidence or recovered_digests:
                        run_trace.add(
                            "tool_recovery_exhausted",
                            "Continuing with recovered evidence after Groq tool_use_failed",
                            details={"error": truncate_text(str(exc), 500)},
                        )
                        investigator_raw = (
                            f"Investigator stopped after Groq tool-call error: "
                            f"{truncate_text(str(exc), 300)}"
                        )
                        break
                    raise

                tool_name, tool_args = parsed
                run_trace.add(
                    "tool_recovery",
                    f"Recovered mangled Groq tool call for {tool_name}",
                    details={"args": tool_args, "attempt": attempt + 1},
                )
                run_trace.tool_start(tool_name, tool_args)
                output = invoke_tool_by_name(tools, tool_name, tool_args)
                run_trace.tool_end(tool_name, output)
                gathered_evidence.append(
                    {
                        "service": tool_args.get("service")
                        or tool_args.get("entity_type")
                        or "index",
                        "source": tool_name,
                        "summary": str(tool_args),
                        "raw_logs": truncate_text(output, 1200),
                    }
                )
                recovered_digests.append(
                    f"[{tool_name}] args={tool_args}\n{truncate_text(output, 900)}"
                )
                user_message = (
                    base_user_message
                    + "\n\nAlready-executed tool results (do NOT repeat these exact calls):\n"
                    + "\n\n".join(recovered_digests)
                    + "\n\nContinue with other tools if needed, then conclude."
                )
                run_trace.handoff(
                    "INVESTIGATOR_ORCHESTRATOR",
                    "INVESTIGATOR_AGENT_RETRY",
                    {
                        "agent_input": user_message,
                    },
                )

        run_trace.add(
            "investigator_summary",
            "Investigator finished tool loop",
            details={
                "tool_results": len(gathered_evidence),
                "recoveries": len(recovered_digests),
            },
        )
        run_trace.node_end("INVESTIGATOR", tools=len(gathered_evidence))
        return {
            "gathered_evidence": gathered_evidence,
            "findings": {"investigator_raw": investigator_raw},
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
        run_trace.handoff(
            "INVESTIGATOR",
            "SYNTHESIZE_RCA",
            {
                "expanded_plan": plan,
                "prefetch_digest": prefetch,
                "investigator_notes": investigator_notes,
                "evidence_text": evidence_text,
            },
        )
        content = synthesize_rca_text(
            plan=plan,
            evidence_text=evidence_text,
            investigator_notes=investigator_notes,
            raw_query=state.get("raw_query", plan.summary),
            prefetch=prefetch,
            llm=llm,
            run_trace=run_trace,
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

    enriched = redact_sensitive_text((prompt or "").strip())
    extras: list[str] = []
    if focus_entity and focus_entity.strip():
        extras.append(f"Focused entity seed: {focus_entity.strip()}")
    rebootish = is_rebootish_query(enriched)
    if include_graph and not rebootish:
        extras.append(
            "Relationship graph is available for questions that ask for relationships "
            "or paths; use it only when it directly answers the question."
        )
    elif rebootish:
        extras.append(
            "Reboot question: establish last boot time, seek local cause, then if "
            "unexplained widen to other cluster hosts in the same window for mentions "
            "of this hostname. Do not invent external-reset causes without evidence."
        )
    style = (answer_style or "Concise RCA").strip()
    if style == "Evidence-heavy":
        extras.append("Answer style: evidence-heavy — cite more digests and hostnames.")
    elif style == "Operation path first" and not rebootish:
        extras.append(
            "Answer style: start with the operation path (VM→port→chassis→host), then RCA."
        )
    elif rebootish:
        extras.append(
            "Answer style: start with last boot/reboot time, then root cause around that window."
        )
    else:
        extras.append("Answer style: concise RCA.")
    # Presentation hints must not alter the text used for LLM planning.
    # They remain UI metadata; mixing them into `enriched` made fallback plans
    # investigate the graph/RCA hint rather than the user's question.
    if False and extras:
        enriched = enriched + "\n\n" + "\n".join(extras)

    run_trace = trace or AgentRunTrace(prompt=enriched)
    run_trace.add("run_start", "Starting LangGraph investigation", details={"db_path": str(db_path)})
    log.info("Investigation start run=%s prompt=%s", run_trace.run_id[:8], truncate_text(enriched, 500))

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
    """Render the answer; observability never exposes internal prompt/evidence dumps."""
    answer = str(final_state.get("final_rca") or "No answer generated.")
    if not include_observability:
        return answer

    restored = AgentRunTrace.from_dict(final_state.get("agent_trace"))
    if restored and restored.events:
        return answer + "\n\n" + restored.render_markdown()
    return answer
