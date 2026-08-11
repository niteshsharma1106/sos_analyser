# planner_graph.py — Planner → (existing tool | Analysis Designer → Static
# Validation → Read-only Verification → Execute) → Evidence Graph → Replanner
#
# This is designed to slot alongside langgraph_investigator.py. It reuses that
# module's conventions (pydantic structured outputs, AgentRunTrace, the
# read-only DatabaseConnector, invoke_tool_by_name) rather than duplicating
# them. Integration points that depend on YOUR concrete schema/tools are
# marked with `# INTEGRATE:`.

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field

from .context_pack import truncate_text
from .observability import AgentObservabilityCallback, AgentRunTrace, get_logger

# Reuse helpers already defined in the existing workflow module instead of
# re-implementing tool dispatch / Groq recovery / RCA synthesis.
from .langgraph_investigator import (
    ExpandedQuery,
    _init_llm,
    fallback_expanded_query,
    invoke_tool_by_name,
    synthesize_rca_text,  # INTEGRATE: extract the body of `synthesize_rca` in
    # langgraph_investigator.py into a standalone `synthesize_rca_text(plan,
    # evidence_text, investigator_notes, raw_query, llm, run_trace) -> str`
    # function so both graphs can call it. Trivial refactor of existing code.
)
from .investigation_tools import build_langchain_tools
from .dbconnector import DatabaseConnector

log = get_logger("planner_graph")

MAX_PLANNER_ITERATIONS = 6
MAX_ANALYSIS_RETRIES = 3
ANALYSIS_ROW_LIMIT = 500

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

PLANNER_SYSTEM_PROMPT = """You are the planning node of an OpenStack RCA investigator.
Given the user's question, the investigation plan, and the evidence graph
gathered so far, decide the SINGLE next best action.

Return ONE compact JSON object only.

action: one of
  - "use_tool"        -> an existing tool can directly answer the next
                          sub-question. Set tool_name and tool_args.
  - "design_analysis" -> no existing tool covers this; a custom read-only
                          DuckDB query over the ingested SOS snapshot is
                          needed. Set analysis_objective (plain English,
                          what the query must determine).
  - "answer_ready"     -> the evidence graph already answers the user's
                          question with sufficient confidence.

Rules:
- Prefer "use_tool" whenever an existing tool plausibly answers the next
  sub-question — custom SQL is a fallback, not a default.
- Never repeat a tool call with identical args already present in evidence.
- reasoning must name what specific uncertainty this action resolves.
"""

ANALYSIS_DESIGNER_SYSTEM_PROMPT = """You design ONE read-only DuckDB SQL query
against an ingested Red Hat OpenStack 17.x sosreport snapshot.

Hard constraints:
- Exactly one statement. SELECT or WITH ... SELECT only.
- No DDL/DML, no PRAGMA/ATTACH/COPY/INSTALL/LOAD/SET/CALL/EXPORT/VACUUM.
- Only reference tables from the provided schema. Never invent tables/columns.
- Always include an explicit LIMIT (<= {row_limit}).
- Prefer WHERE filters over scanning full tables when a hostname, service,
  or time window is known.

Return ONE compact JSON object only: name, purpose, sql, expected_columns.
If given previous_errors, fix the exact problem — do not restate the same
invalid query.
"""

REPLANNER_SYSTEM_PROMPT = """You review the evidence graph gathered so far
against the user's original question and decide whether to continue
investigating or finalize the answer.

Return ONE compact JSON object only.

action: "continue" | "answer_ready"
- "continue": there remains a specific, resolvable uncertainty. Set
  next_focus to the single most important open question.
- "answer_ready": evidence graph supports a direct, evidence-backed answer
  (including "unknown, and here is what was checked" when that is honest).

Do not choose "continue" just to gather more evidence for its own sake.
"""


class PlannerDecision(BaseModel):
    action: Literal["use_tool", "design_analysis", "answer_ready"]
    tool_name: Optional[str] = Field(default=None, max_length=64)
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    analysis_objective: Optional[str] = Field(default=None, max_length=300)
    reasoning: str = Field(default="", max_length=300)


class ProposedAnalysis(BaseModel):
    name: str = Field(max_length=80)
    purpose: str = Field(max_length=200)
    sql: str = Field(max_length=4000)
    expected_columns: List[str] = Field(default_factory=list, max_length=20)


class ReplanDecision(BaseModel):
    action: Literal["continue", "answer_ready"]
    next_focus: Optional[str] = Field(default=None, max_length=200)
    reasoning: str = Field(default="", max_length=300)


class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    source: str  # tool name or "duckdb_analysis"
    objective: str
    entities: List[str] = Field(default_factory=list)  # hostnames/UUIDs/services mentioned
    digest: str  # truncated human-readable result


class EvidenceGraph(BaseModel):
    """Lightweight accumulator, not a full graph DB: nodes = entities seen,
    edges = (entity, evidence_id) 'mentioned_in' relations."""

    items: List[EvidenceItem] = Field(default_factory=list)
    entity_index: Dict[str, List[str]] = Field(default_factory=dict)  # entity -> [evidence ids]

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)
        for entity in item.entities:
            self.entity_index.setdefault(entity, []).append(item.id)

    def render(self, max_items: int = 15) -> str:
        if not self.items:
            return "No evidence gathered yet."
        lines = []
        for item in self.items[-max_items:]:
            ents = ", ".join(item.entities) or "none"
            lines.append(f"[{item.id}] {item.source} :: {item.objective}\n  entities: {ents}\n  {item.digest}")
        return "\n\n".join(lines)

    def seen_signature(self, source: str, args_or_objective: str) -> bool:
        sig = f"{source}::{args_or_objective}".lower()
        return any(f"{it.source}::{it.objective}".lower() == sig for it in self.items)


class PlannerState(TypedDict, total=False):
    raw_query: str
    expanded_plan: Optional[ExpandedQuery]
    allowed_tables: List[str]          # INTEGRATE: populate from your schema
    schema_description: str            # INTEGRATE: DDL / column docs for the designer prompt
    evidence_graph: EvidenceGraph
    planner_decision: Optional[PlannerDecision]
    pending_analysis: Optional[ProposedAnalysis]
    validation_errors: List[str]
    analysis_retries: int
    iterations: int
    final_rca: Optional[str]
    agent_trace: Optional[Dict[str, Any]]
    run_id: str
    _replan_action: str  # internal routing signal set by replanner_node


# --------------------------------------------------------------------------- #
# Static validation (defense layer 1 — before touching DuckDB at all)
# --------------------------------------------------------------------------- #

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|"
    r"ATTACH|DETACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|PRAGMA|SET|CALL|"
    r"VACUUM|CHECKPOINT|GRANT|REVOKE"
    r")\b",
    re.IGNORECASE,
)
_STATEMENT_START_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r"\bFROM\s+([A-Za-z_][\w.]*)|\bJOIN\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
_CTE_NAME_RE = re.compile(r"\b([A-Za-z_][\w]*)\s+AS\s*\(", re.IGNORECASE)


def static_validate_sql(sql: str, allowed_tables: list[str]) -> list[str]:
    """Pure, no I/O. Returns a list of problems (empty = passes layer 1)."""
    errors: list[str] = []
    text = (sql or "").strip().rstrip(";")

    if not text:
        return ["Empty query."]
    if ";" in text:
        errors.append("Only a single statement is allowed (no semicolons inside the query).")
    if not _STATEMENT_START_RE.match(text):
        errors.append("Query must start with SELECT or WITH.")
    forbidden = _FORBIDDEN_KEYWORDS.findall(text)
    if forbidden:
        errors.append(f"Forbidden keyword(s) found: {sorted(set(w.upper() for w in forbidden))}")
    if not re.search(r"\bLIMIT\s+\d+\b", text, re.IGNORECASE):
        errors.append(f"Query must include an explicit LIMIT (<= {ANALYSIS_ROW_LIMIT}).")
    else:
        limit_match = re.search(r"\bLIMIT\s+(\d+)\b", text, re.IGNORECASE)
        if limit_match and int(limit_match.group(1)) > ANALYSIS_ROW_LIMIT:
            errors.append(f"LIMIT exceeds the cap of {ANALYSIS_ROW_LIMIT}.")

    if allowed_tables:
        # CTE aliases (WITH foo AS (...)) are not real tables — exclude them
        # from the whitelist check or every WITH query gets false-flagged.
        cte_names = {m.group(1).lower() for m in _CTE_NAME_RE.finditer(text)}
        allowed_lower = {t.lower() for t in allowed_tables} | cte_names
        referenced = set()
        for m in _TABLE_REF_RE.finditer(text):
            table = (m.group(1) or m.group(2) or "").split(".")[-1].strip('"`').lower()
            if table:
                referenced.add(table)
        unknown = referenced - allowed_lower
        if unknown:
            errors.append(f"Unknown/disallowed table(s) referenced: {sorted(unknown)}")

    return errors


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #

def build_planner_investigation_app(
    db_con,
    tools: list[Any],
    *,
    llm,
    allowed_tables: list[str],
    schema_description: str,
    trace: Optional[AgentRunTrace] = None,
):
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, START, StateGraph

    run_trace = trace or AgentRunTrace()
    tool_catalog = "\n".join(
        f"- {getattr(t, 'name', '?')}: {truncate_text(getattr(t, 'description', ''), 500)}"
        for t in tools
    )

    def _cb(stage: str) -> RunnableConfig:
        return RunnableConfig(callbacks=[AgentObservabilityCallback(run_trace, stage=stage).handler])

    # --- Planner --------------------------------------------------------- #
    def planner_node(state: PlannerState) -> dict:
        run_trace.node_start("PLANNER")
        iterations = state.get("iterations", 0) + 1
        graph = state.get("evidence_graph") or EvidenceGraph()
        plan = state["expanded_plan"]

        if iterations > MAX_PLANNER_ITERATIONS:
            run_trace.add("planner_cap", "Max planner iterations reached; forcing answer_ready")
            decision = PlannerDecision(action="answer_ready", reasoning="Iteration cap reached.")
        else:
            structured_llm = llm.with_structured_output(PlannerDecision)
            prompt = (
                f"{PLANNER_SYSTEM_PROMPT}\n\n"
                f"User question: {state['raw_query']}\n"
                f"Interpreted intent: {getattr(plan, 'summary', '')}\n"
                f"Known entities: {getattr(plan, 'entities', None) and plan.entities.model_dump()}\n"
                f"Available tools:\n{tool_catalog}\n\n"
                f"Evidence graph so far:\n{graph.render()}\n"
            )
            decision = structured_llm.invoke(prompt, config=_cb("planner"))

        run_trace.add(
            "planner_decision",
            f"action={decision.action}",
            details={"reasoning": decision.reasoning, "tool": decision.tool_name},
        )
        run_trace.node_end("PLANNER")
        return {
            "planner_decision": decision,
            "evidence_graph": graph,
            "iterations": iterations,
            "validation_errors": [],
            "analysis_retries": 0,
        }

    def route_after_planner(state: PlannerState) -> str:
        decision = state["planner_decision"]
        if decision.action == "answer_ready":
            return "replanner"  # let replanner do the final sanity check uniformly
        if decision.action == "use_tool":
            return "execute_tool"
        return "analysis_designer"

    # --- Execute existing tool ------------------------------------------ #
    def execute_tool_node(state: PlannerState) -> dict:
        run_trace.node_start("EXECUTE_TOOL")
        decision = state["planner_decision"]
        graph = state["evidence_graph"]
        run_trace.tool_start(decision.tool_name or "unknown", decision.tool_args)
        output = invoke_tool_by_name(tools, decision.tool_name or "", decision.tool_args)
        run_trace.tool_end(decision.tool_name or "unknown", output)

        graph.add(
            EvidenceItem(
                source=decision.tool_name or "unknown",
                objective=decision.reasoning or str(decision.tool_args),
                entities=[v for v in decision.tool_args.values() if isinstance(v, str)][:5],
                digest=truncate_text(output, 1200),
            )
        )
        run_trace.node_end("EXECUTE_TOOL")
        return {"evidence_graph": graph}

    # --- Analysis Designer ------------------------------------------------ #
    def analysis_designer_node(state: PlannerState) -> dict:
        run_trace.node_start("ANALYSIS_DESIGNER")
        # Incremented once per attempt (including the first), so the retry
        # cap checked downstream is based on real accumulated state rather
        # than a value recomputed-and-discarded inside a routing function.
        attempt = state.get("analysis_retries", 0) + 1
        decision = state["planner_decision"]
        errors = state.get("validation_errors", [])
        structured_llm = llm.with_structured_output(ProposedAnalysis)
        prompt = (
            f"{ANALYSIS_DESIGNER_SYSTEM_PROMPT.format(row_limit=ANALYSIS_ROW_LIMIT)}\n\n"
            f"Schema:\n{schema_description}\n\n"
            f"Objective: {decision.analysis_objective or decision.reasoning}\n"
        )
        if errors:
            prompt += f"\nprevious_errors (fix these exactly): {errors}\n"
            prev = state.get("pending_analysis")
            if prev:
                prompt += f"previous_sql: {prev.sql}\n"

        analysis = structured_llm.invoke(prompt, config=_cb("analysis_designer"))
        run_trace.add(
            "analysis_proposed",
            analysis.name,
            details={"sql": truncate_text(analysis.sql, 500), "attempt": attempt},
        )
        run_trace.node_end("ANALYSIS_DESIGNER")
        return {"pending_analysis": analysis, "analysis_retries": attempt, "validation_errors": []}

    # --- Static validation ------------------------------------------------ #
    def static_validation_node(state: PlannerState) -> dict:
        run_trace.node_start("STATIC_VALIDATION")
        analysis = state["pending_analysis"]
        errors = static_validate_sql(analysis.sql, allowed_tables)
        run_trace.add(
            "static_validation",
            "passed" if not errors else "failed",
            details={"errors": errors},
        )
        run_trace.node_end("STATIC_VALIDATION")
        return {"validation_errors": errors}

    def route_after_static_validation(state: PlannerState) -> str:
        if state.get("validation_errors"):
            if state.get("analysis_retries", 0) >= MAX_ANALYSIS_RETRIES:
                return "give_up_on_analysis"
            return "retry_designer"
        return "readonly_verification"

    def give_up_on_analysis_node(state: PlannerState) -> dict:
        run_trace.node_start("ANALYSIS_ABANDONED")
        graph = state["evidence_graph"]
        analysis = state.get("pending_analysis")
        graph.add(
            EvidenceItem(
                source="duckdb_analysis",
                objective=(analysis.purpose if analysis else "custom analysis"),
                digest=f"Abandoned after {MAX_ANALYSIS_RETRIES} failed attempts: {state.get('validation_errors')}",
            )
        )
        run_trace.node_end("ANALYSIS_ABANDONED")
        return {"evidence_graph": graph}

    # --- Read-only verification (defense layer 2 — belt & suspenders) ---- #
    def readonly_verification_node(state: PlannerState) -> dict:
        run_trace.node_start("READONLY_VERIFICATION")
        analysis = state["pending_analysis"]
        sql = analysis.sql.strip().rstrip(";")
        errors: list[str] = []
        try:
            # EXPLAIN catches syntax/reference errors without materializing rows.
            db_con.execute(f"EXPLAIN {sql}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"EXPLAIN failed: {exc}")
        # Belt-and-suspenders: db_con itself must be opened read_only=True by
        # the caller (see DatabaseConnector(..., read_only=True) in
        # langgraph_investigator.investigate_with_langgraph) so any write
        # verb that slipped past static validation still fails here.
        run_trace.add("readonly_verification", "passed" if not errors else "failed", details={"errors": errors})
        run_trace.node_end("READONLY_VERIFICATION")
        return {"validation_errors": errors}

    def route_after_readonly(state: PlannerState) -> str:
        if state.get("validation_errors"):
            if state.get("analysis_retries", 0) >= MAX_ANALYSIS_RETRIES:
                return "give_up_on_analysis"
            return "retry_designer"
        return "execute_analysis"

    # --- Execute on DuckDB -------------------------------------------------- #
    def execute_analysis_node(state: PlannerState) -> dict:
        run_trace.node_start("EXECUTE_ANALYSIS")
        analysis = state["pending_analysis"]
        graph = state["evidence_graph"]
        sql = analysis.sql.strip().rstrip(";")
        try:
            result = db_con.execute(sql).fetchall()
            columns = [d[0] for d in db_con.description] if db_con.description else []
            preview = "\n".join(str(dict(zip(columns, row))) for row in result[:50])
            digest = truncate_text(preview or "(no rows)", 1500)
            run_trace.add("analysis_executed", analysis.name, details={"rows": len(result)})
        except Exception as exc:  # noqa: BLE001
            digest = f"Execution failed: {exc}"
            run_trace.add("analysis_execution_error", str(exc))

        graph.add(
            EvidenceItem(
                source="duckdb_analysis",
                objective=analysis.purpose,
                digest=digest,
            )
        )
        run_trace.node_end("EXECUTE_ANALYSIS")
        return {"evidence_graph": graph}

    # --- Replanner ---------------------------------------------------------- #
    def replanner_node(state: PlannerState) -> dict:
        run_trace.node_start("REPLANNER")
        graph = state["evidence_graph"]
        decision = state.get("planner_decision")
        if decision and decision.action == "answer_ready":
            replan = ReplanDecision(action="answer_ready", reasoning="Planner signaled done.")
        elif state.get("iterations", 0) > MAX_PLANNER_ITERATIONS:
            replan = ReplanDecision(action="answer_ready", reasoning="Iteration cap reached.")
        else:
            structured_llm = llm.with_structured_output(ReplanDecision)
            prompt = (
                f"{REPLANNER_SYSTEM_PROMPT}\n\n"
                f"User question: {state['raw_query']}\n"
                f"Evidence graph:\n{graph.render()}\n"
            )
            replan = structured_llm.invoke(prompt, config=_cb("replanner"))
        run_trace.add("replan_decision", replan.action, details={"next_focus": replan.next_focus})
        run_trace.node_end("REPLANNER")
        # LangGraph conditional edges route purely from returned state, so the
        # decision has to be written here, not passed as a function argument.
        return {
            "planner_decision": None,
            "_replan_action": "synthesize" if replan.action == "answer_ready" else "planner",
        }

    # --- Synthesize (reuse existing RCA synthesis) -------------------------- #
    def synthesize_node(state: PlannerState) -> dict:
        run_trace.node_start("SYNTHESIZE_RCA")
        plan = state["expanded_plan"]
        graph = state["evidence_graph"]
        content = synthesize_rca_text(
            plan=plan,
            evidence_text=graph.render(max_items=50),
            investigator_notes="",
            raw_query=state["raw_query"],
            llm=llm,
            run_trace=run_trace,
        )
        run_trace.node_end("SYNTHESIZE_RCA")
        return {"final_rca": content, "agent_trace": run_trace.to_dict()}

    workflow = StateGraph(PlannerState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("analysis_designer", analysis_designer_node)
    workflow.add_node("static_validation", static_validation_node)
    workflow.add_node("readonly_verification", readonly_verification_node)
    workflow.add_node("execute_analysis", execute_analysis_node)
    workflow.add_node("give_up_on_analysis", give_up_on_analysis_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {"execute_tool": "execute_tool", "analysis_designer": "analysis_designer", "replanner": "replanner"},
    )
    workflow.add_edge("execute_tool", "replanner")
    workflow.add_edge("analysis_designer", "static_validation")
    workflow.add_conditional_edges(
        "static_validation",
        route_after_static_validation,
        {
            "readonly_verification": "readonly_verification",
            "retry_designer": "analysis_designer",
            "give_up_on_analysis": "give_up_on_analysis",
        },
    )
    workflow.add_conditional_edges(
        "readonly_verification",
        route_after_readonly,
        {
            "execute_analysis": "execute_analysis",
            "retry_designer": "analysis_designer",
            "give_up_on_analysis": "give_up_on_analysis",
        },
    )
    workflow.add_edge("execute_analysis", "replanner")
    workflow.add_edge("give_up_on_analysis", "replanner")
    workflow.add_conditional_edges(
        "replanner",
        lambda s: s.get("_replan_action", "planner"),
        {"planner": "planner", "synthesize": "synthesize"},
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def investigate_with_planner_graph(
    db_path: str,
    prompt: str,
    *,
    model: str | None = None,
    model_provider: str | None = None,
    recursion_limit: int = 30,
    trace: AgentRunTrace | None = None,
) -> dict[str, Any]:
    """Run the planner/analysis-designer investigation workflow.

    This is the public CLI entry point corresponding to
    :func:`build_planner_investigation_app`.  The planner graph deliberately
    uses the same read-only connector and LangChain tools as the linear
    investigator so its custom SQL path cannot modify the SOS database.
    """
    from langchain_core.utils.uuid import uuid7

    from .observability import configure_logging
    from .privacy import redact_sensitive_text

    configure_logging()
    raw_query = redact_sensitive_text((prompt or "").strip())
    if not raw_query:
        raise ValueError("An investigation prompt is required.")

    run_trace = trace or AgentRunTrace(prompt=raw_query)
    run_trace.add(
        "run_start",
        "Starting planner investigation",
        details={"db_path": str(db_path)},
    )

    try:
        with DatabaseConnector(db_path, read_only=True) as db:
            connection = db.connect()
            table_rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
            allowed_tables = [row[0] for row in table_rows]
            column_rows = connection.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'main'
                ORDER BY table_name, ordinal_position
                """
            ).fetchall()
            schema_description = "\n".join(
                f"{table}.{column}: {data_type}"
                for table, column, data_type in column_rows
            )
            app = build_planner_investigation_app(
                connection,
                build_langchain_tools(connection),
                llm=_init_llm(model=model, model_provider=model_provider),
                allowed_tables=allowed_tables,
                schema_description=schema_description,
                trace=run_trace,
            )
            initial_state: PlannerState = {
                "raw_query": raw_query,
                # Use the shared deterministic expander here. The planner has
                # its own structured LLM stages and only needs an initial set
                # of entities/keywords to ground those stages.
                "expanded_plan": fallback_expanded_query(raw_query),
                "allowed_tables": allowed_tables,
                "schema_description": schema_description,
                "evidence_graph": EvidenceGraph(),
                "validation_errors": [],
                "analysis_retries": 0,
                "iterations": 0,
                "run_id": run_trace.run_id,
            }
            final_state = app.invoke(
                initial_state,
                config={"run_id": uuid7(), "recursion_limit": recursion_limit},
            )
    except Exception as exc:
        run_trace.error(f"Planner investigation failed: {exc}")
        raise

    result = dict(final_state)
    result["agent_trace"] = run_trace.to_dict()
    result["run_id"] = run_trace.run_id
    run_trace.add("run_end", "Planner investigation complete")
    return result
