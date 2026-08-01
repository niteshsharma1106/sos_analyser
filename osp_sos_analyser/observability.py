# observability.py — backend logging + agent run tracing.
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .context_pack import truncate_text
from .privacy import redact_sensitive_text

LOGGER_NAME = "osp_sos"
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


@dataclass
class TraceEvent:
    kind: str
    message: str
    timestamp: str
    elapsed_ms: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["details"]:
            payload.pop("details")
        return payload


class AgentRunTrace:
    """In-memory timeline for one investigation run."""

    def __init__(self, *, run_id: str | None = None, prompt: str = "") -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.prompt = prompt
        self.started_at = time.perf_counter()
        self.events: list[TraceEvent] = []
        self.logger = logging.getLogger(f"{LOGGER_NAME}.agent")

    def _elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    def add(
        self,
        kind: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        level: int = logging.INFO,
    ) -> TraceEvent:
        event = TraceEvent(
            kind=kind,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=self._elapsed_ms(),
            details=details or {},
        )
        self.events.append(event)
        detail_text = ""
        if event.details:
            detail_text = " " + json.dumps(event.details, default=str, ensure_ascii=True)
        self.logger.log(
            level,
            "run=%s +%sms %s: %s%s",
            self.run_id[:8],
            event.elapsed_ms,
            kind,
            message,
            truncate_text(detail_text, 500),
        )
        return event

    def node_start(self, name: str, **details: Any) -> None:
        self.add("node_start", f"Entering {name}", details=details)

    def node_end(self, name: str, **details: Any) -> None:
        self.add("node_end", f"Finished {name}", details=details)

    def tool_start(self, name: str, args: dict[str, Any] | None = None) -> None:
        self.add(
            "tool_start",
            f"Tool {name}",
            details={"args": _safe_details(args or {})},
        )

    def tool_end(self, name: str, output: str = "") -> None:
        self.add(
            "tool_end",
            f"Tool {name} returned",
            details={"output_preview": truncate_text(redact_sensitive_text(output), 240)},
        )

    def llm_start(self, stage: str, **details: Any) -> None:
        self.add("llm_start", f"LLM call ({stage})", details=_safe_details(details))

    def llm_end(self, stage: str, **details: Any) -> None:
        self.add("llm_end", f"LLM done ({stage})", details=_safe_details(details))

    def error(self, message: str, **details: Any) -> None:
        self.add("error", message, details=_safe_details(details), level=logging.ERROR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "prompt": truncate_text(self.prompt, 300),
            "duration_ms": self._elapsed_ms(),
            "event_count": len(self.events),
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> AgentRunTrace | None:
        if not isinstance(payload, dict):
            return None
        trace = cls(run_id=str(payload.get("run_id") or ""), prompt=str(payload.get("prompt") or ""))
        # Preserve reported duration by shifting started_at backwards.
        duration_ms = int(payload.get("duration_ms") or 0)
        if duration_ms:
            trace.started_at = time.perf_counter() - (duration_ms / 1000.0)
        for raw in payload.get("events") or []:
            if not isinstance(raw, dict):
                continue
            trace.events.append(
                TraceEvent(
                    kind=str(raw.get("kind") or ""),
                    message=str(raw.get("message") or ""),
                    timestamp=str(raw.get("timestamp") or ""),
                    elapsed_ms=int(raw.get("elapsed_ms") or 0),
                    details=dict(raw.get("details") or {}),
                )
            )
        return trace

    def render_markdown(self, *, max_events: int = 80) -> str:
        lines = [
            "## Agent observability",
            f"- run_id: `{self.run_id}`",
            f"- duration_ms: {self._elapsed_ms()}",
            f"- events: {len(self.events)}",
            "",
            "| +ms | kind | message |",
            "| --- | --- | --- |",
        ]
        for event in self.events[:max_events]:
            message = event.message.replace("|", "\\|")
            lines.append(f"| {event.elapsed_ms} | `{event.kind}` | {message} |")
        if len(self.events) > max_events:
            lines.append(f"| … | … | truncated {len(self.events) - max_events} more |")
        tool_calls = [e for e in self.events if e.kind == "tool_start"]
        if tool_calls:
            lines.extend(["", "### Tools used"])
            for event in tool_calls:
                args = event.details.get("args") or {}
                lines.append(f"- `{event.message}` args=`{truncate_text(str(args), 160)}`")
        return "\n".join(lines)


def _safe_details(details: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, str):
            cleaned[key] = truncate_text(redact_sensitive_text(value), 400)
        elif isinstance(value, (int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = truncate_text(str(value), 400)
    return cleaned


class AgentObservabilityCallback:
    """LangChain callback that records LLM/tool activity into an AgentRunTrace."""

    def __init__(self, trace: AgentRunTrace, *, stage: str = "investigator") -> None:
        self.trace = trace
        self.stage = stage
        # Lazy subclass so importing observability does not require langchain.
        from langchain_core.callbacks import BaseCallbackHandler

        outer = self

        class _Handler(BaseCallbackHandler):
            def on_chat_model_start(self, serialized, messages, **kwargs):  # noqa: ANN001
                model = ""
                if isinstance(serialized, dict):
                    model = str(serialized.get("id") or serialized.get("name") or "")
                outer.trace.llm_start(outer.stage, model=model)

            def on_llm_start(self, serialized, prompts, **kwargs):  # noqa: ANN001
                model = ""
                if isinstance(serialized, dict):
                    model = str(serialized.get("id") or serialized.get("name") or "")
                outer.trace.llm_start(outer.stage, model=model)

            def on_llm_end(self, response, **kwargs):  # noqa: ANN001
                outer.trace.llm_end(outer.stage)

            def on_llm_error(self, error, **kwargs):  # noqa: ANN001
                outer.trace.error(f"LLM error in {outer.stage}: {error}")

            def on_tool_start(self, serialized, input_str, **kwargs):  # noqa: ANN001
                name = ""
                if isinstance(serialized, dict):
                    name = str(serialized.get("name") or "")
                name = name or str(kwargs.get("name") or "tool")
                run_id = kwargs.get("run_id")
                if run_id is not None:
                    outer._tool_runs[str(run_id)] = name
                args: dict[str, Any]
                if isinstance(input_str, dict):
                    args = input_str
                else:
                    args = {"input": str(input_str)}
                outer.trace.tool_start(name, args)

            def on_tool_end(self, output, **kwargs):  # noqa: ANN001
                run_id = kwargs.get("run_id")
                name = ""
                if run_id is not None:
                    name = outer._tool_runs.pop(str(run_id), "")
                name = name or str(kwargs.get("name") or "tool")
                outer.trace.tool_end(name, str(output))

            def on_tool_error(self, error, **kwargs):  # noqa: ANN001
                run_id = kwargs.get("run_id")
                name = ""
                if run_id is not None:
                    name = outer._tool_runs.pop(str(run_id), "")
                name = name or str(kwargs.get("name") or "tool")
                outer.trace.error(f"Tool {name} failed: {error}")

        self._tool_runs: dict[str, str] = {}
        self.handler = _Handler()


def configure_logging(
    *,
    level: str | None = None,
    log_file: str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure package logging once for CLI/chat/ingest."""
    logger = logging.getLogger(LOGGER_NAME)
    resolved_level = (level or os.getenv("OSP_SOS_LOG_LEVEL") or "INFO").upper()
    resolved_file = log_file if log_file is not None else os.getenv("OSP_SOS_LOG_FILE")

    if logger.handlers and not force:
        logger.setLevel(resolved_level)
        return logger

    close_logging(logger)
    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if resolved_file:
        path = os.path.abspath(resolved_file)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to file %s", path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logger


def close_logging(logger: logging.Logger | None = None) -> None:
    """Flush, detach, and close package handlers (important on Windows)."""
    target = logger or logging.getLogger(LOGGER_NAME)
    for handler in target.handlers[:]:
        target.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
