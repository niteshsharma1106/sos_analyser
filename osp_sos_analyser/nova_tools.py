from __future__ import annotations

from .analysis import AnalysisStore
from .investigation_tools import indexed_evidence_for_hints
from .models import AgentFinding, EvidenceHint, LogRecord


class NovaAgent:
    name = "Nova Agent"
    services = ("nova",)

    def investigate(self, store: AnalysisStore, hints: EvidenceHint) -> AgentFinding:
        evidence = indexed_evidence_for_hints(
            store._connect(), hints, services=("nova",), limit=10
        )
        if not evidence:
            terms = hints.identifiers or hints.hostnames or hints.keywords
            evidence = list(store.search_logs(service="nova", text_terms=terms[:2], limit=10))
        if not evidence:
            evidence = list(
                store.search_logs(
                    service="nova",
                    levels=("ERROR", "CRITICAL", "WARNING"),
                    limit=10,
                )
            )

        summary = _summary("Nova", evidence)
        recommendations = (
            "Check matching request IDs across neutron and cinder logs.",
            "Inspect nova scheduler and compute errors around the same timestamp.",
            "If instance IDs are present, correlate the full instance lifecycle.",
        )
        return AgentFinding(
            agent=self.name,
            summary=summary,
            evidence=tuple(evidence),
            recommendations=recommendations,
            confidence=_confidence(evidence),
        )


def _summary(label: str, evidence: list[LogRecord]) -> str:
    if not evidence:
        return f"No {label} evidence matched the prompt."
    highest = evidence[0]
    return (
        f"Found {len(evidence)} {label} log event(s); strongest signal is "
        f"{highest.level} in {highest.module}: {highest.message[:160]}"
    )


def _confidence(evidence: list[LogRecord]) -> str:
    if any(item.level in {"ERROR", "CRITICAL"} for item in evidence):
        return "medium"
    if evidence:
        return "low"
    return "none"
