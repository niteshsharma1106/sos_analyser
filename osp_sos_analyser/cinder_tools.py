from __future__ import annotations

from .analysis import AnalysisStore
from .investigation_tools import indexed_evidence_for_hints
from .models import AgentFinding, EvidenceHint, LogRecord


class CinderAgent:
    name = "Cinder Agent"
    services = ("cinder",)

    def investigate(self, store: AnalysisStore, hints: EvidenceHint) -> AgentFinding:
        evidence = indexed_evidence_for_hints(
            store._connect(), hints, services=("cinder",), limit=10
        )
        if not evidence:
            terms = hints.identifiers or hints.hostnames or hints.keywords
            evidence = list(
                store.search_logs(service="cinder", text_terms=terms[:2], limit=10)
            )
        if not evidence:
            evidence = list(
                store.search_logs(
                    service="cinder",
                    levels=("ERROR", "CRITICAL", "WARNING"),
                    limit=10,
                )
            )

        recommendations = (
            "Correlate volume IDs with Nova attach or detach messages.",
            "Check Cinder scheduler and backend-facing errors near the incident time.",
            "Validate backend health from relevant SOS command artifacts.",
        )
        return AgentFinding(
            agent=self.name,
            summary=_summary(evidence),
            evidence=tuple(evidence),
            recommendations=recommendations,
            confidence=_confidence(evidence),
        )


def _summary(evidence: list[LogRecord]) -> str:
    if not evidence:
        return "No Cinder evidence matched the prompt."
    strongest = evidence[0]
    return (
        f"Found {len(evidence)} Cinder log event(s); strongest signal is "
        f"{strongest.level} in {strongest.module}: {strongest.message[:160]}"
    )


def _confidence(evidence: list[LogRecord]) -> str:
    if any(item.level in {"ERROR", "CRITICAL"} for item in evidence):
        return "medium"
    if evidence:
        return "low"
    return "none"
