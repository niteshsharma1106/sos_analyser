from __future__ import annotations

from .analysis import AnalysisStore
from .investigation_tools import indexed_evidence_for_hints
from .models import AgentFinding, EvidenceHint, LogRecord


class NetworkAgent:
    name = "Neutron/OVN Agent"
    services = ("neutron", "ovn")

    def investigate(self, store: AnalysisStore, hints: EvidenceHint) -> AgentFinding:
        evidence: list[LogRecord] = indexed_evidence_for_hints(
            store._connect(),
            hints,
            services=self.services,
            limit=10,
        )
        if not evidence:
            terms = hints.identifiers or hints.hostnames or hints.keywords
            for service in self.services:
                evidence.extend(
                    store.search_logs(service=service, text_terms=terms[:2], limit=8)
                )

        if not evidence:
            for service in self.services:
                evidence.extend(
                    store.search_logs(
                        service=service,
                        levels=("ERROR", "CRITICAL", "WARNING"),
                        limit=8,
                    )
                )

        commands = []
        for pattern in ("ovn", "ovs", "podman_ps"):
            commands.extend(store.get_command_output(pattern, limit=2))

        summary = _summary(evidence, bool(commands))
        recommendations = (
            "Check OVN controller, northd, and ovsdb logs around the incident time.",
            "Correlate Neutron port binding messages with Nova instance events.",
            "Inspect podman and systemctl state for OVN/OVS service restarts.",
        )
        return AgentFinding(
            agent=self.name,
            summary=summary,
            evidence=tuple(evidence[:10] + commands[:4]),
            recommendations=recommendations,
            confidence=_confidence(evidence, bool(commands)),
        )


def _summary(evidence: list[LogRecord], has_commands: bool) -> str:
    if not evidence and not has_commands:
        return "No Neutron, OVN, or OVS evidence matched the prompt."
    if evidence:
        strongest = evidence[0]
        return (
            f"Found {len(evidence)} network log event(s); strongest signal is "
            f"{strongest.level} in {strongest.module}: {strongest.message[:160]}"
        )
    return "Found OVN/OVS command artifacts, but no matching network log events."


def _confidence(evidence: list[LogRecord], has_commands: bool) -> str:
    if any(item.level in {"ERROR", "CRITICAL"} for item in evidence):
        return "medium"
    if evidence and has_commands:
        return "low"
    if evidence or has_commands:
        return "low"
    return "none"
