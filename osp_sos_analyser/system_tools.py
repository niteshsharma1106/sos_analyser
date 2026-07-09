from __future__ import annotations

from .analysis import AnalysisStore
from .models import AgentFinding, CommandRecord, EvidenceHint


class SystemAgent:
    name = "System/Podman Agent"
    services = ("podman",)

    def investigate(self, store: AnalysisStore, hints: EvidenceHint) -> AgentFinding:
        command_patterns = ("podman_ps", "podman", "systemctl", "hostname")
        evidence: list[CommandRecord] = []
        for pattern in command_patterns:
            evidence.extend(store.get_command_output(pattern, limit=3))

        if hints.hostnames:
            for host in hints.hostnames[:2]:
                evidence.extend(store.get_command_output(host, limit=2))

        recommendations = (
            "Check whether impacted service containers restarted around the incident.",
            "Compare podman ps output with expected RHOSP service containers.",
            "Inspect systemctl captures for failed or degraded units.",
        )
        return AgentFinding(
            agent=self.name,
            summary=_summary(evidence),
            evidence=tuple(evidence[:10]),
            recommendations=recommendations,
            confidence="low" if evidence else "none",
        )


def _summary(evidence: list[CommandRecord]) -> str:
    if not evidence:
        return "No podman or system command artifacts matched the prompt."
    return f"Found {len(evidence)} system command artifact(s) useful for service-state context."
