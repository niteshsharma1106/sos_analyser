from __future__ import annotations

from pathlib import Path

from .analysis import AnalysisStore
from .cinder_tools import CinderAgent
from .evidence import extract_evidence_hints
from .models import AgentFinding, EvidenceHint, InvestigationReport
from .network_tools import NetworkAgent
from .nova_tools import NovaAgent
from .system_tools import SystemAgent


class DetectiveCoordinator:
    def __init__(self, db_path: str | Path) -> None:
        self.store = AnalysisStore(db_path)
        self.agents = (
            NovaAgent(),
            NetworkAgent(),
            CinderAgent(),
            SystemAgent(),
        )

    def investigate(self, prompt: str) -> InvestigationReport:
        hints = extract_evidence_hints(prompt)
        selected_agents = self._select_agents(hints)
        findings = tuple(agent.investigate(self.store, hints) for agent in selected_agents)
        services = _timeline_services(hints, findings)
        terms = hints.identifiers or hints.hostnames or hints.keywords
        if hints.identifiers:
            timeline = self.store.find_identifier_events(
                hints.identifiers[0],
                services=services,
                limit=30,
            )
        else:
            timeline = self.store.get_timeline(services=services, text_terms=terms[:3], limit=30)
        if not timeline and services and not hints.identifiers:
            timeline = self.store.get_timeline(services=services, limit=30)

        return InvestigationReport(
            prompt=prompt,
            hints=hints,
            findings=findings,
            timeline=timeline,
            error_summary=self.store.get_error_summary(),
        )

    def _select_agents(self, hints: EvidenceHint):
        if not hints.services:
            return self.agents

        selected = []
        for agent in self.agents:
            if any(service in hints.services for service in agent.services):
                selected.append(agent)

        if "nova" in hints.services and not any(isinstance(agent, NetworkAgent) for agent in selected):
            selected.append(next(agent for agent in self.agents if isinstance(agent, NetworkAgent)))
        if any(service in hints.services for service in ("neutron", "ovn")) and not any(
            isinstance(agent, NovaAgent) for agent in selected
        ):
            selected.append(next(agent for agent in self.agents if isinstance(agent, NovaAgent)))
        system_requested = any(service in {"podman", "neutron", "ovn"} for service in hints.services)
        if system_requested and not any(isinstance(agent, SystemAgent) for agent in selected):
            selected.append(next(agent for agent in self.agents if isinstance(agent, SystemAgent)))

        return tuple(dict.fromkeys(selected))


def investigate_prompt(db_path: str | Path, prompt: str) -> InvestigationReport:
    return DetectiveCoordinator(db_path).investigate(prompt)


def investigate_prompt_offline(db_path: str | Path, prompt: str) -> InvestigationReport:
    return DetectiveCoordinator(db_path).investigate(prompt)


def _timeline_services(
    hints: EvidenceHint, findings: tuple[AgentFinding, ...]
) -> tuple[str, ...]:
    services = set(hints.services)
    for finding in findings:
        for item in finding.evidence:
            service = getattr(item, "service", "")
            if service and service != "unknown":
                services.add(service)
    if "ovn" in services:
        services.add("neutron")
    if "neutron" in services:
        services.add("ovn")
    return tuple(sorted(services))
