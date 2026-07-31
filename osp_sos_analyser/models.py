from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LogEntry:
    timestamp: datetime | None
    pid: int | None
    level: str
    module: str
    message: str
    service: str
    category: str
    source_file: str
    report_name: str
    tags: str
    rhosp_version: str = "17.x"
    hostname: str = ""
    node_role: str = ""
    cluster_id: str = ""


@dataclass(frozen=True)
class CommandArtifact:
    source: str
    command: str
    output: str
    service: str
    category: str
    source_file: str
    report_name: str
    tags: str
    rhosp_version: str = "17.x"
    hostname: str = ""
    node_role: str = ""
    cluster_id: str = ""


@dataclass(frozen=True)
class IngestionStats:
    archives: int = 0
    log_files: int = 0
    log_rows: int = 0
    command_files: int = 0
    command_rows: int = 0

    def add(self, other: "IngestionStats") -> "IngestionStats":
        return IngestionStats(
            archives=self.archives + other.archives,
            log_files=self.log_files + other.log_files,
            log_rows=self.log_rows + other.log_rows,
            command_files=self.command_files + other.command_files,
            command_rows=self.command_rows + other.command_rows,
        )


@dataclass(frozen=True)
class LogRecord:
    timestamp: datetime | None
    level: str
    service: str
    module: str
    message: str
    source_file: str
    report_name: str
    hostname: str = ""
    node_role: str = ""


@dataclass(frozen=True)
class CommandRecord:
    command: str
    service: str
    source_file: str
    output: str
    report_name: str
    hostname: str = ""
    node_role: str = ""


@dataclass(frozen=True)
class EvidenceHint:
    prompt: str
    services: tuple[str, ...]
    identifiers: tuple[str, ...]
    hostnames: tuple[str, ...]
    keywords: tuple[str, ...]
    time_text: str | None = None


@dataclass(frozen=True)
class AgentFinding:
    agent: str
    summary: str
    evidence: tuple[LogRecord | CommandRecord, ...]
    recommendations: tuple[str, ...]
    confidence: str


@dataclass(frozen=True)
class InvestigationReport:
    prompt: str
    hints: EvidenceHint
    findings: tuple[AgentFinding, ...]
    timeline: tuple[LogRecord, ...]
    error_summary: tuple[tuple[str, str, int], ...]
    final_summary: str | None = None
    final_root_cause: str | None = None
    final_confidence: str | None = None
    final_timeline_analysis: str | None = None
    final_recommendations: tuple[str, ...] = ()

    def render_markdown(self) -> str:
        lines = [
            "# SOS Detective Report",
            "",
            "## Summary",
        ]
        if self.final_summary:
            lines.append(f"- {self.final_summary}")
        elif self.findings:
            for finding in self.findings:
                lines.append(f"- {finding.agent}: {finding.summary}")
        else:
            lines.append("- No specialist evidence matched the prompt.")

        lines.extend(["", "## Extracted Hints"])
        lines.append(f"- Services: {', '.join(self.hints.services) or 'not detected'}")
        lines.append(f"- Identifiers: {', '.join(self.hints.identifiers) or 'not detected'}")
        lines.append(f"- Hosts: {', '.join(self.hints.hostnames) or 'not detected'}")
        lines.append(f"- Time: {self.hints.time_text or 'not detected'}")

        lines.extend(["", "## Error Summary"])
        if self.error_summary:
            for service, level, count in self.error_summary:
                lines.append(f"- {service or 'unknown'} {level}: {count}")
        else:
            lines.append("- No ERROR or CRITICAL rows found.")

        lines.extend(["", "## Most Likely Root Cause"])
        lines.append(f"- {self.final_root_cause or self._most_likely_root_cause()}")
        lines.append(f"- Overall confidence: {self.final_confidence or self._overall_confidence()}")

        if self.final_timeline_analysis:
            lines.extend(["", "## Timeline Analysis"])
            lines.append(f"- {self.final_timeline_analysis}")

        lines.extend(["", "## Timeline"])
        if self.timeline:
            for event in self.timeline[:20]:
                stamp = event.timestamp.isoformat(sep=" ") if event.timestamp else "unknown-time"
                lines.append(
                    f"- {stamp} [{event.service}/{event.level}] "
                    f"{event.module}: {event.message[:220]}"
                )
        else:
            lines.append("- No timeline events matched the detected hints.")

        lines.extend(["", "## Specialist Findings"])
        for finding in self.findings:
            lines.append(f"### {finding.agent}")
            lines.append(f"- Confidence: {finding.confidence}")
            lines.append(f"- Finding: {finding.summary}")
            if finding.evidence:
                lines.append("- Evidence:")
                for item in finding.evidence[:5]:
                    if isinstance(item, LogRecord):
                        stamp = item.timestamp.isoformat(sep=" ") if item.timestamp else "unknown-time"
                        lines.append(
                            f"  - {stamp} [{item.service}/{item.level}] "
                            f"{item.module}: {item.message[:220]}"
                        )
                    else:
                        lines.append(
                            f"  - command {item.command} from {item.source_file}: "
                            f"{item.output[:220]}"
                        )
            if finding.recommendations:
                lines.append("- Recommended next checks:")
                for recommendation in finding.recommendations:
                    lines.append(f"  - {recommendation}")

        if self.final_recommendations:
            lines.extend(["", "## Final Recommended Next Checks"])
            for recommendation in self.final_recommendations:
                lines.append(f"- {recommendation}")

        return "\n".join(lines)

    def _most_likely_root_cause(self) -> str:
        if not self.findings:
            return "Insufficient evidence to identify a likely root cause."

        medium_findings = [
            finding
            for finding in self.findings
            if finding.confidence in {"medium", "high"} and finding.evidence
        ]
        candidate = medium_findings[0] if medium_findings else self.findings[0]
        if not candidate.evidence:
            return "No matching evidence was found in the indexed SOS data."

        first = candidate.evidence[0]
        if isinstance(first, LogRecord):
            return (
                f"{candidate.agent} has the strongest current signal: "
                f"{first.level} in {first.module} from {first.source_file}."
            )
        return (
            f"{candidate.agent} has the strongest current signal from command "
            f"{first.command} in {first.source_file}."
        )

    def _overall_confidence(self) -> str:
        levels = [finding.confidence for finding in self.findings]
        if "high" in levels:
            return "high"
        if "medium" in levels:
            return "medium"
        if "low" in levels:
            return "low"
        return "none"
