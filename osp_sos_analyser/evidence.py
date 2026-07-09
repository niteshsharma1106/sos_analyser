from __future__ import annotations

import re

from .models import EvidenceHint


SERVICE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nova", ("nova", "instance", "vm", "spawn", "scheduler", "compute", "migration")),
    ("neutron", ("neutron", "network", "port", "router", "dhcp", "metadata")),
    ("ovn", ("ovn", "ovs", "openvswitch", "chassis", "datapath", "binding", "tunnel")),
    ("cinder", ("cinder", "volume", "attach", "detach", "backend", "storage")),
    ("podman", ("podman", "container", "systemctl", "service", "restart")),
)

IMPORTANT_KEYWORDS = (
    "error",
    "failed",
    "failure",
    "timeout",
    "disconnect",
    "unreachable",
    "binding",
    "no valid host",
    "network",
    "volume",
    "spawn",
    "migration",
    "traceback",
)

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
REQUEST_ID_RE = re.compile(r"\breq-[0-9a-fA-F-]{8,}\b")
HOST_RE = re.compile(r"\b(?:compute|controller|ctl|ctrl|overcloud|cdel)[\w.-]*\b", re.I)
TIME_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}[ T])?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b"
)


def extract_evidence_hints(prompt: str) -> EvidenceHint:
    lowered = prompt.lower()
    services: list[str] = []
    for service, keywords in SERVICE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            services.append(service)
    if "neutron" in services and "ovn" not in services:
        services.append("ovn")

    identifiers = sorted(set(UUID_RE.findall(prompt) + REQUEST_ID_RE.findall(prompt)))
    hostnames = sorted(set(match.group(0) for match in HOST_RE.finditer(prompt)))
    keywords = [
        keyword
        for keyword in IMPORTANT_KEYWORDS
        if keyword in lowered
    ]
    if not keywords:
        keywords = _fallback_keywords(lowered)

    time_match = TIME_RE.search(prompt)

    return EvidenceHint(
        prompt=prompt,
        services=tuple(dict.fromkeys(services)),
        identifiers=tuple(identifiers),
        hostnames=tuple(hostnames),
        keywords=tuple(dict.fromkeys(keywords)),
        time_text=time_match.group(0) if time_match else None,
    )


def _fallback_keywords(lowered: str) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", lowered)
        if token not in {"show", "find", "what", "when", "from", "with", "that", "this"}
    ]
    return tokens[:5]
