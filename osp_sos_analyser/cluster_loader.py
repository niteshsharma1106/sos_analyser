# cluster_loader.py — Layer 0: cluster / node identity from SOS archives.
from __future__ import annotations

import hashlib
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .archive_reader import normalized_member_name, read_text_member

HOSTNAME_BASENAME_RE = re.compile(r"(?:^|/)hostname$", re.I)
UNAME_BASENAME_RE = re.compile(r"(?:^|/)uname(?:_-a)?$", re.I)
INSTALLED_RPMS_RE = re.compile(r"(?:^|/)installed-rpms(?:\.txt)?$", re.I)
SOSREPORT_HOST_RE = re.compile(
    r"sosreport[-_](?P<host>[A-Za-z0-9][A-Za-z0-9._-]{1,63?}?)(?:[-_](?:20\d{2}|case|id)|\.tar)",
    re.I,
)
ROLE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("controller", ("controller", "ctrlplane", "undercloud")),
    ("compute", ("compute", "novacompute")),
    ("storage", ("storage", "ceph", "cinder")),
    ("networker", ("networker", "network")),
)
# Compact RHOSP lab names: ...-ctrl001, ...-comp008 (not matched by "compute").
ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("controller", re.compile(r"(?:^|[-_.])ctrl\d", re.I)),
    ("compute", re.compile(r"(?:^|[-_.])comp\d", re.I)),
    ("storage", re.compile(r"(?:^|[-_.])(?:ceph|storage)\d", re.I)),
)
# First token of `uname -a` / junk hostname files must never become the node id.
INVALID_HOSTNAMES = frozenset(
    {
        "linux",
        "darwin",
        "windows",
        "localhost",
        "localhost.localdomain",
        "unknown",
        "none",
        "null",
        "(none)",
    }
)
OPENSTACK_RPM_RE = re.compile(
    r"^(?P<name>openstack-(?:nova|neutron|cinder|glance|keystone|heat)[^\s]*)\s+(?P<ver>\S+)",
    re.I | re.M,
)
RHOSP_VERSION_RE = re.compile(r"rhosp[-_]?(\d+(?:\.\d+)?)", re.I)


@dataclass
class NodeManifest:
    """Identity facts for one SOS archive / node."""

    archive_name: str
    archive_id: str = ""
    hostname: str = ""
    node_role: str = "unknown"
    rhosp_version: str = "17.x"
    services: set[str] = field(default_factory=set)
    cluster_id: str = ""

    def as_row(self) -> tuple[object, ...]:
        return (
            self.cluster_id,
            self.hostname or _fallback_hostname(self.archive_name),
            self.node_role,
            self.rhosp_version,
            ",".join(sorted(self.services)) if self.services else "",
            self.archive_name,
            self.archive_id,
        )


def guess_hostname_from_archive_name(archive_name: str) -> str:
    match = SOSREPORT_HOST_RE.search(archive_name)
    if match:
        return match.group("host").rstrip(".-_")
    stem = Path(archive_name).name
    for suffix in (".tar.xz", ".tar.gz", ".tar"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"^sosreport[-_]", "", stem, flags=re.I)
    stem = re.sub(r"[-_]20\d{2}.*$", "", stem)
    return stem or archive_name


def infer_node_role(hostname: str, archive_name: str = "") -> str:
    haystack = f"{hostname} {archive_name}".lower()
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(haystack):
            return role
    for role, tokens in ROLE_RULES:
        if any(token in haystack for token in tokens):
            return role
    # Legacy short token: bare "ctrl" still common in older names.
    if re.search(r"(?:^|[-_.])ctrl(?:$|[-_.])", haystack):
        return "controller"
    return "unknown"


def repair_node_roles(conn: Any) -> int:
    """Fix cluster_nodes/os_* roles when hostname implies compute/controller."""
    try:
        rows = conn.execute(
            "SELECT hostname, node_role, archive_name FROM cluster_nodes"
        ).fetchall()
    except Exception:
        return 0
    updated = 0
    for hostname, role, archive_name in rows:
        host = str(hostname or "").strip()
        if not host:
            continue
        inferred = infer_node_role(host, str(archive_name or ""))
        current = str(role or "unknown").strip().lower() or "unknown"
        if inferred == "unknown" or inferred == current:
            continue
        conn.execute(
            "UPDATE cluster_nodes SET node_role = ? WHERE hostname = ?",
            [inferred, host],
        )
        try:
            conn.execute(
                "UPDATE os_logs SET node_role = ? WHERE lower(hostname) = lower(?)",
                [inferred, host],
            )
            conn.execute(
                "UPDATE os_commands SET node_role = ? WHERE lower(hostname) = lower(?)",
                [inferred, host],
            )
        except Exception:
            pass
        updated += 1
    return updated


def is_cluster_manifest_member(member: tarfile.TarInfo, max_file_size: int) -> bool:
    if not member.isreg() or member.size <= 0 or member.size > max_file_size:
        return False
    name = normalized_member_name(member)
    if (
        HOSTNAME_BASENAME_RE.search(name)
        or UNAME_BASENAME_RE.search(name)
        or INSTALLED_RPMS_RE.search(name)
    ):
        return True
    lower = name.lower()
    return lower.endswith("/hostname") or "/sos_commands/host/" in f"/{lower}"


def is_valid_hostname(hostname: str) -> bool:
    candidate = (hostname or "").strip().strip(".")
    if not candidate or len(candidate) > 253:
        return False
    if candidate.lower() in INVALID_HOSTNAMES:
        return False
    # Reject pure OS/kernel tokens and paths.
    if "/" in candidate or " " in candidate:
        return False
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", candidate):
        return False
    return True


def finalize_node_identity(manifest: NodeManifest) -> None:
    """Ensure hostname/role are usable after scanning an archive."""
    if not is_valid_hostname(manifest.hostname):
        manifest.hostname = guess_hostname_from_archive_name(manifest.archive_name)
    if not is_valid_hostname(manifest.hostname):
        manifest.hostname = _fallback_hostname(manifest.archive_name)
    # Re-infer role from the final hostname + archive name so ctrl/compute
    # tokens in the archive name still win after a bad hostname file.
    inferred = infer_node_role(manifest.hostname, manifest.archive_name)
    if inferred != "unknown":
        manifest.node_role = inferred
    elif not manifest.node_role:
        manifest.node_role = "unknown"


def apply_manifest_text(manifest: NodeManifest, source_file: str, text: str) -> None:
    name = source_file.replace("\\", "/").lower()
    body = (text or "").strip()
    if not body:
        return

    basename = Path(name).name
    if HOSTNAME_BASENAME_RE.search(name) or name.endswith("/hostname"):
        # Prefer a real /hostname file over archive-name guess, but never accept
        # `Linux` (common when uname output is mistakenly read as hostname).
        host = _parse_hostname_payload(body)
        if is_valid_hostname(host):
            manifest.hostname = host
            manifest.node_role = infer_node_role(host, manifest.archive_name)
        return

    if UNAME_BASENAME_RE.search(name) or basename.startswith("uname"):
        host = _parse_uname_hostname(body)
        # Only fill hostname from uname when we do not already have a better value.
        if is_valid_hostname(host) and (
            not manifest.hostname or not is_valid_hostname(manifest.hostname)
        ):
            manifest.hostname = host
            manifest.node_role = infer_node_role(host, manifest.archive_name)
        return

    if INSTALLED_RPMS_RE.search(name):
        version = _parse_rhosp_version(body)
        if version:
            manifest.rhosp_version = version
        for match in OPENSTACK_RPM_RE.finditer(body):
            service = match.group("name").split("-")[1].lower() if "-" in match.group("name") else ""
            if service:
                manifest.services.add(service)


def note_service_from_path(manifest: NodeManifest, source_file: str) -> None:
    lower = source_file.lower()
    for service in (
        "nova",
        "neutron",
        "ovn",
        "cinder",
        "glance",
        "keystone",
        "heat",
        "placement",
        "podman",
    ):
        if service in lower:
            manifest.services.add(service)


def resolve_cluster_id(
    conn,
    reports_dir: Path,
    *,
    clear_existing: bool,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    if not clear_existing:
        try:
            row = conn.execute(
                "SELECT cluster_id FROM cluster_nodes WHERE cluster_id IS NOT NULL LIMIT 1"
            ).fetchone()
            if row and row[0]:
                return str(row[0])
        except Exception:
            pass
    digest = hashlib.sha256(str(reports_dir.resolve()).encode("utf-8")).hexdigest()
    return digest[:16]


def new_node_manifest(
    archive_path: Path,
    *,
    archive_id: str,
    cluster_id: str,
) -> NodeManifest:
    hostname = guess_hostname_from_archive_name(archive_path.name)
    return NodeManifest(
        archive_name=archive_path.name,
        archive_id=archive_id,
        hostname=hostname,
        node_role=infer_node_role(hostname, archive_path.name),
        cluster_id=cluster_id,
    )


def absorb_manifest_member(
    manifest: NodeManifest,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
) -> str:
    source_file = normalized_member_name(member)
    text = read_text_member(archive, member)
    apply_manifest_text(manifest, source_file, text)
    return text


def _fallback_hostname(archive_name: str) -> str:
    return guess_hostname_from_archive_name(archive_name)


def _parse_hostname_payload(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip().split()[0] if line.strip() else ""
        if candidate and not candidate.startswith("#") and is_valid_hostname(candidate):
            return candidate
    return ""


def _parse_uname_hostname(text: str) -> str:
    # uname -a: Linux hostname.example.com 5.14.0-...
    parts = text.strip().split()
    if len(parts) >= 2 and parts[0].lower() == "linux":
        host = parts[1]
        # Keep short hostname for cluster matching unless only FQDN is useful.
        short = host.split(".")[0]
        if is_valid_hostname(short):
            return short
        if is_valid_hostname(host):
            return host
    return ""


def _parse_rhosp_version(text: str) -> str | None:
    match = RHOSP_VERSION_RE.search(text)
    if match:
        return match.group(1)
    # Fall back to openstack-nova package epoch-ish version hint.
    nova = re.search(r"^openstack-nova[^\s]*\s+(\d+\.\d+)", text, re.I | re.M)
    if nova:
        return nova.group(1)
    return None
