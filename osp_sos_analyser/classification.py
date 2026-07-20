from __future__ import annotations


SERVICE_RULES: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("nova", "compute", ("nova",), ("compute",)),
    ("cinder", "storage", ("cinder",), ("storage",)),
    ("ovn", "networking", ("ovn", "openvswitch", "ovs"), ("networking", "ovn")),
    ("neutron", "networking", ("neutron",), ("networking",)),
    ("keystone", "identity", ("keystone",), ("identity",)),
    ("glance", "image", ("glance",), ("image",)),
    ("placement", "placement", ("placement",), ("placement",)),
    ("swift", "object-storage", ("swift",), ("object-storage",)),
    ("heat", "orchestration", ("heat",), ("orchestration",)),
    ("horizon", "dashboard", ("horizon",), ("dashboard",)),
    ("ironic", "baremetal", ("ironic",), ("baremetal",)),
    ("manila", "shared-filesystems", ("manila",), ("shared-filesystems",)),
    ("octavia", "load-balancing", ("octavia",), ("load-balancing",)),
    ("barbican", "key-management", ("barbican",), ("key-management",)),
    ("ceilometer", "telemetry", ("ceilometer",), ("telemetry",)),
    ("gnocchi", "telemetry", ("gnocchi",), ("telemetry",)),
    ("aodh", "telemetry", ("aodh",), ("telemetry",)),
    ("tripleo", "deployment", ("tripleo", "director"), ("deployment",)),
    ("podman", "container-runtime", ("podman", "container"), ("container-runtime",)),
)

LOG_SERVICE_TOKENS = tuple(
    sorted({token for _service, _category, tokens, _tags in SERVICE_RULES for token in tokens})
)

COMMAND_TOKENS = LOG_SERVICE_TOKENS + (
    "ip_",
    "ip-",
    "ip.",
    "hostname",
    "systemctl",
    "ss_",
    "ss-",
)


def classify_service(module: str, source_file: str) -> tuple[str, str]:
    source_lower = source_file.lower()
    # Path-based short-circuit for standalone OVN/OVS processes, which have
    # their own dedicated log files distinct from neutron-server's own log.
    ovn_path_markers = ("ovn-metadata-agent.log", "ovn-controller", "openvswitch/", "ovn_controller")
    if any(marker in source_lower for marker in ovn_path_markers):
        return "ovn", "networking"

    combined = f"{module} {source_file}".lower()
    for service, category, tokens, _tags in SERVICE_RULES:
        if service == "ovn":
            continue  # already handled above via path markers
        if any(token in combined for token in tokens):
            return service, category
    return "unknown", "unknown"


def build_tags(service: str, category: str, module: str, source_file: str) -> str:
    combined = f"{service} {category} {module} {source_file}".lower()
    tags = {"openstack", "rhosp17"}

    if "container" in combined or "podman" in combined or "/containers/" in combined:
        tags.add("containerized")
    if "ovn" in combined or "neutron" in combined or "networking" in combined:
        tags.add("networking")
    if "nova" in combined or "compute" in combined:
        tags.add("compute")
    if "cinder" in combined or "storage" in combined:
        tags.add("storage")
    if "tripleo" in combined:
        tags.add("tripleo")
    if service != "unknown":
        tags.add(service)

    return ",".join(sorted(tags))
