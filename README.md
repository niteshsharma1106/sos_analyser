# osp-sos-analyser

This project ingests RHOSP 17.x SOS report archives into a local DuckDB database for downstream analysis.

## What Phase 1 implements

The current pipeline is modular and supports:

- reading one or more SOS archives from the SOS_REPORTS folder
- streaming RHOSP 17-style .tar.xz SOS archives without extracting them to disk
- scanning high-value RHOSP 17 container log paths such as /var/log/containers for Nova, Cinder, Neutron, OVN, and OVS
- parsing oslo.log-style entries into structured fields: timestamp, pid, level, module, and message
- enriching each log and command row with RHOSP-focused metadata such as service, category, source file, report name, tags, and RHOSP version
- loading command-oriented artifacts from SOS command captures into a second DuckDB table for system-state context
- skipping low-value access logs during Phase 1 so ingestion stays practical on large SOS bundles
- writing everything into a local DuckDB database with the tables os_logs and os_commands

## Quick start

1. Place one or more SOS report archives in the SOS_REPORTS folder.
2. Run:

   ```bash
   python main.py --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb
   ```
   OR
   ```
   uv run python main.py --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb --clear-existing --max-file-size-mb 25
   ```

3. The script will ingest the reports and create a DuckDB database at the requested path.

## Example usage

To rebuild the database from scratch:

```bash
python main.py --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb --clear-existing
```

You can also use the explicit ingest subcommand:

```bash
python main.py ingest --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb --clear-existing
```

To run the Phase 2 detective workflow after ingestion:

```bash
python main.py analyze "VMs on compute-03 suddenly lost network connectivity at 14:00" --db-path sos_analysis.duckdb
```

## Current capabilities

- Reads multiple .tar.xz SOS archives directly
- Scans RHOSP 17 container log locations for Nova, Cinder, Neutron, OVN, and OVS
- Parses oslo.log-style logs into structured rows
- Enriches rows with RHOSP service and category metadata
- Captures SOS command-output artifacts into os_commands
- Uses bulk DuckDB loading for faster ingestion
- Verifies the ingestion path with automated tests

## Phase 2 Plan: Multi-Agent Detective

Phase 2 adds a multi-agent investigation layer on top of the DuckDB data created in Phase 1.

The current implementation is dependency-light: the agents are deterministic Python specialist components with restricted DuckDB tools. This makes the workflow usable without API keys and keeps the investigation logic testable. The same specialist tools can later be wrapped by LangGraph or LangChain if an LLM-backed workflow is required.

The goal is to let a user ask an operational question such as:

```text
VMs on compute-03 suddenly lost network connectivity at 14:00.
```

The system should understand the problem, identify the likely services involved, dispatch specialist agents, gather evidence from the indexed SOS data, and return a structured root-cause analysis.

### Proposed architecture

```text
User prompt or pasted logs
        |
Coordinator Agent
        |
Evidence Router
        |
Nova Agent     Neutron/OVN Agent     Cinder Agent     System/Podman Agent
        |
Correlation / Timeline Agent
        |
Final RCA Report
```

### Coordinator Agent

The Coordinator Agent reads the initial user question and extracts investigation hints:

- impacted service or symptom
- hostname or node name
- timestamp or time window
- instance ID, port ID, volume ID, request ID, or traceback
- likely specialist agents to involve

Example routing:

```text
network connectivity issue -> Neutron/OVN Agent + Nova Agent + System/Podman Agent
volume attach failure -> Cinder Agent + Nova Agent
instance spawn failure -> Nova Agent + Neutron/OVN Agent + Cinder Agent
```

### Evidence Router

The Evidence Router handles pasted logs or free-form evidence from the user before specialist agents begin.

It should identify:

- log service, such as nova, neutron, ovn, cinder, or unknown
- timestamp range
- hostnames
- request IDs
- instance, port, network, router, and volume IDs
- traceback blocks
- ERROR, WARNING, and CRITICAL events

### Specialist Agents

Each specialist agent should use restricted query tools rather than writing arbitrary SQL directly. This keeps the agent behavior easier to debug and makes investigations repeatable.

Nova Agent responsibilities:

- search Nova API, scheduler, conductor, and compute logs
- find instance lifecycle failures
- correlate request IDs and instance IDs
- detect spawn, migration, scheduling, metadata, and libvirt-facing errors

Neutron/OVN Agent responsibilities:

- search Neutron server and OVN/OVS logs
- detect port binding, chassis, tunnel, router, and datapath issues
- inspect OVN controller, northd, ovsdb, and openvswitch command artifacts
- correlate networking events with Nova failures

Cinder Agent responsibilities:

- search Cinder API, scheduler, volume, and backup logs
- detect attach, detach, create, delete, backend, and scheduler failures
- correlate volume IDs and request IDs with Nova events

System/Podman Agent responsibilities:

- inspect podman container state from SOS command artifacts
- inspect systemctl status captures
- identify restarted, failed, or unhealthy service containers
- provide host and service inventory context

### Deterministic query tools

Phase 2 includes Python query helpers over DuckDB. Agents call these helpers as tools instead of writing arbitrary SQL directly.

Recommended tool modules:

```text
analysis.py          shared DuckDB query helpers
nova_tools.py        Nova-focused investigation tools
network_tools.py     Neutron, OVN, and OVS tools
cinder_tools.py      Cinder-focused investigation tools
system_tools.py      Podman, systemctl, and host-state tools
detective_graph.py   LangGraph coordinator and specialist workflow
```

Current implemented modules:

```text
analysis.py          shared DuckDB query helpers
evidence.py          prompt and pasted-evidence hint extraction
nova_tools.py        Nova specialist agent
network_tools.py     Neutron, OVN, and OVS specialist agent
cinder_tools.py      Cinder specialist agent
system_tools.py      Podman and host-state specialist agent
detective.py         coordinator and report workflow
```

Example helper functions:

```python
get_error_summary()
search_logs(service="nova", text="NoValidHost")
get_events_around(timestamp, minutes=5)
find_request_id_events(request_id)
find_instance_events(instance_id)
find_port_events(port_id)
get_command_output(command_pattern="podman_ps")
```

### Final RCA report

The final answer should be structured and evidence-backed:

- Summary
- Timeline
- Impacted services
- Key evidence
- Most likely root cause
- Confidence level
- Recommended next checks

### Phase 2 implementation status

Completed:

1. Deterministic DuckDB query helpers in analysis.py.
2. Specialist tool wrappers for Nova, Neutron/OVN, Cinder, and System/Podman.
3. Prompt hint extraction for services, hosts, timestamps, identifiers, and keywords.
4. Lightweight Coordinator Agent.
5. Correlation timeline generation.
6. Markdown RCA report generator.
7. Automated tests for the detective workflow.

Remaining future enhancements:

1. Add richer pasted-log block classification.
2. Add precise time-window filtering for prompts that include timestamps.
3. Add request ID, instance ID, port ID, and volume ID deep-correlation flows.
4. Add optional LangGraph orchestration around the existing specialist tools.
