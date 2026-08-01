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
- tracking completed archive hashes in DuckDB to avoid ingesting the same SOS archive twice
- running row-level deduplication after each archive as a second duplicate guard
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


   # python main.py --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb --clear-existing --max-file-size-mb 25
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

Phase 2 uses a LangChain tool-calling agent by default. Create a local `.env` file first.
Model, provider, and API key are read **only** from `.env` (no hardcoded defaults in code).

Copy the example and edit one provider block:

```bash
cp .env.example .env
```

**Google Gemini**

```env
GOOGLE_API_KEY=your-gemini-api-key-here
OSP_SOS_MODEL=gemini-2.5-pro
OSP_SOS_MODEL_PROVIDER=google_genai
```

**Groq** (for models like `openai/gpt-oss-120b`)

```env
GROQ_API_KEY=your-groq-api-key-here
OSP_SOS_MODEL=openai/gpt-oss-120b
OSP_SOS_MODEL_PROVIDER=groq
```

**OpenAI**

```env
OPENAI_API_KEY=your-openai-api-key-here
OSP_SOS_MODEL=gpt-4o-mini
OSP_SOS_MODEL_PROVIDER=openai
```

Keep a single provider block active. Provider and model must match. Putting a Groq model id under `google_genai` causes a 404.
For local testing without an LLM, use the deterministic fallback:

```bash
python main.py analyze "VM fd27c003-5b78-4abb-a85a-aa90973f7ff0 failed to create" --db-path sos_analysis.duckdb --offline
```

## Multi-node cluster ingest

Place controller and compute SOS archives in the same folder and ingest once into one DuckDB:

```bash
# SOS_REPORTS/ may contain multiple .tar.xz archives
$env:OSP_SOS_SKIP_JOURNALS="1"   # Windows tip
uv run python main.py ingest --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb --clear-existing
```

All nodes share one `cluster_id`. The chat/investigator can then:

- `get_cluster_overview` — list hosts/roles
- `compare_nodes` — ERROR/WARNING counts per host
- `get_entity_evidence` / `search_os_logs` with `hostname=` or `node_role=` filters
- `get_related_entities` / `get_operation_path` — VM↔port↔chassis↔host graph

## Relationship / operation graph

Ingest now builds `entity_relationships` from labelled log cues (`port_id`, `device_id`, `binding:host_id`, `chassis=...`). Ask:

```text
Show the VM → port → chassis → host path for port <uuid>
```

Or in the chat UI set **Focused entity** and **Operation path first**.

## Backend logging & agent observability

Investigation runs emit structured logs and an in-answer timeline of LangGraph nodes, LLM calls, and tool use.

```powershell
# console + file
$env:OSP_SOS_LOG_LEVEL="INFO"
$env:OSP_SOS_LOG_FILE="osp_sos.log"
uv run python main.py chat --db-path sos_analysis.duckdb --log-file osp_sos.log

# CLI analyze also prints the agent timeline under the RCA
uv run python main.py analyze "port UUID failed" --db-path sos_analysis.duckdb --log-file osp_sos.log
```

In the chat UI, keep **Show agent observability** enabled to append the node/tool timeline under each answer.

## Chat UI (ask / answer)




After ingestion, launch an interactive chat UI instead of the CLI:

```bash
uv run python main.py chat --db-path sos_analysis.duckdb
# or
uv run osp-sos-chat --db-path sos_analysis.duckdb
```

Open the printed local URL (default `http://127.0.0.1:7860`). Use **Offline mode** in Investigation settings for deterministic answers without an LLM.

## Implemented capabilities

- Reads multiple .tar.xz SOS archives directly
- Scans RHOSP 17 container log locations for Nova, Cinder, Neutron, OVN, and OVS
- Parses oslo.log-style logs into structured rows
- Enriches rows with RHOSP service and category metadata
- Captures SOS command-output artifacts into os_commands
- Uses bulk DuckDB loading for faster ingestion
- Skips already-ingested archives through the ingested_reports registry table
- Builds an evidence index and VM/port/chassis/host relationship graph
- Provides deterministic offline investigation and an LLM-backed LangGraph RCA workflow
- Includes a local Gradio chat interface and automated test coverage

## Investigation workflow

The application includes a LangGraph investigation layer on top of the DuckDB data created during ingestion.

The default implementation is a LangChain tool-calling coordinator. It reads the user query, selects evidence tools, queries DuckDB, and produces an evidence-backed RCA. A deterministic `--offline` mode is also available when no LLM credentials are configured.

For example, ask an operational question such as:

```text
VMs on compute-03 suddenly lost network connectivity at 14:00.
```

The workflow identifies likely services and entities, gathers indexed SOS evidence, and returns a structured root-cause analysis.

### Runtime architecture

```text
User prompt or pasted logs
        |
LangChain Coordinator Agent
        |
Dynamic tool selection
        |
Nova Agent     Neutron/OVN Agent     Cinder Agent     System/Podman Agent
        |
Correlation / Timeline Agent
        |
Final RCA Report
```

### Coordinator Agent

The Coordinator Agent reads the initial user question and decides which tools to call:

- impacted service or symptom
- hostname or node name
- timestamp or time window
- instance ID, port ID, volume ID, request ID, or traceback
- likely specialist agents to involve

Example reasoning:

```text
network connectivity issue -> call network, nova, and maybe system tools
volume attach failure -> call cinder and nova tools
instance spawn failure -> call nova and network tools, then cinder only if volume context appears
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

### Investigation tools

LangChain exposes restricted DuckDB query helpers as tools instead of allowing arbitrary SQL from the model.

Core modules:

```text
analysis.py          shared DuckDB query helpers
nova_tools.py        Nova-focused investigation tools
network_tools.py     Neutron, OVN, and OVS tools
cinder_tools.py      Cinder-focused investigation tools
system_tools.py      Podman, systemctl, and host-state tools
langgraph_investigator.py LangGraph investigation workflow
```

Current implemented modules:

```text
analysis.py          shared DuckDB query helpers
evidence.py          prompt and pasted-evidence hint extraction
langchain_detective.py LangChain Coordinator Agent and tool definitions
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

### Current implementation status

Implemented:

1. Deterministic DuckDB query helpers in analysis.py.
2. LangChain tool wrappers for Nova, Neutron/OVN, Cinder, System/Podman, timeline, and error summary.
3. `.env` based secret loading with python-dotenv.
4. LangChain model initialization through `init_chat_model`.
5. Correlation timeline generation.
6. Markdown RCA report generator.
7. Automated tests for the detective workflow.

Planned enhancements should be tracked separately from this README so the documented workflow always reflects what users can run today.
