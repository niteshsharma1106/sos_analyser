# OpenStack SOS RCA Agent — End-to-End Working Flow

## Purpose

The RCA Agent turns one or more Red Hat OpenStack SOS reports into an evidence-backed incident answer. It is an investigation assistant: it reads the ingested SOS snapshot, correlates evidence across nodes, and states only conclusions supported by that snapshot.

## Complete operational flow

```mermaid
flowchart TD
    A[SOS report archives<br/>compute / controller / storage] --> B[Ingestion]
    B --> C[(DuckDB evidence snapshot)]
    C --> D[Chat UI or CLI question]
    D --> E{Execution mode}
    E -->|Offline| F[Deterministic search workflow]
    E -->|Linear — default| G[Query expansion]
    E -->|Planner| H[Plan / Act / Replan loop]

    G --> I[Prefetch manifest + relevant evidence]
    I --> J{Reboot / crash question?}
    J -->|Yes| K[Mandatory reboot timeline tool]
    K --> L[Peer and controller correlation]
    J -->|No| M[LLM tool investigation]
    L --> M
    M --> N[Evidence collection]
    N --> O[RCA synthesis]
    H --> P[Use existing tool or design read-only SQL]
    P --> Q[Validate and execute read-only analysis]
    Q --> R[Evidence graph]
    R --> H
    H -->|Answer ready| O
    F --> S[Offline report]
    O --> T[User-facing answer + optional run trace]
    S --> T
```

## Phase-by-phase flow

| Phase | What happens | Output / control |
| --- | --- | --- |
| 1. Ingest | SOS archives are parsed and normalized into host, log, command, entity, and relationship records. | A local DuckDB snapshot. The agent does not query live OpenStack services. |
| 2. Receive question | The chat UI calls `/api/ask`; CLI calls `main.py analyze`. The user asks about a host, VM, port, operation, or error. | Raw incident question and selected engine. |
| 3. Query expansion | The LLM creates a bounded structured plan: intent, entities, hostname, service, keywords, hypotheses, and time window. Fallback parsing protects against invalid provider output. | `ExpandedQuery`. |
| 4. Prefetch | The application retrieves a compact cluster and evidence digest before the tool loop begins. | Fast initial context; it is not treated as a substitute for required evidence tools. |
| 5. Investigation | The agent calls read-only investigation tools one at a time. A reboot question has a mandatory first timeline call. | Tool digests and an observable tool trail. |
| 6. Correlation | Local logs are compared with peer/controller logs in the same time window. Entity and relationship tools link VM, port, request, and host evidence. | A causal timeline, or a clearly stated evidence gap. |
| 7. Synthesis | A final LLM call creates a concise RCA from the original request, evidence, and limitations. | Facts, direct cause, supported root cause, and remaining uncertainty. |
| 8. Presentation | The UI displays the RCA. Observability can display node transitions, LLM calls, tool calls, and handoffs. | Auditable response rather than hidden reasoning. |

## Reboot RCA decision flow

```mermaid
flowchart TD
    A[Question names reboot, crash, panic, watchdog, or power event] --> B[Resolve affected hostname]
    B --> C[Mandatory get_host_reboot_timeline]
    C --> D[Discover journalctl --list-boots / boot boundaries]
    D --> E[Search other ingested hosts in the boot gap]
    E --> F{Positive local crash signature?}
    F -->|Yes| G[Report local cause with timestamp and quote]
    F -->|No| H{Peer monitor / state-loss / fence evidence?}
    H -->|Yes| I[Report causal chain: loss → Pacemaker decision → fence/reboot → boot]
    H -->|No| J[State cause is unknown in available SOS]
    G --> K[State remaining gaps and next evidence to collect]
    I --> K
    J --> K
```

## Example: `comp008` reboot

```mermaid
sequenceDiagram
    participant C as Controller / Pacemaker
    participant H as comp008
    participant F as Fence agent

    C->>H: Remote-node monitor
    Note over C,H: 02:33:45 — monitor connection dropped
    C->>C: Mark node state lost
    Note over C: 02:34:13
    C->>F: Request STONITH reboot
    Note over C,F: 02:34:47 — Pacemaker-controlled reboot completed
    F->>H: Fence/reboot action
    H->>H: New boot begins
    Note over H: 02:38:27
```

This proves that Pacemaker fenced the node after monitoring lost the remote connection. It does **not** by itself prove why that connection was lost; the RCA must retain that distinction.

## Guardrails that make the answer trustworthy

- All investigation tools and temporary analyses use the ingested snapshot in read-only mode.
- Reboot investigations cannot finish with zero evidence-tool calls when a hostname is identified.
- Peer evidence prioritizes the exact resolved hostname, avoiding confusion with similarly named hosts in other racks.
- Monitor-loss and state-loss signals are ordered before fence completion messages, so the response retains trigger and action.
- The response must distinguish observed facts, supported inference, and unknowns.
- Missing local crash evidence is reported as “not found in available SOS,” not converted into an unsupported BMC, hardware, or manual-reset claim.

## Entry points

- Interactive UI: `python main.py chat`
- CLI investigation: `python main.py analyze "why comp008 rebooted"`
- Default engine: `linear` — query expansion → tool agent → synthesis.
- Optional engine: `planner` — iterative planner with validated, read-only DuckDB analysis when existing tools are insufficient.

