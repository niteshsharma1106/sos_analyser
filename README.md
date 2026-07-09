# osp-sos-analyser

This project ingests RHOSP 17.x SOS report archives into a local DuckDB database for downstream analysis.

## What Phase 1 implements

The current pipeline is modular and supports:

- reading one or more SOS archives from the SOS_REPORTS folder
- scanning RHOSP 17-style container log paths such as /var/log/containers and related OpenStack service log locations
- parsing oslo.log-style entries into structured fields: timestamp, pid, level, module, and message
- enriching each log and command row with RHOSP-focused metadata such as service, category, source file, report name, tags, and RHOSP version
- covering the broader RHOSP service surface, including Nova, Cinder, Neutron, OVN, Keystone, Glance, Placement, Swift, Heat, Horizon, Ironic, Manila, Ceilometer, Gnocchi, Aodh, Barbican, Octavia, Sahara, Trove, Zaqar, Mistral, Magnum, TripleO, and container runtime artifacts
- loading command-oriented artifacts from SOS command captures into a second DuckDB table for system-state context
- writing everything into a local DuckDB database with the tables os_logs and os_commands

## Quick start

1. Place one or more SOS report archives in the SOS_REPORTS folder.
2. Run:

   ```bash
   python main.py --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb
   ```

3. The script will ingest the reports and create a DuckDB database at the requested path.

## Example usage

To rebuild the database from scratch:

```bash
python main.py --reports-dir SOS_REPORTS --db-path sos_analysis.duckdb --clear-existing
```

## Current capabilities

- Reads multiple .tar.xz SOS archives directly
- Scans RHOSP 17 container log locations and common OpenStack service names
- Parses oslo.log-style logs into structured rows
- Enriches rows with RHOSP service and category metadata for broad OpenStack service coverage
- Captures SOS command-output artifacts into os_commands
- Verifies the ingestion path with automated tests

## Next steps

- add richer RHOSP-specific enrichment for Nova, Cinder, and OVN
- expand command collection for podman, systemctl, and networking state
- build LangGraph-based analysis workflows on top of the DuckDB tables
