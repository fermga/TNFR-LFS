# Command Line Interface

Install the project in editable mode and run:

```
pip install -e .
tnfr-lfs telemetry.csv --export csv
```

The CLI ingests the telemetry file, computes EPI metrics, and prints the
selected export format to stdout.

## Subcommands

The ``tnfr-lfs`` executable (part of the TNFR × LFS toolkit) organises the workflow into five subcommands:

### ``baseline``

Capture telemetry from live UDP streams or from historical CSV data and
persist it as a baseline for subsequent analysis.  Live capture expects
the simulation to broadcast telemetry through:

* **OutSim UDP** – default port ``4123``
* **OutGauge UDP** – default port ``3000``
* **InSim UDP** – optional control messages through port ``29999`` when a
  simulator requires a handshake (no traffic is sent by the TNFR × LFS toolkit but the
  port must remain available to avoid conflicts)

All clients connect to ``127.0.0.1`` by default and can be redirected via
``--host`` and the corresponding ``--outsim-port``/``--outgauge-port``
options.  When the simulator is not running on the same machine ensure
firewall rules allow inbound UDP traffic on the configured ports.

To ingest prerecorded telemetry, pass ``--simulate`` with the path to a
CSV file containing OutSim-compatible columns.  The ``--format`` option
persists the baseline as ``jsonl`` or ``parquet``.

### ``analyze``

Reads a baseline (captured or simulated) and computes ΔNFR/Sense Index
metrics.  The subcommand uses the core operators to orchestrate
telemetry processing and emits a structured payload via the selected
exporter (``json`` by default).  The payload now includes a
``phase_messages`` array that summarises per-phase ΔNFR↓ deviations with
actionable hints together with a ``reports`` block.  Each invocation
persists ``out/<baseline-stem>/sense_index_map.json`` and
``out/<baseline-stem>/yaw_roll_spectrum.json`` (configurable via the
``paths.output_dir`` setting) so external tooling can reuse the Sense
Index heatmap and yaw/roll spectrum without re-running the CLI.

### ``suggest``

Runs the rule-based recommendation engine on top of an existing baseline
and exports the suggestions using the chosen exporter.  Use
``--car-model`` and ``--track`` to load different recommendation profiles.
The exported payload mirrors ``analyze`` by embedding
``phase_messages`` and the ``reports`` artefacts so the tuning rationale
remains transparent when sharing the recommendations.

### ``report``

Combines the orchestration operators with the exporters registry to
produce explainable ΔNFR/Sense Index reports.  ``--target-delta`` and
``--target-si`` define the desired objectives.  The emitted report now
exposes the same telemetry artefacts as ``analyze``/``suggest`` under the
``reports`` key so downstream automations (dashboards or notebooks) can
ingest the Sense Index map and yaw/roll spectrum while keeping the
command line output concise.

### ``write-set``

Creates a setup plan by blending the optimisation-aware search module
with the rule engine.  The resulting payload follows the
``tnfr_lfs.exporters.setup_plan.SetupPlan`` schema, including:

```
{
  "car_model": "generic_gt",
  "session": "FP1",
  "changes": [
    {"parameter": "rear_wing_angle", "delta": -1.0, "rationale": "Reduce drag", "expected_effect": "Higher top speed"}
  ],
  "rationales": ["Reduce drag"],
  "expected_effects": ["Higher top speed"]
}
```

Exporters normalise the dataclasses into JSON, CSV, or Markdown depending
on the ``--export`` flag.  The Markdown exporter deduplicates rationales
and expected effects to provide a readable handover document.
## Configuration

The CLI resolves defaults from a ``tnfr-lfs.toml`` file located in the
current working directory, the path referenced by the
``TNFR_LFS_CONFIG`` environment variable, or ``~/.config/tnfr-lfs.toml``
as a fallback.  Any explicit CLI flag takes precedence, but the file can
define sensible defaults for ports, exporters, car/track profiles and
report locations:

```toml
[telemetry]
host = "192.168.0.10"
outsim_port = 4125
outgauge_port = 3003

[analyze]
export = "json"

[suggest]
car_model = "gt3"
track = "spa"

[paths]
output_dir = "out"

[limits.delta_nfr]
entry = 0.5
apex = 0.4
exit = 0.6
```

This configuration adjusts the default UDP ports used by ``baseline``,
selects the exporter for analytics/reporting, sets the default
car/track for ``suggest`` and overrides the tolerance used when
highlighting ΔNFR↓ deviations in ``phase_messages``.

Use ``--config`` to point to an alternative file on a per-invocation
basis:

```bash
tnfr-lfs --config configs/tnfr-lfs.stint.toml analyze stint.jsonl
```

## Quickstart script

The repository ships with ``examples/quickstart.sh`` which executes the
end-to-end flow (CSV → baseline → analyze → suggest → report → write-set)
using the bundled synthetic stint.  The script stores artefacts under
``examples/out`` and generates a quick ASCII plot of the Sense Index
series to visualise the lap at a glance.
