# Command Line Interface

Install the project in editable mode and run:

```
pip install -e .
tnfr-lfs telemetry.csv --export csv
```

The CLI ingests the telemetry file, computes EPI metrics, and prints the
selected export format to stdout.

## Subcommands

The ``tnfr-lfs`` executable organises the workflow into five subcommands:

### ``baseline``

Capture telemetry from live UDP streams or from historical CSV data and
persist it as a baseline for subsequent analysis.  Live capture expects
the simulation to broadcast telemetry through:

* **OutSim UDP** – default port ``4123``
* **OutGauge UDP** – default port ``3000``
* **InSim UDP** – optional control messages through port ``29999`` when a
  simulator requires a handshake (no traffic is sent by TNFR-LFS but the
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
exporter (``json`` by default).

### ``suggest``

Runs the rule-based recommendation engine on top of an existing baseline
and exports the suggestions using the chosen exporter.  Use
``--car-model`` and ``--track`` to load different recommendation profiles.

### ``report``

Combines the orchestration operators with the exporters registry to
produce explainable ΔNFR/Sense Index reports.  ``--target-delta`` and
``--target-si`` define the desired objectives.

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
