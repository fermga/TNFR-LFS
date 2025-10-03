# Command Line Interface

Install the project in editable mode and run:

```
pip install -e .
tnfr-lfs telemetry.csv --export csv
```

The CLI ingests the telemetry file, computes EPI metrics, and prints the
selected export format to stdout.

## Subcommands

The ``tnfr-lfs`` executable (part of the TNFR × LFS toolkit) organises the workflow into seven subcommands:

### ``template``

Generates ΔNFR objectives together with slip/yaw windows for a given
car/track profile.  The command resolves the same threshold library used
by the recommendation engine (backed by ``data/threshold_profiles.toml``)
and emits a TOML snippet that can be dropped into ``tnfr-lfs.toml``.

```bash
tnfr-lfs template --car generic_gt --track valencia > presets/valencia.toml
tnfr-lfs --config presets/valencia.toml analyze stint.jsonl
```

The generated file contains ``[analyze.phase_templates]`` and
``[report.phase_templates]`` sections with ΔNFR targets and slip/yaw
windows for each phase of the corner.  It also mirrors the
``limits.delta_nfr`` tolerances used by ``phase_messages`` so that
``analyze`` and ``report`` include the preset alongside the usual
metrics.

### ``diagnose``

Verifies that the Live for Speed ``cfg.txt`` file exposes OutSim/OutGauge data streams and that the UDP
ports are reachable.  Run ``tnfr-lfs diagnose /ruta/a/LFS/cfg.txt`` before a session to receive:

* Advertencias si ``OutSim Mode`` u ``OutGauge Mode`` no están establecidos en ``1``.
* Detalles sobre el puerto ``InSim`` configurado (p. ej. ``29999``) para que puedas lanzar ``/insim 29999``
  desde el chat del simulador cuando toque inicializar la telemetría.
* Una comprobación rápida de sockets que informa si los puertos ``4123`` (OutSim) y ``3000`` (OutGauge)
  pueden reservarse en ``127.0.0.1`` o si otro proceso los está ocupando.

La salida termina en ``Estado: correcto`` cuando todo está disponible; en caso contrario imprime un
resumen de los fallos detectados y devuelve un código de error.

### ``baseline``

Capture telemetry from live UDP streams or from historical CSV data and
persist it as a baseline for subsequent analysis.  Live capture expects
the simulation to broadcast telemetry through:

* **OutSim UDP** – default port ``4123``
* **OutGauge UDP** – default port ``3000``
* **InSim TCP** – optional control channel on port ``29999`` used when enabling the
  ``--overlay`` flag to show contextual information inside Live for Speed

All clients connect to ``127.0.0.1`` by default and can be redirected via
``--host`` and the corresponding ``--outsim-port``/``--outgauge-port``
options.  When the simulator is not running on the same machine ensure
firewall rules allow inbound UDP traffic on the configured ports.

To ingest prerecorded telemetry, pass ``--simulate`` with the path to a
CSV file containing OutSim-compatible columns.  The ``--format`` option
persists the baseline as ``jsonl`` or ``parquet``.

When ``output`` is omitted the command stores the capture under
``runs/<car>_<track>_<YYYYMMDD_HHMMSS_mmmmmm>.jsonl`` using the car/track
names resolved from the configuration.  ``--output-dir`` overrides the
destination directory and an existing folder can be passed as the
positional argument to trigger the same behaviour.  Explicit file paths
continue to be honoured, while ``--force`` skips collision checks when a
file already exists.

```bash
tnfr-lfs baseline --simulate stint.csv
tnfr-lfs baseline runs/session-a --simulate stint.csv
tnfr-lfs baseline custom.parquet --simulate stint.csv --format parquet
```

#### Overlay best practices

The ``--overlay`` flag renders a compact ``IS_BTN`` panel while capturing live
telemetry.  Follow these guidelines to avoid interfering with the driving
session:

* **Avoid permanent buttons** – the overlay automatically disappears when the
  capture ends; do not reuse the button slot for unrelated information.
* **Prefer the HUD margins** – Live for Speed keeps the area around ``L=10`` and
  ``T=10`` mostly free of critical information.  The CLI uses a 180×30 panel in
  that region by default; adjust the layout if another app already occupies it.
* **Keep the copy short** – ``IS_BTN`` packets accept a maximum of 239 visible
  characters.  Use concise status lines (e.g. duration, sample targets) to stay
  within the protocol limits and avoid truncated messages.

### ``analyze``

Reads a baseline (captured or simulated) and computes ΔNFR/Sense Index
metrics.  The subcommand uses the core operators to orchestrate
telemetry processing and emits a structured payload via the selected
exporter (``json`` by default).  The payload now includes a
``phase_messages`` array that summarises per-phase ΔNFR↓ deviations with
actionable hints together with a ``reports`` block.  Each invocation
persists ``out/<baseline-stem>/sense_index_map.json`` and
``out/<baseline-stem>/modal_resonance.json`` (configurable via the
``paths.output_dir`` setting) so external tooling can reuse the Sense
Index heatmap and modal resonance analysis without re-running the CLI.

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
ingest the Sense Index map and modal resonance breakdown while keeping
the command line output concise.

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

### Preparing ``cfg.txt``

Before running the CLI against Live for Speed you must enable the telemetry broadcasters inside
``cfg.txt`` (located in the simulator root directory):

1. Edit the ``OutSim`` block to contain ``Mode 1``, ``Port 4123`` and ``IP 127.0.0.1``.  These values
   match the default options of ``tnfr-lfs baseline`` and can be activated on the fly with the command
   ``/outsim 1 127.0.0.1 4123`` desde el chat del simulador.
2. Update the ``OutGauge`` block with ``Mode 1``, ``Port 3000`` and ``IP 127.0.0.1``; Live for Speed acepta
   ``/outgauge 1 127.0.0.1 3000`` como atajo para la misma configuración.
3. Reserva un puerto ``InSim`` estableciendo ``InSim Port 29999`` (o el valor preferido) y la IP del equipo
   donde se ejecuta TNFR × LFS.  Lanza ``/insim 29999`` al iniciar una sesión para realizar el handshake que
   requieren algunos mods de Live for Speed.

Guarda los cambios y ejecuta ``tnfr-lfs diagnose /ruta/a/cfg.txt`` para confirmar que los valores son
coherentes y que ningún servicio está bloqueando los puertos UDP.

Use ``--config`` to point to an alternative file on a per-invocation
basis:

```bash
tnfr-lfs --config configs/tnfr-lfs.stint.toml analyze stint.jsonl
```

## Quickstart script

The repository ships with ``examples/quickstart.sh`` which executes the
end-to-end flow (CSV → baseline → analyze → suggest → report → write-set)
using the bundled baseline dataset ``data/BL1_XFG_baseline.csv``.  The
script stores artefacts under ``examples/out`` and generates a quick ASCII
plot of the Sense Index series to visualise the lap at a glance.  The
dataset mirrors a short 17-sample stint captured at BL1/XFG pace so that
the quickstart replicates realistic ΔNFR and Sense Index oscillations.
