# Command Line Interface

Install the project in editable mode and run:

```
pip install -e .
tnfr-lfs telemetry.csv --export csv
```

The CLI ingests the telemetry file, computes EPI metrics, and prints the
selected export format to stdout. All TNFR indicators (`ΔNFR`,
`ΔNFR_lat`, `ν_f`, `C(t)` and nodal derivatives) are computed from the
Live for Speed OutSim/OutGauge telemetry streams, so make sure the
simulator exposes both feeds before running any subcommand.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】

## Subcommands

The ``tnfr-lfs`` executable (part of the TNFR × LFS toolkit) organises the workflow into eight subcommands:

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

The threshold library also accepts optional ``[car.track.phase_weights.<phase>]``
tables inside ``data/threshold_profiles.toml``.  Each table defines
multipliers applied to the heuristic phase weighting map derived during
segmentation; values greater than ``1.0`` emphasise the corresponding
node when computing ν_f and Sense Index penalties, while numbers below
``1.0`` soften its influence.  For example, favouring tyres and suspension
in Valencia looks like:

```toml
[generic_gt.valencia.phase_weights.entry]
__default__ = 1.1
tyres = 1.35
suspension = 1.2
```

When the CLI resolves a profile it forwards these weights to
``segment_microsectors`` and the orchestration pipeline so ΔSi/ΔNFR
calculations honour the tuned nodal emphasis.

### ``osd``

Renders the live ΔNFR HUD inside Live for Speed using an ``IS_BTN`` overlay.
Before launching the command ensure the simulator exposes the three
telemetry interfaces:

* ``/outsim 1 127.0.0.1 4123`` – activates the OutSim UDP stream.
* ``/outgauge 1 127.0.0.1 3000`` – exposes the OutGauge dashboard data.
* ``/insim 29999`` – opens the InSim TCP control port required by the HUD.

The HUD cycles through three pages (press the button to advance):

* **Página A** – curva y fase actuales con la insignia ``ν_f`` (muy
  baja/óptima/…), un medidor ΔNFR con tolerancias ``[--^--]`` integrado en la
  línea de objetivos, el arquetipo activo (``hairpin``, ``medium``, ``fast`` o
  ``chicane``) con sus referencias ΔNFR∥/⊥ y pesos de detune, el resumen dinámico
  ``Si↺`` de la memoria sensorial y la banda ``ν_f~▁▃▆`` que muestra la evolución
  reciente de la frecuencia natural.  Debajo se representa la barra de
  coherencia ``C(t) ███░`` normalizada con el objetivo de Si del perfil actual y
  se mantiene el indicador ``Δaero`` que refleja el desequilibrio ΔNFR entre
  ejes a alta velocidad.  Cuando los eventos de operador ``OZ``/``IL`` superan
  sus umbrales ΔNFR contextualizados se añade un medidor «ΔNFR frenada» con la
  superficie dominante (``low_grip``/``neutral``/``high_grip``) y el pico observado
  frente al umbral calculado para el microsector, facilitando el ajuste
  inmediato del ``brake_bias_pct`` desde el propio HUD.  En los microsectores
  donde la densidad estructural y las cargas permanecen por debajo de los
  umbrales configurados aparece además la línea «Silencio …» indicando el
  porcentaje de cobertura y la densidad media observada para ese estado latente.
  Si la racha de microsectores silenciosos se extiende durante varios segmentos
  consecutivos, la cabecera añade un aviso «no tocar» reutilizando el detector
  ``detect_silencio`` para subrayar que conviene mantener los reglajes actuales.
  Bajo los indicadores de amortiguación se añade ahora la línea «CPHI»,
  que colorea cada rueda con 🟢/🟠/🔴 según los umbrales compartidos y añade
  un «+» cuando el valor supera el objetivo verde para atacar la vuelta.
* **Página B** – encabezado con la etiqueta de banda ``ν_f`` y su clasificación,
  junto a la barra de coherencia ``C(t)`` y la onda ``ν_f~`` agregada para la
  última vuelta.  Debajo se listan las top‑3 contribuciones nodales a
  |ΔNFR↓| con barras ASCII de anchura fija y el modo resonante dominante
  (frecuencia, ratio ``ρ`` y clasificación).
* **Página C** – arranca con ``Si plan`` para exponer el estado de memoria del
  piloto y a continuación ofrece la pista rápida para el operador activo: las
  2–3 acciones de setup priorizadas por ``SetupPlanner`` junto a su efecto
  esperado y una guía aero (“Aero …”) que resume si conviene reforzar el alerón
  delantero o trasero en la recta.

The overlay uses the default layout ``left=80, top=20, width=40, height=16``
and every page is trimmed to 239 bytes so it fits within the
``OverlayManager.MAX_BUTTON_TEXT`` limit.  Override the layout with
``--layout-left``/``--layout-top``/``--layout-width``/``--layout-height`` if
another mod already occupies that region.

### Checklist operativo en HUD/CLI

La página C incorpora una línea «Checklist» que valida los objetivos de
operaciones: ``Si`` medio ≥ 0.75, integral ΔNFR ≤ 6.00 kN·s⁻¹, ``Head`` ≥
0.40 y Δμ ≤ 0.12. Cada bloque muestra ✅ cuando la métrica respeta el
umbral y ⚠️ cuando requiere actuación, reutilizando los mismos límites
que emplean el motor de reglas y las métricas de ventana.【F:tnfr_lfs/recommender/rules.py†L604-L615】【F:tnfr_lfs/recommender/search.py†L210-L236】【F:tnfr_lfs/core/metrics.py†L2558-L2575】【F:tnfr_lfs/cli/osd.py†L1477-L1567】

Refer to the [setup equivalence guide](setup_equivalences.md) for a
metric-by-metric breakdown (`ΔNFR_lat`, `ν_f`, `C(t)`) that matches the HUD
widgets and the recommendations emitted by ``osd``.

```bash
tnfr-lfs osd --host 127.0.0.1 --outsim-port 4123 --outgauge-port 3000 --insim-port 29999
```

``--update-rate`` controls the HUD refresh (5–10 Hz recommended) while
``--insim-keepalive`` defines how often keepalive packets are sent.  The
recommendation engine resolves the thresholds and phase hints using the
``--car-model`` and ``--track`` options, matching the behaviour of
``template``/``suggest``.

The phase hint appended to page A now blends operator messages with the
measured phase alignment when the archetype drifts away from its target and
highlights the longitudinal/lateral focus defined by the detune weights. The new
ΔNFR gauge (``[--^--]``) and the ``ν_f~`` sparkline quantify how far the current
stint deviates from the target envelope before the gradient line reports the
dominant frequency extracted from the steer-versus-yaw/lateral cross-spectrum
together with the measured phase offset ``θ`` y su coseno ``Siφ``, the structural
coherence index ``C(t)`` and the frequency badge derived from the vehicle
category (e.g. GT ``1.9–2.2 Hz``). Justo debajo se imprime ``Δaero`` para que el
ingeniero visualice cuánto difiere la carga delantera respecto a la trasera a
alta y baja velocidad.

Natural frequency bands can be customised through
``NaturalFrequencySettings.frequency_bands`` and per-car categories handled by
``ProfileManager``.  The resulting badge (muy baja/óptima/…) and the structural
index ``C(t)`` are scaled with the profile objectives, so GT stints and fórmula
setups share a consistent vocabulary when reviewing telemetry.

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

Each microsector entry under ``filtered_measures`` now embeds a nested ``cphi``
block mirroring the :class:`~tnfr_lfs.core.metrics.CPHIReport`. The export
includes the red/amber/green thresholds and the ``optimal`` flag so downstream
dashboards can reuse the same tyre-health semantics shown in the HUD without
deriving bands manually.

When a configuration pack is available the JSON exporter also emits
``car`` and ``tnfr_targets`` sections. The former mirrors the metadata
resolved via ``data/cars`` (abbreviation, drivetrain, rotation group…)
while the latter surfaces the TNFR objectives declared in
``data/profiles`` so dashboards can expose the active targets alongside
the time-series.

### ``suggest``

Runs the rule-based recommendation engine on top of an existing baseline
and exports the suggestions using the chosen exporter.  Use
``--car-model`` and ``--track`` to load different recommendation profiles.
The exported payload mirrors ``analyze`` by embedding
``phase_messages`` and the ``reports`` artefacts so the tuning rationale
remains transparent when sharing the recommendations.

Pack-aware runs expose the same ``car`` and ``tnfr_targets`` sections as
``analyze`` to capture the resolved TNFR objectives inside the exported
payload.

### ``report``

Combines the orchestration operators with the exporters registry to
produce explainable ΔNFR/Sense Index reports.  ``--target-delta`` and
``--target-si`` define the desired objectives.  The emitted report now
exposes the same telemetry artefacts as ``analyze``/``suggest`` under the
``reports`` key so downstream automations (dashboards or notebooks) can
ingest the Sense Index map, modal resonance breakdown and the new
coherence/operator/bifurcation summaries while keeping the command line
output concise.  Use ``--report-format`` to decide whether those artefacts
are persisted as JSON payloads, Markdown tables or lightweight ASCII
visualisations.

The additional artefacts provide:

* ``coherence_map`` – mean/peak coherence per microsector with the
  reconstructed distance along the lap.
* ``operator_trajectories`` – structural timelines for ``AL``/``OZ``/``IL``/
  ``SILENCIO`` detections extraídos de ``Microsector.operator_events`` junto
  con la cobertura del nuevo estado latente de silencio.
* ``delta_bifurcations`` – ΔNFR sign changes and derivative summaries on
  the structural axis to highlight bifurcation hotspots.

### ``write-set``

Creates a setup plan by blending the optimisation-aware search module
with the rule engine.  The resulting payload follows the
``tnfr_lfs.exporters.setup_plan.SetupPlan`` schema, including:

When the CLI has access to a configuration pack (``paths.pack_root`` or
``--pack-root``) the JSON payload embeds the vehicle metadata and pack
objectives under ``car`` and ``tnfr_targets`` respectively.

```
{
  "car_model": "generic_gt",
  "session": "FP1",
  "changes": [
    {"parameter": "rear_wing_angle", "delta": -1.0, "rationale": "Reduce drag", "expected_effect": "Higher top speed"}
  ],
  "rationales": ["Reduce drag"],
  "expected_effects": ["Higher top speed"],
  "sensitivities": {
    "sense_index": {"rear_wing_angle": 0.042},
    "delta_nfr_integral": {"rear_wing_angle": -0.075}
  },
  "phase_sensitivities": {
    "entry": {"delta_nfr_integral": {"rear_wing_angle": -0.052}}
  },
  "aero_guidance": "Alta velocidad → libera alerón trasero/refuerza delantero",
  "aero_metrics": {"low_speed_imbalance": 0.03, "high_speed_imbalance": -0.35}
}
```

Exporters normalise the dataclasses into JSON, CSV, or Markdown depending
on the ``--export`` flag.  The Markdown exporter deduplicates rationales
and expected effects to provide a readable handover document while also
surfacing the empirical Jacobian (ΔSi/Δp and Δ∫|ΔNFR|/Δp) gathered during
the micro-delta experiments.

Use :meth:`tnfr_lfs.io.profiles.ProfileManager.update_aero_profile` to persist
the ``race`` and ``stint_save`` aero targets mentioned in the HUD.  The rule
engine reads them back via the profile snapshot so the new aero coherence
operator only raises wing tweaks when the stored baseline (baja velocidad)
permanece dentro de tolerancia.
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
pack_root = "~/tnfr-pack"

[limits.delta_nfr]
entry = 0.5
apex = 0.4
exit = 0.6
```

This configuration adjusts the default UDP ports used by ``baseline``,
selects the exporter for analytics/reporting, sets the default
car/track for ``suggest`` and overrides the tolerance used when
highlighting ΔNFR↓ deviations in ``phase_messages``.

When ``pack_root`` points to a TNFR × LFS pack (a directory containing ``config/global.toml`` together with ``data/cars`` and ``data/profiles``) the CLI resolves car metadata and TNFR objectives from that bundle. The ``--pack-root`` flag overrides the configured value for a single invocation.

### Preparing ``cfg.txt``

Before running the CLI against Live for Speed you must enable the telemetry broadcasters inside
``cfg.txt`` (located in the simulator root directory):

1. Edit the ``OutSim`` block to contain ``Mode 1``, ``Port 4123`` y ``IP 127.0.0.1``.  Estos valores
   coinciden con las opciones por defecto de ``tnfr-lfs baseline`` y pueden activarse al vuelo con
   ``/outsim 1 127.0.0.1 4123`` desde el chat del simulador.
2. Añade ``OutSim Opts ff`` para incluir el ID del jugador, las entradas de piloto y el bloque de ruedas
   (fuerzas, cargas, deflexión) que requieren los cálculos de ΔNFR y ν_f.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】
3. Actualiza el bloque ``OutGauge`` con ``Mode 1``, ``Port 3000`` e ``IP 127.0.0.1``; Live for Speed acepta
   ``/outgauge 1 127.0.0.1 3000`` como atajo para la misma configuración.
4. Reserva un puerto ``InSim`` estableciendo ``InSim Port 29999`` (o el valor preferido) y la IP del equipo
   donde se ejecuta TNFR × LFS.  Lanza ``/insim 29999`` al iniciar una sesión para realizar el *handshake* que
   requieren algunos mods de Live for Speed.

Guarda los cambios y ejecuta ``tnfr-lfs diagnose /ruta/a/cfg.txt`` para confirmar que los valores son
coherentes y que ningún servicio está bloqueando los puertos UDP.

### Telemetry field checklist

- **ΔNFR / ΔNFR_lat** – requiere OutSim para exponer cargas verticales,
  aceleraciones, fuerzas y deflexión de rueda junto a OutGauge para
  obtener `rpm`, pedales y luces ABS/TC.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (frecuencia natural)** – depende de la distribución de cargas,
  `slip_ratio`/`slip_angle`, velocidad y `yaw_rate` calculados desde
  OutSim, con señales de estilo (`throttle`, `gear`) procedentes de
  OutGauge.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】
- **C(t) (coherencia estructural)** – se alimenta de la misma mezcla de
  señales OutSim para ΔNFR y de los coeficientes `mu_eff_*`, además de
  las banderas ABS/TC de OutGauge que el fusionador traduce en eventos de
  bloqueo.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】

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
