# Command Line Interface

Install the project in editable mode and run:

```
pip install -e .
tnfr-lfs baseline runs/test1.jsonl data/test1.raf --format jsonl
```

The example converts the bundled RAF capture (`data/test1.raf`) into a
JSONL baseline. The CLI inspects the suffix of every telemetry source
(`.raf`, `.csv`, `.jsonl`, `.parquet`…) and automatically routes it
through the appropriate parser. CSV inputs continue to use the
`--simulate` flag (`tnfr-lfs baseline runs/demo.jsonl --simulate data/BL1_XFG_baseline.csv`),
while RAF captures are parsed natively through `tnfr_lfs.io.read_raf`
and `tnfr_lfs.io.raf_to_telemetry_records`. All TNFR indicators (`ΔNFR`, the
nodal projections `∇NFR∥`/`∇NFR⊥`, `ν_f`, `C(t)` and nodal derivatives)
are computed from the Live for Speed OutSim/OutGauge telemetry streams,
so make sure the simulator exposes both feeds before running any
subcommand. Remember that the `∇NFR∥`/`∇NFR⊥` components capture the
projection of the nodal gradient; whenever you need to interpret
absolute loads, cross-check the recommendations with the direct
`Fz`/`ΔFz` channels from OutSim.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】

Replay Analyzer exports can be ingested with `--replay-csv-bundle`:

```bash
tnfr-lfs baseline runs/test1.jsonl --replay-csv-bundle data/test1.zip --format jsonl
```

The bundle flag accepts either a directory or a ZIP produced by Replay Analyzer and
normalises the signal names to the canonical TNFR × LFS layout (`wheel_load_fl`,
`slip_ratio_rr`, `timestamp`, …). During ingestion the reader also converts the
`distance` index to floats, expands `speed_kmh` into the `speed` channel (m/s) and
maps the G-based accelerations/drift angle into SI units before generating telemetry
records.【F:tnfr_lfs/io/replay_csv_bundle.py†L92-L213】

## Logging

All CLI subcommands emit structured log entries through the standard
library logger. By default the CLI renders each message as a single JSON
object written to **stderr** at the `info` level. Control the verbosity
and destination via the top-level flags:

* `--log-level` – accepts standard logging levels (`debug`, `info`,
  `warning`, `error`, …).
* `--log-output` – choose between `stderr`, `stdout` or a filesystem path
  to append newline-delimited logs.
* `--log-format` – select `json` (default) or `text` for a human-readable
  formatter.

The same options can be persisted inside `tnfr-lfs.toml` under
`[logging]`. See the updated template at the project root for an example
with sane defaults.

### Monitoring capture metrics

Live capture commands emit an `info`-level `capture.completed` entry when
the OutSim/OutGauge loop finishes. The log bundles counters for
`attempts`, `samples`, `dropped_pairs`, the total capture `duration` and
the per-stream timeout/ignored-host totals so dashboards can track data
quality in real time. Follow-up CLI messages (for example the
`baseline.saved` event) also include a `capture_metrics` object with the
same payload to simplify downstream ingestion.

When using JSON logging, pipe the stream directly to your collector or a
CLI filter. For example, to forward the metrics to `jq` before shipping
them to an aggregator:

```bash
tnfr-lfs baseline runs/baseline.jsonl \
  --log-format json \
  --log-output stdout \
  | jq 'select(.event == "capture.completed")'
```

Any warning-level entries emitted by the UDP clients (for exhausted
retries or datagrams from unexpected hosts) can be routed through the
same pipeline so alerting systems receive immediate context about packet
loss or misconfigured endpoints.

## Subcommands

The ``tnfr-lfs`` executable (part of the TNFR × LFS toolkit) organises the workflow into eight subcommands:

### ``template``

Generates ΔNFR objectives together with slip/yaw windows for a given
car/track profile.  The command resolves the same threshold library used
by the recommendation engine (backed by ``data/threshold_profiles.toml``)
and emits a TOML snippet that can be dropped into ``tnfr-lfs.toml``.

```bash
tnfr-lfs template --car FZR --track AS5 > presets/fzr_as5.toml
tnfr-lfs --config presets/fzr_as5.toml analyze stint.jsonl
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
at Aston Grand Prix (AS5) looks like:

```toml
[FZR.AS5.phase_weights.entry]
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

* **Page A** – shows the current corner and phase with the ``ν_f`` badge (very
  low/optimal/…), a ΔNFR gauge with ``[--^--]`` tolerances integrated into the
  target line, the active archetype (``hairpin``, ``medium``, ``fast`` or
  ``chicane``) with its ∇NFR∥/∇NFR⊥ projection references and detune weights, the
  dynamic ``Si↺`` sensory memory summary, and the ``ν_f~▁▃▆`` band that tracks
  the recent evolution of the natural frequency. Below that sits the
  normalised coherence bar ``C(t) ███░`` aligned with the current profile target
  and the ``Δaero`` indicator that reflects the ΔNFR imbalance between axles at
  high speed. When ``OZ``/``IL`` operator events exceed their contextual ΔNFR
  thresholds, the HUD appends a “ΔNFR braking” gauge with the dominant surface
  (``low_grip``/``neutral``/``high_grip``) and the peak observed versus the
  threshold computed for the microsector, enabling immediate ``brake_bias_pct``
  adjustments from the HUD. In microsectors where the structural density and the
  loads stay below the configured thresholds an extra “Silence …” line appears,
  reporting coverage percentage and mean density for that latent state. If the
  silent streak spans several consecutive segments the header adds a “do not
  touch” warning via the ``detect_silencio`` detector to underline that the
  current setup should be preserved. Beneath the damping indicators the new
  “CPHI” line colours each wheel with 🟢/🟠/🔴 according to the shared thresholds
  and appends a “+” when the value clears the green objective before attacking
  the lap.
* **Page B** – displays the ``ν_f`` band label and its classification alongside
  the ``C(t)`` coherence bar and the aggregated ``ν_f~`` waveform for the last
  lap. The section below lists the top-three nodal contributions to |ΔNFR↓|
  using fixed-width ASCII bars together with the dominant resonant mode
  (frequency, ratio ``ρ`` and classification).
* **Page C** – opens with ``Si plan`` to expose the driver memory state and then
  surfaces the quick brief for the active operator: the two or three setup
  actions prioritised by ``SetupPlanner`` with their expected effect and an aero
  cue (“Aero …”) that summarises whether the front or rear wing should be
  reinforced along the straight.

The overlay uses the default layout ``left=80, top=20, width=40, height=16``
and every page is trimmed to 239 bytes so it fits within the
``OverlayManager.MAX_BUTTON_TEXT`` limit.  Override the layout with
``--layout-left``/``--layout-top``/``--layout-width``/``--layout-height`` if
another mod already occupies that region.

### HUD/CLI operations checklist

Page C adds a “Checklist” line that validates the operations targets:
mean ``Si`` ≥ 0.75, ΔNFR integral ≤ 6.00 kN·s⁻¹, ``Head`` ≥ 0.40, and Δμ ≤
0.12. Each block shows ✅ when the metric respects the threshold and ⚠️ when
it requires attention, reusing the same limits enforced by the rule engine and
the window metrics.【F:tnfr_lfs/recommender/rules.py†L604-L615】【F:tnfr_lfs/recommender/search.py†L210-L236】【F:tnfr_lfs/core/metrics.py†L2558-L2575】【F:tnfr_lfs/cli/osd.py†L1477-L1567】

Refer to the [setup equivalence guide](setup_equivalences.md) for a
metric-by-metric breakdown (`∇NFR⊥`, `ν_f`, `C(t)`) that matches the HUD
widgets and the recommendations emitted by ``osd``. That guide also explains
when to back those projections with the `Fz`/`ΔFz` channels if you need
absolute load adjustments.

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
together with the measured phase offset ``θ`` and its cosine ``Siφ``, the
structural coherence index ``C(t)`` and the frequency badge derived from the
vehicle category (e.g. GT ``1.9–2.2 Hz``). Just below it prints ``Δaero`` so the
engineer can see how much the front load differs from the rear at both low and
high speed.

Natural frequency bands can be customised through
``NaturalFrequencySettings.frequency_bands`` and per-car categories handled by
``ProfileManager``.  The resulting badge (very low/optimal/…) and the structural
index ``C(t)`` are scaled with the profile objectives, so GT stints and formula
setups share a consistent vocabulary when reviewing telemetry.

### ``diagnose``

Verifies that the Live for Speed ``cfg.txt`` file exposes OutSim/OutGauge data streams and that the UDP
ports are reachable.  Run ``tnfr-lfs diagnose /path/to/LFS/cfg.txt`` before a session to receive:

* Warnings when ``OutSim Mode`` or ``OutGauge Mode`` are not set to ``1``.
* Details about the configured ``InSim`` port (e.g. ``29999``) so you can launch ``/insim 29999``
  from the simulator chat when it is time to initialise telemetry.
* A quick socket check that reports whether ports ``4123`` (OutSim) and ``3000`` (OutGauge)
  can be reserved on ``127.0.0.1`` or if another process already owns them.

The output ends in ``Status: ok`` when everything is available; otherwise it prints
a summary of the failures detected and returns a non-zero exit code.

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
  ``SILENCIO`` detections extracted from ``Microsector.operator_events`` along
  with the coverage of the new silent latent state.
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
  "car_model": "FZR",
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
  "aero_guidance": "High speed → trim rear wing / reinforce front",
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
operator only raises wing tweaks when the stored baseline (low speed)
remains within tolerance.
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
car_model = "FZR"
track = "AS5"

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

### Brake thermal proxy modes

The brake thermal proxy honours the ``mode`` declared in ``[thermal.brakes]`` and can be overridden per session with the ``TNFR_LFS_BRAKE_THERMAL`` environment variable. The accepted values are ``auto`` (default), ``off`` and ``force``.【F:tnfr_lfs/acquisition/fusion.py†L616-L733】【F:tnfr_lfs/acquisition/fusion.py†L1064-L1126】

- ``auto`` – consumes OutGauge brake temperatures whenever Live for Speed broadcasts plausible values, seeding the estimator with them. When the stream is disabled or sends the ``0 °C`` marker, the proxy keeps integrating brake energy so fade indicators remain continuous.【F:tnfr_lfs/acquisition/fusion.py†L1078-L1107】
- ``off`` – bypasses the proxy and returns the raw OutGauge values (or their last valid samples) even if the feed drops; useful when validating cars with hardware sensors or external thermal models.【F:tnfr_lfs/acquisition/fusion.py†L1107-L1117】
- ``force`` – ignores OutGauge entirely and relies on the estimator for all wheels, ideal for legacy cars without brake sensors or synthetic runs where only OutSim loads are present.【F:tnfr_lfs/acquisition/fusion.py†L1117-L1126】

### Preparing ``cfg.txt``

Before running the CLI against Live for Speed you must enable the telemetry broadcasters inside
``cfg.txt`` (located in the simulator root directory):

1. Edit the ``OutSim`` block to contain ``Mode 1``, ``Port 4123`` and ``IP 127.0.0.1``. These values
   match the defaults used by ``tnfr-lfs baseline`` and can be enabled on the fly with
   ``/outsim 1 127.0.0.1 4123`` from the simulator chat.
2. Add ``OutSim Opts ff`` to include the player ID, driver inputs, and the wheel packet
   (forces, Fz loads, deflection) required to compute ΔNFR and ν_f.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】
3. Update the ``OutGauge`` block with ``Mode 1``, ``Port 3000`` and ``IP 127.0.0.1``; Live for Speed accepts
   ``/outgauge 1 127.0.0.1 3000`` as a shortcut for the same configuration.
4. Reserve an ``InSim`` port by setting ``InSim Port 29999`` (or your preferred value) and the IP of the machine
   running TNFR × LFS. Launch ``/insim 29999`` when starting a session to perform the handshake required by
   some Live for Speed mods.

Save the changes and run ``tnfr-lfs diagnose /path/to/cfg.txt`` to confirm the values are consistent and that no service is blocking the UDP ports.

### Telemetry field checklist

- **ΔNFR (nodal gradient) / ∇NFR⊥ (lateral projection)** – require OutSim to
  expose Fz loads, accelerations, forces, and wheel deflection together with
  OutGauge to obtain `rpm`, pedal positions, and ABS/TC lights. These signals feed
  the nodal gradient; consult the `Fz`/`ΔFz` channels if you need to quantify
  absolute loads before applying an adjustment.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (natural frequency)** – depends on the distribution of Fz loads,
  `slip_ratio`/`slip_angle`, speed, and `yaw_rate` computed from OutSim, along
  with style signals (`throttle`, `gear`) emitted by OutGauge.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】
- **C(t) (structural coherence)** – consumes the same blend of OutSim signals
  used for ΔNFR together with the `mu_eff_*` coefficients and the ABS/TC flags
  from OutGauge that the fusion module translates into lockup events.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】

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
If you prefer to rehearse with real RAF telemetry, swap the dataset in the
script (or call ``tnfr-lfs baseline`` manually) and point it to
``data/test1.raf``; the CLI will pick the right parser automatically.
