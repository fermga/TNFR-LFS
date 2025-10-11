# API Reference

## Configuration utilities

### `tnfr_lfs.ingestion.config_loader`

The configuration loader resolves car manifests and profile bundles shipped
with the project.  Use :func:`tnfr_lfs.ingestion.config_loader.load_cars` and
:func:`tnfr_lfs.ingestion.config_loader.load_profiles` to populate immutable mappings of
cars and tuning profiles, then :func:`tnfr_lfs.ingestion.config_loader.resolve_targets`
to expose the ``targets``/``policy``/``recommender`` sections for the selected
vehicle. 【F:tnfr_lfs/ingestion/config_loader.py†L38-L148】

### `tnfr_lfs.ingestion.track_loader`

The track loader complements the car/profile helpers by exposing the bundled
track manifests, phase-weight profiles, and per-combination modifiers.  The
module exposes:

```python
from tnfr_lfs.ingestion.track_loader import (
    load_track,
    load_track_profiles,
    load_modifiers,
    assemble_session_weights,
)

track = load_track("SO")  # -> tnfr_lfs.ingestion.track_loader.Track
profiles = load_track_profiles()
modifiers = load_modifiers()
session = assemble_session_weights(
    "gtr_mid",  # car profile or group
    track.configs["SO1"].track_profile,
    track_profiles=profiles,
    modifiers=modifiers,
)
```

* :func:`load_track` parses the ``[config.*]`` sections of the requested TOML
  manifest, normalising optional lengths and surface labels into
  :class:`tnfr_lfs.ingestion.track_loader.TrackConfig` instances accessible from the
  returned :class:`tnfr_lfs.ingestion.track_loader.Track`. 【F:tnfr_lfs/ingestion/track_loader.py†L89-L147】
* :func:`load_track_profiles` discovers ``data/track_profiles/*.toml`` and
  returns a mapping keyed by ``meta.id`` (or filename) with normalised phase
  weights and hints ready for consumption. 【F:tnfr_lfs/ingestion/track_loader.py†L149-L175】
* :func:`load_modifiers` scans ``modifiers/combos/*.toml`` and maps each
  ``(car_profile, track_profile)`` pair to its scale factors and optional
  hints. 【F:tnfr_lfs/ingestion/track_loader.py†L178-L210】
* :func:`assemble_session_weights` merges a car profile with a track profile,
  applying any matching modifier scales to produce a ``{"weights", "hints"}``
  payload suitable for the CLI and HUD pipelines. 【F:tnfr_lfs/ingestion/track_loader.py†L213-L247】

## Ingestion

!!! note
    The ingestion surface is consolidated under the
    `tnfr_lfs.ingestion` namespace, which exposes live UDP clients,
    CSV/RAF readers and fusion helpers via modules such as
    `tnfr_lfs.ingestion.live`, `tnfr_lfs.ingestion.replay` and
    `tnfr_lfs.ingestion.fusion`. Import from these modules to wire
    data capture directly into the analytics stack.

### `tnfr_lfs.ingestion.live.OutSimClient`

Reads OutSim-style telemetry from CSV sources and returns a list of
:class:`tnfr_lfs.core.epi.TelemetryRecord` objects.  By default the
client validates the column order, converts values to floats, and keeps
any optional column that is missing in the source as ``math.nan`` instead
of fabricating estimates.【F:tnfr_lfs/ingestion/outsim_client.py†L87-L155】
This allows downstream exporters to surface "no data" when Live for
Speed does not broadcast the extra wheel block.

```
from math import isnan

from tnfr_lfs.ingestion.live import OutSimClient

client = OutSimClient()
records = client.ingest("stint.csv")
# Optional columns remain math.nan when the CSV omits them, so
# downstream metrics can flag "no data".
isnan(records[0].tyre_temp_fl)
```

### `tnfr_lfs.ingestion.live.TelemetryFusion`

Combine OutSim and OutGauge datagrams into enriched
:class:`~tnfr_lfs.core.epi.TelemetryRecord` samples or full
:class:`~tnfr_lfs.core.epi.EPIBundle` payloads.  The fusion layer loads the
default calibration table from ``tnfr_lfs/data/fusion_calibration.toml`` and
merges car/track specific overrides before scaling slip ratios, steering
geometry, μ estimators and suspension weighting for every packet.【F:tnfr_lfs/ingestion/fusion.py†L71-L206】【F:tnfr_lfs/data/fusion_calibration.toml†L1-L52】

Calibration hooks are designed for teams that maintain their own pack
overlays.  ``TelemetryFusion`` searches for a pack root by checking the
``TNFR_LFS_PACK_ROOT`` environment variable, the current working tree and
the installed resource bundle.  When it finds ``config/global.toml`` or a
``data/cars/*.toml`` manifest the fusion logic applies the ``[thermal.brakes]``
entries to seed the brake temperature estimator, allowing per-car defaults
and per-track overrides to live alongside other configuration artefacts.【F:tnfr_lfs/ingestion/fusion.py†L582-L709】
Set ``TNFR_LFS_BRAKE_THERMAL=off`` (or ``auto``/``force``) to coerce the
thermals into the desired mode without editing the pack.【F:tnfr_lfs/ingestion/fusion.py†L675-L709】

Instantiate the fuser once per session, then stream UDP packets into it.
Packets yielded by :class:`~tnfr_lfs.ingestion.outsim_udp.OutSimUDPClient`
and :class:`~tnfr_lfs.ingestion.outgauge_udp.OutGaugeUDPClient` are backed by
reusable pools; call :meth:`release` when you finish consuming a sample or
request immutable dataclasses via ``from_bytes(..., freeze=True)`` when a
copy is required.  ``fuse`` returns the latest ``TelemetryRecord`` while
``fuse_to_bundle`` feeds the bundled EPI extractor so downstream tooling can
immediately reuse the examples from the :mod:`tnfr_lfs.core.epi`
documentation.【F:tnfr_lfs/ingestion/fusion.py†L130-L320】

```python
from collections import deque
from tnfr_lfs.ingestion.live import (
    OutGaugePacket,
    OutGaugeUDPClient,
    OutSimPacket,
    OutSimUDPClient,
    TelemetryFusion,
)
from tnfr_lfs.core.epi import TelemetryRecord

fusion = TelemetryFusion()
records: list[TelemetryRecord] = []

with OutSimUDPClient(port=4123) as outsim, OutGaugeUDPClient(port=3000) as outgauge:
    outsim_buffer: deque[OutSimPacket] = deque()
    outgauge_buffer: deque[OutGaugePacket] = deque()
    while True:
        outsim_buffer.extend(outsim.drain_ready())
        outgauge_buffer.extend(outgauge.drain_ready())
        if not outsim_buffer:
            packet = outsim.recv()
            if packet:
                outsim_buffer.append(packet)
        if not outgauge_buffer:
            packet = outgauge.recv()
            if packet:
                outgauge_buffer.append(packet)

        while outsim_buffer and outgauge_buffer:
            outsim_packet = outsim_buffer.popleft()
            outgauge_packet = outgauge_buffer.popleft()
            try:
                record = fusion.fuse(outsim_packet, outgauge_packet)
                records.append(record)

                # Bridge straight into the EPI workflows used by the examples.
                bundle = fusion.fuse_to_bundle(outsim_packet, outgauge_packet)
                print(bundle.delta_nfr, bundle.sense_index)
            finally:
                outsim_packet.release()
                outgauge_packet.release()

        if outsim.statistics["loss_events"] or outgauge.statistics["loss_events"]:
            print("Loss detected", outsim.statistics, outgauge.statistics)
```

The example above illustrates the minimal glue required to bridge raw UDP
telemetry with the existing EPI tutorials: reuse the same ``TelemetryFusion``
instance, maintain a deque per stream via ``drain_ready`` so that out-of-order
packets can be reinserted automatically, and call ``fuse_to_bundle`` (or
``fusion.extractor.update``) whenever you need the aggregated ΔNFR/SI metrics.
Inspect :attr:`OutSimUDPClient.statistics` or
:attr:`OutGaugeUDPClient.statistics` to surface warnings when suspected loss
events occur.  Both UDP clients accept an optional ``max_batch`` argument to
limit how many datagrams are drained after each readiness notification, which
can help throttle exceptionally chatty streams while still returning already
decoded packets without additional polling.

An asynchronous workflow can drop the polling loop entirely by switching to
:class:`tnfr_lfs.ingestion.outsim_udp.AsyncOutSimUDPClient` and
:class:`tnfr_lfs.ingestion.outgauge_udp.AsyncOutGaugeUDPClient`.  Both classes
share the same reordering buffer and statistics interface, allowing existing
fusion code to run unchanged inside ``async`` tasks:

```python
import asyncio
from collections import deque
from tnfr_lfs.ingestion.live import TelemetryFusion
from tnfr_lfs.ingestion.outgauge_udp import AsyncOutGaugeUDPClient
from tnfr_lfs.ingestion.outsim_udp import AsyncOutSimUDPClient


async def stream_once(fusion: TelemetryFusion) -> None:
    async with AsyncOutSimUDPClient(port=4123) as outsim, AsyncOutGaugeUDPClient(port=3000) as outgauge:
        outsim_buffer = deque(outsim.drain_ready())
        outgauge_buffer = deque(outgauge.drain_ready())

        outsims, outgauges = await asyncio.gather(outsim.recv(), outgauge.recv())
        if outsims:
            outsim_buffer.append(outsims)
        if outgauges:
            outgauge_buffer.append(outgauges)

        if outsim_buffer and outgauge_buffer:
            fusion.fuse(outsim_buffer.popleft(), outgauge_buffer.popleft())


asyncio.run(stream_once(TelemetryFusion()))
```

## Core Analytics

### `tnfr_lfs.core` public surface

The `tnfr_lfs.core` namespace re-exports a curated set of analytics helpers.
Each submodule now defines ``__all__`` so that ``from tnfr_lfs.core.<module>
import *`` only surfaces the documented symbols:

* `tnfr_lfs.core.epi`: ``TelemetryRecord``, ``EPIExtractor``,
  ``NaturalFrequencyAnalyzer``, ``NaturalFrequencySnapshot``,
  ``NaturalFrequencySettings``, ``DeltaCalculator`` and
  ``delta_nfr_by_node``.【F:tnfr_lfs/core/epi.py†L39-L46】
* `tnfr_lfs.core.epi_models`: ``EPIBundle``.【F:tnfr_lfs/core/epi_models.py†L8-L8】
* `tnfr_lfs.core.coherence`: ``compute_node_delta_nfr`` and
  ``sense_index``.【F:tnfr_lfs/core/coherence.py†L12-L12】
* `tnfr_lfs.core.segmentation`: ``Goal``, ``Microsector``,
  ``detect_quiet_microsector_streaks``, ``microsector_stability_metrics`` and
  ``segment_microsectors``.【F:tnfr_lfs/core/segmentation.py†L67-L73】
* `tnfr_lfs.core.resonance`: ``ModalPeak``, ``ModalAnalysis`` and
  ``analyse_modal_resonance``.【F:tnfr_lfs/core/resonance.py†L12-L12】
* `tnfr_lfs.core.structural_time`: ``compute_structural_timestamps`` and
  ``resolve_time_axis``.【F:tnfr_lfs/core/structural_time.py†L10-L10】
* `tnfr_lfs.core.delta_utils` intentionally keeps ``__all__`` empty because the
  helper is not part of the public surface.【F:tnfr_lfs/core/delta_utils.py†L8-L8】

### `tnfr_lfs.core.archetypes`

Archetype targets capture the strategic intent the segmentation and
recommendation pipelines expect for each corner profile.  The module ships a
read-only :data:`~tnfr_lfs.core.archetypes.DEFAULT_ARCHETYPE_PHASE_TARGETS`
mapping covering hairpins, chicanes, medium- and high-speed archetypes and a
lightweight accessor named :func:`~tnfr_lfs.core.archetypes.archetype_phase_targets`
that falls back to the medium archetype for unknown keys.【F:src/tnfr_lfs/core/archetypes.py†L9-L91】

```python
from tnfr_lfs.core.archetypes import archetype_phase_targets

targets = archetype_phase_targets("hairpin")
apex_targets = targets["apex"]
apex_targets.nu_f  # -> 0.55
```

The targets feed the microsector segmentation heuristics documented in
:mod:`tnfr_lfs.core.segmentation` and are reused by the recommendation engine
when synthesising coaching prompts—see :doc:`setup_plan` for an end-to-end
overview.

### `tnfr_lfs.core.constants`

Utility constants centralise the canonical wheel identifiers and their derived
telemetry field names so that ingestion, segmentation and recommendation code
agree on a consistent naming scheme.【F:src/tnfr_lfs/core/constants.py†L9-L39】  The
immutable dictionaries (``MappingProxyType``) include wheel labels as well as
tyre/brake temperature and pressure keys.

```python
from tnfr_lfs.core.constants import (
    BRAKE_TEMPERATURE_MEAN_KEYS,
    TEMPERATURE_MEAN_KEYS,
    WHEEL_LABELS,
)

WHEEL_LABELS["fl"]  # -> "FL"
TEMPERATURE_MEAN_KEYS["rr"]  # -> "tyre_temp_rr"
BRAKE_TEMPERATURE_MEAN_KEYS["fr"]  # -> "brake_temp_fr"
```

### `tnfr_lfs.core.dissonance`

The dissonance helpers quantify whether ΔNFR is improving during high yaw
energy intervals—an important signal when diagnosing OZ operator opportunities
or quiet segments flagged by :func:`tnfr_lfs.core.operator_detection.detect_silence`.
:func:`~tnfr_lfs.core.dissonance.compute_useful_dissonance_stats` compares ΔNFR
gradients against yaw acceleration and returns a tuple of useful sample counts
and their ratio.【F:src/tnfr_lfs/core/dissonance.py†L18-L66】  The
``YAW_ACCELERATION_THRESHOLD`` constant exposes the default 0.5 rad/s² cut-off.

```python
from tnfr_lfs.core.dissonance import compute_useful_dissonance_stats

timestamps = [0.0, 0.1, 0.2, 0.3]
delta_nfr = [20.0, 18.0, 14.0, 11.0]
yaw_rate = [0.0, 0.08, 0.26, 0.5]

useful, energetic, ratio = compute_useful_dissonance_stats(
    timestamps,
    delta_nfr,
    yaw_rate,
)
ratio  # -> 1.0 when ΔNFR decreases during high yaw acceleration
```

### `tnfr_lfs.core.operator_detection`

Operator detectors sweep :class:`~tnfr_lfs.core.epi.TelemetryRecord` sequences
and emit structured events whenever lateral support (AL), oversteer (OZ),
ideal-line deviations (IL) or structural silence conditions appear.  Normalise
operator identifiers with :func:`~tnfr_lfs.core.operator_detection.canonical_operator_label`
or :func:`~tnfr_lfs.core.operator_detection.normalize_structural_operator_identifier`
before presenting them to the driver.【F:src/tnfr_lfs/core/operator_detection.py†L1-L293】

```python
from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.core.operator_detection import detect_al, detect_oz

samples = [
    TelemetryRecord(
        timestamp=index * 0.05,
        vertical_load=620.0,
        slip_ratio=0.0,
        lateral_accel=1.8,
        longitudinal_accel=0.2,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=0.0,
        locking=0.0,
        nfr=42.0,
        si=0.78,
        speed=45.0,
        yaw_rate=0.22,
        slip_angle=0.16,
        steer=0.05,
        throttle=0.6,
        gear=3,
        vertical_load_front=320.0,
        vertical_load_rear=300.0,
        mu_eff_front=0.0,
        mu_eff_rear=0.0,
        mu_eff_front_lateral=0.0,
        mu_eff_front_longitudinal=0.0,
        mu_eff_rear_lateral=0.0,
        mu_eff_rear_longitudinal=0.0,
        suspension_travel_front=0.01,
        suspension_travel_rear=0.02,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
    )
    for index in range(10)
]

al_events = detect_al(samples)
oz_events = detect_oz(samples, slip_threshold=0.12, yaw_threshold=0.2)
```

The resulting dictionaries can be merged into the microsector segmentation
output described in :mod:`tnfr_lfs.core.segmentation` or forwarded to the
recommendation engine so that nudges and prompts appear alongside the ΔNFR/SI
metrics.  Refer to :doc:`examples` for a complete ingestion → detection →
recommendation workflow.

### `tnfr_lfs.core.phases`

Phase helpers coordinate the legacy entry/apex/exit grouping used throughout the
segmentation stack with the more granular phase identifiers expected by the HUD
and recommendation workflows.【F:src/tnfr_lfs/core/phases.py†L1-L76】  Use
:func:`~tnfr_lfs.core.phases.replicate_phase_aliases` when migrating historical
payloads so that ``{"entry": ...}`` dictionaries also expose ``entry1`` and
``entry2`` values.  :func:`~tnfr_lfs.core.phases.expand_phase_alias` returns the
concrete phases associated with a legacy alias, while
:func:`~tnfr_lfs.core.phases.phase_family` reports the canonical family.  The
helpers keep legacy dashboards compatible with the microsector segmentation
output and the recommendation engine described in :doc:`setup_plan`.

```python
from statistics import mean

from tnfr_lfs.core.phases import replicate_phase_aliases

raw = {"entry1": 0.42, "entry2": 0.38, "apex3a": 0.55}
expanded = replicate_phase_aliases(raw, combine=mean)
expanded["entry"]  # -> 0.4 (mean of entry1 and entry2)
```

### `tnfr_lfs.core.utils`

Shared utilities currently expose :func:`~tnfr_lfs.core.utils.normalised_entropy`,
which computes the Shannon entropy of a sequence while normalising the result to
``[0, 1]``.【F:src/tnfr_lfs/core/utils.py†L1-L42】  The helper ignores non-numeric,
non-finite and non-positive values so it can operate directly on raw telemetry
weights.

```python
from tnfr_lfs.core.utils import normalised_entropy

normalised_entropy([1.0, 1.0, 1.0])  # -> 1.0 (uniform distribution)
normalised_entropy([3.0, 0.5, 0.5])  # -> 0.79...
```

### `tnfr_lfs.core.resonance`

Use :class:`tnfr_lfs.core.resonance.ModalPeak` to interpret the dominant
frequencies extracted from a rotational axis.  Each peak records the detected
frequency, its spectral energy and a classification flag that separates
actionable modes from noise.【F:src/tnfr_lfs/core/resonance.py†L14-L64】  Peaks are
labelled as ``"useful"`` when they fall in the 0.05–5 Hz window that covers the
modal range of typical setups and they either lead the spectrum or retain at
least half the dominant energy; all other peaks are marked as
``"parasitic"``.【F:src/tnfr_lfs/core/resonance.py†L70-L101】

:class:`tnfr_lfs.core.resonance.ModalAnalysis` aggregates the spectral
statistics for a single yaw/roll/pitch axis, reporting the sample rate, total
energy, the extracted peaks, the estimated excitation frequency (``nu_exc``) and
the ``rho`` ratio between the excitation and the dominant modal
frequency.【F:src/tnfr_lfs/core/resonance.py†L24-L109】  Values of ``rho`` close to
``1.0`` indicate that the steering/suspension inputs are driving the same mode
identified in the axis spectrum, while small values highlight a mismatch between
the excitation and the structural response.

:func:`tnfr_lfs.core.resonance.analyse_modal_resonance` orchestrates the full
workflow: the helper converts a series of
:class:`~tnfr_lfs.core.epi.TelemetryRecord` samples into per-axis spectra,
collects the modal energy, extracts the most energetic peaks and evaluates the
``rho`` coupling metric described above.【F:src/tnfr_lfs/core/resonance.py†L32-L109】

#### Example: synthetic yaw/roll/pitch sweep

The snippet below synthesises a short series of telemetry samples, evaluates the
modal resonance and prints the dominant peak for each axis:

```python
from math import sin, tau

from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.core.resonance import analyse_modal_resonance

records: list[TelemetryRecord] = []
for index in range(600):
    timestamp = index * 0.02  # 50 Hz
    yaw = 0.03 * sin(tau * 1.5 * timestamp)
    roll = 0.02 * sin(tau * 3.0 * timestamp)
    pitch = 0.01 * sin(tau * 0.5 * timestamp)
    steer = 0.2 * sin(tau * 1.5 * timestamp)
    front_velocity = 0.5 * sin(tau * 1.5 * timestamp + 0.1)
    rear_velocity = 0.5 * sin(tau * 1.5 * timestamp - 0.1)

    records.append(
        TelemetryRecord(
            timestamp=timestamp,
            vertical_load=0.0,
            slip_ratio=0.0,
            lateral_accel=0.0,
            longitudinal_accel=0.0,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            brake_pressure=0.0,
            locking=0.0,
            nfr=0.0,
            si=0.0,
            speed=0.0,
            yaw_rate=0.0,
            slip_angle=0.0,
            steer=steer,
            throttle=0.5,
            gear=3,
            vertical_load_front=0.0,
            vertical_load_rear=0.0,
            mu_eff_front=0.0,
            mu_eff_rear=0.0,
            mu_eff_front_lateral=0.0,
            mu_eff_front_longitudinal=0.0,
            mu_eff_rear_lateral=0.0,
            mu_eff_rear_longitudinal=0.0,
            suspension_travel_front=0.0,
            suspension_travel_rear=0.0,
            suspension_velocity_front=front_velocity,
            suspension_velocity_rear=rear_velocity,
        )
    )

analysis = analyse_modal_resonance(records)

for axis, modal in analysis.items():
    peak = modal.peaks[0] if modal.peaks else None
    print(
        axis,
        f"nu_exc={modal.nu_exc:.2f} Hz",
        f"rho={modal.rho:.2f}",
        f"dominant={peak.frequency:.2f} Hz ({peak.classification})" if peak else "no peaks",
    )
```

Running the script reveals how the helper reports a dominant yaw mode around
1.5 Hz, marks roll as parasitic because its energy sits below the 50% threshold
of the primary peak and describes the pitch response through the ``rho`` ratio.

### `tnfr_lfs.core.coherence`

The coherence helpers distribute ΔNFR magnitudes to the signals that produced
them and collapse the resulting picture into the sense index exposed on the HUD.
Use :func:`tnfr_lfs.core.coherence.compute_node_delta_nfr` when a subsystem emits
multiple raw measurements—wheel loads, aero balance, slip ratios, and so on—yet
only a single ΔNFR value is available.  The helper scales the node-level ΔNFR by
the absolute contribution of every feature in ``feature_map`` so that the
relative weighting remains consistent even when sign changes are handled at the
node level.【F:src/tnfr_lfs/core/coherence.py†L16-L46】

:func:`tnfr_lfs.core.coherence.sense_index` aggregates the ΔNFR map into a
bounded ``[0, 1]`` score that punishes large deviations, prioritises subsystems
with higher natural frequencies and discounts incoherent distributions via an
entropy penalty.【F:src/tnfr_lfs/core/coherence.py†L58-L127】  Three keyword
arguments control the most common tuning points:

* ``w_phase`` selects the phase-dependent weighting profile.  Pass either a
  scalar applied to every node or a mapping where the top-level keys represent
  phases (including aliases) and the values are per-node mappings or scalars.
  The helper resolves the entry for ``active_phase`` and multiplies each node by
  the resulting weight.【F:src/tnfr_lfs/core/coherence.py†L74-L99】
* ``nu_f_targets`` defines the goal natural frequencies.  When provided, the
  targets are resolved using the same phase lookup mechanism as ``w_phase`` and
  mapped to the currently active nodes.  The resulting gain encourages
  convergence towards the target values and compounds with the measured
  frequency gain applied to the ΔNFR magnitude.【F:src/tnfr_lfs/core/coherence.py†L101-L115】
* ``entropy_lambda`` sets the strength of the entropy penalty that drives the
  metric towards evenly distributed ΔNFR contributions.  Higher values increase
  the penalty applied to sharply peaked distributions while keeping the final
  score within ``[0, 1]``.【F:src/tnfr_lfs/core/coherence.py†L117-L127】

```python
from tnfr_lfs.core.coherence import compute_node_delta_nfr, sense_index

# Each node reports a single ΔNFR but tracks different raw signals.
feature_map_suspension = {"lateral": 0.6, "longitudinal": 0.3, "vertical": 0.1}
feature_map_aero = {"front": 0.4, "rear": 0.8}

deltas_by_node = {}
deltas_by_node.update(
    compute_node_delta_nfr("suspension", 0.24, feature_map_suspension)
)
deltas_by_node.update(
    compute_node_delta_nfr("aero", 0.12, feature_map_aero)
)

# Natural frequencies measured for each subsystem.
nu_f_by_node = {"suspension": 0.9, "aero": 0.3}

# Phase weights favour suspension signals during entry while targeting a
# quicker natural frequency for aero elements in the same phase.
w_phase = {
    "__default__": 1.0,
    "entry": {"suspension": 1.4, "aero": 0.8},
}
nu_f_targets = {
    "__default__": 0.5,
    "entry": {"suspension": 0.8, "aero": 1.1},
}

index = sense_index(
    delta_nfr=0.36,
    deltas_by_node={
        "suspension": sum(value for key, value in deltas_by_node.items() if key.startswith("suspension.")),
        "aero": sum(value for key, value in deltas_by_node.items() if key.startswith("aero.")),
    },
    baseline_nfr=1.8,
    nu_f_by_node=nu_f_by_node,
    active_phase="entry",
    w_phase=w_phase,
    nu_f_targets=nu_f_targets,
    entropy_lambda=0.2,
)

print(f"Sense index: {index:.3f}")
```

The example distributes each node's ΔNFR proportionally to its raw
measurements, then aggregates the node-level magnitudes back into a sense index.
Increasing ``w_phase["entry"]["suspension"]`` boosts the suspension penalty,
while the higher ``nu_f_targets`` for aero tighten the goal multiplier and drive
the sense index lower than a neutral configuration would.

### `tnfr_lfs.core.contextual_delta`

Context-aware ΔNFR adjustments rely on two lightweight dataclasses and a loader
that exposes the calibration factors distributed with the library.

* :class:`tnfr_lfs.core.contextual_delta.ContextMatrix` captures the curvature,
  surface and traffic bands together with minimum/maximum multipliers and the
  reference values used to normalise load or speed deltas. The
  ``curve_factor``, ``surface_factor`` and ``traffic_factor`` methods return the
  multiplier associated with each metric, while ``curve_band`` /
  ``surface_band`` / ``traffic_band`` surface the active range label.【F:src/tnfr_lfs/core/contextual_delta.py†L51-L109】
* :class:`tnfr_lfs.core.contextual_delta.ContextFactors` represents the factor
  triplet applied to a ΔNFR and exposes ``multiplier`` to collapse the
  components into a single scalar consumed by the rest of the module.【F:src/tnfr_lfs/core/contextual_delta.py†L35-L48】
* :func:`tnfr_lfs.core.contextual_delta.load_context_matrix` loads the matrix
  from ``context_factors.toml`` (cached in memory), normalises per-track
  datasets and guarantees strictly positive references for traffic terms before
  constructing the :class:`~tnfr_lfs.core.contextual_delta.ContextMatrix`.【F:src/tnfr_lfs/core/contextual_delta.py†L187-L221】
* :func:`tnfr_lfs.core.contextual_delta.apply_contextual_delta` clamps the
  computed multiplier to the thresholds defined by the matrix and returns the
  ΔNFR adjusted for the current context.【F:src/tnfr_lfs/core/contextual_delta.py†L233-L244】

```python
from tnfr_lfs.core.contextual_delta import (
    apply_contextual_delta,
    load_context_matrix,
)

matrix = load_context_matrix()
delta_nfr = 0.18  # raw ΔNFR
factors = {"curve": 1.05, "surface": 0.95, "traffic": 1.1}

contextualised = apply_contextual_delta(delta_nfr, factors, context_matrix=matrix)
print(f"Context-aware ΔNFR: {contextualised:.3f}")
```

### `tnfr_lfs.core.coherence_calibration`

The coherence subsystem keeps per-player/car histories so it can derive
persistent baselines and normal telemetry ranges.

* :class:`tnfr_lfs.core.coherence_calibration.CoherenceCalibrationStore`
  persists a dictionary of :class:`~tnfr_lfs.core.coherence_calibration.CalibrationEntry`
  keyed by ``(player_name, car_model)``, validates decay and lap thresholds, and
  automatically handles loading/saving the associated TOML file.【F:src/tnfr_lfs/core/coherence_calibration.py†L130-L240】
  Each entry accumulates exponentially smoothed means via
  :class:`~tnfr_lfs.core.coherence_calibration.CalibrationMetric`, increments the
  lap counter up to ``max_laps`` and only exposes a derived baseline once
  ``min_laps`` has been satisfied.【F:src/tnfr_lfs/core/coherence_calibration.py†L60-L190】
* :func:`tnfr_lfs.core.coherence_calibration.CoherenceCalibrationStore.snapshot`
  builds an immutable :class:`tnfr_lfs.core.coherence_calibration.CalibrationSnapshot`
  carrying the current baseline and ``(minimum, maximum)`` ranges for each
  metric, making it straightforward to inspect the calibration without exposing
  internal state.【F:src/tnfr_lfs/core/coherence_calibration.py†L181-L272】

```python
from tnfr_lfs.core.coherence_calibration import CoherenceCalibrationStore
from tnfr_lfs.core.epi import DeltaCalculator, TelemetryRecord

store = CoherenceCalibrationStore(decay=0.25, min_laps=2, max_laps=5)

# Build a handful of synthetic telemetry samples.  Populate the key load,
# slip, acceleration and μ fields with plausible values for the car you are
# analysing – the remaining optional channels default to ``math.nan`` when
# omitted.  ``DeltaCalculator.derive_baseline`` averages these records to
# produce a ready-to-use baseline payload.
records = [
    TelemetryRecord(
        timestamp=0.0,
        vertical_load=3900.0,
        slip_ratio=0.02,
        lateral_accel=14.5,
        longitudinal_accel=-1.2,
        yaw=0.01,
        pitch=0.002,
        roll=-0.008,
        brake_pressure=0.0,
        locking=0.0,
        nfr=0.68,
        si=0.74,
        speed=58.0,
        yaw_rate=0.12,
        slip_angle=0.03,
        steer=0.08,
        throttle=0.62,
        gear=3,
        vertical_load_front=2100.0,
        vertical_load_rear=1800.0,
        mu_eff_front=2.2,
        mu_eff_rear=2.1,
        mu_eff_front_lateral=1.9,
        mu_eff_front_longitudinal=1.6,
        mu_eff_rear_lateral=1.8,
        mu_eff_rear_longitudinal=1.5,
        suspension_travel_front=0.042,
        suspension_travel_rear=0.038,
        suspension_velocity_front=0.004,
        suspension_velocity_rear=0.006,
        car_model="fo8",
        track_name="SO1",
    ),
    TelemetryRecord(
        timestamp=0.1,
        vertical_load=4020.0,
        slip_ratio=0.015,
        lateral_accel=15.1,
        longitudinal_accel=-0.9,
        yaw=0.015,
        pitch=0.001,
        roll=-0.006,
        brake_pressure=0.1,
        locking=0.02,
        nfr=0.7,
        si=0.77,
        speed=60.5,
        yaw_rate=0.14,
        slip_angle=0.028,
        steer=0.075,
        throttle=0.66,
        gear=3,
        vertical_load_front=2150.0,
        vertical_load_rear=1870.0,
        mu_eff_front=2.25,
        mu_eff_rear=2.12,
        mu_eff_front_lateral=1.95,
        mu_eff_front_longitudinal=1.64,
        mu_eff_rear_lateral=1.84,
        mu_eff_rear_longitudinal=1.53,
        suspension_travel_front=0.045,
        suspension_travel_rear=0.04,
        suspension_velocity_front=0.005,
        suspension_velocity_rear=0.007,
        car_model="fo8",
        track_name="SO1",
    ),
]

baseline = DeltaCalculator.derive_baseline(records)
store.observe_baseline("Driver 42", "fo8", baseline)

# Once enough laps have been ingested the snapshot exposes the derived baseline.
snapshot = store.snapshot("Driver 42", "fo8")
if snapshot:
    print(snapshot.player_name, snapshot.laps)
```

### `tnfr_lfs.core.cache`

`LRUCache` wraps an internal least-recently-used store so pipelines can opt into
bounded caching while keeping ``maxsize=0`` as a valid way to disable storage at
runtime.【F:src/tnfr_lfs/core/cache.py†L53-L95】  ``cached_delta_nfr_map`` captures
per-record ΔNFR maps using object identifiers (and their ``reference`` links)
while exposing immutable copies to callers, allowing repeated ΔNFR queries to
reuse existing work until a matching record is invalidated.【F:src/tnfr_lfs/core/cache.py†L109-L134】
``cached_dynamic_multipliers`` mirrors the behaviour for ν_f multipliers,
stashing the latest multipliers and their evaluation frequency per
``(car_model, history length, timestamp)`` tuple so dynamic analytics avoid
recalculating identical windows.【F:src/tnfr_lfs/core/cache.py†L144-L185】  Use
``configure_cache`` to toggle ΔNFR caching and resize the dynamic multiplier
store; the helper normalises inputs, tears down caches when set to zero and
restores defaults when omitted.【F:src/tnfr_lfs/core/cache.py†L194-L234】

```python
from tnfr_lfs.core.cache_settings import CacheOptions
from tnfr_lfs.core.cache import configure_cache

options = CacheOptions.from_config({
    "performance": {
        "cache_enabled": True,
        "nu_f_cache_size": 128,
    }
})
normalised = options.with_defaults()
configure_cache(
    enable_delta_cache=normalised.enable_delta_cache,
    nu_f_cache_size=normalised.nu_f_cache_size,
)
```

### `tnfr_lfs.core.cache_settings`

`CacheOptions` is the immutable dataclass consumed throughout the caching
helpers.  :meth:`CacheOptions.from_config` coerces nested ``[performance]`` or
legacy ``[cache]`` mappings into a single payload, sanitising booleans and cache
sizes while carrying defaults for missing entries.【F:src/tnfr_lfs/core/cache_settings.py†L23-L151】
The :meth:`CacheOptions.with_defaults` method re-emits the options with bounded
integers so downstream callers can pass them directly into
:func:`tnfr_lfs.core.cache.configure_cache` via the pattern shown above.【F:src/tnfr_lfs/core/cache_settings.py†L153-L172】
``CacheOptions`` also exposes convenience accessors like ``cache_enabled`` and
``max_cache_size`` for quick eligibility checks when orchestration logic needs to
branch on runtime cache state.【F:src/tnfr_lfs/core/cache_settings.py†L175-L185】

### `tnfr_lfs.core.spectrum`

The spectrum helpers expose consistent utilities for telemetry FFTs and
cross-spectra. Two entry-points power the CLI reports and analytics:

* :func:`tnfr_lfs.core.spectrum.motor_input_correlations` extracts the dominant
  cross-spectrum between driver inputs and chassis responses, returning
  :class:`tnfr_lfs.core.spectrum.PhaseCorrelation` objects keyed by
  ``(control, response)``. The ``dominant_strategy`` flag selects the dense FFT
  evaluation (``"fft"``) or the Goertzel accelerator (``"auto"``/``"goertzel"``)
  when SciPy is installed via ``pip install tnfr_lfs[spectral]``. Provide
  ``candidate_frequencies`` to limit the Goertzel search to known resonant bins.
* :func:`tnfr_lfs.core.spectrum.phase_alignment` mirrors the same configuration
  knobs when estimating the dominant steer-versus-response bin. The helper also
  accepts explicit steer/response sequences so ingest pipelines can reuse
  cached telemetry instead of re-sampling :class:`~tnfr_lfs.core.epi.TelemetryRecord`
  objects.

When the Goertzel path is active the implementation evaluates a fixed list of
candidate bins in :math:`O(n·k)` time, so keeping ``k`` constant makes dominant
frequency lookups scale linearly with the number of samples. The FFT path
remains available whenever SciPy is missing or a full spectrum is required for
post-processing.

```python
from dataclasses import replace
import math

from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.core.spectrum import motor_input_correlations

baseline = TelemetryRecord(
    timestamp=0.0,
    vertical_load=5000.0,
    slip_ratio=0.0,
    lateral_accel=0.0,
    longitudinal_accel=0.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    brake_pressure=0.0,
    locking=0.0,
    nfr=220.0,
    si=0.82,
    speed=45.0,
    yaw_rate=0.0,
    slip_angle=0.0,
    steer=0.0,
    throttle=0.6,
    gear=4,
    vertical_load_front=2500.0,
    vertical_load_rear=2500.0,
    mu_eff_front=1.0,
    mu_eff_rear=1.0,
    mu_eff_front_lateral=1.0,
    mu_eff_front_longitudinal=1.0,
    mu_eff_rear_lateral=1.0,
    mu_eff_rear_longitudinal=1.0,
    suspension_travel_front=0.02,
    suspension_travel_rear=0.02,
    suspension_velocity_front=0.0,
    suspension_velocity_rear=0.0,
)

phase_offset = math.radians(12.0)


def sample_at(index: int, sample_rate: float) -> TelemetryRecord:
    t = index / sample_rate
    steer = 0.3 * math.sin(2.0 * math.pi * 1.25 * t)
    yaw_rate = 0.45 * math.sin(2.0 * math.pi * 1.25 * t + phase_offset)
    return replace(
        baseline,
        timestamp=t,
        steer=steer,
        yaw_rate=yaw_rate,
        lateral_accel=yaw_rate * 5.0,
        longitudinal_accel=0.3 * math.cos(2.0 * math.pi * 0.5 * t),
        throttle=0.65 + 0.1 * math.cos(2.0 * math.pi * 0.5 * t),
        brake_pressure=max(0.0, -0.2 * math.sin(2.0 * math.pi * 0.5 * t)),
        nfr=215.0 + 25.0 * math.sin(2.0 * math.pi * 0.4 * t),
        si=0.8 + 0.05 * math.cos(2.0 * math.pi * 0.4 * t),
        slip_angle=0.02 * math.sin(2.0 * math.pi * 1.25 * t),
    )


records = [sample_at(i, 50.0) for i in range(256)]
correlations = motor_input_correlations(records, sample_rate=50.0, dominant_strategy="fft")

steer_vs_yaw = correlations[("steer", "yaw")]
print(f"Dominant frequency: {steer_vs_yaw.frequency:.2f} Hz")
print(f"Phase alignment: {steer_vs_yaw.alignment:.3f}")
print(f"Latency: {steer_vs_yaw.latency_ms:.1f} ms")
```

The :class:`~tnfr_lfs.core.spectrum.PhaseCorrelation` result exposes the
dominant input/output bin, making the steering-to-yaw relationship in the
example easy to interpret. ``alignment`` close to ``1.0`` indicates the chassis
responds in phase with the steering command, while ``latency_ms`` quantifies the
delay introduced by the yaw-rate response. A positive latency confirms the yaw
motion lags the input, matching the injected twelve-degree offset in the sample
records.【F:src/tnfr_lfs/core/spectrum.py†L32-L47】【F:src/tnfr_lfs/core/spectrum.py†L254-L302】

### `tnfr_lfs.core.epi.EPIExtractor`

Computes :class:`tnfr_lfs.core.epi_models.EPIBundle` objects including EPI,
ΔNFR, entropy-penalised sense index values, and the nodal distribution
calculated by :func:`tnfr_lfs.core.epi.delta_nfr_by_node`.  For additional
subcomponent breakdowns use
:func:`tnfr_lfs.core.coherence.compute_node_delta_nfr` together with
:func:`tnfr_lfs.core.coherence.sense_index` inside custom pipelines.  The
updated Sense Index consumes the natural-frequency map, the active phase, and
the ``w_phase`` weights to evaluate ``1 / (1 + Σ w · |ΔNFR| · g(ν_f)) - λ·H``
with a configurable entropy penalty.

Each :class:`EPIBundle` records the per-node natural frequency ``nu_f`` (Hz),
allowing downstream tooling to weigh ΔNFR contributions using documented
subsystem dynamics.  Bundles now expose the derivative ``dEPI_dt`` and the
cumulatively integrated ``integrated_epi`` value obtained through explicit
Euler integration.  The ΔNFR signal is also decomposed into longitudinal and
lateral components via the ``delta_nfr_proj_longitudinal``/``delta_nfr_proj_lateral``
fields which project the nodal gradient onto each axis, providing immediate
visibility of brake/traction versus balance imbalances while keeping the raw
`Fz`/`ΔFz` channels available for force-based analysis.

### Operator utilities

```
tnfr_lfs.core.operators.emission_operator(target_delta_nfr: float, target_sense_index: float) -> Dict[str, float]
tnfr_lfs.core.operators.reception_operator(records: Sequence[TelemetryRecord], extractor: Optional[EPIExtractor] = None) -> List[EPIBundle]
tnfr_lfs.core.operators.coherence_operator(series: Sequence[float], window: int = 3) -> List[float]
tnfr_lfs.core.operators.dissonance_operator(series: Sequence[float], target: float) -> float
tnfr_lfs.core.operators.dissonance_breakdown_operator(series: Sequence[float], target: float, *, microsectors: Optional[Sequence[Microsector]] = None, bundles: Optional[Sequence[EPIBundle]] = None) -> DissonanceBreakdown
tnfr_lfs.core.operators.coupling_operator(series_a: Sequence[float], series_b: Sequence[float], *, strict_length: bool = True) -> float
tnfr_lfs.core.operators.resonance_operator(series: Sequence[float]) -> float
tnfr_lfs.core.operators.recursive_filter_operator(series: Sequence[float], *, seed: float = 0.0, decay: float = 0.5) -> List[float]
tnfr_lfs.core.operators.orchestrate_delta_metrics(telemetry_segments: Sequence[Sequence[TelemetryRecord]], target_delta_nfr: float, target_sense_index: float, *, coherence_window: int = 3, recursion_decay: float = 0.4, microsectors: Optional[Sequence[Microsector]] = None, phase_weights: Optional[Mapping[str, Mapping[str, float] | float]] = None, operator_state: Optional[Mapping[str, Dict[str, object]]] = None) -> Mapping[str, object]
tnfr_lfs.core.operators.evolve_epi(prev_epi: float, delta_map: Mapping[str, float], dt: float, nu_f_by_node: Mapping[str, float]) -> Tuple[float, float]
```

Legacy Spanish operator names remain available as deprecated aliases to ease the migration.

These functions compose an end-to-end ΔNFR/Sense Index pipeline covering
objective setting (Emission), telemetry reception, smoothing (Coherence),
contrast measurement (Dissonance) with a breakdown of useful vs. parasitic
support events, coupling analysis, resonance estimation, recursive filtering,
and orchestration over segmented telemetry streams. The orchestrator exposes
``microsector_variability`` entries summarising ΔNFR↓ and Sense Index variance
plus population-standard deviation for each microsector, alongside a
``lap_sequence`` describing the lap labels detected in the incoming segments.
The :class:`DissonanceBreakdown` payload now includes the Useful Dissonance
Ratio (UDR) through the ``useful_dissonance_ratio``/``useful_dissonance_samples``
fields, quantifying the fraction of high yaw-acceleration samples where
ΔNFR is already decaying.

```python
from tnfr_lfs.core.operators import (
    emission_operator,
    reception_operator,
    coherence_operator,
    orchestrate_delta_metrics,
)
from tnfr_lfs.core.epi import EPIExtractor

# Reuse the synthetic `records` built in the spectrum example above.
segments = [records[:128], records[128:]]

objectives = emission_operator(target_delta_nfr=-15.0, target_sense_index=0.85)
extractor = EPIExtractor()
bundles = reception_operator(records, extractor=extractor)
delta_trace = [bundle.delta_nfr for bundle in bundles]
smoothed_delta = coherence_operator(delta_trace, window=5)

report = orchestrate_delta_metrics(
    segments,
    target_delta_nfr=objectives["delta_nfr"],
    target_sense_index=objectives["sense_index"],
    coherence_window=5,
)

print(report["delta_nfr"], report["sense_index"])
print(report["stages"]["coherence"]["dissonance_breakdown"].useful_dissonance_ratio)
```

The snippet chains the same helpers invoked inside
:func:`~tnfr_lfs.core.operators.orchestrate_delta_metrics`, making each stage’s
side-effects observable without having to inspect the returned ``"stages"``
payload directly. ``emission_operator`` emits the target map, the reception
stage converts telemetry into :class:`~tnfr_lfs.core.epi_models.EPIBundle`
objects, and ``coherence_operator`` smooths the ΔNFR series before the
orchestrator aggregates nodal, resonance, dissonance and recursive metrics for
the whole stint.【F:src/tnfr_lfs/core/operators.py†L95-L141】【F:src/tnfr_lfs/core/operators.py†L1529-L1645】

``orchestrate_delta_metrics`` also surfaces support metrics derived from
``WindowMetrics``: ``support_effective`` (ΔNFR sustained by tyres and suspension
with structural weighting), ``load_support_ratio`` (normalised by the average
Fz load), and the pairs
``structural_expansion_longitudinal``/``structural_contraction_longitudinal`` and
``structural_expansion_lateral``/``structural_contraction_lateral`` which
describe how the longitudinal/lateral ΔNFR components expand or compress the
structural axis of the analysed window.  The steering budgets
``ackermann_parallel_index`` and ``slide_catch_budget`` are derived exclusively
from the ``slip_angle_*`` channels and the ``yaw_rate`` emitted by OutSim; when
that Live for Speed telemetry is missing the output reports the literal
``"no data"`` marker.  The ``aero_balance_drift`` entry groups the average
rake (pitch plus per-axle travel) and the ``μ_front - μ_rear`` delta for low,
medium, and high speed bands.  Rake relies solely on the ``pitch`` and
suspension-travel channels provided by OutSim, ensuring the aerodynamic drift
matches the native data even if ``AeroCoherence`` appears neutral.

``WindowMetrics`` also publishes the ΔNFR entropy. ``delta_nfr_entropy``
summarises the structural distribution per phase (0 ≙ energy concentrated in a
single phase, 1 ≙ balanced window across all observed phases) while
``node_entropy`` captures the overall nodal diversity using the same normalised
range [0, 1]. The ``phase_delta_nfr_entropy`` map contains the normalised phase
probabilities (summing to 1.0) used to compute entropy, and
``phase_node_entropy`` details the Shannon entropy per phase derived from the
nodal contributions.  Both maps stay within the [0, 1] interval and preserve the
legacy phase aliases so HUDs and the CLI can consume them directly.  The
``entry``/``apex``/``exit`` aliases are slated for removal; consumers relying on
them receive a :class:`DeprecationWarning` to ease the migration towards the
granular identifiers (``entry1``, ``entry2``, ``apex3a``, ``apex3b``, ``exit4``).

``WindowMetrics.cphi`` now yields a :class:`~tnfr_lfs.core.metrics.CPHIReport`
with per-wheel :class:`~tnfr_lfs.core.metrics.CPHIWheel` data and the shared
:class:`~tnfr_lfs.core.metrics.CPHIThresholds`. The thresholds follow a
red/amber/green traffic-light scheme so HUD pages and CLI reports colour tyre
health consistently. Consumers that require the historical flat keys can rely
on :meth:`~tnfr_lfs.core.metrics.CPHIReport.as_legacy_mapping`, which now emits
a :class:`DeprecationWarning` because the flat keys are on a deprecation path.

### `tnfr_lfs.core.metrics`

``tnfr_lfs.core.metrics`` condenses ΔNFR, Sense Index and support diagnostics
for a telemetry window. :func:`~tnfr_lfs.core.metrics.compute_window_metrics`
accepts ordered :class:`~tnfr_lfs.core.epi.TelemetryRecord` samples (plus
optional phase windows and precomputed :class:`~tnfr_lfs.core.epi_models.EPIBundle`
series) and returns a :class:`~tnfr_lfs.core.metrics.WindowMetrics` snapshot
with phase-aligned gradients, structural support ratios, entropy summaries and
motor-input correlations.【F:src/tnfr_lfs/core/metrics.py†L980-L1104】【F:src/tnfr_lfs/core/metrics.py†L1208-L1320】

``WindowMetrics`` aggregates the window averages, ΔNFR spreads and per-phase
decompositions used by the HUD/CLI.  The dataclass enumerates support-efficiency
metrics (``support_effective`` and ``load_support_ratio``), structural expansion
markers for longitudinal and lateral axes, entropy values
(``delta_nfr_entropy``/``node_entropy`` plus per-phase maps), the Useful
Dissonance Ratio, steering budgets and the motor-input correlations exposed by
the HUD.  It also exposes brake headroom, aerodynamics aggregates, nodal ΔNFR
variance and the absolute EPI derivative used to qualify bundles.【F:src/tnfr_lfs/core/metrics.py†L813-L922】【F:src/tnfr_lfs/core/metrics.py†L936-L1018】

The tyre-health view is provided by ``WindowMetrics.cphi``, which returns a
:class:`~tnfr_lfs.core.metrics.CPHIReport`.  Each report bundles per-wheel
:class:`~tnfr_lfs.core.metrics.CPHIWheel` values, component breakdowns and the
shared :class:`~tnfr_lfs.core.metrics.CPHIThresholds` traffic-light scheme, and
offers compatibility helpers such as :meth:`~tnfr_lfs.core.metrics.CPHIReport.as_dict`
and :meth:`~tnfr_lfs.core.metrics.CPHIReport.as_legacy_mapping` for exporters on
the deprecation path.【F:src/tnfr_lfs/core/metrics.py†L312-L360】【F:src/tnfr_lfs/core/metrics.py†L466-L553】

Aerodynamic cues combine ``WindowMetrics.aero_coherence`` and
``WindowMetrics.aero_balance_drift``.  :class:`~tnfr_lfs.core.metrics.AeroCoherence`
splits front/rear load deltas into low/medium/high speed bins (including
dominant-axis helpers), while ``aero_balance_drift`` tracks rake and μ deltas per
speed range so the setup planner can distinguish aero map changes from
mechanical balance shifts.【F:src/tnfr_lfs/core/metrics.py†L262-L340】【F:src/tnfr_lfs/core/metrics.py†L1650-L1735】

When ``compute_window_metrics`` receives an ``EPIBundle`` series it reuses the
smoothed ΔNFR derivative from the extractor, improving Useful Dissonance Ratio
and ``epi_derivative_abs`` calculations.  Without bundles the function derives
gradients from the raw window and still populates the same interface, falling
back to chronological timestamps whenever the structural axis is not
available.【F:src/tnfr_lfs/core/metrics.py†L980-L1040】【F:src/tnfr_lfs/core/metrics.py†L1041-L1104】

```python
from dataclasses import replace
from math import cos, pi, sin

from tnfr_lfs.core.epi import EPIExtractor, TelemetryRecord
from tnfr_lfs.core.metrics import compute_window_metrics

baseline = TelemetryRecord(
    timestamp=0.0,
    vertical_load=4200.0,
    slip_ratio=0.02,
    lateral_accel=5.0,
    longitudinal_accel=0.6,
    yaw=0.05,
    pitch=0.01,
    roll=0.02,
    brake_pressure=0.35,
    locking=0.05,
    nfr=210.0,
    si=0.78,
    speed=62.0,
    yaw_rate=0.18,
    slip_angle=0.03,
    steer=0.11,
    throttle=0.62,
    gear=4,
    vertical_load_front=2080.0,
    vertical_load_rear=2120.0,
    mu_eff_front=1.24,
    mu_eff_rear=1.18,
    mu_eff_front_lateral=1.09,
    mu_eff_front_longitudinal=1.18,
    mu_eff_rear_lateral=1.04,
    mu_eff_rear_longitudinal=1.12,
    suspension_travel_front=0.045,
    suspension_travel_rear=0.052,
    suspension_velocity_front=0.012,
    suspension_velocity_rear=-0.009,
)


def sample_at(step: int) -> TelemetryRecord:
    t = step * 0.12
    return replace(
        baseline,
        timestamp=t,
        nfr=210.0 + 18.0 * sin(2.0 * pi * 0.35 * t),
        si=0.78 + 0.04 * cos(2.0 * pi * 0.35 * t),
        yaw_rate=0.18 + 0.05 * sin(2.0 * pi * 0.55 * t),
        slip_angle=0.03 + 0.01 * sin(2.0 * pi * 0.5 * t),
        steer=0.11 + 0.015 * sin(2.0 * pi * 0.4 * t),
        throttle=0.62 + 0.08 * cos(2.0 * pi * 0.2 * t),
        brake_pressure=0.35 + 0.1 * max(0.0, sin(2.0 * pi * 0.15 * t)),
        vertical_load_front=2060.0 + 160.0 * sin(2.0 * pi * 0.25 * t),
        vertical_load_rear=2140.0 - 160.0 * sin(2.0 * pi * 0.25 * t),
    )


window = [sample_at(i) for i in range(48)]

plain_metrics = compute_window_metrics(window)
bundles = EPIExtractor().extract(window)
bundle_metrics = compute_window_metrics(window, bundles=bundles)

print(f"UDR without bundles: {plain_metrics.useful_dissonance_ratio:.3f}")
print(f"UDR with bundles:    {bundle_metrics.useful_dissonance_ratio:.3f}")
print(
    f"Front-left CPHI: {bundle_metrics.cphi['fl'].value:.2f}",
    f"Brake↔ΔNFR correlation: {bundle_metrics.brake_longitudinal_correlation:.3f}",
    f"Throttle↔ΔNFR correlation: {bundle_metrics.throttle_longitudinal_correlation:.3f}",
)
```

The synthetic window mimics a medium-speed cornering stint. Passing ``bundles``
lets the metrics reuse the smoothed ΔNFR derivative produced by
:class:`~tnfr_lfs.core.epi.EPIExtractor`, which typically raises the Useful
Dissonance Ratio because it filters single-sample spikes.  The ``cphi`` report
exposes each wheel’s contact-patch score alongside the shared thresholds, while
the brake/throttle correlations summarise how strongly driver inputs explain the
longitudinal ΔNFR variation in the analysed window.【F:src/tnfr_lfs/core/metrics.py†L520-L553】【F:src/tnfr_lfs/core/metrics.py†L813-L898】

When the ``operator_state`` shared by ``segment_microsectors`` is supplied, the
orchestrator adds the ``network_memory`` field and a mirror in
``sense_memory["network"]`` with the per-session memory
(``car_model``/``track_name``/``tyre_compound``), including stint histories and
each microsector’s active state.  If OutGauge does not transmit the extended
per-wheel temperature/pressure block, HUD surfaces and exporters show the
literal ``"no data"`` token in those fields to make it clear that LFS telemetry
was unavailable.

### `tnfr_lfs.core.structural_time`

``compute_structural_timestamps`` derives a structural timeline by comparing the
normalised derivatives of high-energy telemetry channels (brake pressure,
throttle, yaw-rate and steer) and feeding them through a moving average window
whose width is controlled by ``window_size``.【F:src/tnfr_lfs/core/structural_time.py†L65-L120】 The blended density inflates
segments rich in chassis activity so that ΔNFR and operator heuristics can weigh
busy events more heavily than coasting samples while still anchoring the first
timestamp to either the provided ``base_timestamp`` or the earliest structural
reference stored on the records.【F:src/tnfr_lfs/core/structural_time.py†L87-L120】

Adjust ``window_size`` when you need to capture sharper activation bursts (small
windows) or smooth out oscillations in long stints (larger windows).  Override
``weights`` to privilege sensors that best represent the chassis state for a
given car or track—e.g. upping brake pressure on endurance stints, or throttle
when analysing traction events.  When the structural axis is partially missing
or not strictly monotonic, ``resolve_time_axis`` falls back to the chronological
``timestamp`` field if ``fallback_to_chronological`` stays enabled, ensuring
downstream metrics still receive an ordered axis.【F:src/tnfr_lfs/core/structural_time.py†L125-L164】

Both helpers integrate directly with :class:`~tnfr_lfs.core.epi.TelemetryRecord`,
which exposes the optional ``structural_timestamp`` attribute so segmentation,
operator pipelines and exporters can persist the denser axis once it has been
computed.【F:src/tnfr_lfs/core/epi.py†L520-L576】  Use ``compute_structural_timestamps`` to populate that
field and rely on ``resolve_time_axis`` whenever consumer code needs to recover
an ordered series even if some records lost their structural markers.

```python
from dataclasses import replace

from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.core.structural_time import (
    compute_structural_timestamps,
    resolve_time_axis,
)


def make_record(timestamp: float, brake: float, throttle: float, steer: float, yaw_rate: float) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=timestamp,
        vertical_load=500.0,
        slip_ratio=0.01,
        lateral_accel=0.30,
        longitudinal_accel=0.10,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        brake_pressure=brake,
        locking=0.0,
        nfr=3.2,
        si=0.5,
        speed=40.0,
        yaw_rate=yaw_rate,
        slip_angle=0.02,
        steer=steer,
        throttle=throttle,
        gear=3,
        vertical_load_front=260.0,
        vertical_load_rear=240.0,
        mu_eff_front=1.2,
        mu_eff_rear=1.1,
        mu_eff_front_lateral=1.0,
        mu_eff_front_longitudinal=1.0,
        mu_eff_rear_lateral=1.0,
        mu_eff_rear_longitudinal=1.0,
        suspension_travel_front=0.05,
        suspension_travel_rear=0.04,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
    )


records = [
    make_record(0.00, brake=0.1, throttle=0.40, steer=0.05, yaw_rate=0.02),
    make_record(0.30, brake=0.4, throttle=0.20, steer=0.10, yaw_rate=0.08),
    make_record(0.60, brake=0.2, throttle=0.60, steer=0.02, yaw_rate=0.04),
]

structural_axis = compute_structural_timestamps(records, window_size=3)
records_with_axis = [
    replace(record, structural_timestamp=value)
    for record, value in zip(records, structural_axis)
]

# Simulate a missing structural timestamp on the final sample.
records_with_axis[-1] = replace(records_with_axis[-1], structural_timestamp=None)

time_axis = resolve_time_axis(records_with_axis)

print(structural_axis)
print(time_axis)
```

The example populates ``TelemetryRecord.structural_timestamp`` and shows how
``resolve_time_axis`` gracefully falls back to the chronological timestamps when
one sample drops its structural marker, keeping operator and segmentation stages
fully aligned.

### `tnfr_lfs.core.segmentation`

```
tnfr_lfs.core.segmentation.segment_microsectors(records: Sequence[TelemetryRecord], bundles: Sequence[EPIBundle], *, phase_weight_overrides: Optional[Mapping[str, Mapping[str, float] | float]] = None) -> List[Microsector]
tnfr_lfs.core.segmentation.Microsector
tnfr_lfs.core.segmentation.Goal
```

The segmentation module aligns telemetry samples with their EPI bundles to
derive microsectors that follow curvature, support events, and ΔNFR
signatures.  Each :class:`Microsector` exposes phase boundaries for entry,
apex, and exit together with an ``active_phase`` selector and the
``dominant_nodes`` mapping that lists the subsystems driving each phase.
The per-phase :class:`Goal` instances define not only the target ΔNFR and
Sense Index averages but also longitudinal/lateral ΔNFR objectives and the
relative weighting between them.  These values are propagated to
:class:`Microsector` objects (``phase_axis_targets``/``phase_axis_weights``)
so downstream recommenders and exporters can highlight whether the
microsector demands longitudinal support (brake bias, differential locking)
or lateral balance (anti-roll bars, toe, alignments).

The ``filtered_measures`` mapping now exposes a structured ``"cphi"`` block
that mirrors :class:`~tnfr_lfs.core.metrics.CPHIReport`, including the
shared traffic-light thresholds so JSON exports and dashboards can reuse
the same red/amber/green semantics without recomputing the bands.

* ``nu_f_target`` – the weighted natural frequency of the fastest nodes in the
  phase.
* ``nu_exc_target`` – the dominant excitation frequency derived from steer and
  suspension inputs inside the phase window.
* ``rho_target`` – the detune ratio ``ν_exc/ν_f`` observed for the archetype,
  used to flag chassis setups that fall out of tune under load.
* ``slip_lat_window`` / ``slip_long_window`` – allowable slip bands derived
  from the telemetry means under lateral/longitudinal load.
* ``yaw_rate_window`` – the expected yaw-rate envelope for the phase.
* ``dominant_nodes`` – the subsystems whose ΔNFR signature anchors the goal.
* ``window_occupancy`` – percentage of telemetry samples that remain within
  each window for entry, apex, and exit.
* ``operator_events`` – grouped by ``AL``/``OZ``/``IL``/``SILENCE`` and enriched
  with the surface type derived from ``context_factors`` plus the contextual
  ΔNFR threshold (``delta_nfr_threshold``) for the microsector.  Each event
  lists the peak/mean ΔNFR observed in the window and its ratio to the
  threshold (``delta_nfr_ratio``), while the aggregate ``SILENCE`` entry adds
  average coverage and structural density to flag low-activation latent states.

```python
from tnfr_lfs.core.segmentation import segment_microsectors

microsectors = segment_microsectors(records, bundles)
first = microsectors[0]

print(first.index, first.active_phase)
for goal in first.goals:
    print(goal.phase, f"target ΔNFR {goal.target_delta_nfr:.1f}", goal.dominant_nodes)

print("Active nodes:", first.dominant_nodes[first.active_phase])
```

Each :class:`~tnfr_lfs.core.segmentation.Microsector` exposes the phase
objectives computed from the telemetry bundles, so a quick inspection of
``microsector.goals`` shows the ΔNFR/Sense Index targets and the dominant
subsystems for every entry/apex/exit slice. The ``dominant_nodes`` mapping keeps
the per-phase node leaders readily available, making it straightforward to cross
reference the active phase with its controlling chassis systems when diagnosing
support events.【F:src/tnfr_lfs/core/segmentation.py†L124-L220】【F:src/tnfr_lfs/core/segmentation.py†L420-L520】

## Recommendation Engine

### `tnfr_lfs.recommender.rules.RecommendationEngine`

Applies load balance, stability index, and coherence rules to produce a
list of :class:`tnfr_lfs.recommender.rules.Recommendation` objects.
Custom rules can be added by implementing the
:class:`tnfr_lfs.recommender.rules.RecommendationRule` protocol. The default
rule-set now evaluates the Useful Dissonance Ratio (UDR) to suggest stiffening
rear support/LSD when yaw impulses fail to tame ΔNFR, or to soften the
offending axle when UDR collapses.

### `tnfr_lfs.recommender.search`

```
tnfr_lfs.recommender.search.DecisionVariable
tnfr_lfs.recommender.search.DecisionSpace
tnfr_lfs.recommender.search.CoordinateDescentOptimizer
tnfr_lfs.recommender.search.objective_score(results: Sequence[EPIBundle], microsectors: Sequence[Microsector] | None = None) -> float
tnfr_lfs.recommender.search.SetupPlanner
tnfr_lfs.recommender.search.Plan
tnfr_lfs.recommender.search.SearchResult
```

The search utilities expose a bound-aware coordinate descent optimiser that
explores decision vectors tied to each car model.  The
:func:`objective_score` helper favours higher Sense Index values while
penalising ΔNFR integrals, and :class:`SetupPlanner` combines the optimiser
with the rule-based engine to emit an explainable :class:`Plan` object that
includes the telemetry trace, recommendations, the resulting decision vector,
and the Sense Control Index (SCI) with its component breakdown.

## Exporters

Two exporters are provided out of the box:

* ``json`` – produces a JSON document ready for storage or pipelines.
* ``csv`` – returns a CSV representation of the EPI series.
* ``markdown`` – renders setup plans as Markdown tables with aggregated
  rationales and expected effects.

```
tnfr_lfs.exporters.setup_plan.SetupChange
tnfr_lfs.exporters.setup_plan.SetupPlan
tnfr_lfs.exporters.setup_plan.serialise_setup_plan(plan: SetupPlan) -> Dict[str, Any]
```

The setup plan dataclasses model the actionable advice produced by the
CLI's ``write-set`` command and by :class:`SetupPlanner`.  Use
:func:`serialise_setup_plan` to convert the plan into a JSON-compatible
payload with deduplicated rationales and expected effects.  The
``sensitivities`` mapping now captures both ΔSi/Δp and
Δ∫|ΔNFR|/Δp entries, while ``phase_sensitivities`` stores the
per-phase Δ∫|ΔNFR|/Δp gradients used by the optimiser and the
profile persistence layer.  The exported mapping also includes the aggregate
SCI value and a fixed-order SCI breakdown so downstream dashboards can report
the optimisation balance directly.
