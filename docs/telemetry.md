# Telemetry guide

TNFR × LFS derives every recommendation from the native OutSim and OutGauge
streams exposed by Live for Speed. This guide expands on the short overview in
the README, detailing the telemetry prerequisites and how each metric consumes
the simulator feeds.

## Required simulator settings

Enable the UDP/TCP broadcasters before running the toolkit:

* `/outsim 1 127.0.0.1 4123`
* `/outgauge 1 127.0.0.1 3000`
* `/insim 29999`

The HUD subcommands reuse the same ports, so the configuration can live either in
`cfg.txt` (persistent) or be set ad-hoc via the in-sim console commands above.

### Extended payloads

TNFR × LFS derives most metrics from the OutSim wheel packet and the OutGauge
extended block. Make sure those richer payloads are available:

* Add `OutSim Opts ff` to `cfg.txt` (or run `/outsim Opts ff`) to expose the
  player index, wheel loads/forces/deflections and driver inputs required by the
  fusion module.
* Enable the OutGauge extensions so the dashboard exporter sends the temperature
  and pressure profiles that feed the tyre/thermal indicators. At minimum set
  `OutGauge Opts OG_EXT_TYRE_TEMP OG_EXT_TYRE_PRESS OG_EXT_BRAKE_TEMP` (or the
  equivalent bitmask value `0x0007`) before launching the session.

When Live for Speed omits any of those blocks, TNFR × LFS keeps the last valid
sample or reports "no data" instead of inventing synthetic readings. CSV-based
imports follow the same rule: the reader preserves missing columns as
``math.nan`` so the downstream metrics surface the absence explicitly.

### Asynchronous ingestion

The synchronous :class:`tnfr_lfs.telemetry.outsim_udp.OutSimUDPClient` and
:class:`tnfr_lfs.telemetry.outgauge_udp.OutGaugeUDPClient` remain available for
tight polling loops, but the toolkit now ships asynchronous counterparts for
applications built around :mod:`asyncio`.  Use
:class:`tnfr_lfs.telemetry.outsim_udp.AsyncOutSimUDPClient` and
:class:`tnfr_lfs.telemetry.outgauge_udp.AsyncOutGaugeUDPClient` when you need to
integrate telemetry into event-driven tasks without blocking the loop:

```python
import asyncio
from tnfr_lfs.telemetry.outsim_udp import AsyncOutSimUDPClient

async def main() -> None:
    async with AsyncOutSimUDPClient(port=4123) as outsim:
        packet = await outsim.recv()
        if packet is not None:
            try:
                print(packet.time, outsim.statistics)
            finally:
                packet.release()

asyncio.run(main())
```

Both async clients expose the same ``statistics`` fields as their synchronous
peers and use the shared circular reordering buffer to provide identical packet
accounting.  Call :meth:`drain_ready` in either mode to fetch packets that are
already sequenced without waiting on additional datagrams.  The decoded packets
come from reusable pools—return them via :meth:`release` once processed or call
``from_bytes(..., freeze=True)`` when you need immutable snapshots for archival
workflows.

## Metric inputs

Every indicator consumes a specific slice of the OutSim/OutGauge feeds. Refer
back to this list when validating telemetry captures or wiring custom data
sources:

* **ΔNFR / ∇NFR∥ / ∇NFR⊥** – integrate the OutSim wheel loads (`Fz`, `ΔFz`), wheel
  forces and suspension deflections together with engine regime, pedal inputs and
  ABS/TC lamps from OutGauge. Whenever a recommendation depends on absolute
  forces, compare the nodal readings against the raw `Fz`/`ΔFz` channels provided
  by the extended OutSim payload.
* **ν_f (natural frequency)** – leverages the load distribution, `slip_ratio_*`
  and `slip_angle_*` channels, `speed` and `yaw_rate` emitted by OutSim combined
  with driver-style cues (`throttle`, `gear`) from OutGauge to classify the
  resonant bands. The fusion pipeline aggregates slip as a load-weighted
  four-wheel average and falls back to the kinematic estimate only when the
  simulator omits the per-wheel data.
* **C(t) (structural coherence)** – built from the nodal ΔNFR distribution, the
  derived adhesion coefficients (`mu_eff_*`) and the ABS/TC activity reported by
  OutGauge. Lock-up events detected in the fusion stage keep coherence aligned
  with the HUD and CLI recommendations.
* **Ackermann budgets / Slide Catch** – depend on the OutSim per-wheel
  `slip_angle_*` channels and the shared `yaw_rate`. Missing angles translate to
  `"no data"` in the CLI and HUD instead of fabricating the geometry.
* **Aero balance drift (`aero_balance_drift`)** – summarises rake from the OutSim
  `pitch` signal, axle suspension travel and the longitudinal μ contrast without
  using synthetic inputs.
* **Tyre temperatures/pressures** – consume the OutGauge extended block when the
  samples are positive and finite. Disable the block and the pipeline freezes the
  last sample or yields `"no data"` to highlight the missing source.
* **CPHI (Contact Patch Health Index)** – requires per-wheel `slip_ratio`,
  `slip_angle`, lateral/longitudinal forces and wheel loads from the extended
  OutSim packet. Without those signals the planner flags the patch as
  `"no data"`, preventing tyre-balance interventions. Reports and the HUD reuse
  the shared 0.62/0.78/0.90 thresholds to colour the indicator.
* **Ventilation / brake fade** – consumes OutGauge brake temperatures when
  available; otherwise the thermal proxy keeps the series continuous.

## Brake thermal proxy

Live for Speed omits brake temperatures for many cars. The TNFR × LFS proxy fills
those gaps by integrating mechanical work (`m·a·v`) weighted by wheel load,
applying convective cooling proportional to vehicle speed and bounding the series
between ambient temperature and a 1200 °C ceiling. Configuration options reside
in the `[thermal.brakes]` section of `config/global.toml`, with per-car overrides
under `data/cars/*.toml`. Override the behaviour at runtime with the
`TNFR_LFS_BRAKE_THERMAL` environment variable (`auto`, `off`, `force`).

### Limitations

* Incoming OutGauge readings are preserved as-is to avoid skewing live sensors.
* Low sampling rates (<10 Hz) can understate peaks when OutSim is the only source.
* The proxy assumes OutSim/OutGauge packets arrive within Live for Speed’s
  typical latency window (one or two packets).

## Further reading

* [CLI guide](cli.md)
* [Setup equivalences](setup_equivalences.md)
* [Brake thermal proxy deep dive](brake_thermal_proxy.md)
