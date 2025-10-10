# Telemetry guide

TNFR × LFS derives every recommendation from the native OutSim and OutGauge
streams exposed by Live for Speed. This guide expands on the short overview in
the README, detailing the telemetry prerequisites and how each metric consumes
the simulator feeds.

## Required simulators settings

Enable the UDP/TCP broadcasters before running the toolkit:

* `/outsim 1 127.0.0.1 4123`
* `/outgauge 1 127.0.0.1 3000`
* `/insim 29999`

Extend the OutSim block with `OutSim Opts ff` in `cfg.txt` (or use `/outsim Opts ff`)
so that TNFR × LFS receives the player ID, wheel packet and driver inputs needed
by the fusion module.

### Asynchronous ingestion

The synchronous :class:`tnfr_lfs.ingestion.outsim_udp.OutSimUDPClient` and
:class:`tnfr_lfs.ingestion.outgauge_udp.OutGaugeUDPClient` remain available for
tight polling loops, but the toolkit now ships asynchronous counterparts for
applications built around :mod:`asyncio`.  Use
:class:`tnfr_lfs.ingestion.outsim_udp.AsyncOutSimUDPClient` and
:class:`tnfr_lfs.ingestion.outgauge_udp.AsyncOutGaugeUDPClient` when you need to
integrate telemetry into event-driven tasks without blocking the loop:

```python
import asyncio
from tnfr_lfs.ingestion.outsim_udp import AsyncOutSimUDPClient

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

* **ΔNFR / ∇NFR∥ / ∇NFR⊥** – combine the OutSim wheel packet loads (`Fz`, `ΔFz`),
  longitudinal/lateral accelerations, suspension deflections and OutGauge engine
  state to contextualise gradient distribution.
* **ν_f (natural frequency)** – uses load distribution, slip ratios/angles,
  yaw rate and driver style (`throttle`, `gear`) to classify each node and tune
  the spectral window.
* **C(t) (structural coherence)** – derives from the nodal ΔNFR distribution and
  ν_f bands, enriched with adhesion coefficients (`mu_eff_*`) and ABS/TC flags
  converted into lockup events by the fusion pipeline.
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
