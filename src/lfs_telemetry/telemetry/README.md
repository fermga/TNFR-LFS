# `lfs_telemetry.telemetry` — comprehensive reference

Capture, persistence, and analysis core. It is **UI-framework agnostic**;
public classes expose `numpy`/`pandas` and can be used from Studio,
notebooks, ML pipelines, or classic controllers.

- CSV schema: `SCHEMA_VERSION = "1.1"` (preamble
  `# lfs-telemetry telemetry schema=1.1`).
- Extended OutSim: `OSO_ALL = 0x1ff` -> `OutSimPack2` (280 B, with
  per-wheel payload and track index).
- Nominal sampling rate: 100 Hz.

---

## 0. Configure LFS

Without telemetry enabled in `cfg.txt`, `LiveTelemetry` / `InSimClient`
will never receive packets. You can configure LFS in three ways:

- `python -m lfs_telemetry.lfs_config "C:\\path\\to\\LFS"`
- Studio: **Tools -> Configure LFS…**
- Manual edit (see root `README.md`).

`lfs_config.patch_cfg(lfs_dir)` inserts/updates required lines, preserves
unrelated values, and creates a `.bak` backup.

---

## 1. Architecture

```text
                ┌─ protocol.packets ─┐   ┌─ protocol.insim ──┐
   UDP 30000 ──►│  OutSimPacket /    │   │ InSimClient (TCP) │◄─ TCP 29999
   UDP 30001 ──►│  OutSimPack2 /     │   │  RaceContext,     │
                │  OutGaugePacket    │   │  PitStopRecord    │
                └──────────┬─────────┘   └─────────┬─────────┘
                           ▼                       ▼
                ┌──────────────────────────────────────────┐
                │   live.LiveTelemetry  (asyncio fuse)     │
                │   -> TelemetrySample (slots dataclass)   │
                └────────────────────┬─────────────────────┘
                                     ▼
   ┌────────────────────────┬───────────────────────┬────────────────────┐
   │ replay.write_csv_      │  lap_summary          │ live_publisher     │
   │ replay / read_csv_*    │  (LapRecord stream)   │ (JSON snapshot)    │
   └──────────┬─────────────┴───────────────────────┴────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ lap.LapTelemetry  ->  stint.StintTelemetry  ->  comparison.Lap*    │
   │                       sectors / fuel_tracker / predict / traffic    │
   │                       derived.enrich_dataframe / damper_histogram   │
   │                       observables.observe_window / calibrate.*      │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Package layout

| Module | Responsibility |
| --- | --- |
| `live` | Async loop that listens to OutSim + OutGauge + InSim and emits `TelemetrySample`. |
| `replay` | Serialize/deserialize stream to CSV 1.1 with metadata header. |
| `lap` | Per-lap view over one CSV (`LapTelemetry`). |
| `stint` | Multi-lap aggregator (`StintTelemetry`). |
| `comparison` | Distance-aligned `LapComparison` + delta time. |
| `sectors` | `Sector` type and split utilities. |
| `lap_slicer` | Canonical lap slicing from `current_lap_dist_m`. |
| `lap_summary` | `LapRecord` built from InSim `IS_LAP/SPX/HLV/OBH`. |
| `lap_cache` | Parquet cache for enriched DataFrames. |
| `predict` | `SplitPredictor` (personal sector best + projected lap). |
| `traffic` | `TrafficSnapshot` for nearby cars. |
| `fuel_tracker` | Online fuel usage/range estimation. |
| `node_delta` | Node-by-node delta against a reference lap. |
| `track_map` | `TrackMap` averaged from several laps. |
| `channels` | `ChannelInfo` registry (~123 columns). |
| `catalog` | CSV discovery/inspection inside a workspace. |
| `derived` | `enrich_dataframe`: adds ~30 derived columns. |
| `damper_histogram` | HS/LS damper histograms per wheel. |
| `observables` | `CarSpec` + structural observation (sample/window). |
| `calibrate` | Mu and mass estimators from raw telemetry. |
| `car_calibration` | Persistent user calibration (`CarSpecStore`) + `RestCalibrator`. |
| `car_info_bin` | Parser for LFS `car_info.bin`. |
| `live_publisher` | JSON snapshot consumed by Studio Live tab. |
| `heading` | Local-frame projection helpers. |
| `protocol/` | Low-level packet parsers (OutSim, OutGauge, InSim). |
| `track/` | `.pth`, `.smx`, `.pin`, `.knw` parsers + geometry/racing line. |

`telemetry/__init__.py` re-exports the public API from `__all__`, so
`from lfs_telemetry.telemetry import ...` is enough.

---

## 3. Live capture (`live.py`)

```python
from lfs_telemetry.telemetry import LiveTelemetry

async def main():
    live = LiveTelemetry(
        outsim_port=30000,
        outgauge_port=30001,
        outsim_opts=0x1ff,
        insim_host="127.0.0.1",
        insim_port=29999,
    )
    async with live.session():
        async for sample in live.stream():
            print(sample.t_capture_s, sample.speed_ms, sample.rpm)
```

`TelemetrySample` is a slots dataclass (~70 fields) with capture/sim
clock, pose, velocities, accelerations, driver inputs, fuel, gear, RPM,
race context, per-wheel data, and projected `current_lap_dist_m`.

---

## 4. CSV schema 1.1 (`replay.py`)

```text
# lfs-telemetry telemetry schema=1.1
# created_utc=2024-…
# car=FOX track=BL1 mass_kg=… mu_lat=… …
t_capture_s,t_sim_s,pos_x,…,current_lap_dist_m,indexed_distance_m,…
```

- `write_csv_replay(samples, path, metadata=...)`
- `read_csv_replay(path)`
- `read_csv_dataframe(path, enrich=True)`
- `detect_schema_version(path)`

Round-trip is lossless for samples generated by `write_csv_replay`.

---

## 5. `LapTelemetry`

```python
from lfs_telemetry.telemetry import LapTelemetry

lap = LapTelemetry.from_csv("captures/BL1_lap03.csv")
df = lap.dataframe()
lap.duration_s
lap.distance_m
lap.average_speed_ms
lap.car
lap.track
lap.metadata
```

`LapTelemetry` keeps lazy cached properties for enriched DataFrame and
headline metrics.

---

## 6. `StintTelemetry`

```python
from lfs_telemetry.telemetry import StintTelemetry

stint = StintTelemetry.from_dir("captures/session_01/")
stint = StintTelemetry.from_csvs(["lap01.csv", "lap02.csv"])
stint = StintTelemetry.from_laps([lap1, lap2])
```

Views and helpers:

- `stint.per_lap`
- `stint.trends(window=3)`
- `stint.average_lap_time(mode="stint"|"clean"|"total"|"rolling")`
- `stint.race_start_lap_indices`, `stint.clean_lap_indices`
- `stint.mark_lap_invalid(idx, reason)`
- `stint.mark_invalid_from_records(lap_records)`

---

## 7. `LapComparison`

```python
from lfs_telemetry.telemetry import LapComparison

cmp = LapComparison.from_laps(lap_ref, lap_chal,
                              channels=("speed_ms", "throttle"))
cmp.distance_m
cmp.reference["speed_ms"]
cmp.challenger["speed_ms"]
cmp.delta_time_s
```

Distance alignment and interpolation are handled internally by
`comparison.py` utilities.

---

## 8. Sectors, splits, and track map

```python
from lfs_telemetry.telemetry import (
    Sector, lap_sectors, sector_times_s, insim_split_distances_m,
    TrackMap,
)

sectors = lap_sectors(lap, splits_m=insim_split_distances_m(record))
times_s = sector_times_s(lap, sectors)

track = TrackMap.from_laps([lap1, lap2, lap3])
xy = track.xy_along_distance(d_m=750.0)
```

---

## 9. Traffic and lap records

```python
from lfs_telemetry.telemetry import traffic_snapshot
snap = traffic_snapshot(local_sample, neighbours)

from lfs_telemetry.telemetry import build_lap_records
records = build_lap_records(insim_events)
```

`LapRecord` merges `IS_LAP`, `IS_SPX`, `IS_HLV`, and `OBH` into
one row per lap, including validity flags.

---

## 10. Channels and catalog

```python
from lfs_telemetry.telemetry import (
    CHANNELS, channel_info, channels_by_group,
    discover_captures, captures_to_dataframe,
)

CHANNELS["speed_ms"].group
channels_by_group()["Tyre"]
```

Groups include: `Driver`, `Engine`, `Vehicle`, `Chassis`, `Suspension`,
`Tyre`, `Derived`, `Aids`, `Lap`, `Context`.

---

## 11. Observables and calibration

```python
from lfs_telemetry.telemetry import (
    car_spec_for, observe_sample, observe_window,
    estimate_mu_lat, estimate_mu_long, estimate_mass_kg,
    calibrate_spec, CarSpecStore,
)

spec = car_spec_for("FOX")
obs = observe_sample(sample, spec)
mu_lat = estimate_mu_lat(df)
```

`observables.py` uses `threading.RLock()` around mutable caches for
safe concurrent access.

---

## 12. Derived channels (`derived.enrich_dataframe`)

`enrich_dataframe(df)` adds ~30 computed columns, including:

- chassis dynamics,
- longitudinal/lateral load transfer,
- friction-use estimates,
- tyre work,
- brake-bias metrics,
- damper velocities,
- dash-light decoding,
- FFB-derived signals,
- track geometry projection.

Only `enrich_dataframe` is public from this module.

---

## 13. `protocol/` subpackage

### `protocol.packets`

Includes packet structs and helpers:

- `OutSimPacket`, `OutSimPack2`, `OutGaugePacket`
- `InSimHeader`, `InSimVersion`
- `decode_dash_lights`, `decode_pit_work`, `penalty_name`, etc.

### `protocol.insim`

- `InSimClient` (async TCP client)
- `RaceContext`
- `PitStopRecord`

---

## 14. `track/` subpackage

- `track.pth`: path/profile parsing (`PthNode`, `Path`, `TrackProfile`)
- `track.smx`: 3D mesh parsing (`SmxObject`, `SmxMesh`)
- `track.pin`: environment bounding metadata
- `track.knw`: AI knowledge files
- `track.geom3d`: banking/surface/corridor geometry helpers
- `track.enrich`: geometry enrichment for DataFrames
- `track.racing_line`: geometric or KNW-driven racing-line generation
- `track.loader`: cached lookup and loading for track geometry assets

---

## 15. Misc utilities

- `lap_cache`: parquet-based cache utilities
- `fuel_tracker.FuelTracker`
- `node_delta.NodeDeltaTracker`
- `damper_histogram`
- `live_publisher.write_snapshot_atomic` (atomic JSON publish)

---

## 16. Usage patterns

### Load a stint and compare laps

```python
from lfs_telemetry.telemetry import StintTelemetry, LapComparison

stint = StintTelemetry.from_dir("captures/session_01/")
best = stint.laps[stint.per_lap["time_s"].idxmin()]
last = stint.laps[-1]
cmp = LapComparison.from_laps(best, last,
                              channels=("speed_ms", "throttle"))
```

### Full live-to-CSV pipeline

```python
import asyncio
from lfs_telemetry.telemetry import LiveTelemetry, write_csv_replay

async def run():
    live = LiveTelemetry(insim_host="127.0.0.1")
    samples = []
    async with live.session():
        async for s in live.stream():
            samples.append(s)
            if len(samples) > 36000:
                break
    write_csv_replay(samples, "captures/session.csv",
                     metadata={"driver": "F", "track": "BL1"})

asyncio.run(run())
```

---

## 17. Implementation conventions

- **Concurrency**: mutable shared caches are lock-protected where needed.
- **Schema versioning**: incompatible CSV changes require schema bump.
- **Public contract**: explicit `__all__` in core modules.
- **Immutability**: protocol/track dataclasses use `slots=True`
  (`frozen=True` where appropriate).
- **Encoding**: UTF-8 CSV read/write and UTF-8-friendly tooling.
