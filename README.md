# LFS Race Engineer

Real-time telemetry capture and **Race Engineer Studio** overlay for
[Live for Speed](https://www.lfs.net/). The package fuses OutSim +
OutGauge (UDP) with InSim (TCP) at 100 Hz into a single typed sample
stream, persists it to a versioned CSV, and renders it in a frameless
PySide6 overlay with live splits, traffic radar, fuel tracker, racing
line, MoTeC-style multi-channel viewer, sector breakdown, damper
histograms, and a setup tab driven by parsed `car_info.bin`.

* Package: `lfs-race-engineer` · entry points: `lfs-race-engineer`
  (Studio) and `lfs-telemetry` (CLI).
* Python ≥ 3.11 · `numpy`, `pandas`, `scipy` (core) ·
  `PySide6`, `pyqtgraph` (studio extra) · `pyinstaller` (build extra).
* CSV replay schema version `1.1`.
* 237 tests collected — 230 pass headless, 7 skipped without LFS or
  PySide6 display server.

```
            ┌──────────── LFS client ────────────┐
            │  UDP 30000   UDP 30001   TCP 29999 │
            │   OutSim      OutGauge      InSim  │
            └─────┬──────────┬────────────┬──────┘
                  ▼          ▼            ▼
            ┌──────────────────────────────────┐
            │  telemetry/protocol  (parsers)   │
            └────────────────┬─────────────────┘
                             ▼
            ┌──────────────────────────────────┐
            │  telemetry/live  (asyncio fuse)  │──► TelemetrySample
            └────────────────┬─────────────────┘            │
                             ▼                              │
            ┌──────────────────────────────────┐            │
            │ replay  (versioned CSV schema)   │◄───────────┘
            └────────────────┬─────────────────┘
                             ▼
                       ┌──────────┐
                       │   Lap /  │  enriched (~30 derived cols)
                       │  Stint / │  comparison, sectors, fuel,
                       │ Compare  │  predict, traffic, observables
                       └────┬─────┘
                            ▼
            ┌──────────────────────────────────┐
            │     studio  (PySide6 overlay)    │
            └──────────────────────────────────┘
```

## Repository layout

```
src/lfs_telemetry/
  __init__.py                 # package metadata (__version__ = 0.2.0)
  cli.py                      # lfs-telemetry: capture | calibrate | reslice
  lfs_config.py               # patch LFS\cfg.txt to emit OutSim/OutGauge/InSim
  telemetry/                  # framework-neutral telemetry core
    protocol/                 # OutSim, OutGauge, InSim wire-format parsers
    track/                    # .pth, .smx, .pin, .knw parsers + enrichment
    live.py                   # asyncio capture + fusion (LiveTelemetry)
    replay.py                 # CSV schema 1.1: write/read/detect
    lap.py                    # LapTelemetry facade
    stint.py                  # StintTelemetry multi-lap aggregator
    comparison.py             # LapComparison (distance-aligned + delta-time)
    sectors.py                # Sector / lap_sectors / sector_times_s
    lap_slicer.py             # canonical wrap-based per-lap splitter
    lap_summary.py            # LapRecord from IS_LAP/IS_SPX/IS_HLV/OBH
    lap_cache.py              # on-disk parquet cache for enriched DFs
    predict.py                # SplitPredictor (SPB + predicted lap)
    traffic.py                # TrafficSnapshot (gaps in m and s)
    fuel_tracker.py           # online fuel consumption + range estimator
    node_delta.py             # node-by-node delta against reference lap
    track_map.py              # averaged TrackMap / TrackBounds
    channels.py               # ChannelInfo registry (~123 columns)
    catalog.py                # discover_captures (workspace browser)
    derived.py                # enrich_dataframe (~30 derived columns)
    damper_histogram.py       # high/low speed damper duty histograms
    observables.py            # CarSpec + per-sample structural projection
    calibrate.py              # μ / mass estimators from raw telemetry
    car_calibration.py        # CarSpecStore (persistent calibration)
    car_info_bin.py           # parse LFS car_info.bin (setup defaults)
    live_publisher.py         # JSON snapshot for the Studio Live tab
    heading.py                # local-frame geometry helpers
  studio/                     # PySide6 race-engineer overlay
    __main__.py               # `python -m lfs_telemetry.studio`
    app.py                    # QApplication boot + signal/timer wiring
    main_window.py            # frameless QMainWindow with all docks
    theme.py                  # dark in-game palette and stylesheets
    signals.py                # cross-widget Qt signal bus
    workspace_state.py        # persisted UI state
    models/                   # Qt models (captures, channels, lap loader)
    charts/                   # pyqtgraph multi-channel chart + decimation
    widgets/                  # docks + center tabs (see studio README)
  app/                        # capture-process support (used by Studio)
    capture_runner.py         # spawn CLI as subprocess, watch stop file
    state.py                  # dataclass mirror of capture state
tests/                        # 237 tests (230 pass, 7 skipped without LFS)
scripts/                      # ops + dev helpers (see Scripts section)
tools/                        # binary-format research helpers
tracks/                       # per-variant elevation profiles + overviews
racing_lines/                 # generated reference racing lines (CSV + PNG)
config/cars.json              # bundled car defaults (mass, μ, geometry)
assets/                       # third-party references and sample stints
installer/                    # Inno Setup script for the Windows installer
build/                        # PyInstaller transient output
dist/                         # PyInstaller frozen Studio bundle
```

## Installation

```powershell
git clone https://github.com/<you>/lfs-race-engineer
cd lfs-race-engineer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[studio,dev]"
```

Run the test suite:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:QT_QPA_PLATFORM  = "offscreen"
python -m pytest -q --ignore="tests/test_racing_line_loader.py" --ignore="tests/studio/test_smoke.py"
# 230 passed, 7 skipped
```

The two ignored modules exercise the on-disk LFS install layout and a
full QApplication smoke test; they pass locally with a real LFS folder
and a display server present.

### Live integration tests against LFS

```powershell
# 1. Launch LFS, patch cfg.txt (see below), join a session.
$env:LFS_TELEMETRY_LIVE_TEST = "1"
python -m pytest tests/test_live_lfs_integration.py -v
```

Optional env vars consumed by the live test:
`LFS_TELEMETRY_HOST`, `LFS_TELEMETRY_INSIM_PORT`,
`LFS_TELEMETRY_OUTSIM_PORT`, `LFS_TELEMETRY_OUTGAUGE_PORT`,
`LFS_TELEMETRY_ADMIN`, `LFS_TELEMETRY_TIMEOUT`.

## Configuring LFS

LFS does not emit telemetry by default. The package can patch
`cfg.txt` automatically:

* **From Studio**: *Tools → Configure LFS…*, point at your LFS folder
  and click *Patch cfg.txt automatically* (LFS must be closed).
* **From a terminal**: `python -m lfs_telemetry.lfs_config "C:\path\to\LFS"`.

Manual snippet for `LFS\cfg.txt`:

```
OutSim Mode 1
OutSim Opts 1ff           ; OSO_ALL → extended OutSimPack2 (280 B)
OutSim Delay 1            ; 10 ms ≈ 100 Hz
OutSim IP 127.0.0.1
OutSim Port 30000
OutSim ID 0

OutGauge Mode 1
OutGauge Delay 1
OutGauge IP 127.0.0.1
OutGauge Port 30001
OutGauge ID 0

InSim Port 29999          ; TCP, optional but required for splits/laps
```

Without `OutSim Opts 1ff`, LFS falls back to the legacy 64-byte
`OutSimPacket`; per-wheel load / slip / suspension data is then
synthesized via quasi-static load transfer in `derived.enrich_dataframe`.

## Command-line interface

```powershell
# Studio overlay
lfs-race-engineer

# Capture 5 flying laps with InSim race context, write per-lap CSVs
lfs-telemetry capture captures\stint.csv --insim-host 127.0.0.1 --laps 5 --per-lap

# Re-slice an existing aggregate CSV into clean line-to-line per-lap files
lfs-telemetry reslice captures\stint.csv --out-dir captures\laps

# Calibrate mass + weight distribution from a rest window
lfs-telemetry calibrate --insim-host 127.0.0.1
```

### `capture` flags

| Flag | Default | Description |
| --- | --- | --- |
| `--outsim-port` / `--outgauge-port` | `30000` / `30001` | UDP ports |
| `--outsim-opts` | `0x1ff` | OutSim Opts mask, must match `cfg.txt` |
| `--insim-host` / `--insim-port` | *(off)* / `29999` | enable InSim TCP |
| `--insim-admin` | *(empty)* | LFS admin password |
| `--seconds N` | `0` | stop after N seconds (0 = no limit) |
| `--laps N` | `0` | stop after N flying laps (requires `--insim-host`) |
| `--warmup-laps K` | `0` | discard K extra laps at session start |
| `--per-lap` | off | also write one CSV per lap (timestamped) |
| `--no-aggregate` | off | with `--per-lap`, skip the combined CSV |
| `--debug-insim` | off | log every InSim packet + 5 s heartbeat |
| `--car` | *(auto)* | force LFS car short name (FOX, FO8, BF1, MRT…) |
| `--stop-file PATH` | — | sentinel file polled by the capture loop |
| `--live-file PATH` | — | JSON snapshot refreshed at ~10 Hz for Studio |

### `reslice` flags

| Flag | Default | Description |
| --- | --- | --- |
| `--out-dir` | *required* | destination directory for per-lap CSVs |
| `--min-drop-m` | `100.0` | min wrap to count as start/finish crossing |
| `--stem` | *(from input)* | filename stem for outputs |
| `--session-tag` | *(timestamp_car_track)* | tag inserted in each filename |

### `calibrate` flags

| Flag | Default | Description |
| --- | --- | --- |
| `--insim-host` | *required* | LFS host (capture cannot proceed without InSim) |
| `--seconds` | `5` | duration of the at-rest sample window |
| `--store` | `~/.lfs-telemetry/cars.json` | output `CarSpecStore` |

## Studio overview

```powershell
lfs-race-engineer
```

The frameless, dark, in-game-style overlay opens with these regions:

* **Captures** (left dock) — workspace browser of every CSV in the
  selected folder (`discover_captures`); double-click loads it.
* **Channels** (right dock) — channel browser grouped by `Vehicle`,
  `Engine`, `Driver`, `Chassis`, `Suspension`, `Tyre`, `Aids`,
  `Derived`, `Lap`, `Context`.
* **Track map** (left dock) — averaged racing line for the active lap
  with start/finish, current position cursor and overlay of reference
  racing line from `racing_lines/<TRACK>_racing.csv`.
* **Elevation** (left dock) — z(s) profile with banking and surface
  classification from `.smx` mesh enrichment.
* **Race dashboard** (right dock) — live splits, predicted lap, gap to
  best, traffic radar, fuel range, lap counter.
* **Central tabs**:
  - **Channels** — pyqtgraph multi-channel viewer with shared
    distance axis (decimated to ≤ 4 000 points/lane).
  - **Stint** — table of `StintTelemetry.per_lap` + trend lines.
  - **Sectors** — per-lap sector splits with best / theoretical-best.
  - **Dampers** — high/low-speed damper histograms per wheel.
  - **Setup** — parsed `car_info.bin` view (geometry, gearing,
    differential, brake bias, wing/aero, weight distribution).
  - **Capture** — start/stop controls for `lfs-telemetry capture`
    (driven via `app/capture_runner.py`).
  - **Live** — race-engineer overlay during a live session, fed by
    the JSON snapshot written by `live_publisher.write_snapshot_atomic`.

See [src/lfs_telemetry/studio/README.md](src/lfs_telemetry/studio/README.md)
for the full widget map and signal flow.

## Telemetry core

The `telemetry/` package is **independent of any UI framework** — it
only depends on `numpy` + `pandas` (and `asyncio` for live capture).
Any application (Studio, Jupyter notebooks, ML pipelines, classical
control) can sit on top of it.

* High-level entry points: `LiveTelemetry`, `LapTelemetry`,
  `StintTelemetry`, `LapComparison`, `SplitPredictor`, `TrafficSnapshot`.
* Persistence: `write_csv_replay`, `read_csv_replay`, `read_csv_dataframe`,
  `SCHEMA_VERSION`, `detect_schema_version` — full round-trip with no
  information loss.
* Per-lap slicer: `slice_into_laps`, `write_per_lap_files`, `reslice_csv`,
  `find_line_crossings` (`current_lap_dist_m` wraparound based).
* Sectors: `Sector`, `lap_sectors`, `sector_times_s`,
  `insim_split_distances_m`.
* Track geometry: parsers for `.pth` (AI driving line), `.smx` (mesh),
  `.pin` (env bbox), `.knw` (per-car AI knowledge) plus
  `compute_profile`, `segment_track`, `assign_segment`, `detect_track`.
* Racing line: `compute_edges`, `compute_geometric_line`,
  `compute_knw_line`, `compute_target_speed`
  (`scripts/racing_line_view.py` uses these to regenerate every
  `_racing.csv` in `racing_lines/`).
* Observables: `CarSpec`, `observe_sample`, `observe_window` (structural
  projection for setup analytics).
* Calibration: `estimate_mu_lat`, `estimate_mu_long`, `estimate_mass_kg`,
  `calibrate_spec`, `RestCalibrator`, `CarSpecStore`.
* Channel + capture metadata: `CHANNELS`, `ChannelInfo`, `channel_info`,
  `channels_by_group`, `CaptureInfo`, `discover_captures`,
  `inspect_capture`, `captures_to_dataframe`.

See [src/lfs_telemetry/telemetry/README.md](src/lfs_telemetry/telemetry/README.md)
for the full module-by-module API reference.

## Scripts and tools

`scripts/` contains operational helpers shipped with the repo:

| Script | Purpose |
| --- | --- |
| `racing_line_view.py` | Generate `<TRACK>_racing.csv` + `.png` for one variant or `--all`. Reads `.pth`, applies `segment_track`, picks the line source (`auto` / `knw` / `heuristic`) and renders a coloured speed map. |
| `track_view.py` | Render a single track outline with edges and centerline. |
| `motec_view.py` | MoTeC-style overlay of one or two CSVs from the CLI. |
| `wheel_view.py` | Per-wheel telemetry plots (load, slip, temperature). |
| `live_inspect.py` | Stream `LiveTelemetry` samples to stdout for diagnostics. |
| `sniff_udp.py` | Raw UDP dump for protocol debugging. |
| `probe_pth.py` | Inspect a `.pth` file (node count, length, climb). |
| `check_real_stint.py` | Sanity-check an aggregate stint CSV. |
| `patch_lfs_cfg.py` | Wrapper around `lfs_config.patch_cfg`. |
| `build_app.ps1` | Build the frozen Studio bundle with PyInstaller. |
| `pyi_runtime_chdir.py` | Frozen-bundle bootstrap so Studio finds `config/` and `racing_lines/`. |

`tools/` contains low-level binary-format research utilities:

| Tool | Purpose |
| --- | --- |
| `decode_knw.py` | Dump a `.knw` AI-knowledge file (per-segment best line). |
| `decode_wld.py` | Dump a `.wld` world-geometry file. |
| `full_lap_plots.py` / `full_lap_report.py` | Build a per-lap PDF/MD report. |
| `sniff_lfs_data.py` | Continuous capture of all UDP/TCP traffic. |
| `sniff_pth_trailing.py` | Probe trailing bytes of `.pth` for hidden fields. |

## Data folders

* `racing_lines/` — `<TRACK>_racing.csv` (per-node table) and `.png`
  (track map + speed band) for every Live for Speed variant.
  Regenerate with `python scripts/racing_line_view.py --all`. The
  aggregate `_racing_summary.csv` lists length, climb, min radius, max
  slope, mean width and speed range per variant. Each per-node CSV
  carries `drive_left_m`/`drive_right_m` (AI usable corridor) and
  `limit_left_m`/`limit_right_m` (track edges) in addition to the
  curvature, radius, slope and segment metadata.
* `tracks/` — per-variant elevation `.csv` profiles and `_overview.png`
  renders.
* `config/cars.json` — bundled `CarSpec` defaults (mass, μ_lat, μ_long,
  geometry, gearing) used when no calibrated entry exists in the user
  store `~/.lfs-telemetry/cars.json`.
* `assets/` — sample stints (`synthetic_*`, `helicorsa_*`, …) plus
  third-party references kept for documentation only.

## Building the Windows installer

```powershell
# Frozen Studio bundle
pyinstaller lfs-race-engineer.spec
# Then run Inno Setup against installer\lfs-race-engineer.iss
```

Outputs land in `dist\lfs-race-engineer\` and `installer\Output\`.

## Licence

MIT. See `pyproject.toml` and individual file headers.
