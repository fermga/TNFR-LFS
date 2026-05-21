# LFS Race Engineer

Real-time telemetry capture and **Race Engineer Studio** overlay for
[Live for Speed](https://www.lfs.net/). The package fuses OutSim +
OutGauge (UDP) with InSim (TCP) at ~100 Hz into a single typed sample
stream, persists it to a versioned CSV, and renders it in a frameless
PySide6 desktop application with a live race-engineer HUD, multi-channel
viewer, sector splits, fuel tracker, racing line, damper histograms and
a stint browser.

* Package: `lfs-race-engineer` · entry points: `lfs-race-engineer`
  (Studio GUI) and `lfs-telemetry` (CLI).
* Python ≥ 3.11 · core: `numpy`, `pandas`, `scipy` · `[studio]` extra:
  `PySide6`, `pyqtgraph` · `[vr]` extra: `openvr` (Windows) · `[build]`
  extra: `pyinstaller` · `[scripts]` extra: `matplotlib` (for the
  stand-alone viewers under `scripts/` and `tools/`).
* Studio UI is fully bilingual (English / Español) via an in-process
  `QTranslator`; the active language is chosen under *View → Language*
  and persisted in `QSettings`.
* CSV replay schema version `1.1` (preamble
  `# lfs-telemetry telemetry schema=1.1`).
* Test suite: **307 passed, 15 skipped** running headless
  (`QT_QPA_PLATFORM=offscreen`). The 15 skips require either a live LFS
  session (`tests/test_live_lfs_integration.py`) or on-disk LFS data
  (`C:\LFS\data\smx`, `…\car_info.bin`, etc.).

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
            │     studio  (PySide6 UI + HUD)   │
            └──────────────────────────────────┘
```

## Repository layout

```
src/lfs_telemetry/
  __init__.py                 # package metadata (__version__ from importlib.metadata)
  __main__.py                 # python -m lfs_telemetry → Studio
  app_paths.py                # bundled-asset / racing-line / doc / config path lookup
  lfs_config.py               # patch LFS\cfg.txt to emit OutSim/OutGauge
  lfs_paths.py                # detect LFS install + data folders
  constants.py                # unit-conversion constants (SPEED_MS_TO_KMH, …)
  cli/                        # `lfs-telemetry` subcommand sub-package
    __init__.py               # main() + argparse dispatch (capture/calibrate/reslice/raf-import)
    _common.py                # _add_lfs_flags, _harden_std_streams, _ResilientTextStream, _request_stop
    _state.py                 # STOP_REQUESTED / CAPTURE_LOOP / CAPTURE_TASK globals
    capture.py                # _cmd_capture (UDP+InSim loop, per-lap writer, live-snapshot writer)
    calibrate.py              # _cmd_calibrate (auto mass + weight distribution)
    reslice.py                # _cmd_reslice (split aggregate CSV into per-lap files)
    raf_import.py             # _cmd_raf_import (LFS .raf replay → per-lap CSVs)
  telemetry/                  # framework-neutral telemetry core
    protocol/                 # OutSim, OutGauge, InSim wire-format parsers
      packets.py              # OutGauge + InSim packets (+ re-exports of OutSim)
      packets_outsim.py       # OutSimPacket / OutSimPack2 / OutSimWheel + OSO_* / OUTSIM_* constants
      insim.py                # InSimClient + IS_* event types
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
    channels.py               # ChannelInfo registry + EN labels (~123 columns)
    channel_interpretations.py # EN + ES interpretation/focus tables for ChannelInfo
    i18n_es.py                # Spanish group/label/description/interpretation tables
    catalog.py                # discover_captures (workspace browser)
    derived.py                # enrich_dataframe (~30 derived columns)
    damper_histogram.py       # high/low speed damper duty histograms
    observables.py            # CarSpec + per-sample structural projection
    calibrate.py              # μ / mass estimators from raw telemetry
    car_calibration.py        # CarSpecStore (persistent calibration)
    car_info_bin.py           # parse LFS car_info.bin (setup defaults)
    raf.py                    # parse LFS .raf replay analyser files
    live_publisher.py         # JSON snapshot for the Studio Live tab
    heading.py                # local-frame geometry helpers
  studio/                     # PySide6 Race Engineer UI
    __main__.py               # `python -m lfs_telemetry.studio`
    app.py                    # QApplication boot + signal/timer wiring
    main_window.py            # frameless QMainWindow with all docks
    theme.py                  # dark in-game palette + semantic colours (LED, proximity, status)
    signals.py                # cross-widget Qt signal bus
    workspace_state.py        # persisted UI state
    i18n.py                   # English/Spanish translation table + tr()
    models/                   # Qt models (captures, channels, lap loader)
    charts/                   # pyqtgraph multi-channel chart + decimation
    widgets/                  # docks + center tabs (see studio README)
      live_modules/           # floating overlay-window sub-package
        _base.py              # _LiveModuleWindow / _LabeledValueWindow base classes
        simple.py             # Position / FuelPct / FuelLapsRemaining / Speed
        inputs.py             # Gear / Rpm / throttle / brake / clutch
        gaps.py               # GapAhead / GapBehind
        session.py            # SessionInfoWindow (dynamic standings)
        diagnostics.py        # Flags / PitLimiter / TcAbs / GMeter
        tyre_risk.py          # per-wheel grip/risk indicator
        compass_map.py        # GapCompass + MiniMap
        radar.py              # 360° traffic radar
        delta_bar.py          # DeltaBar / SpeedDeltaBar
    vr/                       # SteamVR / OpenVR mirror (extra `[vr]`)
      openvr_overlay.py       # IVROverlay panel manager
  app/                        # capture-process support (used by Studio)
    capture_runner.py         # spawn CLI as subprocess, watch stop file
    state.py                  # dataclass mirror of capture state
tests/                        # 322 tests collected (307 pass headless, 15 skipped)
scripts/                      # ops + dev helpers (see Scripts section)
tools/                        # binary-format research helpers
tracks/                       # per-variant elevation profiles + overviews
racing_lines/                 # generated reference racing lines (CSV + PNG)
config/cars.json              # bundled car defaults (mass, μ, geometry)
assets/                       # third-party references, sample stints, icon
installer/                    # Inno Setup script for the Windows installer
build/                        # PyInstaller transient output (gitignored)
dist/                         # PyInstaller frozen Studio bundle (gitignored)
```

## Installation (developer mode)

```powershell
git clone https://github.com/fermga/TNFR-LFS
cd TNFR-LFS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[studio,dev]"
```

Run the test suite headless:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:QT_QPA_PLATFORM  = "offscreen"
python -m pytest -q
# 307 passed, 15 skipped
```

The 15 skipped tests exercise the on-disk LFS install layout
(`C:\LFS\data\smx\*.smx`, `…\car_info.bin`) and the live integration
suite; they run only with a real LFS folder and a live session (see
below).

### Live integration tests against LFS

```powershell
# 1. Launch LFS, patch cfg.txt (see "Configuring LFS"), join a session.
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
OutSim Mode 2             ; 2 = extended OutSimPack2 (280 B)
OutSim Opts 1ff           ; OSO_ALL → full extended payload
OutSim Delay 1            ; 10 ms ≈ 100 Hz
OutSim IP 127.0.0.1
OutSim Port 30000
OutSim ID 0

OutGauge Mode 1
OutGauge Delay 1
OutGauge IP 127.0.0.1
OutGauge Port 30001
OutGauge ID 0
```

InSim has no `cfg.txt` key — start it at runtime with `/insim 29999`
in the LFS console (or launch `LFS.exe /insim=29999`). Writing an
`InSim Port` line into `cfg.txt` makes LFS show a red "unknown
setting" warning.

Without `OutSim Opts 1ff`, LFS falls back to the legacy 64-byte
`OutSimPacket`; per-wheel load / slip / suspension data is then
synthesized via quasi-static load transfer in
`derived.enrich_dataframe`.

## Command-line interface (`lfs-telemetry`)

```powershell
# Studio overlay (graphical app)
lfs-race-engineer

# Capture an open-ended stint, writing every full lap as a CSV
lfs-telemetry capture captures\stint.csv --insim-host 127.0.0.1 --per-lap

# Re-slice an existing aggregate CSV into clean line-to-line per-lap files
lfs-telemetry reslice captures\stint.csv --out-dir captures\laps

# Calibrate mass + weight distribution from a rest window
lfs-telemetry calibrate --insim-host 127.0.0.1

# Convert an LFS .raf replay file into per-lap CSVs (Studio-compatible schema)
lfs-telemetry raf-import path\to\replay.raf --out-dir captures\raf_laps
```

### `capture` flags

| Flag | Default | Description |
| --- | --- | --- |
| `output` (positional) | — | aggregate CSV path. Still required with `--no-csv` (used to derive the live-snapshot folder); nothing is written to it in that case. |
| `--outsim-port` / `--outgauge-port` | `30000` / `30001` | UDP listening ports. |
| `--outsim-opts` | `0x1ff` (`OSO_ALL`) | OutSim Opts bitmask; must match `cfg.txt`. |
| `--insim-host` / `--insim-port` | *(off)* / `29999` | enable InSim TCP client. |
| `--insim-admin` | `""` | LFS admin password (only if the server requires one). |
| `--car` | *(auto)* | force LFS car short name (FOX, FO8, BF1, MRT…). |
| `--seconds N` | `0` | stop after N seconds (0 = no limit). |
| `--laps N` | `0` | stop after N completed flying laps (requires `--insim-host`). |
| `--warmup-laps K` | `0` | discard K extra full laps at session start. |
| `--trim-out-lap` | off | drop samples taken before the first start/finish crossing. Default keeps the out-lap so the user decides later. |
| `--per-lap` | off | also write one CSV per completed lap, tagged with timestamp, car and track. |
| `--no-aggregate` | off | with `--per-lap`, skip the combined CSV. |
| `--include-out-lap` | off | also write the out-lap as `_lap00.csv`. Disables `--warmup-laps`. |
| `--wait-on-track` | off | keep retrying the InSim connection and discard samples below ~3 m/s until the car actually moves. Implied by `--include-out-lap`. |
| `--debug-insim` | off | log every InSim packet plus a 5 s heartbeat with the current race context. |
| `--stop-file PATH` | — | sentinel file polled by the capture loop; capture stops cleanly when the file appears (used by the Studio Capture tab). |
| `--live-file PATH` | — | JSON snapshot refreshed at ~10 Hz with race state (position, splits, gap, fuel, traffic, radar). Consumed by the Studio Overlay tab; implies an InSim MCI subscription. |
| `--no-csv` | off | **overlay-only mode**: connect to LFS and keep `--live-file` updated, but do **not** buffer samples in memory and do **not** write any CSV (per-lap or aggregate). |

`SIGINT` (Ctrl+C) and `SIGBREAK` (`CTRL_BREAK_EVENT` on Windows) both
trigger a clean shutdown that flushes any buffered samples to CSV
before exiting.

### `reslice` flags

| Flag | Default | Description |
| --- | --- | --- |
| `input` (positional) | — | aggregate CSV produced by `lfs-telemetry capture`. |
| `--out-dir` | input's directory | destination directory for per-lap CSVs. |
| `--stem` | input filename stem | filename stem for outputs. |
| `--suffix` | `.csv` | output suffix. |
| `--session-tag` | `""` | optional tag inserted between stem and `lapNN`. |
| `--min-drop-m` | `100.0` | minimum negative jump in `current_lap_dist_m` that counts as a start/finish crossing. |

### `calibrate` flags

| Flag | Default | Description |
| --- | --- | --- |
| `--insim-host` | *required* | LFS host (calibration cannot proceed without InSim). |
| `--insim-port` | `29999` | TCP port for InSim. |
| `--insim-admin` | `""` | LFS admin password if needed. |
| `--car` | *(auto)* | force LFS car short name. |
| `--timeout` | `120.0` | give up if no rest window is detected in N seconds. |
| `--store` | `~/.lfs-telemetry/cars.json` | destination `CarSpecStore` JSON. |
| `--show` | off | just print existing store contents and exit. |

### `raf-import` flags

Converts an LFS Replay Analyser File (`.raf`, v2) into one CSV per
detected lap using the Studio schema (1.1), so RAF replays — including
those recorded by other drivers — can be opened from the Captures dock
for cross-driver comparison.

| Flag | Default | Description |
| --- | --- | --- |
| `input` (positional) | — | `.raf` file produced by LFS. |
| `--out-dir` | `<input>_raf_laps` next to the file | output directory for per-lap CSVs. |
| `--keep-outlap` | off | also export the lead-in partial lap (before the first start/finish crossing). |
| `--min-samples` | `100` | discard lap segments shorter than this many samples. |
| `--inspect N` | `0` | diagnostic: parse but do NOT write CSVs; dump header + first N decoded blocks. |

## Race Engineer Studio

```powershell
lfs-race-engineer
```

The frameless dark in-game-style window opens with these regions:

### Dockable panels

| Position | Dock | Widget | Role |
| --- | --- | --- | --- |
| Left | Captures | `widgets/captures_dock.py` | Workspace browser of every CSV under the current folder (`discover_captures`). Double-click loads a lap; drag-drop changes the workspace. |
| Left | Track map | `widgets/track_map_dock.py` | Averaged racing line of the active lap with start/finish marker, current-position cursor, and overlay of the reference line from `racing_lines/<TRACK>_racing.csv`. |
| Left | Elevation | `widgets/track_elevation_dock.py` | z(s) profile with banking bands and surface classification from the `.smx` mesh. |
| Right | Channels | `widgets/channels_dock.py` | Channel tree grouped by `Vehicle`, `Engine`, `Driver`, `Chassis`, `Suspension`, `Tyre`, `Aids`, `Derived`, `Lap`, `Context`, with a text filter and per-channel toggle for the multi-chart. |
| Right | Race dashboard | `widgets/race_dashboard_dock.py` | Live splits, predicted lap, gap to best, traffic radar, fuel range and lap counter. |

### Central tabs (`widgets/center_tabs.py`)

Order in the build, top to bottom:

1. **Telemetry** — `widgets/charts_dock.py` + `charts/multi_chart.py`.
   MoTeC-style multi-channel viewer over `pyqtgraph`. Lanes share a
   distance- or time-based X axis, the cursor is synchronised across
   lanes, a delta-vs-reference overlay can be toggled, and each lane
   is decimated to ≤ 4 000 points via a min-max-per-bucket algorithm
   that preserves peaks.
2. **Dampers** — `widgets/dampers_tab.py`. High-speed / low-speed
   damper duty histograms per wheel from
   `telemetry.damper_histogram`.
3. **Sectors** — `widgets/sectors_tab.py`. Per-lap sector splits with
   running best and theoretical-best, sourced from `lap_sectors` /
   `sector_times_s` and InSim split distances.
4. **Stint** — `widgets/stint_tab.py`. `StintTelemetry.per_lap` table
   (lap times, fuel, tyre temperatures, suspension load, friction
   usage, damper work) plus rolling trend lines across the loaded
   stint.
5. **Capture** — `widgets/capture_tab.py`. Start/stop controls for
   `lfs-telemetry capture`, driven by `app/capture_runner.py`. The
   form exposes the filename stem, InSim host/port, OutSim port,
   OutGauge port and an **Overlay only (no CSV recording)** checkbox
   that switches the underlying CLI to `--no-csv` so the Overlay tab
   keeps updating in real time without writing any CSV to the
   workspace. A coloured LED reflects InSim status (grey = idle,
   amber = waiting, green = connected), and the embedded log mirrors
   the child-process stderr.
6. **Overlay** — `widgets/live_tab.py`. Race-engineer HUD fed by the
   JSON snapshot written by `live_publisher.write_snapshot_atomic`
   and watched by `widgets/live_data_source.py`. The HUD is composed
   of independent frameless always-on-top windows defined under
   `widgets/live_modules/` (see the studio README for the full module
   list): 360° traffic radar, delta-vs-reference strip, predicted lap,
   gap to best, fuel range, mini-map cursor, plus flag / pit-window /
   penalty decoding. When SteamVR is running, the **VR mirror** group
   in the Live tab pushes every visible overlay to an `IVROverlay`
   panel via `studio/vr/openvr_overlay.py` (extra `[vr]`).

### Car coverage

* **Telemetry capture, replay, lap / stint / sector analytics,
  channels, dampers, racing line and all CSV-driven tooling work
  with every car LFS can drive** — stock cars, mods and unknown
  vehicles alike. They depend only on the OutSim / OutGauge / InSim
  wire formats, which LFS emits identically for any car.
* **The Overlay tab (live HUD) is supported for the stock LFS car
  list and for verified mod footprints only.** The race-engineer
  HUD relies on per-car metadata (mass, fuel tank, gearing, tyre
  compound table, setup defaults) sourced from `config/cars.json`,
  the bundled `assets/source/cars/*.bin` exports of `car_info.bin`
  and the curated `assets/source/mods/*.json` footprints. When a
  car is not present in any of those sources, capture and analysis
  still run normally, but Overlay widgets that need car-specific
  context (fuel range, predicted lap fuel, tyre-wear bars, gear
  indicator scaling) fall back to neutral defaults or stay blank
  until a calibration is recorded with `lfs-telemetry calibrate`.

The menu *Tools* contains:

* *Configure LFS…* — `widgets/lfs_config_dialog.py`. Pick the LFS
  install folder and patch `cfg.txt` in place (with a `.bak` backup).

(The bundled racing line for a track is loaded automatically from
`racing_lines/<TRACK>_racing.csv`; `widgets/racing_line_loader.py`
remains as a helper used internally by the compass / mini-map
renderers, not as a user-facing dialog.)

The menu *View* contains the dock visibility toggles plus *Language*,
which switches between English and Español live; the choice is
persisted in `QSettings`. *Help* exposes *Channel guide…*
(<kbd>F1</kbd>) and *About*.

The `setup_tab.py` and `setup_editor_tab.py` modules exist in the
source tree as the seed of a future setup advisor, but they are
**deliberately not wired into `CenterTabs`** and are **excluded from
the frozen `.exe`** by the PyInstaller spec (see
[Building the Windows installer](#building-the-windows-installer)).

See [src/lfs_telemetry/studio/README.md](src/lfs_telemetry/studio/README.md)
for the full widget map and signal flow.

### Persisted state

`workspace_state.WorkspaceState` writes `~/.lfs-telemetry/studio.json`
with the last opened capture, the workspace path, channel visibility,
split-to-sector mapping and Qt dock geometry (base64-encoded
`saveState()` / `restoreState()`). Cold restarts restore the prior
session exactly.

## Telemetry core

The `telemetry/` package is **independent of any UI framework** — it
only depends on `numpy` + `pandas` (plus `scipy` for a few signal
helpers, and `asyncio` for live capture). Any application (Studio,
Jupyter notebooks, ML pipelines, classical control) can sit on top of
it.

* High-level entry points: `LiveTelemetry`, `LapTelemetry`,
  `StintTelemetry`, `LapComparison`, `SplitPredictor`,
  `TrafficSnapshot`, `FuelTracker`, `NodeDeltaTracker`, `TrackMap`.
* Persistence: `write_csv_replay`, `read_csv_replay`,
  `read_csv_dataframe`, `SCHEMA_VERSION`, `detect_schema_version` —
  lossless round-trip for samples produced by `write_csv_replay`.
* Per-lap slicer: `slice_into_laps`, `write_per_lap_files`,
  `reslice_csv`, `find_line_crossings` (driven by
  `current_lap_dist_m` wraparound).
* Sectors: `Sector`, `lap_sectors`, `sector_times_s`,
  `insim_split_distances_m`.
* Track geometry: parsers for `.pth` (AI driving line), `.smx` (3D
  mesh), `.pin` (env bbox), `.knw` (per-car AI knowledge) plus
  `compute_profile`, `segment_track`, `assign_segment`,
  `detect_track`.
* Racing line: `compute_edges`, `compute_geometric_line`,
  `compute_knw_line`, `compute_target_speed`.
* Observables: `CarSpec`, `observe_sample`, `observe_window`
  (structural projection for setup analytics).
* Calibration: `estimate_mu_lat`, `estimate_mu_long`,
  `estimate_mass_kg`, `calibrate_spec`, `RestCalibrator`,
  `CarSpecStore`.
* Channel + capture metadata: `CHANNELS`, `ChannelInfo`,
  `channel_info`, `channels_by_group`, `CaptureInfo`,
  `discover_captures`, `inspect_capture`, `captures_to_dataframe`.

See [src/lfs_telemetry/telemetry/README.md](src/lfs_telemetry/telemetry/README.md)
for the full module-by-module API reference.

## Scripts and tools

`scripts/` contains operational and developer helpers shipped with
the repo:

| Script | Purpose |
| --- | --- |
| `build_app.ps1` | Full build pipeline: clean, install `[studio,build]`, run PyInstaller, optionally invoke Inno Setup. |
| `build_app_simple.ps1` | Slimmer variant used for routine builds (`-Full` also builds the installer when `iscc.exe` is on `PATH`). |
| `build_installer.ps1` | Stand-alone Inno Setup wrapper. |
| `pyi_runtime_chdir.py` | Frozen-bundle bootstrap so Studio finds `config/` and `racing_lines/` relative to the `.exe`. |
| `racing_line_view.py` | Generate `<TRACK>_racing.csv` + `.png` for one variant or `--all`. Reads `.pth`, applies `segment_track`, picks the line source (`auto` / `knw` / `heuristic`) and renders a coloured speed map. |
| `track_view.py` | Render a single track outline with edges and centerline. |
| `motec_view.py` | MoTeC-style overlay of one or two CSVs from the CLI. |
| `wheel_view.py` | Per-wheel telemetry plots (load, slip, temperature). |
| `live_inspect.py` | Stream `LiveTelemetry` samples to stdout for diagnostics. |
| `sniff_udp.py` | Raw UDP dump for protocol debugging. |
| `probe_pth.py` | Inspect a `.pth` file (node count, length, climb). |
| `check_real_stint.py` | Sanity-check an aggregate stint CSV. |
| `patch_lfs_cfg.py` | Wrapper around `lfs_config.patch_cfg`. |

`tools/` contains low-level binary-format research utilities:

| Tool | Purpose |
| --- | --- |
| `decode_knw.py` | Dump a `.knw` AI-knowledge file (per-segment best line). |
| `decode_wld.py` | Dump a `.wld` world-geometry file. |
| `full_lap_plots.py` / `full_lap_report.py` | Build a per-lap PDF/MD report. |
| `sniff_lfs_data.py` | Continuous capture of all UDP/TCP traffic. |
| `sniff_pth_trailing.py` | Probe trailing bytes of `.pth` for hidden fields. |

## Development tooling

* **Continuous integration** — `.github/workflows/test.yml` runs
  `ruff check .` and `pytest -q` on every push and pull request. The
  Linux matrix covers Python 3.11 / 3.12 / 3.13 (with the
  `libgl1 / libegl1 / libxkbcommon0 / libxcb-cursor0 / libdbus /
  libfontconfig1` system libs needed by PySide6 under
  `QT_QPA_PLATFORM=offscreen`); a Windows job runs Python 3.13 as a
  smoke check.
* **Pre-commit** — `.pre-commit-config.yaml` wires
  [`pre-commit-hooks`](https://github.com/pre-commit/pre-commit-hooks)
  v5 (trailing whitespace, end-of-file fixer, YAML/TOML/merge-conflict
  checks, LF line endings) and
  [`ruff-pre-commit`](https://github.com/astral-sh/ruff-pre-commit)
  v0.7.4 (`ruff --fix` + `ruff format`). Install with
  `pip install pre-commit && pre-commit install`.
* **Linting** — `ruff` selects
  `E, F, I, UP, B, SIM, C4, PIE, RUF, ERA, ASYNC`; `RUF001/002/003`
  (Greek letters ν, Δ used in TNFR notation plus the Spanish i18n
  tables) and `ERA001` (annotated physics derivations look like
  commented-out code) are ignored project-wide. `tools/**` and
  `scripts/**` get per-file relaxations so the diagnostic one-liners
  there don't trip the broader rules.
* **Type checking** — `[tool.mypy]` in `pyproject.toml` ships a
  gradual configuration (`python_version = "3.13"`,
  `ignore_missing_imports = true`, `disallow_untyped_defs = false`).
  Stricter knobs (`disallow_untyped_defs`,
  `no_implicit_optional`, `check_untyped_defs`) are intended to be
  enabled per-module as coverage improves; `build/`, `dist/`,
  `installer/Output/`, `tools/`, `scripts/` and `tests/` are
  excluded.

## Data folders

* `racing_lines/` — `<TRACK>_racing.csv` (per-node table) and `.png`
  (track map + speed band) for every Live for Speed variant.
  Regenerate with `python scripts/racing_line_view.py --all`. The
  aggregate `_racing_summary.csv` lists length, climb, min radius,
  max slope, mean width and speed range per variant. Each per-node
  CSV carries `drive_left_m`/`drive_right_m` (AI usable corridor) and
  `limit_left_m`/`limit_right_m` (track edges) in addition to
  curvature, radius, slope and segment metadata.
* `tracks/` — per-variant elevation `.csv` profiles and
  `_overview.png` renders.
* `config/cars.json` — bundled `CarSpec` defaults (mass, μ_lat,
  μ_long, geometry, gearing) used when no calibrated entry exists in
  the user store `~/.lfs-telemetry/cars.json`.
* `assets/` — synthetic sample stints (`synthetic_*`) used by the
  test suite, the bundled `source/mods` + `source/cars` seed
  databases consumed by Studio for unknown-car detection, the TNFR
  theoretical reference (`AGENTS.md`), and the application icon
  (`icon.ico`) embedded into the frozen `.exe`.
* `reports/` — generated lap / stint markdown reports produced by
  `tools/full_lap_report.py` and friends.

## Building the Windows installer

The frozen Studio bundle and the Inno Setup installer are produced by
the PowerShell scripts under `scripts/`. The build pulls the version
from `pyproject.toml` (single source of truth) and forwards it to
PyInstaller and Inno Setup.

### Quick build

```powershell
# 1. From the repo root, ensure the venv exists and is activated.
.\.venv\Scripts\Activate.ps1

# 2. Build the .exe only.
.\scripts\build_app_simple.ps1

# 3. Build .exe + Inno Setup installer.
.\scripts\build_app_simple.ps1 -Full
```

`build_app_simple.ps1`:

* parses `version = "x.y.z"` from `pyproject.toml`;
* deletes `build/` and `dist/`;
* installs `.[studio,build]` quietly into the active venv;
* runs `pyinstaller lfs-race-engineer.spec --noconfirm --clean`;
* reports the bundle path and size at
  `dist\lfs-race-engineer\lfs-race-engineer.exe`;
* with `-Full`, looks up `iscc.exe` on `PATH` and runs Inno Setup
  against `installer\lfs-race-engineer.iss`, producing
  `installer\Output\lfs-race-engineer-setup-<ver>.exe`.

If `iscc.exe` is not on `PATH`, invoke Inno Setup explicitly:

```powershell
& "$env:LocalAppData\Programs\Inno Setup 6\iscc.exe" `
    "/DMyAppVersion=$version" `
    "installer\lfs-race-engineer.iss"
```

### PyInstaller spec (`lfs-race-engineer.spec`)

One-folder layout (no onefile bootstrapper):

```
dist/lfs-race-engineer/
  lfs-race-engineer.exe       (entry point: studio.__main__:main)
  _internal/                  (Python runtime, PySide6, scipy, …)
  config/cars.json            (bundled CarSpec defaults)
  racing_lines/*.csv          (reference racing lines)
  tracks/*.csv                (elevation profiles)
```

Key behaviours encoded in the spec:

* All `lfs_telemetry.*` submodules are collected as hidden imports
  (so dynamic `importlib` lookups survive), **except**
  `lfs_telemetry.studio.widgets.setup_tab` and
  `lfs_telemetry.studio.widgets.setup_editor_tab`, which are kept
  out of the bundle until the Setup workflow is wired in.
* `pyqtgraph` and `scipy` submodules are also collected to cover
  their lazy imports.
* `assets/source/mods/*.json` (mod-car footprints) and
  `assets/source/cars/*.bin` (stock `car_info.bin` exports) ship
  alongside the static data folders so the radar and setup readers
  work out of the box.
* A runtime hook (`scripts/pyi_runtime_chdir.py`) chdir's to the
  `.exe` directory at startup so cwd-based lookups under `config/`
  and `racing_lines/` keep working.

### Inno Setup installer (`installer/lfs-race-engineer.iss`)

* Inno Setup 6.3+, modern wizard style, English + Spanish.
* `ArchitecturesAllowed=x64compatible` and
  `ArchitecturesInstallIn64BitMode=x64compatible` (current Inno
  Setup recommendation; the legacy `x64` token is deprecated).
* Supports both per-user and per-machine installs
  (`PrivilegesRequired=lowest` +
  `PrivilegesRequiredOverridesAllowed=dialog commandline`).
* Compression: `lzma2/ultra` + `SolidCompression=yes`.
* Layout: copies the entire `dist\lfs-race-engineer\` tree to
  `{app}`, plus a `README.txt` derived from the repo `README.md`.
* Creates user-writable `{app}\captures` and `{app}\exports` folders.
* Optional file association: `.csv` → *LFS Race Engineer Telemetry
  Replay* (off by default; opt-in task).
* Uninstaller deletes `studio.log` / `telemetry.log` and any empty
  app directories, but **does not** touch user-created files under
  `captures/` and `exports/`.
* `MyAppVersion` is sourced from `/DMyAppVersion=x.y.z` on the
  command line; the hard-coded fallback inside the script exists
  only for ad-hoc manual builds and is kept in sync with
  `pyproject.toml`.

## Licence

MIT. See `pyproject.toml` and individual file headers.

## Credits

This project was inspired by prior community work on Live for Speed
telemetry tools. See [CREDITS.md](CREDITS.md) for full attribution to
**LFSTelemetry** (Cyril Bissey, MIT), **helicorsa** (Jens Lohmann,
MIT), **Detect&Monitor** (KingOfIce, proprietary — credited for
inspiration only), and the Live for Speed track-geometry formats.
