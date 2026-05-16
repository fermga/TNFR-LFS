"""Framework-neutral telemetry layer for Live for Speed.

This package owns *everything* needed to obtain, persist, replay and
analyze telemetry from LFS, with **no coupling to any analysis engine**.
Any analysis or visualization framework (PySide6 Studio, classical
control, ML, MoTeC-style plotters, …) can sit on top of
:mod:`lfs_telemetry.telemetry` by depending only on ``numpy`` + ``pandas``.

Quick start (offline / standalone app)
--------------------------------------

::

    from lfs_telemetry.telemetry import LapTelemetry

    lap = LapTelemetry.from_csv("stint_bl1_fbm_lap01.csv")
    print(lap.summary)        # {'lap_time_s': 86.5, 'top_speed_kmh': 207, ...}
    df = lap.enriched         # raw + 28 derived signals (friction, FFB, …)

Quick start (live capture)
--------------------------

::

    import asyncio
    from lfs_telemetry.telemetry import LiveTelemetry, write_csv_replay

    async def main():
        async with LiveTelemetry(insim_host="127.0.0.1") as live:
            samples = []
            async for s in live.samples():
                samples.append(s)
                if len(samples) > 6000:
                    break
        write_csv_replay("stint.csv", samples)

    asyncio.run(main())

Sub-packages
------------

* :mod:`lfs_telemetry.telemetry.protocol` — LFS wire protocol (OutSim,
  OutGauge, InSim).
* :mod:`lfs_telemetry.telemetry.track` — track geometry (.pth/.smx parsers,
  spatial enrichment, racing line).

Modules
-------

* :mod:`.live` — UDP capture + InSim fusion (asyncio).
* :mod:`.replay` — CSV persistence schema (versioned).
* :mod:`.lap` — :class:`LapTelemetry` facade for app consumers.
* :mod:`.stint` — :class:`StintTelemetry` aggregator for multi-lap
  session trends (pace drop-off, fuel use, tyre work, aid usage).
* :mod:`.comparison` — :class:`LapComparison` for distance-aligned
  overlay + delta-time (the MoTeC-style headline view).
* :mod:`.channels` — :class:`ChannelInfo` registry mapping columns to
  display label, units and group for the channel browser.
* :mod:`.catalog` — :func:`discover_captures` for the workspace
  browser (lightweight metadata-only scan of a capture folder).
* :mod:`.derived` — physics-derived columns on top of the raw schema.
* :mod:`.observables` — :class:`CarSpec` and per-sample structural
  projection.
* :mod:`.calibrate` — μ/mass estimators from telemetry.
* :mod:`.car_calibration` — persistent CarSpec store.
* :mod:`.lap_summary` — :class:`LapRecord` per-lap session metadata
  (from IS_LAP/IS_SPX/IS_HLV).
* :mod:`.traffic` — :class:`TrafficSnapshot` from IS_MCI.
"""

from __future__ import annotations

__version__ = "0.2.0"

# --- Live capture -----------------------------------------------------------
from .live import LiveTelemetry, TelemetrySample

# --- Persistence (versioned CSV schema) ------------------------------------
from .replay import (
    SCHEMA_VERSION,
    detect_schema_version,
    read_csv_dataframe,
    read_csv_replay,
    write_csv_replay,
)

# --- High-level lap facade (preferred entry point for analysis apps) -------
from .lap import LapTelemetry
from .stint import StintTelemetry
from .comparison import LapComparison
from .sectors import Sector, insim_split_distances_m, lap_sectors, sector_times_s
from .track_map import TrackBounds, TrackMap
from .lap_slicer import (
    LapSlice,
    find_line_crossings,
    reslice_csv,
    slice_into_laps,
    write_per_lap_files,
)
from .predict import SplitPredictor

# --- Channel registry & capture catalog (UI plumbing) ----------------------
from .channels import CHANNELS, ChannelInfo, channel_info, channels_by_group
from .catalog import CaptureInfo, captures_to_dataframe, discover_captures, inspect_capture

# --- Physics-derived signals -----------------------------------------------
from .derived import enrich_dataframe

# --- Car spec / observables ------------------------------------------------
from .observables import CarSpec, car_spec_for, observe_sample, observe_window

# --- Calibration -----------------------------------------------------------
from .calibrate import (
    calibrate_spec,
    calibration_report,
    estimate_mass_kg,
    estimate_mu_lat,
    estimate_mu_lat_curve,
    estimate_mu_long,
)
from .car_calibration import CarCalibration, CarSpecStore, RestCalibrator

# --- Race-context derivations ----------------------------------------------
from .lap_summary import (
    LapRecord,
    build_lap_records,
    dump_lap_records,
    load_lap_records,
)
from .traffic import TrafficSnapshot, traffic_snapshot

__all__ = [
    "__version__",
    # live
    "LiveTelemetry", "TelemetrySample",
    # persistence
    "SCHEMA_VERSION", "detect_schema_version",
    "read_csv_dataframe", "read_csv_replay", "write_csv_replay",
    # facade
    "LapTelemetry",
    "StintTelemetry",
    "LapComparison",
    "Sector", "lap_sectors", "sector_times_s", "insim_split_distances_m",
    "TrackMap", "TrackBounds",
    "LapSlice", "find_line_crossings", "slice_into_laps",
    "write_per_lap_files", "reslice_csv",
    "SplitPredictor",
    # channel registry / catalog
    "CHANNELS", "ChannelInfo", "channel_info", "channels_by_group",
    "CaptureInfo", "captures_to_dataframe",
    "discover_captures", "inspect_capture",
    # derived
    "enrich_dataframe",
    # observables
    "CarSpec", "car_spec_for", "observe_sample", "observe_window",
    # calibration
    "calibrate_spec", "calibration_report",
    "estimate_mass_kg", "estimate_mu_lat",
    "estimate_mu_lat_curve", "estimate_mu_long",
    "CarCalibration", "CarSpecStore", "RestCalibrator",
    # race context
    "LapRecord", "build_lap_records",
    "dump_lap_records", "load_lap_records",
    "TrafficSnapshot", "traffic_snapshot",
]
