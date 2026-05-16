"""CSV replay: persist live captures and replay them offline.

The CSV schema is kept stable so that captures from earlier runs remain
compatible with newer analyzer versions. One row per fused sample.

The schema additionally covers extended OutSim (per-wheel) and InSim
race-context columns; these are written empty when not available, and the
reader tolerates older CSVs that lack them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator

from .live import TelemetrySample
from .protocol.packets import (
    OutGaugePacket,
    OutSimPack2,
    OutSimPacket,
    OutSimWheel,
    WHEEL_ORDER,
)


# Bumped whenever the canonical column set changes. Old readers ignore the
# preamble comment line; new readers can warn or migrate as needed.
SCHEMA_VERSION = "1.1"
_SCHEMA_HEADER = f"# lfs-telemetry telemetry schema={SCHEMA_VERSION}"


# CORNERS used in the CSV column names: F[L/R] / R[L/R].
_WHEEL_FIELDS = (
    "susp_deflect_m",
    "vertical_load_n",
    "slip_ratio",
    "tan_slip_angle",
    "x_force_n",
    "y_force_n",
    "ang_vel_rads",
    "lean_rel_road_rad",
    "air_temp_c",
    "slip_fraction",
    "touching",
    "steer_rad",
)

_RACE_FIELDS = (
    "ctx_track",
    "ctx_weather",
    "ctx_wind",
    "ctx_view_plid",
    "ctx_view_car",
    "ctx_race_in_progress",
    "ctx_race_laps",
    "ctx_qual_minutes",
    "ctx_lfs_version",
    "ctx_view_lap_count",
    "ctx_view_last_lap_ms",
    "ctx_view_last_split1_ms",
    "ctx_view_last_split2_ms",
    "ctx_view_last_split3_ms",
    "ctx_view_last_hlv_code",
    "ctx_view_last_hlv_name",
    "ctx_view_last_hlv_speed_ms",
    "ctx_view_handicap_mass_kg",
    "ctx_view_handicap_t_res",
    "ctx_view_tyre_compounds",
    "ctx_obh_count",
    "ctx_pit_stop_count",
)

_OUTSIM2_FIELDS = (
    "current_lap_dist_m",
    "indexed_distance_m",
    "steer_torque_nm",
    "engine_ang_vel_rads",
    "max_torque_at_vel_nm",
    "input_throttle",
    "input_brake",
    "input_steer",
    "input_clutch",
    "input_handbrake",
)


def _wheel_columns() -> list[str]:
    cols: list[str] = []
    for c in WHEEL_ORDER:
        for f in _WHEEL_FIELDS:
            cols.append(f"wheel_{c}_{f}")
    return cols


_FIELDS = [
    "time_ms",
    # OutSim
    "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "heading", "pitch", "roll",
    "accel_x", "accel_y", "accel_z",
    "vel_x", "vel_y", "vel_z",
    "pos_x", "pos_y", "pos_z",
    # OutGauge
    "car", "gear", "speed_ms", "rpm",
    "throttle", "brake", "clutch",
    "fuel", "eng_temp_c", "oil_temp_c", "oil_pressure_bar",
    "turbo_bar",
    "og_flags", "dash_lights", "show_lights",
    "og_player_id",
    # OutSimPack2 scalar extensions
    *_OUTSIM2_FIELDS,
    # OutSimPack2 per-wheel (4 × 12 = 48 columns)
    *_wheel_columns(),
    # InSim race context snapshot
    *_RACE_FIELDS,
]


def write_csv_replay(path: str | Path, samples: Iterable[TelemetrySample]) -> int:
    """Write fused samples to ``path`` and return the row count.

    The first line is a ``# lfs-telemetry telemetry schema=<v>`` comment that
    :func:`csv.DictReader` and :func:`pandas.read_csv` (with the helper
    :func:`read_csv_dataframe` below) both skip transparently.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as fp:
        fp.write(_SCHEMA_HEADER + "\n")
        writer = csv.DictWriter(fp, fieldnames=_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            if not sample.is_complete:
                continue
            writer.writerow(_sample_to_row(sample))
            rows += 1
    return rows


def read_csv_replay(path: str | Path) -> Iterator[TelemetrySample]:
    """Stream fused samples back from a previously captured CSV."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(_iter_no_preamble(fp))
        for row in reader:
            yield _row_to_sample(row)


def read_csv_dataframe(path: str | Path):
    """Load a captured CSV directly as a :class:`pandas.DataFrame`.

    Much faster than iterating :func:`read_csv_replay` when the consumer
    only needs a tabular view (plotting, MoTeC-style analyzers, ML).
    Pandas is imported lazily so the live capture path does not pay the
    cost.
    """
    import pandas as pd  # local import keeps the live path lightweight

    return pd.read_csv(path, comment="#")


def detect_schema_version(path: str | Path) -> str | None:
    """Return the schema version embedded in the CSV preamble, or ``None``
    if the file predates schema versioning.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        first = fp.readline().strip()
    if first.startswith("# lfs-telemetry telemetry schema="):
        return first.split("=", 1)[1]
    return None


def _iter_no_preamble(fp):
    """Drop leading ``#``-prefixed lines so DictReader sees a normal CSV."""
    for line in fp:
        if not line.startswith("#"):
            yield line
            break
    yield from fp


def _sample_to_row(sample: TelemetrySample) -> dict[str, object]:
    os_pkt = sample.outsim
    og_pkt = sample.outgauge
    assert os_pkt is not None and og_pkt is not None
    row: dict[str, object] = {
        "time_ms": sample.time_ms,
        "ang_vel_x": os_pkt.ang_vel[0],
        "ang_vel_y": os_pkt.ang_vel[1],
        "ang_vel_z": os_pkt.ang_vel[2],
        "heading": os_pkt.heading,
        "pitch": os_pkt.pitch,
        "roll": os_pkt.roll,
        "accel_x": os_pkt.accel[0],
        "accel_y": os_pkt.accel[1],
        "accel_z": os_pkt.accel[2],
        "vel_x": os_pkt.vel[0],
        "vel_y": os_pkt.vel[1],
        "vel_z": os_pkt.vel[2],
        "pos_x": os_pkt.pos[0],
        "pos_y": os_pkt.pos[1],
        "pos_z": os_pkt.pos[2],
        "car": og_pkt.car,
        "gear": og_pkt.gear,
        "speed_ms": og_pkt.speed_ms,
        "rpm": og_pkt.rpm,
        "throttle": og_pkt.throttle,
        "brake": og_pkt.brake,
        "clutch": og_pkt.clutch,
        "fuel": og_pkt.fuel,
        "eng_temp_c": og_pkt.eng_temp_c,
        "oil_temp_c": og_pkt.oil_temp_c,
        "oil_pressure_bar": og_pkt.oil_pressure_bar,
        "turbo_bar": og_pkt.turbo_bar,
        "og_flags": og_pkt.flags,
        "dash_lights": og_pkt.dash_lights,
        "show_lights": og_pkt.show_lights,
        "og_player_id": og_pkt.player_id,
    }
    # Pre-fill empty cells for the extended schema.
    for col in _OUTSIM2_FIELDS:
        row[col] = ""
    for col in _wheel_columns():
        row[col] = ""
    for col in _RACE_FIELDS:
        row[col] = ""
    if sample.outsim2 is not None:
        _fill_outsim2(row, sample.outsim2)
    if sample.race_context is not None:
        _fill_race_context(row, sample.race_context)
    return row


def _fill_outsim2(row: dict[str, object], pkt2: OutSimPack2) -> None:
    if pkt2.current_lap_dist_m is not None:
        row["current_lap_dist_m"] = pkt2.current_lap_dist_m
    if pkt2.indexed_distance_m is not None:
        row["indexed_distance_m"] = pkt2.indexed_distance_m
    if pkt2.steer_torque_nm is not None:
        row["steer_torque_nm"] = pkt2.steer_torque_nm
    if pkt2.engine_ang_vel_rads is not None:
        row["engine_ang_vel_rads"] = pkt2.engine_ang_vel_rads
    if pkt2.max_torque_at_vel_nm is not None:
        row["max_torque_at_vel_nm"] = pkt2.max_torque_at_vel_nm
    if pkt2.throttle is not None:
        row["input_throttle"] = pkt2.throttle
        row["input_brake"] = pkt2.brake
        row["input_steer"] = pkt2.input_steer
        row["input_clutch"] = pkt2.clutch
        row["input_handbrake"] = pkt2.handbrake
    if pkt2.wheels is not None and len(pkt2.wheels) == 4:
        for c, w in zip(WHEEL_ORDER, pkt2.wheels):
            prefix = f"wheel_{c}_"
            row[prefix + "susp_deflect_m"] = w.susp_deflect_m
            row[prefix + "vertical_load_n"] = w.vertical_load_n
            row[prefix + "slip_ratio"] = w.slip_ratio
            row[prefix + "tan_slip_angle"] = w.tan_slip_angle
            row[prefix + "x_force_n"] = w.x_force_n
            row[prefix + "y_force_n"] = w.y_force_n
            row[prefix + "ang_vel_rads"] = w.ang_vel_rads
            row[prefix + "lean_rel_road_rad"] = w.lean_rel_road_rad
            row[prefix + "air_temp_c"] = w.air_temp_c
            row[prefix + "slip_fraction"] = w.slip_fraction
            row[prefix + "touching"] = w.touching
            row[prefix + "steer_rad"] = w.steer_rad


def _fill_race_context(row: dict[str, object], ctx) -> None:
    snap = ctx.snapshot()
    row["ctx_track"] = snap.get("track") or ""
    row["ctx_weather"] = snap.get("weather") if snap.get("weather") is not None else ""
    row["ctx_wind"] = snap.get("wind") if snap.get("wind") is not None else ""
    row["ctx_view_plid"] = snap.get("view_player_id") if snap.get("view_player_id") is not None else ""
    row["ctx_view_car"] = snap.get("view_player") or ""
    row["ctx_race_in_progress"] = snap.get("race_in_progress") if snap.get("race_in_progress") is not None else ""
    row["ctx_race_laps"] = snap.get("race_laps") if snap.get("race_laps") is not None else ""
    row["ctx_qual_minutes"] = snap.get("qual_minutes") if snap.get("qual_minutes") is not None else ""
    row["ctx_lfs_version"] = snap.get("lfs_version") or ""
    row["ctx_view_lap_count"] = snap.get("view_lap_count") if snap.get("view_lap_count") is not None else ""
    row["ctx_view_last_lap_ms"] = snap.get("view_last_lap_ms") if snap.get("view_last_lap_ms") is not None else ""
    plid = ctx.view_player_id
    splits = ctx.last_split_ms.get(plid, {}) if plid is not None else {}
    row["ctx_view_last_split1_ms"] = splits.get(1, "")
    row["ctx_view_last_split2_ms"] = splits.get(2, "")
    row["ctx_view_last_split3_ms"] = splits.get(3, "")
    last_hlv = ctx.last_hlv.get(plid) if plid is not None else None
    if last_hlv is not None:
        from .protocol.packets import hlvc_name  # local import avoids cycle
        row["ctx_view_last_hlv_code"] = last_hlv.hlvc
        row["ctx_view_last_hlv_name"] = hlvc_name(last_hlv.hlvc)
        row["ctx_view_last_hlv_speed_ms"] = last_hlv.car_speed_ms
    else:
        row["ctx_view_last_hlv_code"] = ""
        row["ctx_view_last_hlv_name"] = ""
        row["ctx_view_last_hlv_speed_ms"] = ""
    player = ctx.players.get(plid) if plid is not None else None
    if player is not None:
        row["ctx_view_handicap_mass_kg"] = player.handicap_mass_kg
        row["ctx_view_handicap_t_res"] = player.handicap_t_res
        row["ctx_view_tyre_compounds"] = "|".join(str(t) for t in player.tyres)
    else:
        row["ctx_view_handicap_mass_kg"] = ""
        row["ctx_view_handicap_t_res"] = ""
        row["ctx_view_tyre_compounds"] = ""
    row["ctx_obh_count"] = len(ctx.obh_events)
    row["ctx_pit_stop_count"] = len(ctx.pit_stops)


def _row_to_sample(row: dict[str, str]) -> TelemetrySample:
    t = int(row["time_ms"])
    os_pkt = OutSimPacket(
        time_ms=t,
        ang_vel=(float(row["ang_vel_x"]), float(row["ang_vel_y"]), float(row["ang_vel_z"])),
        heading=float(row["heading"]),
        pitch=float(row["pitch"]),
        roll=float(row["roll"]),
        accel=(float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])),
        vel=(float(row["vel_x"]), float(row["vel_y"]), float(row["vel_z"])),
        pos=(float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])),
    )
    og_pkt = OutGaugePacket(
        time_ms=t,
        car=row["car"],
        flags=int(_fnum(row.get("og_flags", 0))),
        gear=int(row["gear"]),
        player_id=int(_fnum(row.get("og_player_id", 0))),
        speed_ms=float(row["speed_ms"]),
        rpm=float(row["rpm"]),
        turbo_bar=float(row.get("turbo_bar", 0.0) or 0.0),
        eng_temp_c=float(row["eng_temp_c"]),
        fuel=float(row["fuel"]),
        oil_pressure_bar=float(row["oil_pressure_bar"]),
        oil_temp_c=float(row["oil_temp_c"]),
        dash_lights=int(_fnum(row.get("dash_lights", 0))),
        show_lights=int(_fnum(row.get("show_lights", 0))),
        throttle=float(row["throttle"]),
        brake=float(row["brake"]),
        clutch=float(row["clutch"]),
        display1="",
        display2="",
    )
    pkt2 = _row_to_outsim2(row, t)
    return TelemetrySample(time_ms=t, outsim=os_pkt, outgauge=og_pkt, outsim2=pkt2)


def _row_to_outsim2(row: dict[str, str], time_ms: int) -> OutSimPack2 | None:
    """Reconstruct an :class:`OutSimPack2` from a CSV row, or ``None``."""
    # Detect presence by looking for the first wheel column.
    probe = row.get(f"wheel_{WHEEL_ORDER[0]}_vertical_load_n", "")
    if probe in (None, ""):
        return None
    try:
        wheels = [
            OutSimWheel(
                susp_deflect_m=_fnum(row[f"wheel_{c}_susp_deflect_m"]),
                steer_rad=_fnum(row.get(f"wheel_{c}_steer_rad", 0.0)),
                x_force_n=_fnum(row.get(f"wheel_{c}_x_force_n", 0.0)),
                y_force_n=_fnum(row.get(f"wheel_{c}_y_force_n", 0.0)),
                vertical_load_n=_fnum(row[f"wheel_{c}_vertical_load_n"]),
                ang_vel_rads=_fnum(row.get(f"wheel_{c}_ang_vel_rads", 0.0)),
                lean_rel_road_rad=_fnum(row.get(f"wheel_{c}_lean_rel_road_rad", 0.0)),
                air_temp_c=int(_fnum(row.get(f"wheel_{c}_air_temp_c", 0))),
                slip_fraction_byte=int(round(
                    _fnum(row.get(f"wheel_{c}_slip_fraction", 0.0)) * 255.0)),
                touching=int(_fnum(row.get(f"wheel_{c}_touching", 0))),
                slip_ratio=_fnum(row[f"wheel_{c}_slip_ratio"]),
                tan_slip_angle=_fnum(row[f"wheel_{c}_tan_slip_angle"]),
            )
            for c in WHEEL_ORDER
        ]
    except (KeyError, ValueError):
        return None
    pkt2 = OutSimPack2(opts=0, time_ms=time_ms, wheels=wheels)
    # Restore scalar OutSim2 channels that were persisted to CSV but
    # would otherwise be lost on reload.
    cld = row.get("current_lap_dist_m", "")
    if cld not in (None, ""):
        try:
            pkt2.current_lap_dist_m = float(cld)
        except (TypeError, ValueError):
            pass
    ixd = row.get("indexed_distance_m", "")
    if ixd not in (None, ""):
        try:
            pkt2.indexed_distance_m = float(ixd)
        except (TypeError, ValueError):
            pass
    return pkt2


def _fnum(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
