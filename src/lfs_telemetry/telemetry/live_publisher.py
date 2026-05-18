"""Live snapshot publisher for the Studio overlay.

Builds a small, JSON-serialisable dict that summarises the current race
state from a :class:`RaceContext` plus per-tick state (current lap time,
last sample). The capture CLI writes this to a ``live.json`` file every
~100 ms; the Studio Live tab tail-reads it to drive a helicorsa-style
radar and a Detect&Monitor-style gauge stack.

Design notes
------------
* Pure functions only — no I/O, no global state. Easy to unit-test.
* Coordinate frame for the radar is "view car centred, view-heading
  rotated up": we transform every other car's world XY into the view
  car's local frame so the radar always points where the driver is
  looking, exactly like helicorsa / acRadar.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lap_averages import compute_lap_averages
from .protocol.insim import RaceContext
from .protocol.packets import CompCar
from .traffic import _build_snapshot

_LOG = logging.getLogger(__name__)
_LFS_COLOR_RE = re.compile(r"\^[0-9]")


def _strip_lfs_color_codes(text: str) -> str:
    cleaned = _LFS_COLOR_RE.sub("", text or "")
    return cleaned.strip()


def _player_name_for(ctx: RaceContext, plid: int) -> str:
    p = ctx.players.get(plid)
    if p is None:
        return f"PLID {plid}"
    name = _strip_lfs_color_codes(p.player_name)
    return name or f"PLID {plid}"


def _build_standings(
    ctx: RaceContext,
    cars: Iterable[CompCar],
    *,
    view_plid: int | None,
    session_mode: str,
) -> list[dict[str, Any]]:
    cars_list = list(cars)
    if session_mode == "race":
        ordered = sorted(
            cars_list,
            key=lambda c: (
                c.position if c.position > 0 else 10_000,
                c.player_id,
            ),
        )
    else:
        # Practice/qualifying: leaderboard by best lap.
        # If a driver has no best yet, push down and use latest lap,
        # then race position as a final stable tie-breaker.
        def _best_for(c: CompCar) -> int:
            laps = ctx.lap_times_ms.get(c.player_id, [])
            return min(laps) if laps else 10**9

        def _last_for(c: CompCar) -> int:
            return int(ctx.last_lap_ms.get(c.player_id) or 10**9)

        ordered = sorted(
            cars_list,
            key=lambda c: (
                _best_for(c),
                _last_for(c),
                c.position if c.position > 0 else 10_000,
                c.player_id,
            ),
        )
    out: list[dict[str, Any]] = []
    for c in ordered:
        laps = ctx.lap_times_ms.get(c.player_id, [])
        best = min(laps) if laps else None
        out.append(
            {
                "pos": int(c.position),
                "plid": int(c.player_id),
                "name": _player_name_for(ctx, int(c.player_id)),
                "lap": int(c.lap),
                "last_lap_ms": ctx.last_lap_ms.get(c.player_id),
                "best_lap_ms": int(best) if best is not None else None,
                "speed_kmh": round(float(c.speed_ms) * 3.6, 1),
                "rank_mode": session_mode,
                "view": (view_plid is not None and c.player_id == view_plid),
            }
        )
    return out


def _session_mode(ctx: RaceContext) -> str:
    # LFS InSim convention:
    # race_in_progress: 0=none, 1=race, 2=qualifying.
    rip = int(ctx.race_in_progress or 0)
    if rip == 1:
        return "race"
    if rip == 2:
        return "qualifying"
    return "practice"


@dataclass(slots=True)
class RadarCar:
    """A single opponent on the radar in the view car's local frame.

    ``x_local`` is right-positive, ``y_local`` is forward-positive (so
    a car directly ahead has ``y_local > 0`` and ``x_local ≈ 0``).
    """

    plid: int
    x_local_m: float
    y_local_m: float
    distance_m: float
    relative_speed_ms: float
    is_view: bool

    def to_dict(self) -> dict:
        return {
            "plid": self.plid,
            "x": round(self.x_local_m, 2),
            "y": round(self.y_local_m, 2),
            "d": round(self.distance_m, 2),
            "rel_v": round(self.relative_speed_ms, 2),
            "view": self.is_view,
        }


def project_to_local(
    view: CompCar, other: CompCar
) -> tuple[float, float]:
    """Re-exported from :mod:`lfs_telemetry.telemetry.heading`.

    Kept here for backward compatibility with callers (and tests) that
    import the symbol from ``live_publisher``. The canonical
    implementation lives in ``heading.py``; both this module and
    ``traffic.py`` import from there so a heading-convention change
    only needs to happen in one place.
    """
    from .heading import project_to_local as _impl
    return _impl(view, other)


def build_radar_cars(
    view: CompCar, cars: Iterable[CompCar]
) -> list[RadarCar]:
    """Project every car (including ``view``) into the view-local frame."""
    out: list[RadarCar] = []
    for c in cars:
        if c is view or c.player_id == view.player_id:
            out.append(RadarCar(
                plid=c.player_id, x_local_m=0.0, y_local_m=0.0,
                distance_m=0.0,
                relative_speed_ms=0.0,
                is_view=True,
            ))
            continue
        x_l, y_l = project_to_local(view, c)
        d = math.hypot(x_l, y_l)
        out.append(RadarCar(
            plid=c.player_id, x_local_m=x_l, y_local_m=y_l,
            distance_m=d,
            relative_speed_ms=c.speed_ms - view.speed_ms,
            is_view=False,
        ))
    return out


def build_snapshot(
    ctx: RaceContext | None,
    *,
    armed: bool,
    samples_count: int,
    current_lap_ms: int | None = None,
    last_sample_speed_ms: float | None = None,
    last_sample_rpm: float | None = None,
    last_sample_gear: int | None = None,
    last_sample_fuel_pct: float | None = None,
    last_sample_throttle: float | None = None,
    last_sample_brake: float | None = None,
    last_sample_clutch: float | None = None,
    last_sample_handbrake: float | None = None,
    last_sample_accel_lat_ms2: float | None = None,
    last_sample_accel_lon_ms2: float | None = None,
    last_sample_max_slip: float | None = None,
    last_sample_max_slip_ratio: float | None = None,
    last_sample_tyres: list[dict[str, Any]] | None = None,
    monotonic_ts: float = 0.0,
    delta_to_best_ms: int | None = None,
    predicted_lap_ms: int | None = None,
    spb_ms: int | None = None,
    fuel_laps_remaining: float | None = None,
    fuel_burn_pct_per_lap: float | None = None,
    ghost_node: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready snapshot dict from the live race state."""
    snap: dict[str, Any] = {
        "ts": round(float(monotonic_ts), 3),
        "armed": bool(armed),
        "samples": int(samples_count),
        "view_plid": None,
        "view_position": None,
        "view_lap": None,
        "view_speed_ms": None,
        "view_speed_kmh": None,
        "view_rpm": None,
        "view_gear": None,
        "view_fuel_pct": None,
        "view_throttle": None,
        "view_brake": None,
        "view_clutch": None,
        "view_handbrake": None,
        "view_accel_lat_ms2": None,
        "view_accel_lon_ms2": None,
        "view_max_slip": None,
        "view_max_slip_ratio": None,
        "tyres": [],
        "view_x_m": None,
        "view_y_m": None,
        "view_heading_rad": None,
        "view_node": None,
        "current_lap_ms": int(current_lap_ms) if current_lap_ms else None,
        "last_lap_ms": None,
        "best_lap_ms": None,
        "delta_vs_best_ms": None,
        "predicted_lap_ms": (
            int(predicted_lap_ms) if predicted_lap_ms is not None else None
        ),
        "spb_ms": int(spb_ms) if spb_ms is not None else None,
        "fuel_laps_remaining": (
            round(float(fuel_laps_remaining), 2)
            if fuel_laps_remaining is not None else None
        ),
        "fuel_burn_pct_per_lap": (
            round(float(fuel_burn_pct_per_lap), 3)
            if fuel_burn_pct_per_lap is not None else None
        ),
        "ghost_node": int(ghost_node) if ghost_node is not None else None,
        "lap_averages_ms": {"stint": None, "clean": None, "total": None},
        "track": None,
        "weather": None,
        "race_in_progress": None,
        "session_mode": "practice",
        "traffic": None,
        "standings": [],
        "cars": [],
        "cars_world": [],
    }
    if last_sample_speed_ms is not None:
        snap["view_speed_ms"] = round(float(last_sample_speed_ms), 3)
        snap["view_speed_kmh"] = round(float(last_sample_speed_ms) * 3.6, 1)
    if last_sample_rpm is not None:
        snap["view_rpm"] = round(float(last_sample_rpm), 0)
    if last_sample_gear is not None:
        snap["view_gear"] = int(last_sample_gear)
    if last_sample_fuel_pct is not None:
        snap["view_fuel_pct"] = round(float(last_sample_fuel_pct), 2)
    if last_sample_throttle is not None:
        snap["view_throttle"] = round(float(last_sample_throttle), 3)
    if last_sample_brake is not None:
        snap["view_brake"] = round(float(last_sample_brake), 3)
    if last_sample_clutch is not None:
        snap["view_clutch"] = round(float(last_sample_clutch), 3)
    if last_sample_handbrake is not None:
        snap["view_handbrake"] = round(float(last_sample_handbrake), 3)
    if last_sample_accel_lat_ms2 is not None:
        snap["view_accel_lat_ms2"] = round(float(last_sample_accel_lat_ms2), 3)
    if last_sample_accel_lon_ms2 is not None:
        snap["view_accel_lon_ms2"] = round(float(last_sample_accel_lon_ms2), 3)
    if last_sample_max_slip is not None:
        snap["view_max_slip"] = round(float(last_sample_max_slip), 3)
    if last_sample_max_slip_ratio is not None:
        snap["view_max_slip_ratio"] = round(
            float(last_sample_max_slip_ratio), 3
        )
    if last_sample_tyres:
        tyres: list[dict[str, Any]] = []
        for row in last_sample_tyres:
            try:
                tyres.append(
                    {
                        "corner": str(row.get("corner") or "?"),
                        "temp_c": (
                            round(float(row.get("temp_c")), 1)
                            if row.get("temp_c") is not None else None
                        ),
                        "slip_frac": (
                            round(float(row.get("slip_frac")), 3)
                            if row.get("slip_frac") is not None else None
                        ),
                        "slip_ratio": (
                            round(float(row.get("slip_ratio")), 3)
                            if row.get("slip_ratio") is not None else None
                        ),
                        "load_n": (
                            round(float(row.get("load_n")), 1)
                            if row.get("load_n") is not None else None
                        ),
                        "tan_slip": (
                            round(float(row.get("tan_slip")), 4)
                            if row.get("tan_slip") is not None else None
                        ),
                        "fx_n": (
                            round(float(row.get("fx_n")), 1)
                            if row.get("fx_n") is not None else None
                        ),
                        "fy_n": (
                            round(float(row.get("fy_n")), 1)
                            if row.get("fy_n") is not None else None
                        ),
                        "touching": bool(row.get("touching", False)),
                    }
                )
            except (TypeError, ValueError):
                continue
        snap["tyres"] = tyres
    if ctx is None:
        return snap
    snap["track"] = ctx.track
    snap["weather"] = ctx.weather
    snap["race_in_progress"] = ctx.race_in_progress
    snap["session_mode"] = _session_mode(ctx)
    plid = ctx.view_player_id
    if plid is not None:
        snap["view_plid"] = int(plid)
        lap_count = ctx.lap_count.get(plid)
        if lap_count is not None:
            snap["view_lap"] = int(lap_count)
        last = ctx.last_lap_ms.get(plid)
        if last is not None:
            snap["last_lap_ms"] = int(last)
        laps = ctx.lap_times_ms.get(plid)
        if laps:
            best = min(laps)
            snap["best_lap_ms"] = int(best)
            if current_lap_ms is not None:
                snap["delta_vs_best_ms"] = int(current_lap_ms) - int(best)
            pit_in_laps = [
                p.laps_done for p in ctx.pit_stops
                if p.player_id == plid
            ]
            snap["lap_averages_ms"] = compute_lap_averages(
                laps, pit_in_laps
            )
    # If the caller supplies a per-node interpolated delta (the
    # continuous Detect&Monitor-style signal), prefer it over the
    # crude ``current_lap_ms - best_lap_ms`` lap-time difference.
    if delta_to_best_ms is not None:
        snap["delta_vs_best_ms"] = int(delta_to_best_ms)
    mci = ctx.last_mci
    if mci is not None and mci.cars:
        snap["standings"] = _build_standings(
            ctx,
            mci.cars,
            view_plid=plid,
            session_mode=str(snap["session_mode"]),
        )
    if mci is not None and mci.cars and plid is not None:
        view: CompCar | None = next(
            (c for c in mci.cars if c.player_id == plid),
            None,
        )
        if view is not None:
            snap["view_position"] = int(view.position)
            snap["view_x_m"] = round(float(view.x_m), 2)
            snap["view_y_m"] = round(float(view.y_m), 2)
            snap["view_heading_rad"] = round(float(view.heading_rad), 4)
            snap["view_node"] = int(view.node)
            traffic = _build_snapshot(view, mci.cars)
            snap["traffic"] = {
                "ahead_plid": traffic.car_ahead_plid,
                "ahead_pos": traffic.car_ahead_position,
                "ahead_gap_m": traffic.gap_to_ahead_m,
                "ahead_gap_s": traffic.gap_to_ahead_s,
                "behind_plid": traffic.car_behind_plid,
                "behind_pos": traffic.car_behind_position,
                "behind_gap_m": traffic.gap_to_behind_m,
                "behind_gap_s": traffic.gap_to_behind_s,
                "blue_flag": traffic.blue_flag_for_view,
                "yellow_flag": traffic.yellow_flag_active,
                "num_cars": traffic.num_cars,
            }
            snap["cars"] = [
                c.to_dict() for c in build_radar_cars(view, mci.cars)
            ]
            snap["cars_world"] = [
                {
                    "plid": int(c.player_id),
                    "x": round(float(c.x_m), 2),
                    "y": round(float(c.y_m), 2),
                    "pos": int(c.position),
                    "view": c.player_id == plid,
                }
                for c in mci.cars
            ]
    return snap


def write_snapshot_atomic(path: Path, snap: dict[str, Any]) -> None:
    """Write ``snap`` as JSON to ``path`` atomically (write+os.replace).

    The Studio Live tab can read the file at any time without ever
    catching a half-written line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, separators=(",", ":"))
        # On Windows ``os.replace`` can fail with PermissionError if a
        # reader (Studio overlay) holds the file open without
        # FILE_SHARE_DELETE.  Retry a couple of times before giving up;
        # the next ~100 ms tick will retry naturally anyway.
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                os.replace(tmp_name, str(path))
                return
            except PermissionError as exc:
                last_exc = exc
        # Couldn't replace — clean tmp and surface as a debug log so
        # repeated failures (overlay holding the file open without
        # FILE_SHARE_DELETE, antivirus locking, etc.) become diagnosable
        # without flooding stderr. Missing one snapshot frame is not
        # fatal; the next ~100 ms tick retries naturally.
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        if last_exc is not None:
            _LOG.debug(
                "live snapshot atomic replace failed after retries: %s",
                last_exc,
            )
            return
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


__all__ = [
    "RadarCar",
    "build_radar_cars",
    "build_snapshot",
    "project_to_local",
    "write_snapshot_atomic",
]
