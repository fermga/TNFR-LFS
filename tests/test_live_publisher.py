"""Tests for the live snapshot publisher (radar projection + JSON I/O)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from lfs_telemetry.telemetry import live_publisher
from lfs_telemetry.telemetry.live_publisher import (
    RadarCar,
    build_radar_cars,
    build_snapshot,
    project_to_local,
    write_snapshot_atomic,
)
from lfs_telemetry.telemetry.protocol.insim import RaceContext
from lfs_telemetry.telemetry.protocol.packets import CompCar, InSimMCI


def _car(
    *, plid: int, x: float = 0.0, y: float = 0.0, heading_rad: float = 0.0,
    speed_ms: float = 0.0, position: int = 1, lap: int = 0,
) -> CompCar:
    return CompCar(
        node=0, lap=lap, player_id=plid, position=position, info=0,
        x_m=x, y_m=y, z_m=0.0,
        speed_ms=speed_ms, direction_rad=0.0,
        heading_rad=heading_rad, ang_vel_rads=0.0,
    )


# ---------------------------------------------------------------- projection

def test_project_to_local_north_facing_returns_world_offset() -> None:
    """heading=0 (facing +y world) → local frame == world offset."""
    view = _car(plid=1, x=0.0, y=0.0, heading_rad=0.0)
    other = _car(plid=2, x=0.0, y=10.0)
    x_l, y_l = project_to_local(view, other)
    assert math.isclose(x_l, 0.0, abs_tol=1e-9)
    assert math.isclose(y_l, 10.0, abs_tol=1e-9)


def test_project_to_local_west_facing_swaps_axes() -> None:
    """heading=π/2 (LFS = facing world -x, i.e. west) projects correctly.

    LFS InSim heading convention: 0 = world +y (north), increases
    anticlockwise. So at h=π/2 the car forward axis points to -x.
    A car at world +x is therefore directly behind the view car
    (y_local < 0, x_local = 0); a car at world -y is to the driver's
    right (x_local > 0, y_local = 0).
    """
    view = _car(plid=1, x=0.0, y=0.0, heading_rad=math.pi / 2)
    behind = _car(plid=2, x=10.0, y=0.0)
    x_l, y_l = project_to_local(view, behind)
    assert math.isclose(x_l, 0.0, abs_tol=1e-9)
    assert math.isclose(y_l, -10.0, abs_tol=1e-9)
    # And a car at world south (dy=-10): the view faces west (-x), so
    # its right-hand side points north (+y). A car at -y is therefore
    # on the driver's LEFT (x_local < 0).
    left = _car(plid=3, x=0.0, y=-10.0)
    x_l, y_l = project_to_local(view, left)
    assert math.isclose(x_l, -10.0, abs_tol=1e-9)
    assert math.isclose(y_l, 0.0, abs_tol=1e-9)


def test_build_radar_cars_marks_view_and_computes_distance() -> None:
    view = _car(plid=1, x=0.0, y=0.0, heading_rad=0.0, speed_ms=30.0)
    other = _car(plid=2, x=3.0, y=4.0, speed_ms=25.0)
    radar = build_radar_cars(view, [view, other])
    by_plid = {c.plid: c for c in radar}
    assert by_plid[1].is_view is True
    assert by_plid[1].distance_m == 0.0
    assert by_plid[2].is_view is False
    assert math.isclose(by_plid[2].distance_m, 5.0, abs_tol=1e-9)
    assert math.isclose(by_plid[2].relative_speed_ms, -5.0, abs_tol=1e-9)


def test_radar_car_to_dict_uses_short_keys() -> None:
    rc = RadarCar(plid=7, x_local_m=1.234, y_local_m=-2.345,
                  distance_m=2.6, relative_speed_ms=1.1, is_view=False,
                  node=42, lap=3)
    d = rc.to_dict()
    assert set(d) == {"plid", "x", "y", "d", "rel_v", "view",
                      "node", "lap"}
    assert d["plid"] == 7
    assert d["view"] is False
    assert d["node"] == 42
    assert d["lap"] == 3


# ---------------------------------------------------------------- snapshot

def test_build_snapshot_no_ctx_only_view_sample_fields() -> None:
    snap = build_snapshot(
        None, armed=False, samples_count=0,
        last_sample_speed_ms=20.0, last_sample_rpm=4500.0,
        last_sample_gear=3, last_sample_fuel_pct=50.0,
        last_sample_throttle=0.5, last_sample_brake=0.0,
        monotonic_ts=1.5,
    )
    assert snap["ts"] == 1.5
    assert snap["armed"] is False
    assert snap["samples"] == 0
    assert snap["view_speed_ms"] == 20.0
    assert snap["view_speed_kmh"] == 72.0
    assert snap["view_rpm"] == 4500
    assert snap["view_gear"] == 3
    assert snap["view_fuel_pct"] == 50.0
    assert snap["cars"] == []
    assert snap["traffic"] is None
    # Must round-trip through JSON.
    assert json.loads(json.dumps(snap)) == snap


def test_build_snapshot_with_mci_populates_cars_and_traffic() -> None:
    ctx = RaceContext()
    ctx.track = "BL1"
    ctx.weather = 0
    ctx.race_in_progress = 1
    ctx.view_player_id = 1
    ctx.lap_count = {1: 2}
    ctx.last_lap_ms = {1: 90_500}
    ctx.lap_times_ms = {1: [92_000, 90_500]}
    view = _car(plid=1, x=0.0, y=0.0, heading_rad=0.0, position=2, lap=2)
    behind = _car(plid=2, x=0.0, y=-15.0, position=3, lap=2)
    ctx.last_mci = InSimMCI(cars=[view, behind])

    snap = build_snapshot(
        ctx, armed=True, samples_count=42,
        current_lap_ms=89_900, monotonic_ts=10.0,
    )
    assert snap["track"] == "BL1"
    assert snap["view_plid"] == 1
    assert snap["view_position"] == 2
    assert snap["view_lap"] == 2
    assert snap["last_lap_ms"] == 90_500
    assert snap["best_lap_ms"] == 90_500
    # Without a per-node ``delta_to_best_ms`` from NodeDeltaTracker
    # the snapshot leaves delta_vs_best_ms unset (None); the crude
    # ``current_lap_ms - best_lap_ms`` fallback was removed because it
    # behaves like "time-to-start/finish-line" rather than a real
    # pace-vs-PB signal.
    assert snap["delta_vs_best_ms"] is None
    cars = snap["cars"]
    assert len(cars) == 2
    # JSON-serialisable.
    json.dumps(snap)
    # Traffic dict has at least the basic shape.
    traffic = snap["traffic"]
    assert traffic is not None
    assert traffic["num_cars"] == 2


# ---------------------------------------------------------------- I/O

def test_write_snapshot_atomic_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "live.json"
    snap = {"ts": 0.0, "armed": True, "samples": 1, "cars": []}
    write_snapshot_atomic(target, snap)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == snap

    # Overwriting must still work (os.replace handles existing target).
    snap2 = dict(snap, samples=2)
    write_snapshot_atomic(target, snap2)
    assert json.loads(target.read_text(encoding="utf-8")) == snap2


def test_write_snapshot_atomic_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "live.json"
    write_snapshot_atomic(target, {"hello": "world"})
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"hello": "world"}


def test_module_exposes_expected_public_api() -> None:
    for name in (
        "RadarCar", "project_to_local", "build_radar_cars",
        "build_snapshot", "write_snapshot_atomic",
    ):
        assert hasattr(live_publisher, name), name
