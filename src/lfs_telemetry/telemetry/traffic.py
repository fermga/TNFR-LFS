"""Traffic snapshots from IS_MCI multi-car packets.

A small helper layer over :class:`lfs_telemetry.telemetry.protocol.packets.InSimMCI`
that turns a CompCar list into derived metrics relative to the *view* car:
gap_m to car ahead/behind, closing speed, race position.

Use ``RaceContext.last_mci`` (kept fresh by the InSim client when the
``request_mci=True`` flag is on) and call :func:`traffic_snapshot`.

Example::

    snap = traffic_snapshot(client.context)
    print(snap.gap_to_ahead_m, snap.car_ahead_plid)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .heading import project_to_local
from .protocol.insim import RaceContext
from .protocol.packets import CCI_BLUE, CCI_YELLOW, CompCar


@dataclass(slots=True)
class TrafficSnapshot:
    """Snapshot of every other car relative to the view car."""

    view_player_id: int
    view_position: int
    view_lap: int
    view_speed_ms: float
    car_ahead_plid: int | None = None
    car_ahead_position: int | None = None
    gap_to_ahead_m: float | None = None
    gap_to_ahead_s: float | None = None
    closing_speed_to_ahead_ms: float | None = None
    car_behind_plid: int | None = None
    car_behind_position: int | None = None
    gap_to_behind_m: float | None = None
    gap_to_behind_s: float | None = None
    closing_speed_to_behind_ms: float | None = None
    blue_flag_for_view: bool = False
    yellow_flag_active: bool = False
    num_cars: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _euclidean(a: CompCar, b: CompCar) -> float:
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def _find_nearest_neighbours(
    view: CompCar, cars: Iterable[CompCar],
) -> tuple[CompCar | None, CompCar | None]:
    """Return ``(ahead, behind)`` cars by spatial proximity around ``view``.

    Uses helicorsa-style local-frame projection: any car in the forward
    half-plane (``y_local > 0``) is "ahead", anything in the rear
    half-plane is "behind", and the closest one in each half-plane
    wins. Ties exactly on ``y_local == 0`` are broken by ``x_local``
    sign so a car directly to the right counts as ahead and a car
    directly to the left counts as behind, which matches the natural
    overtake semantics. Works in every session type — including solo
    practice, qualifying and hot-lapping where race positions don't
    discriminate cars.
    """
    nearest_ahead: tuple[float, CompCar] | None = None
    nearest_behind: tuple[float, CompCar] | None = None
    for other in cars:
        if other.player_id == view.player_id:
            continue
        x_l, y_l = project_to_local(view, other)
        d = math.hypot(x_l, y_l)
        is_ahead = y_l > 0.0 or (y_l == 0.0 and x_l > 0.0)
        if is_ahead:
            if nearest_ahead is None or d < nearest_ahead[0]:
                nearest_ahead = (d, other)
        else:
            if nearest_behind is None or d < nearest_behind[0]:
                nearest_behind = (d, other)
    return (
        nearest_ahead[1] if nearest_ahead is not None else None,
        nearest_behind[1] if nearest_behind is not None else None,
    )


def traffic_snapshot(
    ctx: RaceContext, *, view_player_id: int | None = None
) -> TrafficSnapshot | None:
    """Compute a :class:`TrafficSnapshot` from the latest IS_MCI in ``ctx``.

    Returns ``None`` if no MCI packet has arrived yet or the view car is not
    in the array.
    """
    mci = ctx.last_mci
    if mci is None or not mci.cars:
        return None
    plid = view_player_id if view_player_id is not None else ctx.view_player_id
    if plid is None:
        return None
    view: CompCar | None = next((c for c in mci.cars if c.player_id == plid), None)
    if view is None:
        return None
    return _build_snapshot(view, mci.cars)


def _build_snapshot(view: CompCar, cars: Iterable[CompCar]) -> TrafficSnapshot:
    cars_list = list(cars)
    snap = TrafficSnapshot(
        view_player_id=view.player_id,
        view_position=view.position,
        view_lap=view.lap,
        view_speed_ms=view.speed_ms,
        blue_flag_for_view=bool(view.info & CCI_BLUE),
        yellow_flag_active=any(c.info & CCI_YELLOW for c in cars_list),
        num_cars=len(cars_list),
    )
    # Primary signal: race-position ordering (works in races where every
    # car has a unique, ranked position). This preserves classical
    # "car-in-front-in-the-standings" semantics that fans of timing
    # screens expect.
    by_pos = {
        c.position: c
        for c in cars_list
        if c.player_id != view.player_id and c.position > 0
    }
    ahead = by_pos.get(view.position - 1) if view.position > 1 else None
    behind = by_pos.get(view.position + 1)
    # Fallback: helicorsa-style spatial proximity. Triggered whenever
    # race-position can't tell us who's around (solo practice, qualy,
    # hot-lap, broken position field, lapped opponents that share a
    # position number). Without this, the overlay's ahead/behind
    # indicators stay blank in the very sessions where they're most
    # useful for finding clear track.
    if ahead is None or behind is None:
        spatial_ahead, spatial_behind = _find_nearest_neighbours(
            view, cars_list,
        )
        if ahead is None:
            ahead = spatial_ahead
        if behind is None:
            behind = spatial_behind
    if ahead is not None:
        snap.car_ahead_plid = ahead.player_id
        snap.car_ahead_position = ahead.position
        snap.gap_to_ahead_m = _euclidean(view, ahead)
        snap.closing_speed_to_ahead_ms = view.speed_ms - ahead.speed_ms
        # Time gap: closure / view_speed; canonical when both cars
        # are roughly on the same racing line.
        if view.speed_ms > 0.5:
            snap.gap_to_ahead_s = snap.gap_to_ahead_m / view.speed_ms
    if behind is not None:
        snap.car_behind_plid = behind.player_id
        snap.car_behind_position = behind.position
        snap.gap_to_behind_m = _euclidean(view, behind)
        snap.closing_speed_to_behind_ms = behind.speed_ms - view.speed_ms
        if behind.speed_ms > 0.5:
            snap.gap_to_behind_s = snap.gap_to_behind_m / behind.speed_ms
    return snap


__all__ = ["TrafficSnapshot", "traffic_snapshot"]
