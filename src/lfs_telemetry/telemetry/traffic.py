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
from collections.abc import Iterable
from dataclasses import asdict, dataclass

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


def _gap_on_track_m(
    view: CompCar,
    other: CompCar,
    *,
    forward: bool,
    node_to_s_m: list[float] | None = None,
    track_length_m: float = 0.0,
) -> float:
    """Best-effort on-track distance from ``view`` to ``other``.

    If a per-node arclength table is provided (one entry per LFS path
    node, e.g. loaded from ``racing_lines/<TRACK>_racing.csv``), the
    gap is computed along the track using ``CompCar.node`` and wraps
    around the start/finish line. Otherwise falls back to straight-line
    euclidean distance, which is correct on straights but underestimates
    the real gap inside curves (chicanes, hairpins).

    ``forward=True`` returns ``(s_other - s_view) mod L``  (other is
    ahead). ``forward=False`` returns ``(s_view - s_other) mod L``
    (other is behind).
    """
    eu = _euclidean(view, other)
    if (
        node_to_s_m
        and track_length_m > 0.0
        and len(node_to_s_m) > 0
    ):
        n = len(node_to_s_m)
        vi = int(view.node) % n
        oi = int(other.node) % n
        s_view = node_to_s_m[vi]
        s_other = node_to_s_m[oi]
        gap = s_other - s_view if forward else s_view - s_other
        if gap < 0.0:
            gap += track_length_m
        # Lap-mismatch / wrap artefact: when an opponent is one lap
        # apart from us but physically right next to us, node arclength
        # wraps to ~track_length while euclidean shows them within a
        # few metres. Trust euclidean in that close-range case so the
        # behind/ahead gauge doesn't jump to ~3 km on the grid or just
        # after the start/finish line.
        if eu < 30.0 and gap > eu + 50.0:
            return eu
        # Cars sharing exactly the same node fall through to euclidean
        # so the radar's near-field readout still has sub-node
        # resolution (LFS nodes are several metres apart on most
        # tracks).
        if gap > 0.5:
            return gap
    return eu


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

    When the view car is moving we ignore stationary opponents (cars
    with ``speed_ms < 0.5``). LFS exposes no explicit "in pit" flag in
    ``CompCar.Info`` so a pitting / spectating / just-spawned car would
    otherwise lock the ahead/behind gauge to a car that isn't actually
    racing us. On a standing grid (view also stationary) the filter
    self-disables so we still see neighbours at the start.
    """
    view_moving = view.speed_ms > 0.5
    nearest_ahead: tuple[float, CompCar] | None = None
    nearest_behind: tuple[float, CompCar] | None = None
    for other in cars:
        if other.player_id == view.player_id:
            continue
        if view_moving and other.speed_ms < 0.5:
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
    ctx: RaceContext, *, view_player_id: int | None = None,
    node_to_s_m: list[float] | None = None,
    track_length_m: float = 0.0,
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
    return _build_snapshot(
        view, mci.cars,
        node_to_s_m=node_to_s_m, track_length_m=track_length_m,
    )


def _build_snapshot(
    view: CompCar, cars: Iterable[CompCar],
    *,
    node_to_s_m: list[float] | None = None,
    track_length_m: float = 0.0,
) -> TrafficSnapshot:
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
    # car has a unique, ranked position). Use ``max(p < view)`` /
    # ``min(p > view)`` instead of strict ``view.position ± 1`` so that
    # a disconnect/DNF leaving a gap in the position table (e.g. 1, 2,
    # 4, 5) still resolves to the *actually* adjacent car instead of
    # silently falling through to the spatial fallback.
    by_pos = {
        c.position: c
        for c in cars_list
        if c.player_id != view.player_id and c.position > 0
    }
    ahead = behind = None
    if view.position > 0 and by_pos:
        above = [p for p in by_pos if p < view.position]
        below = [p for p in by_pos if p > view.position]
        if above:
            ahead = by_pos[max(above)]
        if below:
            behind = by_pos[min(below)]
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
        snap.gap_to_ahead_m = _gap_on_track_m(
            view, ahead, forward=True,
            node_to_s_m=node_to_s_m, track_length_m=track_length_m,
        )
        snap.closing_speed_to_ahead_ms = view.speed_ms - ahead.speed_ms
        # Time gap: closure / view_speed; canonical when both cars
        # are roughly on the same racing line.
        if view.speed_ms > 0.5:
            snap.gap_to_ahead_s = snap.gap_to_ahead_m / view.speed_ms
    if behind is not None:
        snap.car_behind_plid = behind.player_id
        snap.car_behind_position = behind.position
        snap.gap_to_behind_m = _gap_on_track_m(
            view, behind, forward=False,
            node_to_s_m=node_to_s_m, track_length_m=track_length_m,
        )
        snap.closing_speed_to_behind_ms = behind.speed_ms - view.speed_ms
        # Convention: both ahead/behind time gaps use the view car's
        # speed as reference (timing-tower style "time at line").
        # Previously we divided by ``behind.speed_ms`` which produced
        # a different physical quantity and made the two gaps not
        # directly comparable. Falls back to behind's speed only when
        # the view car is essentially stopped.
        ref_speed = view.speed_ms if view.speed_ms > 0.5 else behind.speed_ms
        if ref_speed > 0.5:
            snap.gap_to_behind_s = snap.gap_to_behind_m / ref_speed
    return snap


__all__ = ["TrafficSnapshot", "traffic_snapshot"]
