"""Heading-aligned frame projection (single source of truth).

LFS InSim uses an unsigned 16-bit ``heading`` value where ``0`` points to
world +y (north) and rotation is **anticlockwise** (``32768`` is south,
``16384`` is west). At heading ``h`` the car forward unit vector in
world XY is ``(-sin h,  cos h)`` and the right-hand vector is
``( cos h,  sin h)``.

This module exposes the canonical :func:`project_to_local` used by both
the live snapshot publisher (radar overlay) and the traffic analyser
(helicorsa-style proximity fallback). Keeping a single implementation
prevents the two consumers from drifting apart if the convention is
ever revisited.
"""

from __future__ import annotations

import math

from .protocol.packets import CompCar


def project_to_local(view: CompCar, other: CompCar) -> tuple[float, float]:
    """Project ``other``'s world XY into ``view``'s heading-aligned frame.

    Returns ``(x_local, y_local)`` in metres, right- and forward-positive,
    matching the helicorsa / acRadar convention so opponents always
    appear on the side they really are when the view car is yawed in a
    corner.
    """
    dx = other.x_m - view.x_m
    dy = other.y_m - view.y_m
    h = view.heading_rad
    cos_h = math.cos(h)
    sin_h = math.sin(h)
    x_local = dx * cos_h + dy * sin_h
    y_local = -dx * sin_h + dy * cos_h
    return x_local, y_local


__all__ = ["project_to_local"]
