"""Phase 9.B — Closed-form sensitivity narrator.

Given a :class:`PhysicalAction` proposed by the advisor, estimate the
order-of-magnitude lap-time impact per unit change using deterministic
physical formulas (no surrogate model, no fitting).

The output is **never** authoritative for tuning — it is an
explanatory enrichment for the *Why* column. Each estimate carries a
``confidence`` flag (``high`` / ``medium`` / ``low``) so the UI can
hedge weak ones.

All sentences are jargon-free (see ``docs/TNFR_SETUP_ADVISOR.md``
\u00a710.3): only physical engineering language.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from lfs_telemetry.telemetry.car_info_bin import CarInfoBin, CarInfoWheel

from .operators import PhysicalAction


Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class SensitivityEstimate:
    """One physical-narrator sentence + magnitude tag for an action."""

    action_kind: str
    target: str
    sentence: str
    lap_time_ms_per_unit: float | None
    confidence: Confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_AXLE_WHEELS: dict[str, tuple[str, ...]] = {
    "FL": ("FL",), "FR": ("FR",), "RL": ("RL",), "RR": ("RR",),
    "front": ("FL", "FR"), "rear": ("RL", "RR"),
    "global": ("FL", "FR", "RL", "RR"),
}


def _resolve_wheels(
    baseline: CarInfoBin, target: str,
) -> tuple[CarInfoWheel, ...]:
    names = _AXLE_WHEELS.get(target, ())
    by_name = {w.name: w for w in baseline.wheels}
    out: list[CarInfoWheel] = []
    for n in names:
        w = by_name.get(n)
        if w is not None:
            out.append(w)
    return tuple(out)


def _corner_mass_kg(baseline: CarInfoBin, target: str) -> float:
    """Approximate corner sprung mass for the resolved wheels."""
    if not baseline.wheels:
        return baseline.mass_kg / 4.0
    total = baseline.mass_kg
    front_frac = baseline.weight_dist_front
    wheels = _resolve_wheels(baseline, target)
    if not wheels:
        return total / 4.0
    # 50/50 left/right split assumed; weight_dist_front does L/R averaged.
    fractions = []
    for w in wheels:
        if w.name.startswith("F"):
            fractions.append(front_frac / 2.0)
        else:
            fractions.append((1.0 - front_frac) / 2.0)
    return total * sum(fractions) / max(len(wheels), 1)


# ---------------------------------------------------------------------------
# Per-kind estimators
# ---------------------------------------------------------------------------


def _estimate_damper(
    action: PhysicalAction, baseline: CarInfoBin,
) -> SensitivityEstimate:
    wheels = _resolve_wheels(baseline, action.target)
    if not wheels:
        return SensitivityEstimate(
            action.kind, action.target,
            "Damping adjustment expected to stabilise wheel-load oscillation.",
            None, "low",
        )
    # LFS damper "click" ~= 5% of nominal damping coefficient (heuristic).
    click_step = 0.05
    avg_damping = sum(
        w.damping_rebound if "rebound" in action.kind else w.damping_comp
        for w in wheels
    ) / len(wheels)
    avg_spring = sum(w.spring_const for w in wheels) / len(wheels)
    m_corner = _corner_mass_kg(baseline, action.target)
    if avg_spring <= 0 or m_corner <= 0:
        return SensitivityEstimate(
            action.kind, action.target,
            "Damping adjustment expected to stabilise wheel-load oscillation.",
            None, "low",
        )
    c_crit = 2.0 * math.sqrt(avg_spring * m_corner)
    zeta_before = avg_damping / c_crit
    delta_c = action.delta * click_step * avg_damping
    zeta_after = (avg_damping + delta_c) / c_crit
    # Empirical rule of thumb: ~30 ms/lap per 0.05 absolute change in zeta
    # when the corner is the limit, conservative outside the limit.
    lap_ms_per_unit = (zeta_after - zeta_before) / max(action.delta, 1e-9) \
        * 600.0  # ms per click
    sentence = (
        f"Damping ratio at the corner shifts from "
        f"{zeta_before:.2f} to {zeta_after:.2f} (target 0.5-0.6 on dry "
        f"asphalt); expect ~{abs(lap_ms_per_unit):.0f} ms/lap per click "
        f"while the corner is the limiting one."
    )
    return SensitivityEstimate(
        action.kind, action.target, sentence,
        lap_time_ms_per_unit=lap_ms_per_unit,
        confidence="medium",
    )


def _estimate_spring(
    action: PhysicalAction, baseline: CarInfoBin,
) -> SensitivityEstimate:
    wheels = _resolve_wheels(baseline, action.target)
    if not wheels:
        return SensitivityEstimate(
            action.kind, action.target,
            "Spring change expected to retune ride frequency.",
            None, "low",
        )
    avg_spring = sum(w.spring_const for w in wheels) / len(wheels)  # N/m
    m_corner = _corner_mass_kg(baseline, action.target)
    if m_corner <= 0 or avg_spring <= 0:
        return SensitivityEstimate(
            action.kind, action.target,
            "Spring change expected to retune ride frequency.",
            None, "low",
        )
    # action.delta is in N/mm per docs; convert to N/m.
    delta_k_nm = action.delta * 1000.0
    f_before = math.sqrt(avg_spring / m_corner) / (2.0 * math.pi)
    f_after = math.sqrt(max(avg_spring + delta_k_nm, 1.0) / m_corner) \
        / (2.0 * math.pi)
    sentence = (
        f"Ride frequency at the corner moves from {f_before:.2f} Hz to "
        f"{f_after:.2f} Hz (asphalt target 1.5-2.5 Hz on a road car, "
        f"2.5-3.5 Hz on a stiff race car)."
    )
    return SensitivityEstimate(
        action.kind, action.target, sentence,
        lap_time_ms_per_unit=None, confidence="medium",
    )


def _estimate_arb(
    action: PhysicalAction, _baseline: CarInfoBin,
) -> SensitivityEstimate:
    # ARB delta is signed N·mm/deg (kept abstract). Map directly to a
    # qualitative balance shift sentence.
    direction = "stiffer" if action.delta > 0 else "softer"
    side = action.target if action.target in ("front", "rear") else "axle"
    bal = "less" if (action.delta > 0) == (side == "rear") else "more"
    sentence = (
        f"Anti-roll bar on the {side} becomes {direction}; in steady-state "
        f"cornering the {side} will gain lateral load transfer, so the "
        f"car should oversteer {bal} on power-on."
    )
    return SensitivityEstimate(
        action.kind, action.target, sentence,
        lap_time_ms_per_unit=None, confidence="medium",
    )


def _estimate_tyre_pressure(
    action: PhysicalAction, baseline: CarInfoBin,
) -> SensitivityEstimate:
    wheels = _resolve_wheels(baseline, action.target)
    if not wheels:
        return SensitivityEstimate(
            action.kind, action.target,
            "Pressure change shifts contact patch area and tyre stiffness.",
            None, "low",
        )
    avg_p = sum(w.tyre_pressure_kpa for w in wheels) / len(wheels)
    new_p = avg_p + action.delta
    # Contact patch area scales roughly as 1/p (Hertz/lumped-membrane).
    area_ratio = avg_p / max(new_p, 1.0)
    sentence = (
        f"Average pressure at the corner moves from {avg_p:.0f} kPa to "
        f"{new_p:.0f} kPa; static contact-patch area changes by "
        f"{(area_ratio - 1.0) * 100.0:+.1f}% (Hertz approximation)."
    )
    return SensitivityEstimate(
        action.kind, action.target, sentence,
        lap_time_ms_per_unit=None, confidence="medium",
    )


def _estimate_brake_bias(
    action: PhysicalAction, baseline: CarInfoBin,
) -> SensitivityEstimate:
    base_pct = baseline.brake_balance_front * 100.0
    new_pct = base_pct + action.delta
    side = "rear" if action.delta < 0 else "front"
    sentence = (
        f"Brake bias moves from {base_pct:.1f}% front to "
        f"{new_pct:.1f}% front; under threshold braking the {side} "
        f"axle will start to limit deceleration sooner."
    )
    return SensitivityEstimate(
        action.kind, action.target, sentence,
        lap_time_ms_per_unit=None, confidence="high",
    )


def _estimate_alignment(
    action: PhysicalAction, baseline: CarInfoBin,
) -> SensitivityEstimate:
    wheels = _resolve_wheels(baseline, action.target)
    if not wheels:
        return SensitivityEstimate(
            action.kind, action.target,
            f"{action.kind.capitalize()} change retunes contact patch "
            f"alignment under load.", None, "low",
        )
    if action.kind == "camber":
        avg_deg = (sum(w.camber_rad for w in wheels) / len(wheels)) \
            * 180.0 / math.pi
        new_deg = avg_deg + action.delta
        sentence = (
            f"Static camber at the corner moves from {avg_deg:+.2f}\u00b0 to "
            f"{new_deg:+.2f}\u00b0; expect tyre temperatures to redistribute "
            f"across the tread and peak lateral grip to shift "
            f"{'inboard' if action.delta < 0 else 'outboard'}."
        )
    else:  # toe
        avg_deg = (sum(w.toe_in_rad for w in wheels) / len(wheels)) \
            * 180.0 / math.pi
        new_deg = avg_deg + action.delta
        sentence = (
            f"Static toe-in at the corner moves from {avg_deg:+.2f}\u00b0 to "
            f"{new_deg:+.2f}\u00b0; turn-in response sharpens with more "
            f"toe-out and straight-line stability improves with more toe-in."
        )
    return SensitivityEstimate(
        action.kind, action.target, sentence,
        lap_time_ms_per_unit=None, confidence="medium",
    )


def _estimate_ride_height(
    action: PhysicalAction, baseline: CarInfoBin,
) -> SensitivityEstimate:
    """Ride-height retune: reframe as chassis-attitude / aero-balance.

    LFS' setup screen ride height is in millimetres above the chassis
    rest pose. Lowering one axle by Δh (mm) shifts the static rake
    angle by approximately ``Δh / wheelbase`` radians, which moves
    aerodynamic balance and mechanical roll-centre height. We surface
    a deterministic physical sentence; no lap-time mapping (highly
    car/track dependent).
    """
    wb_mm = max(baseline.wheelbase_m * 1000.0, 1.0)
    # Rake-angle change in milli-radians (≈ degrees × 17.45).
    rake_change_mrad = (
        (-action.delta) / wb_mm * 1000.0 if action.target == "front"
        else action.delta / wb_mm * 1000.0
    )
    direction = (
        "lower" if action.delta < 0 else "raise"
    )
    side = action.target if action.target in ("front", "rear") else "axle"
    nose_dir = "nose-down" if rake_change_mrad > 0 else "nose-up"
    sentence = (
        f"Ride height at the {side} moves by {action.delta:+.1f} mm "
        f"({direction} the {side}); static rake shifts "
        f"{abs(rake_change_mrad):.1f} mrad {nose_dir}, which moves "
        f"aerodynamic balance and lowers the mechanical roll-centre at "
        f"that axle."
    )
    return SensitivityEstimate(
        action.kind, action.target, sentence,
        lap_time_ms_per_unit=None, confidence="low",
    )


_KIND_DISPATCH = {
    "damper_rebound": _estimate_damper,
    "damper_bump": _estimate_damper,
    "spring": _estimate_spring,
    "arb": _estimate_arb,
    "tyre_pressure": _estimate_tyre_pressure,
    "brake_bias": _estimate_brake_bias,
    "camber": _estimate_alignment,
    "toe": _estimate_alignment,
    "ride_height": _estimate_ride_height,
}


def estimate_action_sensitivity(
    action: PhysicalAction, baseline: CarInfoBin,
) -> SensitivityEstimate:
    """Return a deterministic physical sentence for ``action``."""
    fn = _KIND_DISPATCH.get(action.kind)
    if fn is None:
        return SensitivityEstimate(
            action.kind, action.target,
            f"Adjustment to {action.kind.replace('_', ' ')} applied to "
            f"the {action.target}.",
            None, "low",
        )
    return fn(action, baseline)


__all__ = (
    "Confidence",
    "SensitivityEstimate",
    "estimate_action_sensitivity",
)
