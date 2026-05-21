"""User-editable setup overrides on top of a parsed ``CarInfoBin``.

The LFS in-game F11 garage exposes a fixed catalogue of tunable
parameters (brake balance, gear ratios, springs, dampers, ARBs,
camber/toe, tyre pressures, ...). Those exact same numbers live inside
the ``<car>_CAR_info.bin`` export LFS writes: when LFS produces the
file it serialises the current garage state of the car, so a freshly
exported bin already encodes the user's current setup.

The catch is that re-exporting the bin after every garage tweak is
tedious and easy to forget. To close the loop we let the user edit the
same fields in-app via :class:`SetupEditorTab`. This module is the
pure data layer:

* :class:`SetupOverrides` — a flat, mutable dataclass with one optional
  field per editable LFS F11 parameter (``None`` means *leave the
  baseline alone*). Per-axle quantities (camber, toe, springs, dampers,
  ARBs, tyre pressures) are stored once per axle and applied
  symmetrically to both wheels of that axle, mirroring how LFS surfaces
  them in the garage UI.
* :func:`from_baseline` — pre-fills every override field with the
  baseline value so the editor opens on the current numbers (the user
  then tweaks individual fields).
* :func:`apply` — returns a *new* :class:`CarInfoBin` with the
  overrides merged in. The function is pure and leaves the input bin
  untouched, so it is safe to call from the UI thread on every edit.

Unit convention: every field is stored in the same units as the
underlying ``CarInfoBin`` field (SI: radians, kPa, N/m, N·s/m, Nm,
fractions in 0..1). The editor widget is responsible for converting
to/from display units (degrees, psi, N/mm, percentages).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .car_info_bin import CarInfoBin, CarInfoWheel

__all__ = ["SetupOverrides", "apply", "from_baseline"]


# Wheel index convention from ``CarInfoBin.wheels``: (RL, RR, FL, FR).
_RL, _RR, _FL, _FR = 0, 1, 2, 3


@dataclass(slots=True)
class SetupOverrides:
    """Mutable bag of optional LFS F11 setup overrides.

    Every field defaults to ``None``, meaning "keep whatever the
    baseline ``CarInfoBin`` already has". :func:`from_baseline` returns
    an instance with every field populated from a baseline bin, so the
    editor can present the user with the current numbers as starting
    point.

    Per-axle fields apply the same value to both wheels of that axle
    (LFS garage default). Asymmetric setups are not exposed: they are
    a footgun for the user with negligible payoff.
    """

    # ---- Chassis / fuel ------------------------------------------------
    passengers: int | None = None
    weight_dist_front: float | None = None      # 0..1 fraction
    fuel_capacity_l: float | None = None        # tank size, litres

    # ---- Brakes / steering --------------------------------------------
    brake_strength_nm: float | None = None
    brake_balance_front: float | None = None    # 0..1 fraction
    parallel_steer: float | None = None         # 0..1 fraction

    # ---- Drivetrain ---------------------------------------------------
    final_drive: float | None = None
    # Forward gears only (excluding the reverse slot at index 0 in
    # ``CarInfoBin.gear_ratios``). Length must equal
    # ``baseline.forward_gears`` when applied.
    gear_ratios: tuple[float, ...] | None = None
    drivetrain_efficiency: float | None = None  # 0..1
    torque_split: float | None = None           # AWD only, 0..1

    # ---- Suspension geometry (per axle, symmetric) --------------------
    front_camber_rad: float | None = None
    rear_camber_rad: float | None = None
    front_toe_in_rad: float | None = None
    rear_toe_in_rad: float | None = None

    # ---- Suspension rates (per axle, symmetric) -----------------------
    front_spring_const: float | None = None     # N/m
    rear_spring_const: float | None = None
    front_damping_comp: float | None = None     # N·s/m
    rear_damping_comp: float | None = None
    front_damping_rebound: float | None = None
    rear_damping_rebound: float | None = None
    front_anti_roll: float | None = None        # N/m
    rear_anti_roll: float | None = None

    # ---- Tyres (per axle, symmetric) ----------------------------------
    front_tyre_pressure_kpa: float | None = None
    rear_tyre_pressure_kpa: float | None = None


def from_baseline(baseline: CarInfoBin) -> SetupOverrides:
    """Return a :class:`SetupOverrides` pre-filled from ``baseline``.

    Per-axle fields take the value from the *left* wheel of each axle
    (RL for rear, FL for front). The right wheel is assumed to mirror
    it; if the baseline is asymmetric the left value wins, matching the
    F11 garage's per-axle UI.
    """
    fl = baseline.wheels[_FL]
    rl = baseline.wheels[_RL]
    forward_gears = tuple(baseline.gear_ratios[1:])  # drop reverse slot
    return SetupOverrides(
        passengers=baseline.passengers,
        weight_dist_front=baseline.weight_dist_front,
        fuel_capacity_l=baseline.fuel_capacity_l,
        brake_strength_nm=baseline.brake_strength_nm,
        brake_balance_front=baseline.brake_balance_front,
        parallel_steer=baseline.parallel_steer,
        final_drive=baseline.final_drive,
        gear_ratios=forward_gears,
        drivetrain_efficiency=baseline.drivetrain_efficiency,
        torque_split=baseline.torque_split,
        front_camber_rad=fl.camber_rad,
        rear_camber_rad=rl.camber_rad,
        front_toe_in_rad=fl.toe_in_rad,
        rear_toe_in_rad=rl.toe_in_rad,
        front_spring_const=fl.spring_const,
        rear_spring_const=rl.spring_const,
        front_damping_comp=fl.damping_comp,
        rear_damping_comp=rl.damping_comp,
        front_damping_rebound=fl.damping_rebound,
        rear_damping_rebound=rl.damping_rebound,
        front_anti_roll=fl.anti_roll,
        rear_anti_roll=rl.anti_roll,
        front_tyre_pressure_kpa=fl.tyre_pressure_kpa,
        rear_tyre_pressure_kpa=rl.tyre_pressure_kpa,
    )


def _patch_wheel(
    w: CarInfoWheel,
    *,
    camber_rad: float | None,
    toe_in_rad: float | None,
    spring_const: float | None,
    damping_comp: float | None,
    damping_rebound: float | None,
    anti_roll: float | None,
    tyre_pressure_kpa: float | None,
) -> CarInfoWheel:
    """Return a copy of ``w`` with the given non-None fields replaced."""
    patch: dict = {}
    if camber_rad is not None:
        patch["camber_rad"] = float(camber_rad)
    if toe_in_rad is not None:
        patch["toe_in_rad"] = float(toe_in_rad)
    if spring_const is not None:
        patch["spring_const"] = float(spring_const)
    if damping_comp is not None:
        patch["damping_comp"] = float(damping_comp)
    if damping_rebound is not None:
        patch["damping_rebound"] = float(damping_rebound)
    if anti_roll is not None:
        patch["anti_roll"] = float(anti_roll)
    if tyre_pressure_kpa is not None:
        patch["tyre_pressure_kpa"] = float(tyre_pressure_kpa)
    if not patch:
        return w
    return replace(w, **patch)


def apply(baseline: CarInfoBin, overrides: SetupOverrides) -> CarInfoBin:
    """Return a new :class:`CarInfoBin` with ``overrides`` merged in.

    Pure function: ``baseline`` is never mutated. Fields left as
    ``None`` in ``overrides`` keep their baseline value. ``gear_ratios``
    overrides must have the same length as ``baseline.forward_gears``;
    a ``ValueError`` is raised otherwise (the editor enforces this).
    """
    # Per-wheel patches: front pair shares the front_* values, rear
    # pair shares the rear_* values.
    rl = _patch_wheel(
        baseline.wheels[_RL],
        camber_rad=overrides.rear_camber_rad,
        toe_in_rad=overrides.rear_toe_in_rad,
        spring_const=overrides.rear_spring_const,
        damping_comp=overrides.rear_damping_comp,
        damping_rebound=overrides.rear_damping_rebound,
        anti_roll=overrides.rear_anti_roll,
        tyre_pressure_kpa=overrides.rear_tyre_pressure_kpa,
    )
    rr = _patch_wheel(
        baseline.wheels[_RR],
        camber_rad=overrides.rear_camber_rad,
        toe_in_rad=overrides.rear_toe_in_rad,
        spring_const=overrides.rear_spring_const,
        damping_comp=overrides.rear_damping_comp,
        damping_rebound=overrides.rear_damping_rebound,
        anti_roll=overrides.rear_anti_roll,
        tyre_pressure_kpa=overrides.rear_tyre_pressure_kpa,
    )
    fl = _patch_wheel(
        baseline.wheels[_FL],
        camber_rad=overrides.front_camber_rad,
        toe_in_rad=overrides.front_toe_in_rad,
        spring_const=overrides.front_spring_const,
        damping_comp=overrides.front_damping_comp,
        damping_rebound=overrides.front_damping_rebound,
        anti_roll=overrides.front_anti_roll,
        tyre_pressure_kpa=overrides.front_tyre_pressure_kpa,
    )
    fr = _patch_wheel(
        baseline.wheels[_FR],
        camber_rad=overrides.front_camber_rad,
        toe_in_rad=overrides.front_toe_in_rad,
        spring_const=overrides.front_spring_const,
        damping_comp=overrides.front_damping_comp,
        damping_rebound=overrides.front_damping_rebound,
        anti_roll=overrides.front_anti_roll,
        tyre_pressure_kpa=overrides.front_tyre_pressure_kpa,
    )

    # Top-level scalar patches.
    patch: dict = {"wheels": (rl, rr, fl, fr)}
    if overrides.passengers is not None:
        patch["passengers"] = int(overrides.passengers)
    if overrides.weight_dist_front is not None:
        patch["weight_dist_front"] = float(overrides.weight_dist_front)
    if overrides.fuel_capacity_l is not None:
        patch["fuel_capacity_l"] = float(overrides.fuel_capacity_l)
    if overrides.brake_strength_nm is not None:
        patch["brake_strength_nm"] = float(overrides.brake_strength_nm)
    if overrides.brake_balance_front is not None:
        patch["brake_balance_front"] = float(overrides.brake_balance_front)
    if overrides.parallel_steer is not None:
        patch["parallel_steer"] = float(overrides.parallel_steer)
    if overrides.final_drive is not None:
        patch["final_drive"] = float(overrides.final_drive)
    if overrides.drivetrain_efficiency is not None:
        patch["drivetrain_efficiency"] = float(
            overrides.drivetrain_efficiency
        )
    if overrides.torque_split is not None:
        patch["torque_split"] = float(overrides.torque_split)
    if overrides.gear_ratios is not None:
        new_forward = tuple(float(g) for g in overrides.gear_ratios)
        if len(new_forward) != baseline.forward_gears:
            raise ValueError(
                f"gear_ratios override has {len(new_forward)} gears,"
                f" baseline expects {baseline.forward_gears}"
            )
        # Preserve the reverse slot at index 0 from the baseline.
        reverse = baseline.gear_ratios[0]
        patch["gear_ratios"] = (reverse, *new_forward)

    return replace(baseline, **patch)
