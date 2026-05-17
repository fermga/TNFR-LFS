"""Tests for ``telemetry.setup_overrides``.

The module is the data layer that backs the in-app garage editor: it
turns optional user edits into a modified :class:`CarInfoBin` that the
:class:`SetupAdvisor` can consume as baseline. These tests pin the
contract end-to-end:

* :func:`from_baseline` round-trips through :func:`apply` to an
  identical bin (no edits ⇒ no change).
* Each per-axle edit propagates to both wheels of the axle.
* Scalar edits land on the right top-level field.
* Gear ratio overrides preserve the reverse slot and reject wrong
  lengths.
* The baseline bin is never mutated.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from lfs_telemetry.telemetry.car_info_bin import CarInfoBin, CarInfoWheel
from lfs_telemetry.telemetry.setup_overrides import (
    SetupOverrides,
    apply,
    from_baseline,
)


def _synthetic_baseline() -> CarInfoBin:
    wheels = tuple(
        CarInfoWheel(
            name=n, tyre_type=0,
            contact_x_m=0.75 * (1 if "R" in n else -1),
            contact_y_m=1.20 if n.startswith("F") else -1.20,
            contact_z_m=0.0, unsprung_mass_kg=18.0,
            tyre_width_m=0.22, sidewall_height_prop=0.55,
            rim_radius_m=0.20, rim_width_m=0.18,
            spring_const=42000.0, damping_comp=3200.0,
            damping_rebound=3800.0, anti_roll=18000.0,
            camber_rad=-0.035, inclination_rad=0.0, caster_rad=0.10,
            scrub_radius_m=0.01,
            moment_inertia=1.2, susp_deflection_m=0.05,
            max_susp_deflection_m=0.12,
            tyre_vert_spring=180000.0, tyre_vert_deflection=0.008,
            tyre_pressure_kpa=160.0, air_temp_c=25.0, toe_in_rad=0.0,
        )
        for n in ("RL", "RR", "FL", "FR")
    )
    return CarInfoBin(
        file_version=2, short_name="FBM", passengers=1,
        cg_x_m=0.0, cg_y_m=0.0, cg_z_m=0.30,
        cg_x_rel=0.5, cg_y_rel=0.5, cg_z_rel=0.30,
        fuel_tank_x_m=0.0, fuel_tank_y_m=-0.5, fuel_tank_z_m=0.25,
        max_torque_nm=180.0, max_torque_rpm=5500.0,
        max_power_kw=110.0, max_power_rpm=7800.0,
        fuel_capacity_l=40.0, mass_kg=525.0, wheelbase_m=2.40,
        weight_dist_front=0.50, forward_gears=6, drive="RWD",
        torque_split=0.0, drivetrain_efficiency=0.93,
        gear_ratios=(-3.0, 3.0, 2.1, 1.6, 1.3, 1.1, 0.9),
        final_drive=4.0, parallel_steer=0.5,
        brake_strength_nm=1100.0, brake_balance_front=0.60,
        wheels=wheels,
    )


def test_from_baseline_populates_every_field() -> None:
    """No field should remain ``None`` after pre-filling from a bin."""
    baseline = _synthetic_baseline()
    ov = from_baseline(baseline)
    for name in (
        "passengers", "weight_dist_front", "fuel_capacity_l",
        "brake_strength_nm", "brake_balance_front", "parallel_steer",
        "final_drive", "gear_ratios", "drivetrain_efficiency",
        "torque_split",
        "front_camber_rad", "rear_camber_rad",
        "front_toe_in_rad", "rear_toe_in_rad",
        "front_spring_const", "rear_spring_const",
        "front_damping_comp", "rear_damping_comp",
        "front_damping_rebound", "rear_damping_rebound",
        "front_anti_roll", "rear_anti_roll",
        "front_tyre_pressure_kpa", "rear_tyre_pressure_kpa",
    ):
        assert getattr(ov, name) is not None, name
    # Reverse slot stripped from the forward gears view.
    assert ov.gear_ratios == (3.0, 2.1, 1.6, 1.3, 1.1, 0.9)


def test_apply_with_empty_overrides_is_noop_clone() -> None:
    """All-None overrides ⇒ a fresh bin equal to the baseline."""
    baseline = _synthetic_baseline()
    out = apply(baseline, SetupOverrides())
    assert out is not baseline
    assert out == baseline


def test_apply_does_not_mutate_baseline() -> None:
    baseline = _synthetic_baseline()
    snapshot = replace(baseline, wheels=tuple(baseline.wheels))
    apply(baseline, SetupOverrides(brake_balance_front=0.80))
    assert baseline == snapshot


def test_apply_roundtrip_from_baseline_equals_baseline() -> None:
    """Editor opens on baseline → user saves without edits → no diff."""
    baseline = _synthetic_baseline()
    out = apply(baseline, from_baseline(baseline))
    assert out == baseline


def test_per_axle_camber_applies_to_both_wheels_of_axle() -> None:
    baseline = _synthetic_baseline()
    out = apply(
        baseline,
        SetupOverrides(front_camber_rad=-0.07, rear_camber_rad=-0.02),
    )
    # Wheel order: RL=0, RR=1, FL=2, FR=3.
    assert out.wheels[2].camber_rad == pytest.approx(-0.07)
    assert out.wheels[3].camber_rad == pytest.approx(-0.07)
    assert out.wheels[0].camber_rad == pytest.approx(-0.02)
    assert out.wheels[1].camber_rad == pytest.approx(-0.02)


def test_per_axle_tyre_pressure_and_springs() -> None:
    baseline = _synthetic_baseline()
    out = apply(
        baseline,
        SetupOverrides(
            front_tyre_pressure_kpa=175.0,
            rear_tyre_pressure_kpa=185.0,
            front_spring_const=50000.0,
            rear_spring_const=55000.0,
            front_anti_roll=22000.0,
            rear_anti_roll=12000.0,
            front_damping_comp=3500.0,
            front_damping_rebound=4200.0,
        ),
    )
    assert out.wheels[2].tyre_pressure_kpa == pytest.approx(175.0)
    assert out.wheels[3].tyre_pressure_kpa == pytest.approx(175.0)
    assert out.wheels[0].tyre_pressure_kpa == pytest.approx(185.0)
    assert out.wheels[1].tyre_pressure_kpa == pytest.approx(185.0)
    assert out.wheels[2].spring_const == pytest.approx(50000.0)
    assert out.wheels[0].spring_const == pytest.approx(55000.0)
    assert out.wheels[3].anti_roll == pytest.approx(22000.0)
    assert out.wheels[1].anti_roll == pytest.approx(12000.0)
    assert out.wheels[2].damping_comp == pytest.approx(3500.0)
    assert out.wheels[3].damping_rebound == pytest.approx(4200.0)
    # Rear dampers untouched ⇒ keep baseline values.
    assert out.wheels[0].damping_comp == pytest.approx(3200.0)
    assert out.wheels[1].damping_rebound == pytest.approx(3800.0)


def test_scalar_overrides_brakes_and_drivetrain() -> None:
    baseline = _synthetic_baseline()
    out = apply(
        baseline,
        SetupOverrides(
            brake_strength_nm=1250.0,
            brake_balance_front=0.55,
            parallel_steer=0.75,
            final_drive=4.3,
            drivetrain_efficiency=0.90,
            passengers=2,
            weight_dist_front=0.48,
            fuel_capacity_l=35.0,
        ),
    )
    assert out.brake_strength_nm == pytest.approx(1250.0)
    assert out.brake_balance_front == pytest.approx(0.55)
    assert out.parallel_steer == pytest.approx(0.75)
    assert out.final_drive == pytest.approx(4.3)
    assert out.drivetrain_efficiency == pytest.approx(0.90)
    assert out.passengers == 2
    assert out.weight_dist_front == pytest.approx(0.48)
    assert out.fuel_capacity_l == pytest.approx(35.0)


def test_gear_ratios_override_preserves_reverse_slot() -> None:
    baseline = _synthetic_baseline()
    new_forward = (3.2, 2.3, 1.7, 1.35, 1.10, 0.92)
    out = apply(baseline, SetupOverrides(gear_ratios=new_forward))
    assert out.gear_ratios[0] == pytest.approx(-3.0)   # reverse kept
    assert out.gear_ratios[1:] == pytest.approx(new_forward)


def test_gear_ratios_wrong_length_raises() -> None:
    baseline = _synthetic_baseline()  # forward_gears == 6
    with pytest.raises(ValueError, match="gear_ratios override"):
        apply(baseline, SetupOverrides(gear_ratios=(3.0, 2.0, 1.5)))


def test_advisor_consumes_overridden_baseline() -> None:
    """Sanity: SetupAdvisor accepts the patched bin and the baseline
    hash reflects the edits (different from raw baseline)."""
    from lfs_telemetry.tnfr_racing.advisor import _hash_baseline

    baseline = _synthetic_baseline()
    edited = apply(
        baseline,
        SetupOverrides(brake_balance_front=0.55, front_spring_const=50000.0),
    )
    h_before = _hash_baseline(baseline)
    h_after = _hash_baseline(edited)
    assert h_before != h_after
    assert len(h_after) == 16
