"""Tests for :mod:`lfs_telemetry.telemetry.fuel_tracker`."""

from __future__ import annotations

from lfs_telemetry.telemetry.fuel_tracker import FuelTracker


def test_no_estimate_before_first_lap():
    ft = FuelTracker()
    ft.observe_fuel(50.0)
    assert ft.avg_burn_pct_per_lap is None
    assert ft.laps_remaining(50.0) is None


def test_burn_per_lap_simple():
    ft = FuelTracker(window=3)
    ft.observe_lap(50.0)  # initial
    ft.observe_lap(48.0)  # 2.0 burn
    ft.observe_lap(46.0)  # 2.0 burn
    assert abs(ft.avg_burn_pct_per_lap - 2.0) < 1e-9
    assert abs(ft.laps_remaining(46.0) - 23.0) < 1e-9


def test_window_limits_history():
    ft = FuelTracker(window=2)
    ft.observe_lap(60.0)
    ft.observe_lap(58.0)  # 2
    ft.observe_lap(54.0)  # 4
    ft.observe_lap(50.0)  # 4
    # window=2 → average over last two: (4+4)/2 = 4
    assert abs(ft.avg_burn_pct_per_lap - 4.0) < 1e-9


def test_refuel_ignored():
    ft = FuelTracker()
    ft.observe_lap(40.0)
    ft.observe_lap(38.0)  # 2 burn
    ft.observe_lap(80.0)  # refuel → ignore
    ft.observe_lap(78.0)  # 2 burn
    assert abs(ft.avg_burn_pct_per_lap - 2.0) < 1e-9


def test_observe_fuel_does_not_advance_when_lap_seen():
    ft = FuelTracker()
    ft.observe_fuel(50.0)
    ft.observe_lap(48.0)  # 2 burn vs the initial fuel observation
    assert abs(ft.avg_burn_pct_per_lap - 2.0) < 1e-9


def test_laps_remaining_zero_burn_safe():
    ft = FuelTracker()
    ft.observe_lap(50.0)
    ft.observe_lap(50.0)  # zero burn → not added
    assert ft.avg_burn_pct_per_lap is None
    assert ft.laps_remaining(50.0) is None
