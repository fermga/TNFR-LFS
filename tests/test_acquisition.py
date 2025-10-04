"""Acquisition pipeline regression tests."""

from __future__ import annotations

from io import StringIO

import pytest

from tnfr_lfs.acquisition.outsim_client import DEFAULT_SCHEMA, OutSimClient


def test_outsim_ingest_captures_per_wheel_slip_and_radius() -> None:
    schema = DEFAULT_SCHEMA
    header = ",".join(schema.columns)
    values = {
        "timestamp": "0.1",
        "structural_timestamp": "0.1",
        "vertical_load": "5200",
        "slip_ratio": "0.02",
        "slip_ratio_fl": "0.03",
        "slip_ratio_fr": "0.01",
        "slip_ratio_rl": "0.02",
        "slip_ratio_rr": "0.02",
        "lateral_accel": "1.5",
        "longitudinal_accel": "-0.2",
        "yaw": "0.05",
        "pitch": "0.01",
        "roll": "0.0",
        "brake_pressure": "12.5",
        "locking": "0.0",
        "nfr": "510.0",
        "si": "0.82",
        "speed": "45.0",
        "yaw_rate": "0.25",
        "slip_angle": "0.08",
        "slip_angle_fl": "0.11",
        "slip_angle_fr": "0.06",
        "slip_angle_rl": "0.05",
        "slip_angle_rr": "0.04",
        "steer": "0.12",
        "throttle": "0.3",
        "gear": "3",
        "vertical_load_front": "2600",
        "vertical_load_rear": "2600",
        "mu_eff_front": "1.2",
        "mu_eff_rear": "1.1",
        "mu_eff_front_lateral": "1.15",
        "mu_eff_front_longitudinal": "1.05",
        "mu_eff_rear_lateral": "1.08",
        "mu_eff_rear_longitudinal": "1.0",
        "suspension_travel_front": "0.03",
        "suspension_travel_rear": "0.04",
        "suspension_velocity_front": "0.5",
        "suspension_velocity_rear": "0.4",
        "tyre_temp_fl": "88.0",
        "tyre_temp_fr": "87.5",
        "tyre_temp_rl": "85.0",
        "tyre_temp_rr": "84.5",
        "tyre_pressure_fl": "1.45",
        "tyre_pressure_fr": "1.43",
        "tyre_pressure_rl": "1.42",
        "tyre_pressure_rr": "1.41",
        "instantaneous_radius": "14.0",
        "front_track_width": "1.46",
        "wheelbase": "2.74",
    }
    payload = ",".join(values[column] for column in schema.columns)
    buffer = StringIO(f"{header}\n{payload}\n")
    client = OutSimClient(schema=schema)
    records = client.ingest(buffer)
    assert len(records) == 1
    record = records[0]
    assert record.slip_angle_fl == pytest.approx(0.11)
    assert record.slip_angle_fr == pytest.approx(0.06)
    assert record.instantaneous_radius == pytest.approx(14.0)
    assert record.front_track_width == pytest.approx(1.46)
    assert record.wheelbase == pytest.approx(2.74)
