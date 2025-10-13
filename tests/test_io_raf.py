from __future__ import annotations

import math

import pytest

from tnfr_lfs.telemetry.offline import raf_to_telemetry_records, read_raf


@pytest.fixture(scope="session")
def _raf_sample(raf_sample_path):
    return read_raf(raf_sample_path)


def test_read_raf_parses_sample_metadata(_raf_sample):
    raf = _raf_sample
    header = raf.header

    assert header.magic == "LFSRAF"
    assert header.player_name == "NE[BRM] naiqun"
    assert header.car_model == "XF GTI"
    assert header.track_name == "Blackwood"
    assert header.track_layout == "GP TRACK"
    assert header.interval_seconds == pytest.approx(0.0100078125, rel=1e-9)
    assert header.frame_count == 9586
    assert len(raf.frames) == header.frame_count

    first = raf.frames[0]
    assert first.index == 0
    assert first.gear == 4
    assert first.engine_rpm == pytest.approx(7042.3544437, rel=1e-6)
    assert first.distance == pytest.approx(0.0620864816, rel=1e-6)

    fl, fr, rl, rr = first.wheels
    assert fl.vertical_load == pytest.approx(3056.1862793, rel=1e-6)
    assert fr.vertical_load == pytest.approx(1965.0721436, rel=1e-6)
    assert rl.longitudinal_force == pytest.approx(830.2507324, rel=1e-6)
    assert rr.suspension_deflection == pytest.approx(0.1091849208, rel=1e-6)

    last = raf.frames[-1]
    assert last.index == header.frame_count - 1
    assert last.gear == 4
    assert last.distance == pytest.approx(3296.1804199, rel=1e-6)


def test_raf_to_telemetry_records_exposes_wheel_data(_raf_sample):
    records = raf_to_telemetry_records(_raf_sample)

    assert len(records) == _raf_sample.header.frame_count

    first = records[0]
    assert first.gear == 4
    assert first.vertical_load_front == pytest.approx(5021.2584229, rel=1e-6)
    assert first.vertical_load_rear == pytest.approx(6064.3896484, rel=1e-6)
    assert first.wheel_load_fl == pytest.approx(3056.1862793, rel=1e-6)
    assert first.wheel_longitudinal_force_rr == pytest.approx(901.4348755, rel=1e-6)
    assert first.suspension_travel_front == pytest.approx(0.0124925393, rel=1e-6)
    assert math.isnan(first.tyre_temp_fl)
    assert math.isnan(first.tyre_temp_rr)
