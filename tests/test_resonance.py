from __future__ import annotations

import math

import pytest

from tnfr_lfs.core.epi import TelemetryRecord
from tnfr_lfs.core.resonance import analyse_modal_resonance


def _make_record(timestamp: float, yaw: float, pitch: float, roll: float) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=timestamp,
        vertical_load=0.0,
        slip_ratio=0.0,
        lateral_accel=0.0,
        longitudinal_accel=0.0,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        brake_pressure=0.0,
        locking=0.0,
        nfr=0.0,
        si=0.0,
        speed=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
        steer=0.0,
        throttle=0.0,
        gear=0,
        vertical_load_front=0.0,
        vertical_load_rear=0.0,
        mu_eff_front=0.0,
        mu_eff_rear=0.0,
        suspension_travel_front=0.0,
        suspension_travel_rear=0.0,
        suspension_velocity_front=0.0,
        suspension_velocity_rear=0.0,
    )


@pytest.mark.parametrize(
    "freq_yaw,freq_roll,freq_pitch",
    [(1.25, 6.0, 0.35)],
)
def test_resonance_identifies_modal_peaks(
    freq_yaw: float, freq_roll: float, freq_pitch: float
) -> None:
    sample_rate = 20.0
    dt = 1.0 / sample_rate
    samples = 512
    records = []
    for index in range(samples):
        t = index * dt
        yaw = 0.6 * math.sin(2.0 * math.pi * freq_yaw * t)
        yaw += 0.1 * math.sin(2.0 * math.pi * 3.5 * t)
        roll = 0.4 * math.sin(2.0 * math.pi * freq_roll * t)
        roll += 0.05 * math.sin(2.0 * math.pi * 1.2 * t)
        pitch = 0.5 * math.sin(2.0 * math.pi * freq_pitch * t)
        pitch += 0.05 * math.sin(2.0 * math.pi * 2.5 * t)
        records.append(_make_record(t, yaw=yaw, pitch=pitch, roll=roll))

    analysis = analyse_modal_resonance(records)

    yaw_peaks = analysis["yaw"].peaks
    assert yaw_peaks
    dominant_yaw = yaw_peaks[0]
    assert dominant_yaw.classification == "useful"
    assert dominant_yaw.frequency == pytest.approx(freq_yaw, rel=0.1)

    roll_peaks = analysis["roll"].peaks
    assert roll_peaks
    dominant_roll = roll_peaks[0]
    assert dominant_roll.classification == "parasitic"
    assert dominant_roll.frequency == pytest.approx(freq_roll, rel=0.1)

    pitch_peaks = analysis["pitch"].peaks
    assert pitch_peaks
    dominant_pitch = pitch_peaks[0]
    assert dominant_pitch.classification == "useful"
    assert dominant_pitch.frequency == pytest.approx(freq_pitch, rel=0.1)

    # The secondary components should be classified as parasitic modes due to
    # their reduced energy contribution compared to the dominant frequency.
    if len(pitch_peaks) > 1:
        assert any(peak.classification == "parasitic" for peak in pitch_peaks[1:])

    assert analysis["yaw"].sample_rate == pytest.approx(sample_rate, rel=0.05)
