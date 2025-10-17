from __future__ import annotations

import math

import numpy as np

import pytest

from tests.helpers import build_resonance_record

from tnfr_core.resonance import analyse_modal_resonance
from tnfr_core.metrics import resonance as resonance_module


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
        records.append(build_resonance_record(t, yaw=yaw, pitch=pitch, roll=roll))

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


def test_estimate_excitation_frequency_uses_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rate = 25.0
    dt = 1.0 / sample_rate
    frequency = 1.5
    records = []
    for index in range(256):
        t = index * dt
        steer = math.sin(2.0 * math.pi * frequency * t)
        suspension = 0.5 * math.sin(2.0 * math.pi * frequency * t + math.pi / 4.0)
        records.append(
            build_resonance_record(
                t,
                steer=steer,
                suspension_velocity_front=suspension,
                suspension_velocity_rear=suspension,
            )
        )

    from tnfr_core.metrics import spectrum as spectrum_module

    captured: dict[str, object] = {}
    real_power_spectrum = spectrum_module.power_spectrum

    def _tracking_power_spectrum(values, rate, *, xp_module=None):
        captured["xp_module"] = xp_module
        return real_power_spectrum(values, rate, xp_module=xp_module)

    monkeypatch.setattr(resonance_module, "power_spectrum", _tracking_power_spectrum)

    dominant = resonance_module.estimate_excitation_frequency(records, sample_rate)

    assert dominant > 0.0
    assert captured.get("xp_module") is resonance_module.xp


def test_extract_peaks_classification_is_preserved() -> None:
    spectrum = [
        (0.8, 20.0),
        (3.2, 12.0),
        (7.5, 15.0),
        (1.5, 5.0),
    ]

    peaks = resonance_module._extract_peaks(spectrum, max_peaks=3)

    assert [peak.classification for peak in peaks] == [
        "useful",
        "parasitic",
        "useful",
    ]
    assert [pytest.approx(peak.energy) for peak in peaks] == [
        pytest.approx(20.0),
        pytest.approx(15.0),
        pytest.approx(12.0),
    ]

    reduced_energy_spectrum = [
        (0.8, 20.0),
        (3.2, 9.5),
        (1.5, 5.0),
    ]

    reduced_peaks = resonance_module._extract_peaks(reduced_energy_spectrum, max_peaks=3)

    assert [peak.classification for peak in reduced_peaks] == [
        "useful",
        "parasitic",
        "parasitic",
    ]


def test_extract_peaks_large_input_uses_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    xp_backend = resonance_module.xp
    original_argpartition = xp_backend.argpartition
    call_counter = {"count": 0}

    def _tracking_argpartition(values, kth, *args, **kwargs):
        call_counter["count"] += 1
        return original_argpartition(values, kth, *args, **kwargs)

    monkeypatch.setattr(xp_backend, "argpartition", _tracking_argpartition)

    total_points = 16_384
    frequencies = np.linspace(0.1, 250.0, total_points)
    energies = np.linspace(1.0, float(total_points), total_points)
    spectrum = list(zip(frequencies.tolist(), energies.tolist()))

    peaks = resonance_module._extract_peaks(spectrum, max_peaks=5)

    assert call_counter["count"] >= 1

    expected_indices = np.argsort(energies)[-5:][::-1]
    expected_frequencies = frequencies[expected_indices]
    expected_energies = energies[expected_indices]

    assert [pytest.approx(peak.frequency) for peak in peaks] == [
        pytest.approx(value) for value in expected_frequencies
    ]
    assert [pytest.approx(peak.energy) for peak in peaks] == [
        pytest.approx(value) for value in expected_energies
    ]
