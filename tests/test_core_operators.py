"""Unit tests for the high-level TNFR-LFS operators."""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev

import pytest

from tnfr_lfs.core import TelemetryRecord
from tnfr_lfs.core.operators import (
    acoplamiento_operator,
    coherence_operator,
    dissonance_operator,
    evolve_epi,
    emission_operator,
    orchestrate_delta_metrics,
    recepcion_operator,
    recursividad_operator,
    resonance_operator,
)


def _build_record(
    timestamp: float,
    vertical_load: float,
    slip_ratio: float,
    lateral_accel: float,
    longitudinal_accel: float,
    nfr: float,
    si: float,
) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=timestamp,
        vertical_load=vertical_load,
        slip_ratio=slip_ratio,
        lateral_accel=lateral_accel,
        longitudinal_accel=longitudinal_accel,
        nfr=nfr,
        si=si,
    )


def test_coherence_operator_reduces_jitter_without_bias():
    raw_series = [0.2, 0.8, 0.1, 0.9, 0.2]
    smoothed = coherence_operator(raw_series, window=3)

    assert mean(smoothed) == pytest.approx(mean(raw_series), rel=1e-9)
    assert pstdev(smoothed) < pstdev(raw_series)


def test_evolve_epi_runs_euler_step():
    prev = 0.45
    deltas = {"tyres": 1.2, "suspension": -0.6, "driver": 0.4}
    nu_map = {"tyres": 0.18, "suspension": 0.14, "driver": 0.05}
    new_epi, derivative = evolve_epi(prev, deltas, 0.1, nu_map)

    expected_derivative = 0.18 * 1.2 + 0.14 * -0.6 + 0.05 * 0.4
    assert derivative == pytest.approx(expected_derivative, rel=1e-9)
    assert new_epi == pytest.approx(prev + expected_derivative * 0.1, rel=1e-9)


def test_orchestrator_pipeline_builds_consistent_metrics():
    segment_a = [
        _build_record(0.0, 5200.0, 0.05, 1.2, 0.6, 0.82, 0.91),
        _build_record(1.0, 5100.0, 0.04, 1.1, 0.5, 0.81, 0.92),
    ]
    segment_b = [
        _build_record(2.0, 5000.0, 0.03, 1.0, 0.4, 0.80, 0.90),
        _build_record(3.0, 4950.0, 0.02, 0.9, 0.35, 0.79, 0.88),
    ]

    results = orchestrate_delta_metrics(
        [segment_a, segment_b],
        target_delta_nfr=0.0,
        target_sense_index=0.9,
    )

    assert results["objectives"]["sense_index"] == pytest.approx(0.9)
    assert len(results["bundles"]) == 4
    assert len(results["delta_nfr_series"]) == 4
    assert len(results["sense_index_series"]) == 4
    assert results["dissonance"] >= 0.0
    assert -1.0 <= results["coupling"] <= 1.0
    assert 0.0 <= results["resonance"] <= 1.0
    assert len(results["recursive_trace"]) == 4


def test_orchestrator_consumes_fixture_segments(synthetic_records):
    segments = [synthetic_records[:9], synthetic_records[9:]]

    report = orchestrate_delta_metrics(
        segments,
        target_delta_nfr=0.5,
        target_sense_index=0.82,
    )

    assert report["objectives"]["delta_nfr"] == pytest.approx(0.5)
    assert len(report["bundles"]) == len(synthetic_records)
    assert pytest.approx(report["sense_index"], rel=1e-6) == mean(report["sense_index_series"])
    assert len(report["recursive_trace"]) == len(synthetic_records)


def test_emission_operator_clamps_sense_index():
    objectives = emission_operator(target_delta_nfr=0.5, target_sense_index=1.5)

    assert objectives["delta_nfr"] == pytest.approx(0.5)
    assert objectives["sense_index"] == 1.0


def test_recepcion_operator_wraps_epi_extractor():
    records = [
        _build_record(0.0, 5000.0, 0.1, 1.0, 0.5, 0.8, 0.9),
        _build_record(1.0, 5050.0, 0.09, 1.1, 0.6, 0.81, 0.91),
    ]

    bundles = recepcion_operator(records)

    assert len(bundles) == 2
    assert isinstance(bundles[0].epi, float)


def test_recursividad_operator_requires_decay_in_range():
    with pytest.raises(ValueError):
        recursividad_operator([0.1, 0.2], decay=1.0)


def test_acoplamiento_and_resonance_behaviour():
    series_a = [0.1, 0.2, 0.3, 0.4]
    series_b = [0.1, 0.15, 0.25, 0.35]

    coupling = acoplamiento_operator(series_a, series_b)
    resonance = resonance_operator(series_b)
    dissonance = dissonance_operator(series_a, target=0.25)

    assert coupling > 0
    expected_resonance = sqrt(mean(value * value for value in series_b))
    assert resonance == pytest.approx(expected_resonance, rel=1e-9)
    assert dissonance == pytest.approx(mean(abs(value - 0.25) for value in series_a))

